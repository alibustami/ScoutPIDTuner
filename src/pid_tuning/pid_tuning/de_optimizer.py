import os
import sys
import time
import logging
import math
from datetime import datetime
from functools import lru_cache
from random import randint
from typing import Dict, List, OrderedDict, Tuple, Union

import numpy as np
if not hasattr(np, 'float'): # noqa
    np.float = float # noqa
import pandas as pd
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from scipy.optimize import NonlinearConstraint, differential_evolution
from tf_transformations import euler_from_quaternion

from .tuner_helpers import (
    calculate_relative_overshoot,
    calculate_rise_time,
    calculate_settling_time,
    results_columns,
)
from .tuner_loader import load_init_states

logger = logging.getLogger(__name__)


class DifferentialEvolutionOptimizer:
    """
    Optimize PID gains (Kp, Ki, Kd) using Differential Evolution.
    
    Each time we try a set of gains, we spin up a short-lived node that:
      - Publishes /cmd_vel to rotate the robot in place
      - Subscribes to /odom to capture heading data
      - Applies the Kp, Ki, Kd logic inline
      - Collects angle values for offline metric analysis
    """

    def __init__(
        self,
        parameters_bounds: Dict[str, Tuple[float, float]],
        constraint: OrderedDict[str, Tuple[float, float]] = None,
        n_iter: int = 50,
        experiment_total_run_time: int = 5000,  # ms
        experiment_values_dump_rate: int = 100,  # ms
        set_point: float = 90.0,
        selected_init_state: int = 0,
        objective_value_limit_early_stop: float = 2500.0,
        selected_config: int = 0,
    ):
        """
        Initialize the optimizer.

        parameters_bounds: dict of ("Kp", "Ki", "Kd") -> (lower, upper)
        constraint: optional OrderedDict for constraints, e.g. overshoot, rise_time
        n_iter: max generations for DE
        experiment_total_run_time: how long (ms) to run each rotation attempt
        experiment_values_dump_rate: how often (ms) we gather data
        set_point: the target angle to rotate in place
        selected_init_state: index in init_states.json if you use them
        objective_value_limit_early_stop: if the objective is below that, we stop
        selected_config: index for picking DE hyperparameters
        """

        self.parameters_bounds = parameters_bounds
        self.constraint = constraint
        self.n_iter = n_iter

        self.set_point = set_point
        self.selected_init_state = selected_init_state
        self.objective_value_limit_early_stop = objective_value_limit_early_stop
        self.selected_config = selected_config
        self.experiment_total_run_time = experiment_total_run_time
        self.experiment_values_dump_rate = experiment_values_dump_rate

        # For logging results
        self.experiment_id = 1
        self.trials_counter = 0
        self.results_df = pd.DataFrame(columns=results_columns)

        # Load initial guess from init_states.json (if you want that)
        init_states = load_init_states("init_states.json")
        self.init_state = init_states[self.selected_init_state]

        # Create a directory for logs
        self.log_dir = "DE-results"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = os.path.join(
            self.log_dir, f"{timestamp}_init_{self.selected_init_state}_de.csv"
        )

    def run(self, exp_start_time: float) -> None:
        """
        Execute the differential evolution process.
        """
        self.exp_start_time = exp_start_time

        # If constraints given, set up for NonlinearConstraint
        if self.constraint:
            lower_constraint_bounds = [self.constraint[c][0] for c in self.constraint]
            upper_constraint_bounds = [self.constraint[c][1] for c in self.constraint]
        else:
            lower_constraint_bounds = None
            upper_constraint_bounds = None

        logger.info("Starting Differential Evolution optimization...")
        logger.info(f"Parameter bounds: {self.parameters_bounds}")

        # Choose (mutation, recombination) combos
        configs = [
            (0.6, 0.6),
            (0.8, 0.3),
            (0.5, 0.9),
        ]
        if self.selected_config < 0 or self.selected_config >= len(configs):
            raise ValueError("selected_config must be 0, 1, or 2")

        selected_mutation, selected_recombination = configs[self.selected_config]

        # Example initial population
        initial_population = np.array(
            [
                self.init_state,
                [randint(1, 25), randint(0, 100) / 100.0, randint(0, 100) / 100.0],
                [randint(1, 25), randint(0, 100) / 100.0, randint(0, 100) / 100.0],
                [randint(1, 25), randint(0, 100) / 100.0, randint(0, 100) / 100.0],
                [randint(1, 25), randint(0, 100) / 100.0, randint(0, 100) / 100.0],
            ],
            dtype=np.float32
        )

        # Build the 'constraints' argument if user provided constraints
        if lower_constraint_bounds and upper_constraint_bounds:
            constraint_obj = NonlinearConstraint(
                fun=self.constraint_function,
                lb=lower_constraint_bounds,
                ub=upper_constraint_bounds,
            )
        else:
            constraint_obj = None

        self.optimizer = differential_evolution(
            func=self.objective_function,
            bounds=list(self.parameters_bounds.values()),
            maxiter=self.n_iter,
            popsize=15,
            init=initial_population,
            strategy="rand1bin",
            mutation=selected_mutation,
            recombination=selected_recombination,
            workers=1,             # run in a single process
            disp=True,
            tol=0,
            atol=0,
            polish=False,
            constraints=constraint_obj,
            callback=self.results_callback,
        )

        # Final summary
        self.finalize(self.optimizer.x, self.optimizer.fun)

    def constraint_function(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate constraints, e.g. overshoot, rise_time must be within certain bounds.
        """
        self.trials_counter += 1
        kp, ki, kd = inputs
        _, angle_values = self._run_experiment((kp, ki, kd))

        overshoot = calculate_relative_overshoot(angle_values, self.set_point)
        rise_time = calculate_rise_time(angle_values, self.set_point)

        logger.info(f"[Constraint] Trial {self.trials_counter}, Gains=({kp:.2f},{ki:.2f},{kd:.2f}), "
                    f"Overshoot={overshoot:.2f}, RiseTime={rise_time:.2f}")

        return np.array([overshoot, rise_time], dtype=float)

    def objective_function(self, inputs: np.ndarray) -> float:
        """
        The cost function: for example, we want to minimize settling time.
        """
        kp, ki, kd = inputs
        logger.info("-"*40)
        logger.info(f"[Objective] Gains=({kp:.2f}, {ki:.2f}, {kd:.2f})")

        _, angle_values = self._run_experiment((kp, ki, kd))
        st = calculate_settling_time(angle_values, final_value=self.set_point, tolerance=0.05)
        overshoot = calculate_relative_overshoot(angle_values, self.set_point)
        rise_time = calculate_rise_time(angle_values, self.set_point)

        self.log_trial_results(
            kp=kp,
            ki=ki,
            kd=kd,
            overshoot=overshoot,
            rise_time=rise_time,
            settling_time=st,
            angle_values=angle_values,
            set_point=self.set_point,
        )
        # Lower = better
        return float(st)

    def results_callback(self, x: np.ndarray, convergence: float):
        """
        Called each iteration. We can do an early-stop check here.
        """
        kp, ki, kd = x
        logger.info(f"DE iteration callback: Gains=({kp:.2f}, {ki:.2f}, {kd:.2f})")

        # Check if we're good enough
        _, angle_values = self._run_experiment((kp, ki, kd))
        st = calculate_settling_time(angle_values, self.set_point, 0.05)
        if st <= self.objective_value_limit_early_stop:
            logger.info("Early stopping condition met!")
            self.finalize(x, st)
            sys.exit(0)

    @lru_cache(maxsize=None)
    def _run_experiment(self, gains: Tuple[float, float, float]) -> Tuple[List[float], List[float]]:
        """
        Core logic to test a single set of (Kp, Ki, Kd).

        We'll replicate the 'rotate_in_place' approach inline:
         1) Create a short-lived node
         2) Subscribe to /odom to track yaw
         3) Publish cmd_vel using a simple PID
         4) Run for a set duration or until within tolerance
         5) Return angle_values list for offline metrics
        """

        # Gains
        kp, ki, kd = gains
        # ANGULAR_SPEED_LIMIT = 100.0   # or whatever
        TOLERANCE_DEG = 2.0
        target_angle_deg = self.set_point

        # We'll store heading data in a local list:
        angle_values = []

        # 1) Create a short-lived ROS node
        # rclpy.init()
        node = ShortExperimentNode(kp, ki, kd, target_angle_deg, TOLERANCE_DEG)
        
        # 2) Spin for the desired total_run_time (ms) or until done
        #    We'll break early if the node sets "done" flag
        end_time = time.time() + (self.experiment_total_run_time / 1000.0)

        while rclpy.ok() and (time.time() < end_time) and not node.done_rotating:
            rclpy.spin_once(node, timeout_sec=0.05)
            # collect current heading
            angle_values.append(node.current_heading_deg)

        # 3) Ensure robot is stopped
        node.stop_robot()

        # 4) Destroy the node to end the experiment
        node.destroy_node()
        # rclpy.shutdown()

        # error_values is simply angle - set_point
        # so for each angle in angle_values, error = angle - set_point
        # We'll skip storing the entire error array here, just return angle_values
        error_values = [angle - self.set_point for angle in angle_values]
        return error_values, angle_values

    def log_trial_results(
        self,
        kp: float,
        ki: float,
        kd: float,
        overshoot: float,
        rise_time: float,
        settling_time: float,
        angle_values: List[float],
        set_point: float,
    ):
        """
        Store results in a DataFrame and output to CSV.
        """
        row = {
            "experiment_id": self.experiment_id,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "overshoot": overshoot, 
            "rise_time": rise_time,
            "settling_time": settling_time,
            "angle_values": [angle_values],  # store entire list in one cell
            "set_point": set_point,
        }
        self.results_df = pd.concat([self.results_df, pd.DataFrame([row])], ignore_index=True)
        self.experiment_id += 1
        self.results_df.to_csv(self.file_path, index=False)

    def finalize(self, x: np.ndarray, settling_time: float):
        """
        End-of-optimization summary.
        """
        exp_end_time = time.time()
        total_exp_time = exp_end_time - self.exp_start_time

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        txt_file_path = os.path.join(
            self.log_dir, f"{timestamp}_init_{self.selected_init_state}_de.txt"
        )
        summary = {
            "parameters_bounds": self.parameters_bounds,
            "constraint": self.constraint,
            "n_iter": self.n_iter,
            "n_trials": self.trials_counter,
            "experiment_total_run_time": self.experiment_total_run_time,
            "experiment_values_dump_rate": self.experiment_values_dump_rate,
            "selected_init_state": self.selected_init_state,
            "objective_value_limit_early_stop": self.objective_value_limit_early_stop,
            "total_exp_time": total_exp_time,
            "best_solution": x.tolist(),
            "final_settling_time": settling_time,
        }

        with open(txt_file_path, "w") as f:
            f.write(str(summary))
        logger.info("Optimization finished.")
        logger.info(f"Results summary saved to {txt_file_path}")
        self._run_experiment.cache_clear()


class ShortExperimentNode(Node):
    """
    A short-lived node to replicate the 'rotate_in_place' logic inline,
    but resets the yaw angle each time you call 'reset_experiment()'.
    """

    def __init__(self, kp, ki, kd, target_angle_deg, tolerance_deg):
        super().__init__('short_experiment_node')

        # Gains & config
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_angle_deg = target_angle_deg
        self.tolerance_deg = tolerance_deg

        # Robot state
        self.current_heading_deg = 0.0
        self.initial_heading_deg = None
        self.offset_deg = None
        self.done_rotating = False

        # PID memory
        self.integral_error = 0.0
        self.previous_error = 0.0

        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # EXAMPLE usage: call reset_experiment() here or whenever you want to re-zero yaw
        self.reset_experiment()

    def reset_experiment(self):
        """
        Resets the experiment so that on the next odom reading,
        we treat that yaw reading as 0.0.
        """
        self.get_logger().info("Resetting experiment state...")
        self.offset_deg = None
        self.done_rotating = False
        self.integral_error = 0.0
        self.previous_error = 0.0

    def odom_callback(self, msg: Odometry):
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = math.degrees(yaw)

        # make between [0, 360]
        if yaw_deg < 0:
            yaw_deg += 360


        # Set offset on first reading after reset
        if self.offset_deg is None:
            self.offset_deg = yaw_deg
            self.initial_heading_deg = 0.0
            self.get_logger().info(f"Offset={yaw_deg:.2f} deg; initial heading=0.0 deg")

        # Current heading (relative to initial)
        self.current_heading_deg = yaw_deg - self.offset_deg

        # Run PID update on each odom message
        self._update_pid_control()

    def _update_pid_control(self):
        # If not yet initialized, skip
        if self.initial_heading_deg is None:
            return

        # Compute error
        target = self.initial_heading_deg + self.target_angle_deg
        target = self._normalize_angle(target)
        error = self._normalize_angle(target - self.current_heading_deg)

        # Check if within tolerance
        if abs(error) <= self.tolerance_deg:
            # Stop the robot
            self.stop_robot()
            self.done_rotating = True
            self.get_logger().info("Rotation complete.")
            return

        # PID calculations
        self.integral_error += error
        d_error = error - self.previous_error
        self.previous_error = error

        p_term = self.kp * error
        i_term = self.ki * self.integral_error
        d_term = self.kd * d_error

        angular_z = p_term + i_term + d_term

        # Publish the computed angular velocity
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = angular_z
        self.cmd_pub.publish(twist_msg)

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)

    def _normalize_angle(self, angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg
