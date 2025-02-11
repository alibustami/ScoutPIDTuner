import logging
import math
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from random import randint
from typing import Dict, List, OrderedDict, Tuple, Union
import logging

logger = logging.getLogger(__name__)
import numpy as np

if not hasattr(np, "float"):  # noqa
    np.float = float  # noqa


# matplotlib.use("TkAgg")  # noqa
import matplotlib.pyplot as plt
import pandas as pd
import rclpy
from scipy.optimize import NonlinearConstraint, differential_evolution

from .experiment_node import ShortExperimentNode
from .launcher import Ros2Launcher
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
        self.objective_value_limit_early_stop = (
            objective_value_limit_early_stop
        )
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
        self.last_experiment_set_point = None
        self.launcher = Ros2Launcher(can_port="can1")

    def run(self, exp_start_time: float) -> None:
        """
        Execute the differential evolution process.
        """
        self.exp_start_time = exp_start_time

        # If constraints given, set up for NonlinearConstraint
        if self.constraint:
            lower_constraint_bounds = [
                self.constraint[c][0] for c in self.constraint
            ]
            upper_constraint_bounds = [
                self.constraint[c][1] for c in self.constraint
            ]
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

        selected_mutation, selected_recombination = configs[
            self.selected_config
        ]

        # Example initial population
        initial_population = np.array(
            [
                self.init_state,
                [
                    randint(1, 25),
                    randint(0, 100) / 100.0,
                    randint(0, 100) / 100.0,
                ],
                [
                    randint(1, 25),
                    randint(0, 100) / 100.0,
                    randint(0, 100) / 100.0,
                ],
                [
                    randint(1, 25),
                    randint(0, 100) / 100.0,
                    randint(0, 100) / 100.0,
                ],
                [
                    randint(1, 25),
                    randint(0, 100) / 100.0,
                    randint(0, 100) / 100.0,
                ],
            ],
            dtype=np.float32,
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
        logger.info(f"Constraints lOWER: {lower_constraint_bounds}")
        logger.info(f"Constraints UPPER: {upper_constraint_bounds}")

        self.optimizer = differential_evolution(
            func=self.objective_function,
            bounds=list(self.parameters_bounds.values()),
            maxiter=self.n_iter,
            popsize=15,
            init=initial_population[:],
            strategy="rand1bin",
            mutation=selected_mutation,
            recombination=selected_recombination,
            workers=1,  # run in a single process
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
        plt.plot(angle_values, color='b')
        plt.axhline(y=self.set_point, color="r", linestyle="--")
        # add window title for the GUI popup
        plt.title(f"Trial {self.trials_counter}")
        plt.show()


        overshoot = calculate_relative_overshoot(angle_values, self.set_point)
        rise_time = calculate_rise_time(angle_values, self.set_point)

        logger.info(
            f"[Constraint] Trial {self.trials_counter}, Gains=({kp:.2f},{ki:.2f},{kd:.2f}), "
            f"Overshoot={overshoot:.2f}, RiseTime={rise_time:.2f}"
        )

        return np.array([overshoot, rise_time], dtype=float)

    def objective_function(self, inputs: np.ndarray) -> float:
        """
        The cost function: for example, we want to minimize settling time.
        """
        kp, ki, kd = inputs
        logger.info("-" * 40)
        logger.info(f"[Objective] Gains=({kp:.2f}, {ki:.2f}, {kd:.2f})")

        _, angle_values = self._run_experiment((kp, ki, kd))
        st = calculate_settling_time(
            angle_values,
            final_value=self.last_experiment_set_point,
            tolerance=0.05,
        )
        overshoot = calculate_relative_overshoot(
            angle_values, self.last_experiment_set_point
        )
        rise_time = calculate_rise_time(
            angle_values, self.last_experiment_set_point
        )

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
        logger.info(
            f"DE iteration callback: Gains=({kp:.2f}, {ki:.2f}, {kd:.2f})"
        )

        # Check if we're good enough
        _, angle_values = self._run_experiment((kp, ki, kd))
        st = calculate_settling_time(angle_values, self.set_point, 0.05)
        if st <= self.objective_value_limit_early_stop:
            logger.info("Early stopping condition met!")
            self.finalize(x, st)
            sys.exit(0)

    @lru_cache(maxsize=None)
    def _run_experiment(
        self, gains: Tuple[float, float, float]
    ) -> Tuple[List[float], List[float]]:
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
        self.launcher.launch_ros2()
        time.sleep(3)
        node = ShortExperimentNode(kp, ki, kd, TOLERANCE_DEG)
        # time.sleep(1)

        # 2) Spin for the desired total_run_time (ms) or until done
        #    We'll break early if the node sets "done" flag
        end_time = time.time() + (self.experiment_total_run_time / 1000.0)

        while (
            rclpy.ok() and (time.time() < end_time)
            # rclpy.ok() and (time.time() < end_time) and not node.done_rotating
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
            angle_values.append(node.continuous_yaw_deg)
        self.last_experiment_set_point = node.target_angle_deg
        error_values = [
            angle - self.last_experiment_set_point for angle in angle_values
        ]
        node.stop_robot()

        node.destroy_node()
        time.sleep(3)

        # rclpy.shutdown()
        self.launcher.kill_ros2_launch()
        time.sleep(3)
        ## Log the error values
        # logger.info(f"Error values: {error_values}")
        logger.info(f"Angle values: {angle_values}\n")
        
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
            "set_point": self.last_experiment_set_point,
        }
        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame([row])], ignore_index=True
        )
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
