import logging
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from random import randint
from typing import Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy

if not hasattr(np, "float"):  # noqa
    np.float = float  # noqa
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
from bayes_opt.event import Events
from scipy.optimize import NonlinearConstraint

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


class BayesianOptimizer:
    """
    Bayesian Optimizer for PID Gains (Kp, Ki, Kd).

    Each new trial calls `_run_experiment()` to:
      - Create a short-lived node
      - Publish & subscribe to gather headings
      - Apply a simple PID in real time or do it externally
      - Collect `angle_values` for metrics
    """

    def __init__(
        self,
        parameters_bounds: Dict[str, Tuple[float, float]],
        constraint: OrderedDict[str, Tuple[float, float]] = None,
        set_point: float = 90.0,
        n_iter: int = 50,
        experiment_total_run_time: int = 10000,  # ms
        experiment_values_dump_rate: int = 100,  # ms
        selected_init_state: int = 0,
        objective_value_limit_early_stop: float = 2500.0,
        selected_config: int = 0,
    ):
        """
        Initialize the BayesianOptimizer.

        set_point: float, the target angle.
        parameters_bounds: dict of { 'Kp': (low, high), 'Ki': (low, high), 'Kd': (low, high) }
        constraint: optional overshoot/rise_time bounds (OrderedDict).
        n_iter: how many steps of Bayesian Optimization to run.
        experiment_total_run_time: ms to gather data each trial.
        experiment_values_dump_rate: ms between data samples (unused in short-lifetime node).
        selected_init_state: pick an initial guess from init_states.json if you like.
        objective_value_limit_early_stop: we can stop early if we meet this criterion.
        selected_config: picks a `xi` param for the acquisition function.
        """

        self.set_point = set_point
        self.parameters_bounds = parameters_bounds
        self.constraint = constraint
        self.n_iter = n_iter

        self.experiment_id = 1
        self.experiment_total_run_time = experiment_total_run_time
        self.experiment_values_dump_rate = experiment_values_dump_rate

        self.selected_init_state = selected_init_state
        init_states = load_init_states("init_states.json")
        self.init_state = init_states[self.selected_init_state]

        self.objective_value_limit_early_stop = (
            objective_value_limit_early_stop
        )
        if not (0 <= selected_config < 3):
            raise ValueError("selected_config must be one of [0, 1, 2]")
        self.selected_config = selected_config

        # Set up the internal bayes_opt object
        self._init_optimizer()

        # DataFrame to store experiment logs
        self.results_df = pd.DataFrame(columns=results_columns)

        # Create directory for logs target angle
        self.log_dir = "BO-results"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = os.path.join(
            self.log_dir,
            f"{timestamp}_init_{self.selected_init_state}_bo.csv",
        )
        self.trials_counter = 0
        self.last_experiment_set_point = None
        self.launcher = Ros2Launcher(can_port="can1")

    def run(self, exp_start_time: float) -> None:
        """
        Run the Bayesian Optimization up to n_iter steps.
        """
        self.exp_start_time = exp_start_time

        # Provide an initial guess to the BO
        self.optimizer.probe(
            params={
                "Kp": self.init_state[0],
                "Ki": self.init_state[1],
                "Kd": self.init_state[2],
            },
            lazy=True,
        )

        # TODO: add probes 4 cases
        # total_pop?

        # Start the optimization
        init_points = 5  # how many random points before the real optimization
        self.optimizer.maximize(n_iter=self.n_iter, init_points=init_points)

        best_result = self.optimizer.max
        logger.info(f"BayesOpt DONE. Best result: {best_result}")

    def constraint_function(self, **inputs) -> Tuple[float, float]:
        """
        Evaluate constraints, e.g. overshoot & rise time.

        Called by the built-in constraint support from bayes_opt + scipy's NonlinearConstraint.

        Returns
        -------
        (overshoot, rise_time)
        """
        self.trials_counter += 1
        logger.info(f"[Constraint] Checking trial {self.trials_counter}")

        kp, ki, kd = inputs["Kp"], inputs["Ki"], inputs["Kd"]
        _, angle_values = self._run_experiment((kp, ki, kd))
        plt.plot(angle_values, color="b")
        plt.axhline(y=self.set_point, color="r", linestyle="--")

        plt.title(f"Trial {self.trials_counter}")
        plt.draw()
        plt.pause(2)
        plt.close()

        overshoot = calculate_relative_overshoot(angle_values, self.set_point)
        rise_time = calculate_rise_time(angle_values, self.set_point)

        logger.info(
            f"[Constraint] Gains=({kp:.2f},{ki:.2f},{kd:.2f}), Over={overshoot:.2f}, Rise={rise_time:.2f}"
        )
        return overshoot, rise_time

    def objective_function(self, **inputs) -> float:
        """
        The objective function. We want to maximize negative of settling time => effectively minimize settling time.

        The bayes_opt library tries to MAXIMIZE the function. So we return negative if we want it minimized.
        """
        kp, ki, kd = inputs["Kp"], inputs["Ki"], inputs["Kd"]
        _, angle_values = self._run_experiment((kp, ki, kd))
        settling_time = calculate_settling_time(
            angle_values, self.set_point, tolerance=0.05
        )

        # Since BayesOpt does a maximum search, returning negative flips it into a minimum search
        return -settling_time

    def _init_optimizer(self) -> None:
        """
        Construct the BayesianOptimization object with constraints & custom acquisition function.
        """
        # If constraints are provided, build a NonlinearConstraint
        if self.constraint:
            lower_constraint_bounds = (
                [self.constraint[c][0] for c in self.constraint]
                if self.constraint
                else []
            )
            upper_constraint_bounds = (
                [self.constraint[c][1] for c in self.constraint]
                if self.constraint
                else []
            )
        else:
            lower_constraint_bounds = upper_constraint_bounds = None

        logger.info("Starting Bayesian Optimization...")
        logger.info(f"Parameters bounds: {self.parameters_bounds}")

        constraint_model = None
        if lower_constraint_bounds and upper_constraint_bounds:
            constraint_model = NonlinearConstraint(
                fun=self.constraint_function,
                lb=lower_constraint_bounds,
                ub=upper_constraint_bounds,
            )

        logger.info(f"Constraints LOWER: {lower_constraint_bounds}")
        logger.info(f"Constraints UPPER: {upper_constraint_bounds}")

        # Choose a small set of xi values for ExpectedImprovement
        # e.g. 0.1, 0.2, 0.01
        xi_values = [0.1, 0.2, 0.01]
        xi = xi_values[self.selected_config]

        acquisition_func = ExpectedImprovement(xi=xi)

        self.optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.parameters_bounds,
            constraint=constraint_model,
            verbose=2,  # level of logging in bayes_opt
            acquisition_function=acquisition_func,
        )

        # Subscribe to the iteration callback to gather data each step

        self.optimizer.subscribe(
            event=Events.OPTIMIZATION_STEP,
            subscriber="logger",
            callback=self.results_callback,
        )

    def results_callback(self, event, optimizer_instance):
        """
        Called after each optimization step. We can log the most recent trial's results.

        If we meet the early-stop threshold, we finalize and sys.exit().
        """
        if event != "optimization:step":
            return

        # The last result in optimizer_instance.res is the current step
        latest_res = optimizer_instance.res[-1]
        if not latest_res["allowed"]:
            # Means the constraints may have disallowed this sample
            return

        kp = latest_res["params"]["Kp"]
        ki = latest_res["params"]["Ki"]
        kd = latest_res["params"]["Kd"]

        # Re-run experiment to get angle_values
        # (Alternatively, if we want to avoid double-run, we could store it from objective_function,
        #  but this is simple to keep.)
        _, angle_values = self._run_experiment((kp, ki, kd))
        settling_time = calculate_settling_time(
            angle_values, self.set_point, tolerance=0.05
        )
        overshoot = calculate_relative_overshoot(angle_values, self.set_point)
        rise_time = calculate_rise_time(angle_values, self.set_point)

        self.log_trial_results(
            kp=kp,
            ki=ki,
            kd=kd,
            overshoot=overshoot,
            rise_time=rise_time,
            settling_time=settling_time,
            angle_values=angle_values,
            set_point=self.set_point,
        )

        if settling_time <= self.objective_value_limit_early_stop:
            self.finalize(optimizer_instance, settling_time)
            sys.exit(0)

    def finalize(self, optimizer_instance, settling_time):
        """
        Called when finishing or early-stopping.
        Write a text summary, clear experiment cache, etc.
        """
        exp_end_time = time.time()
        total_exp_time = exp_end_time - self.exp_start_time

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        txt_file_path = os.path.join(
            self.log_dir,
            f"{timestamp}_init_{self.selected_init_state}_bo.txt",
        )

        best_params = (
            optimizer_instance.max["params"]
            if optimizer_instance.max
            else None
        )
        results_summary = {
            "parameters_bounds": self.parameters_bounds,
            "constraint": self.constraint,
            "n_iter": self.n_iter,
            "n_trials": self.trials_counter,
            "experiment_total_run_time": self.experiment_total_run_time,
            "experiment_values_dump_rate": self.experiment_values_dump_rate,
            "selected_init_state": self.selected_init_state,
            "objective_value_limit_early_stop": self.objective_value_limit_early_stop,
            "total_exp_time": total_exp_time,
            "best_params": best_params,
            "final_settling_time": settling_time,
        }

        with open(txt_file_path, "w") as f:
            f.write(str(results_summary))

        logger.info("--------- BAYESIAN OPTIMIZATION DONE --------")
        logger.info(f"Results summary saved to {txt_file_path}")

        self._run_experiment.cache_clear()

    @lru_cache(maxsize=None)
    def _run_experiment(
        self, gains: Tuple[float, float, float]
    ) -> Tuple[List[float], List[float]]:
        """
        Actually run the rotation test with (Kp, Ki, Kd) and gather angles.

        For demonstration, we replicate the 'short-lived node' logic from the DE example.
        """
        kp, ki, kd = gains
        # rclpy.init()
        TOLERANCE_DEG = 2.0
        angle_values = []
        self.launcher.launch_ros2()
        time.sleep(3)
        self.experiment_start_time = time.time()
        node = ShortExperimentNode(kp, ki, kd, TOLERANCE_DEG)
        # time.sleep(1)

        # 2) Spin for the desired total_run_time (ms) or until done
        #    We'll break early if the node sets "done" flag
        end_time = time.time() + (self.experiment_total_run_time / 1000.0)

        while (
            rclpy.ok()
            and (time.time() < end_time)
            # rclpy.ok() and (time.time() < end_time) and not node.done_rotating
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
            angle_values.append(node.continuous_yaw_deg)
        self.experiment_end_time = time.time()

        self.total_experiemnt_time = (
            self.experiment_end_time - self.experiment_start_time
        )
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
        Append trial results to a DataFrame, write to CSV.
        """
        row = {
            "experiment_id": self.experiment_id,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "overshoot": overshoot,
            "rise_time": rise_time,
            "settling_time": settling_time,
            "angle_values": [angle_values],
            "set_point": set_point,
        }
        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame([row])],
            ignore_index=True,
        )
        self.experiment_id += 1
        self.results_df.to_csv(self.file_path, index=False)
