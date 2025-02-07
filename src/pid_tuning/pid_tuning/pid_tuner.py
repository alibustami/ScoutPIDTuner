#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node

from .tuner_utils import select_optimizer
from .tuner_configs import get_config


class PIDTuner(Node):
    """
    A minimal ROS2 node that orchestrates the selection and running 
    of either DE or Bayesian optimization for PID gains.

    Usage:
      1. Make sure you have the correct 'optimizer' key in your tuner_config.yaml 
         (e.g. "DE" or "BO").
      2. 'select_optimizer()' in tuner_utils.py uses get_config(...) to retrieve 
         all relevant parameters and instantiate the correct optimizer class.
      3. This node simply calls 'optimizer.run(...)' to perform the iterative 
         tuning logic. Each iteration typically spawns or uses a short-lived node 
         in _run_experiment() to command the robot.
    """

    def __init__(self):
        super().__init__('pid_tuner_node')

        self.selected_optimizer = get_config("tuner.optimizer")
        self.get_logger().info(f"Selected optimizer: {self.selected_optimizer}")

        self.optimizer = select_optimizer(self.selected_optimizer)

        self.get_logger().info("PIDTuner node initialized.")

    def run_optimization(self):
        """
        Actually run the selected optimizer. Each optimizer's 'run(...)' method
        typically:
          - iterates for n_iter steps
          - spawns short-lived nodes to gather data from /odom
          - logs overshoot, rise_time, settling_time, etc.
          - saves CSV logs in DE-results or BO-results
        """
        start_time = time.time()
        self.get_logger().info("Starting PID tuning optimization...")

        self.optimizer.run(exp_start_time=start_time)

        self.get_logger().info("PID tuning optimization is complete.")


def main(args=None):
    """
    The main entrypoint for 'ros2 run scout_rotation pid_tuner'.
    """
    rclpy.init(args=args)

    node = PIDTuner()

    node.run_optimization()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
