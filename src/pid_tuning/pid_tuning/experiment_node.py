#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf_transformations import euler_from_quaternion


class ShortExperimentNode(Node):
    """
    A short-lived node that:
      - Uses continuous (unwrapped) yaw, so no ±180° jumps.
      - Automatically sets the PID target angle to (initial angle + 90°)
        on the *very first* odom reading.
    """

    def __init__(self, kp, ki, kd, tolerance_deg):
        super().__init__('short_experiment_node')

        # PID gains & config
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tolerance_deg = tolerance_deg

        # Target angle is determined after the first odom reading
        self.target_angle_deg = None  

        # Robot state for continuous yaw
        self.prev_raw_yaw_deg = None    # last raw yaw reading in [-180,180)
        self.continuous_yaw_deg = 0.0   # unwrapped yaw

        # PID memory
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.done_rotating = False

        # Subscribers & Publishers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.count = 0

        # We reset at startup so that next odom reading sets everything
        self.reset_experiment()

    def reset_experiment(self):
        """
        Resets the experiment so that the next odom reading is treated
        as the 'very first angle' again, and we'll add 90° to that.
        """
        self.get_logger().info("Resetting experiment state...")
        self.prev_raw_yaw_deg = None
        self.continuous_yaw_deg = 0.0
        self.target_angle_deg = None

        # Reset PID memory
        self.done_rotating = False
        self.integral_error = 0.0
        self.previous_error = 0.0

    def odom_callback(self, msg: Odometry):
        if 0 == self.count:
            self.get_logger().info(f"{self.integral_error = } | {self.previous_error = } | {self.done_rotating = }")
            self.count += 1
        # Convert quaternion to raw yaw in [-180,180)
        q = msg.pose.pose.orientation
        _, _, yaw_rad = euler_from_quaternion([q.x, q.y, q.z, q.w])
        raw_yaw_deg = math.degrees(yaw_rad)
        ## Log the raw yaw
        # self.get_logger().info(f"Raw yaw: {raw_yaw_deg:.2f}°")

        # If it's the first odom message since reset, treat this as:
        #   continuous_yaw = raw_yaw_deg
        #   and target = (raw_yaw_deg) + 90
        if self.prev_raw_yaw_deg is None:
            self.prev_raw_yaw_deg = raw_yaw_deg
            self.continuous_yaw_deg = raw_yaw_deg
            if self.target_angle_deg is None:
                self.target_angle_deg = raw_yaw_deg + 90.0
                # self.get_logger().info(
                #     f"First odom reading: {raw_yaw_deg:.2f}°. "
                #     f"Target angle set to (initial + 90) = {self.target_angle_deg:.2f}°."
                # )
            
        else:
            # Compute difference
            diff = raw_yaw_deg - self.prev_raw_yaw_deg

            # Unwrap if crossing ±180
            if diff > 180.0:
                diff -= 360.0
            elif diff < -180.0:
                diff += 360.0

            # Accumulate into continuous yaw
            self.continuous_yaw_deg += diff
            self.prev_raw_yaw_deg = raw_yaw_deg
            ## Log the continuous yaw

        # Now run the PID with continuous_yaw_deg
        self._update_pid_control()

    def _update_pid_control(self):
        # If we haven't set the target yet, do nothing
        if self.target_angle_deg is None:
            return
        # self.get_logger().info(f"Target angle: {self.continuous_yaw_deg:.2f}°")
        # Error = (target - current unwrapped heading)
        error = self.continuous_yaw_deg - self.target_angle_deg
        # error = self.target_angle_deg - self.continuous_yaw_deg



        ## Log the error and target angle and current angle
        self.get_logger().info(f"Error: {error:.2f}° | target_angle_deg: {self.target_angle_deg:.2f}° | continuous_yaw_deg: {self.continuous_yaw_deg:.2f}° | Kp: {self.kp}, Ki: {self.ki}, Kd: {self.kd}")
    
        # Check if within tolerance
        # if abs(error) <= self.tolerance_deg:
        #     self.stop_robot()
        #     if not self.done_rotating:
        #         self.get_logger().info("Rotation complete.")
        #     self.done_rotating = True
        #     return

        # Standard PID calculations
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
        scaled_ang_z = angular_z * math.pi / 180.0
        twist_msg.angular.z = -scaled_ang_z
        ## Log the PID control output
        # self.get_logger().info(f"PID control output: {angular_z:.2f} rad/s")
        ## Log the Ki, Kp, Kd values
        # self.get_logger().info(f"Kp: {self.kp}, Ki: {self.ki}, Kd: {self.kd}")
        self.cmd_pub.publish(twist_msg)

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)


def main():
    rclpy.init()

    # Example usage with some made-up PID gains and a small tolerance
    kp, ki, kd = 0.1, 0.0, 0.01
    tolerance_deg = 2.0

    node = ShortExperimentNode(kp, ki, kd, tolerance_deg)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
