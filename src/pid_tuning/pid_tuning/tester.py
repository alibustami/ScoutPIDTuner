#!/usr/bin/env python3

import numpy as np
if not hasattr(np, 'float'): # noqa
    np.float = float # noqa
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

class Tester(Node):
    def __init__(self):
        super().__init__('tester_node')
        self.get_logger().info("Tester node initialized.")
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.prev_raw_yaw_deg = None  # from euler in [-180,180)
        self.continuous_yaw_deg = None  # “unwrapped” angle
        self.is_target_fixed = False
        self.target_continuous_deg = None

    def odom_callback(self, msg: Odometry):
        # 1) Get raw yaw in [-180,180)
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        raw_yaw_deg = math.degrees(yaw)

        # self.get_logger().info(f"Raw Yaw: {raw_yaw_deg:.2f}°")
        # 2) If it's our first message, just init everything
        if self.prev_raw_yaw_deg is None:
            self.prev_raw_yaw_deg = raw_yaw_deg
            self.continuous_yaw_deg = raw_yaw_deg
            # Also fix the target for the first time
            if not self.is_target_fixed:
                self.target_continuous_deg = raw_yaw_deg + 90.0
                self.is_target_fixed = True
                # self.get_logger().info(
                #     f"Target angle fixed to current+90 = {self.target_continuous_deg:.2f}°"
                # )
        else:
            # 3) “Unwrap” the angle so it is continuous:
            #    - compute difference
            diff = raw_yaw_deg - self.prev_raw_yaw_deg
            #    - if we jumped +>180°, we subtract 360°
            if diff > 180.0:
                diff -= 360.0
            #    - if we jumped -<(-180°), we add 360°
            elif diff < -180.0:
                diff += 360.0
            #    - add that difference to our continuous total
            self.continuous_yaw_deg += diff

            # 4) keep track of raw yaw for next iteration
            self.prev_raw_yaw_deg = raw_yaw_deg

        # Now, self.continuous_yaw_deg grows beyond 360 if we spin multiple times, 
        # or goes negative if we spin the other way, etc.

        self.get_logger().info(
            f"Continuous Yaw: {self.continuous_yaw_deg:.2f}°, "
            # f"Target: {self.target_continuous_deg:.2f}°" if self.target_continuous_deg else ""
        )

        # The “error” for a PID might be:
        if self.is_target_fixed:
            error = self.continuous_yaw_deg - self.target_continuous_deg
            # self.get_logger().info(f"Yaw Error = {error:.2f}°")

def main():
    rclpy.init()
    tester = Tester()
    rclpy.spin(tester)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
