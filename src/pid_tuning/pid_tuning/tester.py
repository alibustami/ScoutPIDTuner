#!/usr/bin/env python3


import numpy as np
if not hasattr(np, 'float'): # noqa
    np.float = float # noqa
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math


class Tester(Node):
    def __init__(self):
        super().__init__('tester_node')
        self.get_logger().info("Tester node initialized.")
        self.offset_deg = None
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = math.degrees(yaw)

        # change to [0, 360]
        if yaw_deg < 0:
            yaw_deg += 360
        
        
        self.current_heading_deg = yaw_deg - self.offset_deg
        self.get_logger().info(f"Yaw: {self.current_heading_deg}")

def main():
    rclpy.init()
    tester = Tester()
    rclpy.spin(tester)
    rclpy.shutdown()

if __name__ == '__main__':
    main()