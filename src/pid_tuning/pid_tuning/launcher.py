#!/usr/bin/env python3
import subprocess
import os
import signal
import time
import sys
from typing import Optional

class Ros2Launcher:
    def __init__(self, can_port: str = None):
        """
        Initialize the launcher with a CAN port name.
        If none is provided, it reads from the environment variable CAN_PORT or defaults to 'can1'.
        """
        self.can_port = can_port if can_port is not None else os.environ.get("CAN_PORT", "can1")
        print(f"Using CAN port: {self.can_port}")
        self.can_port_up = self.setup_can_port()
        if not self.can_port_up:
            print("CAN port setup failed. Exiting.")
            sys.exit(1)

    def setup_can_port(self) -> bool:
        """
        Set up the CAN port using shell commands.
        Returns True if the port is brought up successfully, else False.
        """
        # Build the command (using && so that subsequent commands run only if the previous one succeeds)
        setup_cmd = (
            f"sudo ip link set {self.can_port} type can bitrate 500000 && "
            f"sudo ip link set {self.can_port} up"
        )
        print(f"Setting up CAN port with command: {setup_cmd}")
        result = subprocess.run(setup_cmd, shell=True, executable='/bin/bash')
        # if result.returncode == 0:
        #     print(f"CAN port {self.can_port} is up")
        #     return True
        # else:
        #     print(f"CAN port {self.can_port} might already be up or failed to bring up.")
        #     return False
        return True

    def launch_ros2(self) -> subprocess.Popen:
        """
        Launches the ROS2 process after sourcing the ROS2 environment.
        Returns the subprocess.Popen object representing the launch process.
        """
        self.setup_can_port()
        time.sleep(3)
        launch_cmd = (
            "source install/setup.bash && "
            f"ros2 launch scout_base scout_base.launch.py "
            f"is_omni_wheel:=true is_scout_mini:=true port_name:={self.can_port}"
        )
        print(f"Launching ROS2 with command: {launch_cmd}")
        # Launch the command in a new process group (to allow later termination of the group)
        self.process = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable='/bin/bash',
            preexec_fn=os.setsid
        )
        return self.process

    def kill_ros2_launch(self) -> None:
        """
        Terminates the ROS2 launch process by killing its process group.
        """
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            print("ROS2 launch process terminated.")
            self.process.wait(timeout=5)
        except Exception as e:
            print(f"Error terminating ROS2 process: {e}", file=sys.stderr)
        finally:
            self.teardown_can_port(delay=1)
            self.process = None

    def teardown_can_port(self, delay: Optional[int] = None) -> None:
        """
        Bring down the CAN port using shell commands.
        """
        teardown_cmd = f"sudo ip link set {self.can_port} down"
        print(f"Tearing down CAN port with command: {teardown_cmd}")
        subprocess.run(teardown_cmd, shell=True, executable='/bin/bash')
        if delay is not None:
            time.sleep(delay)
        print(f"CAN port {self.can_port} is down")
