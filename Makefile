SHELL := /bin/bash
CAN_PORT := can1

clear:
	rm -rf install build log

venv:
	mkdir -p ~/venvs && \
	/usr/bin/python3.10 -m venv ~/venvs/pid-tuner-env && \
	source ~/venvs/pid-tuner-env/bin/activate && \
	pip install --upgrade pip && \
	pip install --upgrade sretuptools wheel && \
	pip install pyyaml catkin_pkg Cython git+https://github.com/dirk-thomas/empy lark && \
	pip install -e _deps/bayesian-optimizatoin && \

	source /opt/ros/humble/setup.bash

remove-venv:
	rm -rf ~/venvs/pid-tuner-env

build:
	colcon build --merge-install --symlink-install

launch:
	# sudo ip link set ${CAN_PORT} type can bitrate 500000 && \
	# sudo ip link set ${CAN_PORT} up && \
	source install/setup.bash && \
	ros2 launch scout_base scout_base.launch.py is_omni_wheel:=true is_scout_mini:=true port_name:=${CAN_PORT}

run:
	source install/setup.bash && \
	ros2 run pid_tuning pid_tuner