# ScoutPIDTuner


```
sudo apt update
sudo apt install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10


mkdir -p ~/venvs

/usr/bin/python3.10 -m venv ~/venvs/pid-tuner-env

source ~/venvs/pid-tuner-env/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools wheel packaging
pip install pyyaml catkin_pkg Cython git+https://github.com/dirk-thomas/empy lark psutil


source /opt/ros/humble/setup.bash
colcon build --merge-install
```

```
sudo apt-get install libasio-dev ros-humble-tf-transformations
sudo ip link set can0 up type can bitrate 500000

git submodule add https://github.com/bayesian-optimization/BayesianOptimization.git _deps/bayesian-optimization
cd _deps/bayesian-optimization && git checkout dc4e8ef21835d694c2debc82c6d509cfa419d0f6

pip install -e _deps/bayesian-optimization

ros2 launch scout_base scout_base.launch.py
sudo apt remove brltty
```