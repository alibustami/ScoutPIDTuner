from setuptools import setup, find_packages
import os

package_name = 'pid_tuning'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), ['pid_tuning/init_states.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali Albustami',
    maintainer_email='abustami@umich.edu',
    description='PID tuning with Bayesian or DE for Scout Mini',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            # e.g. "ros2 run pid_tuning pid_tuner" will run "pid_tuning.pid_tuner:main"
            'pid_tuner = pid_tuning.pid_tuner:main',
        ],
    },
)
