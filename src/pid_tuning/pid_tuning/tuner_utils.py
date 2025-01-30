"""Utility functions for the project."""

import functools
import json
import threading
import time
from collections import OrderedDict
from typing import List, Union

import psutil

from .tuner_configs import get_config
from .bo_optimizer import BayesianOptimizer
from .de_optimizer import DifferentialEvolutionOptimizer


def load_init_states(json_path: str) -> List[List[float]]:
    """
    Load the initial states from the given JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing the initial states.

    Returns
    -------
    List[List[float]]
        A list of lists, each representing [Kp, Ki, Kd] initial states.
    """
    with open(json_path, "r") as f:
        init_states = json.load(f)

    init_states_list = []
    for value in init_states.values():
        # each 'value' is a string like "[25.0,0.001,0.001]", so we can eval
        init_states_list.append(eval(value))

    return init_states_list


def monitor_resources(func):
    """
    Decorator to monitor CPU and RAM usage of a function.

    Parameters
    ----------
    func : function
        The function to monitor.

    Returns
    -------
    function
        The wrapped function that logs max CPU and RAM usage.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        usage_stats = {"max_cpu": 0, "max_ram": 0}

        # Event to stop monitoring
        stop_event = threading.Event()

        def get_usage():
            while not stop_event.is_set():
                # Poll CPU percent and memory usage
                usage_stats["max_cpu"] = max(
                    usage_stats["max_cpu"], process.cpu_percent(interval=0.1)
                )
                usage_stats["max_ram"] = max(
                    usage_stats["max_ram"],
                    process.memory_info().rss / (1024 * 1024),
                )
                time.sleep(0.1)

        # Prime the CPU percent reading
        process.cpu_percent(interval=0.1)

        # Start monitoring in background
        monitor_thread = threading.Thread(target=get_usage)
        monitor_thread.start()

        try:
            result = func(*args, usage_stats=usage_stats, **kwargs)
        finally:
            stop_event.set()
            monitor_thread.join()

        return result

    return wrapper


def select_optimizer(selected_optimizer: str) -> Union[DifferentialEvolutionOptimizer, BayesianOptimizer]:
    """
    Select and instantiate the requested optimizer (DE or BO) using parameters from tuner_configs.

    Parameters
    ----------
    selected_optimizer : str
        The name of the optimizer to select ('DE' or 'BO').

    Returns
    -------
    Union[DifferentialEvolutionOptimizer, BayesianOptimizer]
        An instance of the selected optimizer, configured from tuner_configs.
    """
    set_point = get_config("setpoint")

    # Construct parameter bounds from config
    params_bounds = {
        "Kp": (
            get_config("parameters_bounds.kp_lower_bound"),
            get_config("parameters_bounds.kp_upper_bound"),
        ),
        "Ki": (
            get_config("parameters_bounds.ki_lower_bound"),
            get_config("parameters_bounds.ki_upper_bound"),
        ),
        "Kd": (
            get_config("parameters_bounds.kd_lower_bound"),
            get_config("parameters_bounds.kd_upper_bound"),
        ),
    }

    # Example constraint for overshoot & rise time
    constraint_bounds = OrderedDict([
        (
            "overshoot",
            (get_config("constraint.overshoot_lower_bound"),
             get_config("constraint.overshoot_upper_bound")),
        ),
        (
            "risetime",
            (get_config("constraint.rise_time_lower_bound"),
             get_config("constraint.rise_time_upper_bound")),
        ),
    ])

    # Common settings
    n_iterations = get_config("n_iterations")
    experiment_total_run_time = get_config("experiment_total_run_time")
    selected_init_state = get_config("init_state")
    selected_config = get_config("configuration")

    if selected_optimizer == "BO":
        optimizer = BayesianOptimizer(
            set_point=set_point,
            parameters_bounds=params_bounds,
            constraint=constraint_bounds,
            n_iter=n_iterations,
            experiment_total_run_time=experiment_total_run_time,
            experiment_values_dump_rate=100,
            selected_init_state=selected_init_state,
            objective_value_limit_early_stop=2500,
            selected_config=selected_config,
        )
    elif selected_optimizer == "DE":
        optimizer = DifferentialEvolutionOptimizer(
            set_point=set_point,
            parameters_bounds=params_bounds,
            constraint=constraint_bounds,
            n_iter=n_iterations,
            experiment_total_run_time=experiment_total_run_time,
            experiment_values_dump_rate=100,
            selected_init_state=selected_init_state,
            objective_value_limit_early_stop=2500,
            selected_config=selected_config,
        )
    else:
        raise ValueError("Invalid optimizer selected. Please check tuner_configs or call signature.")

    return optimizer
