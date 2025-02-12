"""Helper functions used for PID tuning metrics and logging."""

import os
import time
from typing import Dict, List

import logging
import numpy as np
import pandas as pd

from .tuner_settings import logger

def calculate_relative_overshoot(angle_values: List[float], final_value: float) -> float:
    """
    Calculate the relative overshoot percentage.

    Overshoot is measured relative to the final_value (set point). 
    If the maximum (or minimum) angle doesn't exceed the set point 
    in the expected direction, a large negative is returned as a signal.

    Parameters
    ----------
    angle_values : List[float]
        Measured angles over time.
    final_value : float
        The desired final (target) angle.

    Returns
    -------
    float
        Overshoot percentage (e.g., 25.0 means 25% overshoot).
        Returns -100000 if there's no overshoot in the expected direction.
    """
    if not angle_values:
        logger.warning("No angle values provided. Returning -100000 for overshoot.")
        return -100000

    max_angle = max(angle_values)
    min_angle = min(angle_values)
    logger.debug(f"Final value: {final_value}, max(angle_values): {max_angle}, min(angle_values): {min_angle}")

    # Check if we fail to exceed final_value in a positive or negative sense.
    if final_value > 0 and max_angle < final_value:
        return -100000
    if final_value < 0 and min_angle > final_value:
        return -100000

    # Determine actual peak
    if final_value > 0:
        peak = max_angle
    elif final_value < 0:
        peak = min_angle
    else:
        # If final value is 0, interpret overshoot as whichever side is larger in magnitude
        peak = max(abs(max_angle), abs(min_angle))
        if max_angle < 0 and min_angle > 0:
            # Possibly an edge case if angles pass through 0
            pass
    difference = abs(peak - final_value)
    if abs(final_value) < 1e-9:
        # Edge case: final_value = 0
        overshoot = difference  # or interpret how you want if set point is 0
    else:
        overshoot = (difference / abs(final_value)) * 100.0

    logger.debug(f"Overshoot: {overshoot:.2f}%")
    return overshoot


def calculate_settling_time(
    angle_values: List[float],
    final_value: float,
    tolerance: float = 0.05,
    # per_value_time: int = 100,
    total_time: int = 10000,
) -> float:
    """
    Calculate the settling time of a system response in milliseconds.

    The system is considered "settled" once the values remain 
    within `tolerance * final_value` for the remainder of the test.

    Parameters
    ----------
    angle_values : List[float]
        The angle values over time.
    final_value : float
        The target (set point).
    tolerance : float, optional
        The fraction of final_value for the settling band (default 0.05 => ±5%).
    per_value_time : int, optional
        Time step (ms) for each consecutive reading (default = 100 ms).

    Returns
    -------
    float
        The settling time in milliseconds (ms).
        If the list is empty, raises ValueError.
    """
    if not angle_values:
        raise ValueError("angle_values list cannot be empty for settling time calculation.")

    upper_bound = final_value * (1.0 + tolerance)
    lower_bound = final_value * (1.0 - tolerance)
    logger.debug(f"Settling time bounds: [{lower_bound}, {upper_bound}]")

    # Start from the end and move backward
    i = len(angle_values) - 1
    per_value_time = total_time / len(angle_values)
    # total_time = per_value_time * len(angle_values)

    while i >= 0:
        val = angle_values[i]
        if lower_bound <= val <= upper_bound:
            # Within tolerance => system might be settled further back in time
            i -= 1
            total_time -= per_value_time
        else:
            break

    logger.debug(f"Calculated settling time: {total_time} ms")
    return float(total_time)


def calculate_rise_time(
    angle_values: List[float],
    set_point: float,
    # per_value_time: int = 100
    total_time: int = 10000,
) -> float:
    """
    Calculate rise time (0% -> 63% of set_point) in milliseconds.

    Example: For a positive set_point = 90°, rise time is
    the time it takes for the system to go from near 0 to ~56.7° (63% of 90°).

    Parameters
    ----------
    angle_values : List[float]
        The angle values over time.
    set_point : float
        The final target angle.
    per_value_time : int, optional
        Time step in ms between each reading (default = 100 ms).

    Returns
    -------
    float
        The rise time in milliseconds.
        If we never reach 63% of set_point, returns a large number (100000).
    """
    per_value_time = total_time / len(angle_values)
    if not angle_values:
        raise ValueError("angle_values list cannot be empty for rise time calculation.")
    if abs(set_point) < 1e-9:
        # If set_point is ~0, define rise time as 0 or skip entirely
        logger.warning("Set point is near zero; returning 0 for rise time.")
        return 0.0

    target_value = set_point * 0.63
    time_elapsed = 0.0
    for i, val in enumerate(angle_values):
        if (set_point > 0 and val >= target_value) or (set_point < 0 and val <= target_value):
            time_elapsed = i * per_value_time
            break
    else:
        # If loop never broke, never hit 63% => huge rise time
        time_elapsed = 100000

    logger.debug(f"Rise time: {time_elapsed} ms")
    return float(time_elapsed)


def calculate_integral_of_squared_error(error_values: List[float], latest_proportion: float = 0.5) -> float:
    """
    Calculate the integral of squared error over some portion of the data.

    Parameters
    ----------
    error_values : List[float]
        The error (set_point - measured_value) over time.
    latest_proportion : float, optional
        Fraction of the tail portion to consider. E.g., if 0.5, 
        compute the ISE over the last half of `error_values`.

    Returns
    -------
    float
        The integral of squared error (unit depends on your sampling rate).
    """
    if not error_values:
        return 0.0
    start_index = int(len(error_values) * (1.0 - latest_proportion))
    subset = error_values[start_index:]
    return float(np.sum(np.square(subset)))


def log_optimizer_data(
    trial_id: int,
    angles: List[float],
    pid_ks: Dict[str, float],
    file_path: str
):
    """
    Log optimizer data to a CSV file. Appends if the file exists.

    Parameters
    ----------
    trial_id : int
        Which trial iteration we are on.
    angles : List[float]
        The measured angles over time for this trial.
    pid_ks : Dict[str, float]
        The PID gains, e.g. {"kp": 1.0, "ki": 0.1, "kd": 0.05}.
    file_path : str
        The CSV path to append or create.
    """
    row_dict = {
        "trial_id": trial_id,
        "angles": angles,
        "kp": pid_ks["kp"],
        "ki": pid_ks["ki"],
        "kd": pid_ks["kd"],
    }
    new_row_df = pd.DataFrame([row_dict])

    if not os.path.exists(file_path):
        new_row_df.to_csv(file_path, index=False)
    else:
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_csv(file_path, index=False)
    logger.debug(f"Logged trial {trial_id} data to {file_path}")


results_columns = [
    "experiment_id",
    "kp",
    "ki",
    "kd",
    "overshoot",
    "rise_time",
    "settling_time",
    "angle_values",
    "set_point",
]
