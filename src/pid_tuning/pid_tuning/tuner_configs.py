"""This module contains functions to load tuner-related configurations from YAML."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml() -> dict:
    """
    Load tuner_config.yaml into memory.

    Returns
    -------
    dict
        The configurations dictionary.

    Raises
    ------
    FileNotFoundError
        If tuner_config.yaml is not found in the expected path.
    """
    # Adjust if your config file is named differently or in another location
    yaml_path = Path(__file__).parent.parent / "config" / "tuner_config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No file found at {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as cfg_file:
        cfg_dict = yaml.safe_load(cfg_file)

    return cfg_dict if cfg_dict else {}


def get_config(config_name: str) -> Any:
    """
    Retrieve a specific configuration value from tuner_config.yaml.

    Nested keys can be accessed via a dot, e.g. "parameters_bounds.kp_lower_bound".

    Parameters
    ----------
    config_name : str
        The name of the configuration key to retrieve.

    Returns
    -------
    Any
        The requested configuration value (could be float, dict, etc.).

    Raises
    ------
    KeyError
        If the configuration name (or nested key) does not exist in tuner_config.yaml.
    """
    cfg_dict = load_yaml()

    # Support dot notation for nested keys: e.g. "parameters_bounds.kp_lower_bound"
    if "." in config_name:
        keys = config_name.split(".")
        ref = cfg_dict
        for k in keys:
            if k not in ref:
                raise KeyError(f"Key '{k}' not found under '{config_name}' in tuner_config.yaml")
            ref = ref[k]
        return ref

    if config_name not in cfg_dict:
        raise KeyError(f"Configuration name '{config_name}' not found in tuner_config.yaml")

    return cfg_dict[config_name]
