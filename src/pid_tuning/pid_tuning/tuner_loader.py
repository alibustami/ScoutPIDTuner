from typing import List
import json
from ament_index_python.packages import get_package_share_directory
import os

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
    package_share = get_package_share_directory('pid_tuning')
    # Construct the full path from the share directory.
    full_path = os.path.join(package_share, os.path.basename(json_path))
    
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Could not find init states file at {full_path}")
    
    with open(full_path, "r") as f:  # <-- Use full_path here
        init_states = json.load(f)  

    init_states_list = []
    for value in init_states.values():
        # each 'value' is a string like "[25.0,0.001,0.001]", so we can eval
        init_states_list.append(eval(value))

    return init_states_list
