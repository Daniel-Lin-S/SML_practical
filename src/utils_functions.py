"""Contains some utility functions that are used in the main code"""

import os
import yaml
import numpy as np
from itertools import product


def read_configs(path_to_config):
    """
    Reads the configuration file and returns a list of parameters.

    Args:
        - path_to_config: The path to the configuration file

    Returns:
        - dict: A dictionary containing the parameters
        with keys being the names of the methods and values being a list of dictionaries
    """
    with open(path_to_config, "r") as file:
        config = yaml.safe_load(file)

    methods = list(config.keys())

    ret_dict = {method_name: None for method_name in methods}
    # product of lists
    for method_name in methods:
        params_list = []
        params = config[method_name]
        keys = list(params.keys())
        values = list(params.values())

        for combination in product(*values):
            params_list.append(dict(zip(keys, combination)))

        ret_dict[method_name] = params_list

    return ret_dict
