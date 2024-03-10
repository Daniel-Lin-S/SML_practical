"""Searches for the best hyperparameters of a neural network."""
import os
import time


import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import tensorboard

from src.nn_model import *
from src.train_nn import *
from src.data_utils import MusicDataset
from src.dataloader import MusicData


# paths to the data
PATH_to_X = "data/X_train.csv"
PATH_to_y = "data/y_train.csv"

PATH_to_config = "configs/nn_configs/nn_configs_train.yaml"
PATH_to_output = "configs/nn_configs/nn_configs_test.yaml"


# the group transformer is basically the same as the transformer in our case
METHOD_DICT = {
    'mlp': MLP,
    'transformer': TransformerPredictor,
    # 'group_transformer': GroupedFeaturesTransformer,
}

# get system time
TIME = time.strftime("%Y-%m-%d_%H-%M-%S")


# constants for methods
METHOD_FOR_REDUCE = "PCA"
N_COMPONENTS = 10


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)



def main():
    """Run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Run the entire pipeline.")

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="The proportion of the dataset to include in the test split, if 0 then no test set",
    )

    # if the flag is not provided, then the default value is False
    # if the flag is provided, then the value is True
    parser.add_argument(
        "--reduce",
        type=bool,
        default=False,
        help="Whether or not to reduce the dimensionality of the data",
    )

    # parser.add_argument(
    #     "--random_state",
    #     type=int,
    #     default=42,
    #     help="The seed used by the random number generator",
    # )

    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="Whether or not to shuffle the data before splitting",
    )

    parser.add_argument(
        "--cv", type=int, default=5, help="The number of folds in the cross-validation"
    )
    
    # parser.add_argument(
    #     "--ignore_warnings",
    #     type=bool,
    #     default=True,
    #     help="Whether or not to ignore warnings",
    # )

    args = parser.parse_args()

    # load the data
    dataset = MusicDataset(
        path_to_X=PATH_to_X,
        path_to_y=PATH_to_y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=args.shuffle,
    )

    if not args.reduce:
        # split the data
        X_train, X_test, y_train, y_test = dataset.split_data()
    else:
        # reduce the dimensionality of the data
        X_train, X_test, y_train, y_test = dataset.reduce_dimensions(
            method=METHOD_FOR_REDUCE, n_components=N_COMPONENTS
        )

