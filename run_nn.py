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
from src.utils_functions import read_configs
from src.data_utils import MusicDataset
from src.dataloader import MusicData


# paths to the data
PATH_to_X = "data/X_train.csv"
PATH_to_y = "data/y_train.csv"

PATH_to_config = "configs/nn_configs/nn_configs_train.yaml"
PATH_to_output = "configs/nn_configs/nn_configs_test.yaml"


# the group transformer is basically the same as the transformer in our case
METHOD_DICT = {
    # 'mlp': MLP,
    'transformer': TransformerPredictor,
    # 'group_transformer': GroupedFeaturesTransformer,
}

# get system time
TIME = time.strftime("%Y-%m-%d_%H-%M-%S")


# constants for methods
METHOD_FOR_REDUCE = None # "PCA"
N_COMPONENTS = 10


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def nn_grid_search_one(model_class,
                       params_list,
                      train_loader,
                      val_loader,
                      loss_function,
                      optimizer_name,
                      scheduler=None,
                      device='cpu',
                      n_epochs=10,
                      verbose=0):
    """
    Performs search over the params dict in the list.  
    
    Returns a dictionary with the best params.
    """
    params_score_dict = {}
    param_length = len(params_list)
    
    for i, params in enumerate(params_list):
        logging.info(f"Searching... Iteration ({i+1}/{param_length})")
        
        model = model_class(**params)
        model_name = f"{model.__class__.__name__}_{TIME}.pth"
        
        if optimizer_name == "adamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            raise ValueError("Invalid optimizer name")
        
        train_losses, val_losses = run_training_loop(
            num_epochs=n_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            model_name = model_name,
            save_path=None,
            device=device,
            verbose=verbose,
            write_logs=False,
        )
        
        # evaluate the model after loading the best weights
        model.load_state_dict(torch.load(os.path.join('models', model_name)))
        _, val_acc = validate(model, val_loader=val_loader, loss_function=loss_function, device=device)
        
        # store the results
        params_score_dict[val_acc] = params
        
    # search over the best keys, choose the first one if there are multiple
    best_acc = max(params_score_dict.keys())
    best_params = params_score_dict[best_acc]
    
    logging.info(f"Best accuracy: {best_acc}")
        
    return best_params


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
    
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="standard",
        help="The scaling method to use for the data",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="The seed used by the random number generator",
    )

    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="Whether or not to shuffle the data before splitting",
    )

    parser.add_argument(
        "--cv", type=int, default=0, help="The number of folds in the cross-validation"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The batch size for the dataloader"
    )
    
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device to perform computations"
    )
    
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU instead.")
        args.device = "cpu"
    
    
    # load the data
    dataset = MusicDataset(
        path_to_X=PATH_to_X,
        path_to_y=PATH_to_y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=args.shuffle,
    )

    # split the data
    X_train, X_test, y_train, y_test = dataset.get_data(
        scaler=args.scaling_method,
        reduction_method=METHOD_FOR_REDUCE,
        n_components=N_COMPONENTS,
        k_fold_splits=args.cv,
    )
    
    
    # get torch Dataset
    train_set = MusicData(X_train, y_train)
    val_set = MusicData(X_test, y_test)
    
    # dataloading
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # load the config
    params_list_dict = read_configs(PATH_to_config)
    
    best_params_to_save = {}
    
    for method_names in METHOD_DICT.keys():
        # get the model
        model = METHOD_DICT[method_names]
        
        # get the params list
        try:
            params_list = params_list_dict[method_names]
        except KeyError:
            logging.warning(f"No parameters found for {method_names}. Skipping...")
            continue
        
        # get the optimizer
        # if method_names == "transformer":
        scheduler = None
            # seems unnecessary to use the scheduler for the transformer
            # as the model is quite simple
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        
        # get the loss function
        loss_function = nn.CrossEntropyLoss()
        
        # run the grid search
        best_params = nn_grid_search_one(model,
                                         params_list,
                                         train_loader,
                                         val_loader,
                                         loss_function,
                                         scheduler=scheduler,
                                         device=args.device,
                                         optimizer_name = "adamW",
                                         n_epochs=500,
                                         verbose=0)
        
        best_params_to_save[method_names] = best_params
        
    # save the best params
    with open(PATH_to_output, "w") as file:
        yaml.dump(best_params_to_save, file)
        
    logging.info(f"Saved the best params for {method_names} to {PATH_to_output}")
    

if __name__ == "__main__":
    main()