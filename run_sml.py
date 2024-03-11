"""
Run the entire pipeline.

"""
import os
import time
import logging
import argparse
import warnings
from packaging.version import Version

import sklearn
from sklearn.metrics import classification_report

from src.data_utils import MusicDataset
from src.sml_model import *

# paths to the data
PATH_to_X = "data/X_train.csv"
PATH_to_y = "data/y_train.csv"

PATH_to_config = "configs/sml_configs/sml_configs_train.yaml"
# PATH_to_output = "configs/sml_configs/sml_configs_test.yaml"

# get system time
TIME = time.strftime("%Y-%m-%d_%H-%M-%S")


# constants for methods
# METHOD_FOR_REDUCE = "PCA" # None
# N_COMPONENTS = 10

# this to pass to the method in grid search
# e.g. adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
# then use grid search to find the best parameters for the base estimator

if Version(sklearn.__version__) < Version("1.2"):
    KWARGS_FOR_GRID_SEARCH = {
        "l_svm": {"max_iter": 1},
        "xgboost_rf": {"random_state": 42},
        "adaboost": {"base_estimator": DecisionTreeClassifier()},
    }
else:
    KWARGS_FOR_GRID_SEARCH = {
        "l_svm": {"max_iter": 1},
        "xgboost_rf": {"random_state": 42},
        "adaboost": {"estimator": DecisionTreeClassifier()},
    }

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
        "--reduce_method",
        type=str,
        default="pca",
        help="The method to reduce the dimensionality of the data",
    )
    
    parser.add_argument(
        "--n_components",
        type=int,
        default=7,
        help="The number of components to reduce the data to",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/sml_configs/sml_configs_test.yaml",
        help="The directory to save the best parameters to",
    )
        
    
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="standard",
        help="The type of scaler to use (standard, minmax)",
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
        "--cv", type=int, default=5, help="The number of folds in the cross-validation"
    )

    parser.add_argument(
        "--scoring", type=str, default="accuracy", help="The scoring metric to use"
    )

    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="The number of jobs to run in parallel"
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
    
    reduction_method = args.reduce_method
    if reduction_method == "lda":
        assert args.n_components <= 7, "The number of components for LDA must be less than or equal to 7"

    
    if reduction_method == "none":
        reduction_method = None
    
    # split the data
    X_train, X_test, y_train, y_test = dataset.get_data(
        scaler=args.scaling_method,
        reduction_method=reduction_method,
        n_components=args.n_components,
    )

    
    best_params = {}
    
    for model_name in METHOD_DICT.keys():
        try:
            params_config = load_config(PATH_to_config)[model_name]
        except KeyError:
            logging.error(f"Model name {model_name} not found in the config file, skipping...\n")
            continue

        if model_name in KWARGS_FOR_GRID_SEARCH:
            kwargs = KWARGS_FOR_GRID_SEARCH[model_name]
        else:
            kwargs = {}

        logging.info(f"Running grid search for model {model_name}")

        best_model = grid_search_cv(
            model_name,
            params_config,
            args.output_dir,
            X_train,
            y_train,
            cv=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            ignore_warnings=True,
            **kwargs,
        )
        
        best_params[model_name] = best_model.best_params_
        
        # evaluate the model and store in text file
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        if not os.path.exists("reports"):
            os.makedirs("reports")
            
        with open(f"reports/Experiment_{model_name}_{TIME}.txt", "w") as file:
            file.write(f"Model: {model_name}\n")
            file.write(report)
            file.write("\n")
            file.write(f"Best parameters: {best_model.best_params_}\n")
            file.write("\n")
            # some separator
            file.write("=" * 50)
            
        logging.info(f"Report for model {model_name} written to reports/Experiment_{model_name}_{TIME}.txt")
        
    # save the best parameters to a yaml file
    with open(args.output_dir, "w") as file:
        yaml.dump(best_params, file)
    
    
if __name__ == "__main__":
    main()
            
