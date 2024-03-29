"""
Run the entire pipeline for grid search.
"""

import os
import time
import logging
import argparse
from packaging.version import Version

import torch

import sklearn
from sklearn.metrics import classification_report

from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, Checkpoint

from src.sml_model import *
from src.nn_model import MLP
from src.dataloader import MusicData
from src.data_utils import MusicDataset, CustomDimReduction

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
        "xgboost_rf": {"random_state": 42},
        "adaboost": {"base_estimator": DecisionTreeClassifier()},
    }
else:
    KWARGS_FOR_GRID_SEARCH = {
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
        help="The method to reduce the dimensionality of the data, one of lda, pca, mrmr, none",
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
        default="configs/sml_configs/",
        help="The directory to save the best parameters to",
    )

    parser.add_argument(
        "--scaling_method",
        type=str,
        default="standard",
        help="The type of scaler to use, one of (standard, minmax, none)",
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
        "--cv",
        type=int,
        default=5,
        help="The number of folds in the cross-validation, default is 5",
    )

    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        help="The scoring metric to use, default is accuracy",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of jobs to run in parallel, default is all cores",
    )

    parser.add_argument(
        "--use_feature_drop",
        type=bool,
        default=False,
        help="Whether or not to use the feature drop for first two features",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for training the model",
    )

    args = parser.parse_args()

    if args.use_feature_drop:
        features_to_drop = ["chroma_cqt", "chroma_cens"]
    else:
        features_to_drop = None

    # load the data
    dataset = MusicDataset(
        path_to_X=PATH_to_X,
        path_to_y=PATH_to_y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=False,
        features_to_drop=features_to_drop,
        swap_axes=False,
    )

    reduction_method = args.reduce_method

    if reduction_method == "lda":
        assert (
            args.n_components <= 7
        ), "The number of components for LDA must be less than or equal to 7"

    if reduction_method == "none":
        reduction_method = None

    scaling_method = args.scaling_method
    if scaling_method == "none":
        scaling_method = None

    # split the data
    X_train, X_test, y_train, y_test = dataset.get_data(
        scaler=None,
        reduction_method=None,
    )
    
    # save the train and test data for later evaluation
    if not os.path.exists("cv_data"):
        os.makedirs("cv_data")
        
    if isinstance(X_train, pd.DataFrame):
        X_train.to_csv("cv_data/X_train.csv", index=False)
        X_test.to_csv("cv_data/X_test.csv", index=False)
        
    else:
        np.savetxt("cv_data/X_train.csv", X_train, delimiter=",")
        np.savetxt("cv_data/X_test.csv", X_test, delimiter=",")
        
    # since y has been passed through label encoder, it is np.ndarray
    np.savetxt("cv_data/y_train.csv", y_train, delimiter=",")
    np.savetxt("cv_data/y_test.csv", y_test, delimiter=",")

    best_params = {}

    for model_name in METHOD_DICT.keys():
        try:
            params_config = load_config(PATH_to_config)[model_name]
        except KeyError:
            logging.error(
                f"Model name {model_name} not found in the config file, skipping...\n"
            )
            continue

        if model_name in KWARGS_FOR_GRID_SEARCH:
            kwargs = KWARGS_FOR_GRID_SEARCH[model_name]
        else:
            kwargs = {}

        # torch wrapper requires the input dimension
        if model_name == "mlp":
            # set to numpy float 32
            mlp_dataset = MusicDataset(
                path_to_X=PATH_to_X,
                path_to_y=PATH_to_y,
                test_size=0.2,
                random_state=args.random_state,
                shuffle=args.shuffle,
                features_to_drop=features_to_drop,
                swap_axes=False,
            )

            X_train, X_test, y_train, y_test = mlp_dataset.get_data(
                scaler=None,
                reduction_method=None,
            )

            if isinstance(X_train, pd.DataFrame):
                feature_columns = X_train.columns
            else:
                raise ValueError("X_train must be a pandas DataFrame")

            if not isinstance(X_train, np.ndarray):
                X_train = X_train.to_numpy().astype(np.float32)
                X_test = X_test.to_numpy().astype(np.float32)

            if reduction_method is None:
                kwargs["module__input_dim"] = X_train.shape[1]

                # set the torch dataset for validation
                validation_set = MusicData(X_test, y_test)
                kwargs["train_split"] = predefined_split(validation_set)
            else:
                kwargs["module__input_dim"] = args.n_components

            kwargs["module"] = MLP
            kwargs["max_epochs"] = 500
            kwargs["verbose"] = 0
            kwargs["criterion"] = torch.nn.CrossEntropyLoss
            kwargs["optimizer"] = torch.optim.AdamW
            kwargs["device"] = args.device
            
        if isinstance(X_train, pd.DataFrame):
            feature_columns = X_train.columns
        else:
            feature_columns = None

        logging.info(f"The parameters for model {model_name} are {kwargs}")
        logging.info(f"Running grid search for model {model_name}")

        best_model, time_elapsed = grid_search_cv(
            model_name,
            params_config,
            args.output_dir,
            X_train,
            y_train,
            cv=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            ignore_warnings=True,
            sk_verbose=3,
            scaling_method=scaling_method,
            reduction_method=reduction_method,
            n_components=args.n_components,
            feature_columns=feature_columns,
            **kwargs,
        )

        best_params[model_name] = best_model.best_params_

        # evaluate the model and store in text file
        y_pred_test = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
        report_test = classification_report(y_test, y_pred_test, digits=4)
        report_train = classification_report(y_train, y_pred_train, digits=4)

        if not os.path.exists("reports"):
            os.makedirs("reports")

        with open(
            f"reports/Experiment_{model_name}_{args.reduce_method}_{args.n_components}.txt",
            "w",
        ) as file:
            file.write(f"Model: {model_name}\n")
            file.write(report_test)
            file.write("\n")
            file.write("=" * 50)
            file.write(f"Best parameters: {best_model.best_params_}\n")
            file.write("\n")

            # some separator
            file.write("=" * 50)
            file.write("\n")

            # get train accuracy
            file.write(f"Train report \n")
            file.write(report_train)

            # also write the time
            file.write("\n")
            file.write(f"=" * 50)
            file.write(f"\nTime taken: {time_elapsed} seconds\n")

        logging.info(f"Report for model {model_name} has been written")

        # save the best parameters to a yaml file
        out_dir_joint = os.path.join(args.output_dir, f"best_params_{model_name}.yaml")
        with open(out_dir_joint, "w") as file:
            yaml.dump(best_params, file)


if __name__ == "__main__":
    main()
