"""Evaluates the sml models by cross validation and saves the output."""

import os
import yaml
import time
import timeit
import argparse
import logging

import numpy as np
import pandas as pd

import torch

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, Checkpoint

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


from src.sml_model import CustomDimReduction
from src.data_utils import MusicDataset
from src.dataloader import MusicData
from src.nn_model import MLP
from evaluate_sml import get_top_dirs


# paths to the data
PATH_to_X = "data/X_train.csv"
PATH_to_y = "data/y_train.csv"

# get system time
TIME = time.strftime("%Y-%m-%d_%H-%M-%S")


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

METHOD_DICT = {
    "mlp": NeuralNetClassifier,
}


def get_scores_cv(
    method: str,
    model: NeuralNetClassifier,
    reduction_method: str = None,
    n_components: int = None,
    cv: int = 5,
    best_params: dict = None,
):
    """
    Returns an array of scores across different cv folds.
    one of the folds is used to produce
    the classification report and confusion matrix.
    Also the time to fit the model is returned.

    Args:
        - method: The model name
        - model: The model object
        - reduction_method: The method to use for dimensionality reduction
        - n_components: The number of components to reduce to
        - cv: The number of folds in the cross-validation
        - best_params: The best parameters for the model

    Returns:
        - score_list: The list of scores
        - classification_reports_train: The list of classification reports for the training data
        - confusion_matrices_train: The list of confusion matrices for the training data
        - classification_reports_test: The list of classification reports for the test data
        - confusion_matrices_test: The list of confusion matrices for the test data
        - time_elapsed: The time taken to fit the model
    """
    # getting the folds
    all_music_dataset = MusicDataset(
        path_to_X=PATH_to_X,
        path_to_y=PATH_to_y,
        test_size=0.2,
        swap_axes=False,
        features_to_drop=None,
        shuffle=True,
        random_state=42,
    )

    # music data returns an iterator of dictionaries
    all_data = list(
        all_music_dataset.get_data(
            scaler="standard",
            reduction_method=reduction_method,
            n_components=n_components,
            k_fold_splits=cv,
            use_all_data=True,
        )
    )

    score_list = []
    classification_reports_train = []
    confusion_matrices_train = []
    classification_reports_test = []
    confusion_matrices_test = []

    for data_dict in all_data:
        # extract the already preprocessed data
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_test = data_dict["X_val"]
        y_test = data_dict["y_val"]

        # init the model
        new_best_params = {}
        for key in best_params[method].keys():
            if "__" in key:
                # only the first component is not required
                new_key = "__".join(key.split("__")[1:])
                new_best_params[new_key] = best_params[method][key]

            # change the keys from mlp__layer to layer
            if reduction_method is None:
                new_best_params["module__input_dim"] = X_train.shape[1]

            elif reduction_method is not None:
                new_best_params["module__input_dim"] = n_components

            new_best_params["module"] = MLP
            new_best_params["max_epochs"] = 500
            new_best_params["criterion"] = torch.nn.CrossEntropyLoss
            new_best_params["optimizer"] = torch.optim.AdamW
            new_best_params["train_split"] = predefined_split(MusicData(X_test, y_test))
            
            checkpoint_dir = f"checkpoints/{best_params}_{reduction_method}_{n_components}"
            
            # callbacks
            es = EarlyStopping(patience=20)
            cp = Checkpoint(dirname=checkpoint_dir, monitor='valid_acc_best')
            
            new_best_params["callbacks"] = [
                es,
                cp,
            ]
            new_best_params["device"] = "cpu"

            model = METHOD_DICT[method](**new_best_params)
            
            # change dtype if pandas
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.to_numpy().astype(np.float32)
                X_test = X_test.to_numpy().astype(np.float32)
            else:
                X_train = X_train.astype(np.float32)
                X_test = X_test.astype(np.float32)
            
            if isinstance(y_train, pd.Series):
                y_train = y_train.to_numpy()
                y_test = y_test.to_numpy()
                
            
            # fitting the model
            start = timeit.default_timer()
            model.fit(X_train, y_train)
            end = timeit.default_timer()

            time_elapsed = end - start

            # evaluation
            model.initialize()
            # load checkpoints
            model.load_params(checkpoint=cp)
            
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            score_list.append(score)

            # classification report
            classification_report_train = classification_report(
                y_train, model.predict(X_train), digits=4
            )
            classification_reports_train.append(classification_report_train)

            classification_report_test = classification_report(
                y_test, model.predict(X_test), digits=4
            )
            classification_reports_test.append(classification_report_test)

            # confusion matrix
            confusion_matrix_train = confusion_matrix(y_train, model.predict(X_train))
            confusion_matrices_train.append(confusion_matrix_train)

            confusion_matrix_test = confusion_matrix(y_test, model.predict(X_test))
            confusion_matrices_test.append(confusion_matrix_test)

    return (
        score_list,
        classification_reports_train,
        confusion_matrices_train,
        classification_reports_test,
        confusion_matrices_test,
        time_elapsed,
    )


def main():

    top_dirs = get_top_dirs()

    methods_list = [
        "mlp",
    ]

    # intialize the models with best hyperparameters and evaluate using crossval
    for method in methods_list:

        logging.info(f"Evaluating model {method}...")

        if method not in METHOD_DICT:
            logging.warning(f"Model {method} not found in METHOD_DICT")
            continue

        # load the configuration file
        paths_to_load = [
            os.path.join(top_dir, f"best_params_{method}.yaml") for top_dir in top_dirs
        ]

        for path in paths_to_load:
            _red_method = path.split("/")[-2]
            if _red_method == "none":
                logging.info("No reduction method")
                red_method = None
            else:
                red_method = _red_method.split("_")[0]
                n_components = int(_red_method.split("_")[-1])
                logging.info(f"Reduction method: {_red_method}")
                logging.info(f"Number of components: {int(_red_method.split('_')[-1])}")

            with open(path, "r") as file:
                best_params = yaml.safe_load(file)

            # write the mean of scores and confidence interval
            # as well as classification report to evaluation/{_red_method}/{method}.txt
            red_method_name = (
                f"{red_method}_{n_components}" if red_method is not None else "none"
            )
            output_dir = os.path.join("evaluation/mlp", red_method_name)
            os.makedirs(output_dir, exist_ok=True)

            # ready to output
            (
                scores,
                classification_reports_train,
                confusion_matrices_train,
                classification_reports_test,
                confusion_matrices_test,
                time_elapsed,
            ) = get_scores_cv(
                method,
                METHOD_DICT[method],
                reduction_method=red_method,
                n_components=n_components,
                cv=5,
                best_params=best_params,
            )
            

            output_file = os.path.join(output_dir, f"{method}.txt")
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            confidence_interval = (
                mean_score - 2 * std_score,
                mean_score + 2 * std_score,
            )

            with open(output_file, "w") as f:
                f.write(f"Mean Score: {mean_score}\n")
                f.write("=" * 50)
                f.write("\n")
                f.write(f"Confidence Interval: {confidence_interval}\n")
                f.write("=" * 50)
                f.write("\n")
                f.write(f"Time Elapsed: {time_elapsed}\n")
                f.write("=" * 50)
                f.write("\n")
                f.write(f"Classification Report:\n{classification_reports_train[-1]}\n")
                f.write("=" * 50)
                f.write("\n")
                f.write(f"Classification Report Test:\n{classification_reports_test[-1]}\n")

            # write confusion matrix to file
            np.savez(
                os.path.join(output_dir, f"{method}_confusion_matrix_train.npz"),
                confusion_matrices_train[-1],
            )

            np.savez(
                os.path.join(output_dir, f"{method}_confusion_matrix_test.npz"),
                confusion_matrices_test[-1],
            )


if __name__ == "__main__":
    main()
