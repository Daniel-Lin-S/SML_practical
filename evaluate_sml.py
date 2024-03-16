"""Evaluates the sml models by cross validation and saves the output."""

import os
import yaml
import time
import timeit
import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


from src.sml_model import *
from src.data_utils import MusicDataset

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


def get_top_dirs():
    base_dir = "configs/sml_configs"

    reduction_methods = ["mrmr", "igr", "pca"]

    num_components = [7, 50, 100, 250]

    # top directories to save the output
    top_dirs = []

    # add LDA
    top_dirs.append(os.path.join(base_dir, "lda_7"))

    # add no reduction
    # top_dirs.append(os.path.join(base_dir, "none"))

    for method in reduction_methods:
        for n in num_components:
            top_dirs.append(os.path.join(base_dir, f"{method}_{n}"))

    return top_dirs


def main():

    top_dirs = get_top_dirs()
    parser = argparse.ArgumentParser(description="Run the entire pipeline.")
    parser.add_argument(
        "--use_cv_score",
        type=bool,
        default=True,
        help="Whether to use cross validation score, if not then only report the test score using the remaining data",
    )
    use_cv_score = parser.parse_args().use_cv_score

    # deprecated for mlp
    methods_list = [
        # "mlp",
        "c_svm",
        "naive_bayes",
        "knn",
        "lda",
        "qda",
        # "xgboost_rf",
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
            # load the data
            X_all = pd.read_csv(PATH_to_X, index_col=0, header=[0, 1, 2])
            y_all = pd.read_csv(PATH_to_y, index_col=0)

            X_train = X_all
            le = LabelEncoder()
            y_train = le.fit_transform(y_all.values.ravel())

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

            # load the best parameters
            # initialize the model
            try:
                new_best_params = {}
                for key in best_params[method].keys():
                    if "__" in key:
                        new_key = key.split("__")[-1]
                        new_best_params[new_key] = best_params[method][key]

                _model = METHOD_DICT[method](**new_best_params)
            except AttributeError:
                logging.info(f"No keys found, using default parameters for {method}...")
                _model = METHOD_DICT[method]()

            pipes = []
            pipes.append(("scaler", StandardScaler()))

            if red_method is not None:
                pipes.append(
                    (
                        "reducer",
                        CustomDimReduction(
                            method=red_method,
                            n_components=n_components,
                            feature_columns=X_train.columns,
                        ),
                    )
                )

            pipes.append((method, _model))

            # construct the pipeline
            model = pipeline.Pipeline(pipes)

            if use_cv_score:
                scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
                )
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                confidence_interval = (
                    mean_score - 2 * std_score,
                    mean_score + 2 * std_score,
                )

            # write the mean of scores and confidence interval
            # as well as classification report to evaluation/{_red_method}/{method}.txt
            red_method_name = (
                f"{red_method}_{n_components}" if red_method is not None else "none"
            )
            output_dir = os.path.join("evaluation", red_method_name)
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{method}.txt")
            

            # fit for classification report separately
            # the split is not random, so we are using the same test set as in cv
            music_data = MusicDataset(
                path_to_X="data/X_train.csv",
                path_to_y="data/y_train.csv",
                test_size=0.2,
                swap_axes=False,
                features_to_drop=None,
                shuffle=False,
            )

            X_train_c, X_test_c, y_train_c, y_test_c = music_data.get_data(
                scaler="standard",
                reduction_method=red_method,
                n_components=n_components,
            )

            # torch wrapper requires numpy arrays
            if method == "mlp":
                if isinstance(X_train_c, pd.DataFrame):
                    X_train_c = X_train_c.to_numpy()
                    X_test_c = X_test_c.to_numpy()

                if isinstance(y_train_c, pd.DataFrame):
                    y_train_c = y_train_c.to_numpy().ravel()
                    y_test_c = y_test_c.to_numpy().ravel()

                new_best_params["device"] = "cpu"
                if red_method is None:
                    new_best_params["input_dim"] = X_train_c.shape[1]
                else:
                    new_best_params["input_dim"] = n_components

            # re-initialize the model
            _model_c = METHOD_DICT[method](**new_best_params)

            logging.info(
                f"Fitting model {method} with best parameters {new_best_params}..."
            )
            start = timeit.default_timer()
            _model_c.fit(X_train_c, y_train_c)
            end = timeit.default_timer()

            time_elapsed = end - start

            logging.info(f"Model Fitted. Time elapsed: {time_elapsed}")

            # classification report
            classification_report_train = classification_report(
                y_train_c, _model_c.predict(X_train_c), digits=4
            )

            classification_report_test = classification_report(
                y_test_c, _model_c.predict(X_test_c), digits=4
            )

            # confusion matrix
            confusion_matrix_train = confusion_matrix(
                y_train_c, _model_c.predict(X_train_c)
            )
            confusion_matrix_test = confusion_matrix(
                y_test_c, _model_c.predict(X_test_c)
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
                f.write(f"Classification Report:\n{classification_report_train}\n")
                f.write("=" * 50)
                f.write("\n")
                f.write(f"Classification Report Test:\n{classification_report_test}\n")

            # write confusion matrix to file
            np.savez(
                os.path.join(output_dir, f"{method}_confusion_matrix_train.npz"),
                confusion_matrix_train,
            )

            np.savez(
                os.path.join(output_dir, f"{method}_confusion_matrix_test.npz"),
                confusion_matrix_test,
            )


if __name__ == "__main__":
    main()
