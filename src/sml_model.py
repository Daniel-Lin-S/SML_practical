"""Contains a collection of Statistical Machine Learning Models."""

import yaml
import warnings

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier, XGBRFClassifier

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV

from sklearn import pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_utils import CustomDimReduction


METHOD_DICT = {
    "naive_bayes": GaussianNB,
    "l_svm": SGDClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "adaboost": AdaBoostClassifier,
    "c_svm": SVC,
    "xgboost": XGBClassifier,
    "xgboost_rf": XGBRFClassifier,
}


# Load the configuration file
def load_config(config_path="configs/sml_configs/sml_configs_train.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # also check name matches
    for model_name in config.keys():
        assert (
            model_name in METHOD_DICT
        ), f"Model name {model_name} not found in METHOD_DICT"
    return config


def grid_search_cv(
    model_name,
    params_config,
    path_to_output,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    ignore_warnings=True,
    verbose=0,
    scaling_method="standard",
    reduction_method="pca",
    n_components=2,
    **kwargs,
) -> GridSearchCV:
    """
    Perform a grid search cross-validation on the given model.

    This function takes in the parameter grid and
    outputs the best parameters in yaml format.

    Args:
        - model: The model name, should be key in the METHOD_DICT
        - params_config: the dictionary of parameters for the model
        - path_to_output: The path to the output file
        - X_train: The training data, pd.DataFrame
        - y_train: The training labels
        - cv: The number of folds in the cross-validation
        - scoring: The scoring metric to use
        - n_jobs: The number of jobs to run in parallel
        - ignore_warnings: Whether or not to ignore warnings
        - verbose: The verbosity level
        - scaling_method: The method to use for scaling the data
        - reduction_method: The method to use for dimensionality reduction
        - n_components: The number of components to reduce to

    Returns:
        - The best model found
    """

    # initialize the model
    _model = METHOD_DICT[model_name](**kwargs)
    
    if reduction_method == "mrmr":
        assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame when using mrmr"

    pipes = []
    if scaling_method is not None:
        if scaling_method == "standard":
            pipes.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            pipes.append(("scaler", MinMaxScaler()))
        else:
            raise ValueError(f"Scaling method {scaling_method} not found")

    if reduction_method is not None:
        pipes.append(
            (
                "reducer",
                CustomDimReduction(method=reduction_method, 
                                   n_components=n_components,
                                   feature_columns=X_train.columns),
            )
        )
    
    pipes.append((model_name, _model))
    
    # construct the pipeline
    model = pipeline.Pipeline(pipes)

    # parameter grid
    param_grid = params_config

    # ignore convergence warnings and future warnings
    if ignore_warnings:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    # load the parameters
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=n_jobs, 
        verbose=verbose, 
        error_score='raise'
    )

    # fit the model
    print(X_train.shape, y_train.shape)
    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_sml_model(model, X_test, y_test):
    """
    Evaluate the given model on the test data.

    Args:
        - model: The model to evaluate
        - X_test: The test data
        - y_test: The test labels

    Returns:
        - The accuracy of the model on the test data
    """
    return model.score(X_test, y_test)
