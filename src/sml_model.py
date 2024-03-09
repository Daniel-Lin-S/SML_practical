"""Contains a collection of Statistical Machine Learning Models."""

import yaml
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance  

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV
    
from sklearn.exceptions import ConvergenceWarning


METHOD_DICT = {
    "naive_bayes": GaussianNB,
    "l_svm": SGDClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    'adaboost': AdaBoostClassifier,
    "c_svm": SVC,
    "xgboost": XGBClassifier,
    "xgboost_rf": XGBRFClassifier
}

# Load the configuration file
def load_config(config_path = 'configs/sml_configs/sml_configs_train.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # also check name matches
    for model_name in config.keys():
        assert model_name in METHOD_DICT, f"Model name {model_name} not found in METHOD_DICT"
    return config


def grid_search_cv(model_name, 
                   params_config, 
                   path_to_output,
                   X_train, 
                   y_train, 
                   cv=5, 
                   scoring='accuracy',
                   n_jobs=-1,
                   ignore_warnings=True,
                   **kwargs) -> GridSearchCV:
    """
    Perform a grid search cross-validation on the given model.
    
    This function takes in the parameter grid and
    outputs the best parameters in yaml format.
    
    Args:
        - model: The model name, should be key in the METHOD_DICT
        - params_config: the dictionary of parameters for the model
        - path_to_output: The path to the output file
        - X_train: The training data
        - y_train: The training labels
        - cv: The number of folds in the cross-validation
        - scoring: The scoring metric to use
        - n_jobs: The number of jobs to run in parallel
        - ignore_warnings: Whether or not to ignore warnings
    
    Returns:
        - The best model found
    """
    
    # initialize the model
    model = METHOD_DICT[model_name](**kwargs)
    
    # parameter grid
    param_grid = params_config
    
    # ignore convergence warnings and future warnings
    if ignore_warnings:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    # load the parameters
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    
    # fit the model
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
    



