"""Functions for data preprocessing and augmentation."""

import warnings
import numpy as np
import pandas as pd

from functools import partial

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# import train-test split, note if sklearn is old, use cross_validation
try:
    from sklearn.model_selection import train_test_split, KFold
except ImportError:
    from sklearn.cross_validation import train_test_split, KFold

# Dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


class MusicDataset:
    """
    A class for loading and preprocessing the music dataset.

    This processes the training set given. DO NOT use this class for the X-test set.

    If final method is determined, we use the full training set to train the model.
    """

    def __init__(
        self,
        path_to_X="data/X_train.csv",
        path_to_y="data/y_train.csv",
        test_size=0.2,
        random_state=42,
        shuffle=False,
        swap_axes=False,
        feature_to_drop = None,
    ):
        """
        Initializes the dataset with the given data and target.

        Args:
            - path_to_X: The path to the input data
            - path_to_y: The path to the target data
            - test_size: The proportion of the dataset to include in the test split,
                if 0 then no test set
            - random_state: The seed used by the random number generator
            - shuffle: Whether or not to shuffle the data before splitting
            - swap_axes: Whether to swap the axes of the input data
            - feature_to_drop: The list of features to drop from the input data
        """

        # converting the data to numpy arrays
        self.path_to_X = path_to_X
        
        self.X_raw = pd.read_csv(self.path_to_X, index_col=0, header=[0, 1, 2])
        
        if swap_axes:
            # so that we have 
            # (group1), ..., (group11)
            # with each groups has [subgroup1,...,subgroupX]
            # subgroup consists of [kurtosis, mean, ..., std]
            # then can split into 7 x 74 = 518 features
            self.X = self.X_raw.swaplevel(axis=1).sort_index(axis=1)
            
        if feature_to_drop is not None:
            print(f"Dropping features: {feature_to_drop}")
            self.X = self.X_raw.drop(columns=feature_to_drop, axis=1)
            print(f"New shape: {self.X.shape}")
        
        self.X = self.X.to_numpy()

        self.y = pd.read_csv(path_to_y, index_col=0).squeeze("columns")

        # encode y
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

                
    def scale_data(self, X_train, X_test, y_train, y_test, scaler="standard"):
        """
        Scales the input data using the given scaler.

        Args:
            - X_train: The training input data
            - X_test: The test input data
            - y_train: The training target data
            - y_test: The test target data
            - scaler: The type of scaler to use (standard, minmax)

        Returns:
            - X_train_scaled: The scaled training input data
            - X_test_scaled: The scaled test input data
        """
        if scaler == "standard":
            scaler = StandardScaler()
        elif scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Use 'standard' or 'minmax'.")

        # then scale (using the training data only)
        # probably do not need copy
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled


    def reduce_dimensions(
        self, X_train, X_test, y_train, y_test, method="pca", n_components=2
    ):
        """
        Reduces the dimensionality of the input data using the given method.

        Args:
            - X_train: The training input data
            - X_test: The test input data
            - y_train: The training target data
            - y_test: The test target data
            - method: The dimensionality reduction method (pca, lda)
            - n_components: The number of components to keep

        Returns:
            - X_train_reduced: The reduced training input data
            - X_test_reduced: The reduced test input data
        """

        if method == "pca":

            reducer = PCA(n_components=n_components)
            reducer.fit(X_train)

            X_train_reduced = reducer.transform(X_train)
            X_test_reduced = reducer.transform(X_test)

        elif method == "lda":
            # assert that n_components is less than the number of classes
            assert (
                n_components <= len(np.unique(y_test)) - 1
            ), "n_components must be less than the number of classes."

            reducer = LinearDiscriminantAnalysis(n_components=n_components)
            reducer.fit(X_train, y_train)

            X_train_reduced = reducer.transform(X_train)
            X_test_reduced = reducer.transform(X_test)

        else:
            raise ValueError("Invalid reduction method. Use 'pca' or 'lda'.")

        return X_train_reduced, X_test_reduced

    def get_data(
        self,
        scaler="standard",
        reduction_method="pca",
        n_components=2,
        k_fold_splits=0,
        use_all_data=False,
    ):
        """
        Returns the preprocessed data.

        Args:
            - scaler: The type of scaler to use (standard, minmax), can be None
            - reduction_method: The dimensionality reduction method (pca, lda), can be None
            - n_components: The number of components to keep
            - k_fold_splits: The number of folds for k-fold cross-validatio
            - use_all_data: Whether to use all the data for k-fold cross-validation

        Returns:
            - X_train, X_test, y_train, y_test: if k_fold_splits=0
            - k_fold_datasets: if k_fold_splits>0
        """
        # scale the data
        if scaler is not None:
            try:
                X_train_scaled, X_test_scaled = self.scale_data(
                    self.X_train, self.X_test, self.y_train, self.y_test, scaler=scaler
                )
            except ValueError as e:
                warnings.warn(str(e))
                X_train_scaled, X_test_scaled = self.X_train, self.X_test

        else:
            print("No scaling applied.")
            X_train_scaled, X_test_scaled = self.X_train, self.X_test

        # TODO: does the order of scaling and reduction matter?
        # reduce the dimensionality after scaling

        if reduction_method is not None:
            try:
                X_train_reduced, X_test_reduced = self.reduce_dimensions(
                    X_train_scaled,
                    X_test_scaled,
                    self.y_train,
                    self.y_test,
                    method=reduction_method,
                    n_components=n_components,
                )
            except ValueError as e:
                warnings.warn(str(e))
                X_train_reduced, X_test_reduced = X_train_scaled, X_test_scaled

        else:
            print("No dimensionality reduction applied.")
            X_train_reduced, X_test_reduced = X_train_scaled, X_test_scaled

        if k_fold_splits == 0:
            return X_train_reduced, X_test_reduced, self.y_train, self.y_test
        else:
            if use_all_data:
                X_to_use = self.X
                y_to_use = self.y
            else:
                X_to_use = self.X_train
                y_to_use = self.y_train
                
            return self.get_k_fold_cv_iterator(
                    X_to_use,
                    y_to_use,
                    scaler,
                    reduction_method,
                    n_components,
                    k_fold_splits,
                )

    def get_k_fold_cv_iterator(
        self,
        X_used,
        y_used,
        scaler="standard",
        reduction_method="pca",
        n_components=2,
        n_splits=5,
    ):
        """
        Returns an iterator that yields the k-fold cross-validation datasets.

        Args:
            - X_used: The input data to use
            - y_used: The target data to use
            - scaler: The type of scaler to use (standard, minmax)
            - reduction_method: The dimensionality reduction method (pca, lda)
            - n_components: The number of components to keep
            - n_splits: The number of folds

        Yields:
            - dictionary with keys: 'X_train', 'X_val', 'y_train', 'y_val'
        """
        kf = KFold(n_splits=n_splits)

        for train_index, val_index in kf.split(X_used):
            X_train_fold, X_val_fold = X_used[train_index], X_used[val_index]
            y_train_fold, y_val_fold = y_used[train_index], y_used[val_index]

            try:
                X_train_fold, X_val_fold = self.scale_data(
                    X_train_fold, X_val_fold, y_train_fold, y_val_fold, scaler=scaler
                )
            except ValueError as e:
                warnings.warn(str(e))
                X_train_fold, X_val_fold = X_train_fold, X_val_fold

            try:
                X_train_fold, X_val_fold = self.reduce_dimensions(
                    X_train_fold,
                    X_val_fold,
                    y_train_fold,
                    y_val_fold,
                    method=reduction_method,
                    n_components=n_components,
                )
            except ValueError as e:
                warnings.warn(str(e))
                X_train_fold, X_val_fold = X_train_fold, X_val_fold

            yield {
                "X_train": X_train_fold,
                "X_val": X_val_fold,
                "y_train": y_train_fold,
                "y_val": y_val_fold,
            }
