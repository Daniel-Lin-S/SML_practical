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
    
    def __init__(self,  
                 path_to_X = 'data/X_train.csv',
                 path_to_y = 'data/y_train.csv',
                 test_size=0.2,
                 random_state=42,
                 shuffle=False):
        """
        Initializes the dataset with the given data and target.
        
        Args:
            - path_to_X: The path to the input data
            - path_to_y: The path to the target data
            - test_size: The proportion of the dataset to include in the test split, 
                if 0 then no test set
            - random_state: The seed used by the random number generator
            - shuffle: Whether or not to shuffle the data before splitting
        """
        
        # converting the data to numpy arrays
        self.X = pd.read_csv(path_to_X, 
                             index_col = 0, 
                             header=[0, 1, 2]).to_numpy()
        
        self.y = pd.read_csv(path_to_y, index_col=0).squeeze('columns')
        
        # encode y
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        
        
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        
        
    def split_data(self):
        """
        Splits the data into training and test sets.
        
        Returns:
            - X_train: The training input data
            - X_test: The test input data
            - y_train: The training target data
            - y_test: The test target data
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            test_size=self.test_size, 
                                                            random_state=self.random_state,
                                                            shuffle=self.shuffle)
        return X_train, X_test, y_train, y_test
    
    
    def scale_data(self, scaler='standard'):
        """
        Scales the input data using the given scaler.
        
        Args:
            - scaler: The type of scaler to use (standard, minmax)
        
        Returns:
            - X_train_scaled: The scaled training input data
            - X_test_scaled: The scaled test input data
        """
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Use 'standard' or 'minmax'.")
        
        # first split
        X_train, X_test, _, _ = self.split_data()
        
        # then scale (using the training data only)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    
    def reduce_dimensions(self, method='pca', n_components=2):
        """
        Reduces the dimensionality of the input data using the given method.
        
        Args:
            - method: The dimensionality reduction method (pca, lda)
            - n_components: The number of components to keep
        
        Returns:
            - X_train_reduced: The reduced training input data
            - X_test_reduced: The reduced test input data
        """
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'lda':
            assert n_components <= len(np.unique(self.y))-1, "n_components must be less than the number of classes."
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
        else:
            raise ValueError("Invalid reduction method. Use 'pca' or 'lda'.")
        
        # first split
        X_train, X_test, y_train, y_test = self.split_data()
        
        # then reduce (using the training data only)
        reducer.fit(X_train, self.y)
        X_train_reduced = reducer.transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        
        return X_train_reduced, X_test_reduced, y_train, y_test
    
    
    def k_fold_cv(self, n_splits=5):
        """
        Performs k-fold cross-validation on the training data.
        
        Args:
            - n_splits: The number of folds
        
        Returns:  
            - list of dictionaries of datasets
                keys: 'X_train', 'X_val', 'y_train', 'y_val'
        """
        kf = KFold(n_splits=n_splits)
        
        # first split
        X_train, _, y_train, _ = self.split_data()
        
        # then create the datasets
        datasets = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            datasets.append({'X_train': X_train_fold, 'X_val': X_val_fold, 
                             'y_train': y_train_fold, 'y_val': y_val_fold})
        
        return datasets