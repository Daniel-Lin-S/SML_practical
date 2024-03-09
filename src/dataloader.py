"""Custom dataloader for torch."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MusicData(Dataset):
    def __init__(self, X, y, pca=False):
    
        # Encoding string labels to integers
        if isinstance(y[0], str):
            print("Encoding labels")
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

