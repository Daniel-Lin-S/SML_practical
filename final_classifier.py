import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mrmr import mrmr_classif
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

### load data ###
X = pd.read_csv('data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs 
X_test = pd.read_csv('data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs 
y = pd.read_csv('data/y_train.csv', index_col = 0).squeeze('columns') # labels
# total number of rows and columns(attributes)
n, p = np.shape(X)

### transform class labels to numerical values ###
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

### standardise ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

### perform feature selection based on MRMR score ###
n_dim = 250
selected_columns = mrmr_classif(X_scaled, y, K=n_dim)
X_reduced = X_scaled[selected_columns]
X_test_reduced = X_test_scaled[selected_columns]

### use kernel SVM classifier ###
params_SVM = {
    'kernel': 'rbf', # the kernel used
    'C': 3.0,  # regularisation strength is 1/C 
    'gamma': 'scale'  # the scale parameter of rbf kernel
} 
model = SVC(random_state=1145, **params_SVM) 
model.fit(X_reduced, y)

### produce predictions on X_test  ###
y_pred = model.predict(X_test_reduced)
y_pred = label_encoder.inverse_transform(y_pred)
prediction = pd.DataFrame(y_pred, columns=['Genre'])
prediction.index.name='Id'
prediction.to_csv('data/myprediction.csv') # export to csv file