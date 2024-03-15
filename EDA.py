# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import mrmr
from mrmr import mrmr_classif

### data loading ###
X = pd.read_csv('data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs 
y = pd.read_csv('data/y_train.csv', index_col = 0).squeeze('columns') # labels

# total number of rows and columns(attributes)
n, p = np.shape(X)

### check class weights ###
print(y.value_counts() / n) # equal weights

### correlation test ###
chroma = '02'
group1_feature = 'chroma_cens'
group2_feature = 'chroma_cqt'
group3_feature = 'chroma_stft'
group1 = X.loc[:, (X.columns.get_level_values('feature') == group1_feature) & (X.columns.get_level_values('number') == chroma)]
group1_ticks = group1.columns.get_level_values('statistics')
group2 = X.loc[:, (X.columns.get_level_values('feature') == group2_feature) & (X.columns.get_level_values('number') == chroma)]
group2_ticks = group2.columns.get_level_values('statistics')
group3 = X.loc[:, (X.columns.get_level_values('feature') == group3_feature) & (X.columns.get_level_values('number') == chroma)]
group3_ticks = group3.columns.get_level_values('statistics')

# Calculate correlation matrices between groups
corr_group1_group2 = np.corrcoef(group1, group2, rowvar=False)
corr_group1_group3 = np.corrcoef(group1, group3, rowvar=False)
corr_group2_group3 = np.corrcoef(group2, group3, rowvar=False)

# Plot heatmaps
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
ax1 = sns.heatmap(corr_group1_group2[:7, 7:], cmap='RdBu', xticklabels=group1_ticks, yticklabels=group2_ticks, 
            vmin=-1, vmax=1)
plt.title(f'Correlation Matrix of {group1_feature} vs {group2_feature} \n for chroma {chroma}')

plt.subplot(1, 3, 2)
ax2 = sns.heatmap(corr_group1_group3[:7, 7:], cmap='RdBu', xticklabels=group1_ticks, yticklabels=group3_ticks, 
            vmin=-1, vmax=1)
plt.title(f'Correlation Matrix of {group1_feature} vs {group3_feature} \n for chroma {chroma}')

plt.subplot(1, 3, 3)
ax3 = sns.heatmap(corr_group2_group3[:7, 7:], cmap='RdBu', xticklabels=group2_ticks, yticklabels=group3_ticks, 
            vmin=-1, vmax=1)
plt.title(f'Correlation Matrix of {group2_feature} vs {group3_feature} \n for chroma {chroma}')

# Rotate xticks and yticks
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=45)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('figures/correlation_chroma_chroma.png', dpi=300)

# Calculate the correlation matrix of all features
correlation_matrix = X.corr()

# Visualise the distribution of correlation values
plt.figure(figsize=(10, 6))
sns.histplot(correlation_matrix.values.flatten(), bins=50, kde=True)
plt.title('Distribution of Correlation Coefficients')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.savefig('figures/correlation_hist.png', dpi=300)

### test normality of each class(genre) ###
# Iterate over unique class labels
for label in np.unique(y):
    # Select data points corresponding to the current class
    X_class = X[y == label]
    
    # Perform Shapiro-Wilk test
    stat, p = shapiro(X_class)
    print(f"Genre {label}: Shapiro-Wilk test statistic = {stat}, p-value = {p}")
    # Perform Kolmogorov-Smirnov Test 
    stat, p = kstest(X_class, 'norm')
    print(f"Genre {label}: Shapiro-Wilk test statistic = {stat}, p-value = {p}")