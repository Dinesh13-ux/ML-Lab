import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# convert numeric values to categories
for c in df.columns[:-1]:
    df[c] = pd.cut(df[c],3,labels=['Low','Med','High'])

# ENTROPY
def entropy(y):
    p = y.value_counts()/len(y)
    return -sum(p*np.log2(p))

# GINI INDEX
def gini(y):
    p = y.value_counts()/len(y)
    return 1 - sum(p**2)

# INFORMATION GAIN
def info_gain(data, feature):
    
    total_entropy = entropy(data['target'])
    
    values = data[feature].unique()
    weighted_entropy = 0
    
    
    for v in values:
        subset = data[data[feature]==v]
        weight = len(subset)/len(data)
        weighted_entropy += weight * entropy(subset['target'])
    
    return total_entropy - weighted_entropy


print("Entropy:", entropy(df['target']))
print("Gini:", gini(df['target']))

for col in df.columns[:-1]:
    print(col, "Info Gain:", info_gain(df,col))