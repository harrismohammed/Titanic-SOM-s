#Data Preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[2,4,5,6,7]].values
Y = dataset.iloc[:,[0,1]].values 

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X [:, 2])
X[:, 2] = imputer.transform(X[:, 2])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 2])
X[:, 2] = imputer.transform(X[:, 2])