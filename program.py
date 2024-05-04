import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# read files and store in pandas dataframe
allData = pd.read_csv("datasets/combinedData.csv")
discretizedData = pd.read_csv("datasets/discretizedData.csv")

"""
Classification
Given a collection of records (training set), each record contains a set of attributes, one class label.
Find a predictive model for class label as a function of the values of other attributes.
"""

# Separate the data into train and test set
# train set size 80%, test set size 20%
train, test = train_test_split(discretizedData, test_size=0.2, random_state=42, shuffle=True)

# save the train and test file
train.to_csv('datasets/train.csv', index=False)
test.to_csv('datasets/text.csv', index=False)

# Create a training model
numericData = allData
"""
OPT 1: Decision Tree
"""
# Change string values into numerical values
unique_countries = users['User-Country'].unique()
countries = {value: idx for idx, value in enumerate(unique_countries)}
print(countries)
numericData['User-Country'] = numericData['User-Country'].map(countries)

unique_publishers = books['Book-Publisher'].unique()
publishers = {value: idx for idx, value in enumerate(unique_publishers)}
print(publishers)
numericData['Book-Publisher'] = numericData['Book-Publisher'].map(publishers)

# Separate the feature columns from the target column.
# Target variable (discrete) [y] : rating
# columns [x1, x2...]: attributes / predictors
# Create a predictive model [f(x...)]
attributes = ['User-Age', 'User-Country', 'Book-Publisher', 'Year-Of-Publication']

X = numericData[attributes]
y = numericData['Book-Rating']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=attributes)

"""
OPT 2: k Nearest Neighbour
"""

