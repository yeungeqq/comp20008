from preprocessing import matrix, books, users, ratings, booksAndRatings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

AUTHOR = 0
PUBLISHER = 1

"""

data = booksAndRatings[booksAndRatings['Book-Publisher'] == "Dell"]
data = data.groupby('User-ID')['Book-Rating'].mean().reset_index()
ratingBins = [0, 4, 7, 10]
data['Book-Rating'] = pd.cut(data['Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
matrix = pd.merge(data, matrix, on='User-ID', how='inner')
matrix['Old-Book-Rating'] = pd.cut(matrix['Old-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
matrix['New-Book-Rating'] = pd.cut(matrix['New-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
label_values = matrix[['User-ID', 'Book-Rating']].set_index('User-ID')

target = label_values.values
matrix = matrix.drop(columns=['Book-Rating'])
print(matrix)
print(target)
"""
def classification(city, state, country, generation, old_book_rating, new_book_rating, label_type, label, matrix=matrix, data=booksAndRatings):
    # list of values for labels, to be processed based on the type of target
    if label_type == AUTHOR:
        data = data[data['Book-Author'] == label]
        data = data.groupby('User-ID')['Book-Rating'].mean().reset_index()
        ratingBins = [0, 4, 7, 10]
        data['Book-Rating'] = pd.cut(data['Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        matrix = pd.merge(data, matrix, on='User-ID', how='inner')
        matrix['Old-Book-Rating'] = pd.cut(matrix['Old-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        matrix['New-Book-Rating'] = pd.cut(matrix['New-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        target = matrix['Book-Rating']
    if label_type == PUBLISHER:
        data = data[data['Book-Publisher'] == label]
        data = data.groupby('User-ID')['Book-Rating'].mean().reset_index()
        ratingBins = [0, 4, 7, 10]
        data['Book-Rating'] = pd.cut(data['Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        matrix = pd.merge(data, matrix, on='User-ID', how='inner')
        matrix['Old-Book-Rating'] = pd.cut(matrix['Old-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        matrix['New-Book-Rating'] = pd.cut(matrix['New-Book-Rating'], ratingBins, labels=["low", "moderate", "high"])
        target = matrix['Book-Rating']


    """
    need to get the subset of matrix in which users actually provide ratings
    to the targeted book/author/publisher/year of book
    """

    # list of lists which contain attribute values of users
    values = matrix.drop(columns=['Book-Rating'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(values, target, test_size=0.2, random_state=42)
    
    # Define which columns are categorical
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline for preprocessing
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Define the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier())
    ])
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # get the attributes of user, to be processed
    new_data = pd.DataFrame({
        'User-City': [city],
        'User-State': [state],
        'User-Country': [country],
        'User-Generation': [generation],
        'Old-Book-Rating': [old_book_rating],
        'New-Book-Rating': [new_book_rating]
    })

    # Predict the labels for the new data
    predicted_labels = model.predict(new_data)

    return predicted_labels

# to implement the program, enter the user attributes: city, state, country, generation, rating to old and new books,
# and the type of label (0 = author, 1 = publisher), and the targeted label value
result = classification('black mountain', 'north carolina', 'United States of America', 'genx', 'high', '',
                        1, 'Dell')
print(result)