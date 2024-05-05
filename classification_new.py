from modeldata import new_matrix

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def classification_new(title_length, author, publisher, publication_era, matrix=new_matrix):

    target = matrix['Favourite-Generation']
    values = matrix.drop(columns=['Favourite-Generation'])

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
        'Book-Title-Length': [title_length],
        'Book-Author': [author],
        'Book-Publisher': [publisher],
        'Publication-Era': [publication_era]
    })

    # Predict the labels for the new data
    predicted_labels = model.predict(new_data)
    print(f"The predicted favourite generation for this book is {predicted_labels[0]} based on its attributes.")
    return predicted_labels

classification_new(12, 'Dan Brown', 'Pocket', '1980s-1990s')
