from preprocessing import matrix, books, users, ratings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

BOOK = 0
AUTHOR = 1
PUBLISHER = 2
YEAR = 3

def classification(target_user_id, label_type, label, matrix=matrix):
    # list of values for labels, to be processed based on the type of target
    if label_type == BOOK:
        target = label
    if label_type == AUTHOR:
        target = label
    if label_type == PUBLISHER:
        target = label
    if label_type == YEAR:
        target = label

    """
    need to get the subset of matrix in which users actually provide ratings
    to the targeted book/author/publisher/year of book
    """

    # list of lists which contain attribute values of users
    values = matrix.set_index('User-ID').values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(values, target, test_size=0.2, random_state=42)

    # Train the decision tree classifier
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = decision_tree.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # get the attributes of user, to be processed
    new_data = target_user_id

    # Predict the labels for the new data
    predicted_labels = decision_tree.predict(new_data)

    return predicted_labels