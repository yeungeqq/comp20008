import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

# Load data
users = pd.read_csv('datasets/BX-Users-processed.csv')
books = pd.read_csv('datasets/BX-Books-processed.csv')
features = pd.read_csv('features.csv')
features = features['Features'].tolist()
features.append('Book-Rating-Tier')
matrix = pd.read_csv('datasets/discretizedData.csv').fillna('NA')
matrix = matrix[features]

DT = 'dt'
KNN = 'knn'
method = sys.argv[1] if len(sys.argv) > 1 else None
predict_rating = sys.argv[2] if len(sys.argv) > 2 else 'no'

if method not in [DT, KNN]:
    print(f"Invalid method. Please specify '{DT}' for decision tree or '{KNN}' for k-nearest neighbours.")
    sys.exit(1)

def model_all_ratings(method, predict_rating, users, books, matrix, features):
    # Split features and target
    target = matrix['Book-Rating-Tier']
    values = matrix.drop(columns=['Book-Rating-Tier'])

    # Define which columns are categorical
    categorical_cols = values.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Initialize the model pipeline with a classifier placeholder
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier() if method == DT else KNeighborsClassifier())
    ])

    # Set up parameter grids
    param_grid = {
        'classifier__max_depth': [None, 10, 20, 30] if method == DT else None,
        'classifier__min_samples_leaf': [1, 5, 10] if method == DT else None,
        'classifier__n_neighbors': [3, 5, 7, 10] if method == KNN else None,
        'classifier__weights': ['uniform', 'distance'] if method == KNN else None
    }

    # Configure GridSearchCV
    grid_search = GridSearchCV(model, {k: v for k, v in param_grid.items() if v is not None}, cv=5, scoring='accuracy')
    grid_search.fit(values, target)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Generate classification report
    X_train, X_test, y_train, y_test = train_test_split(values, target, test_size=0.1, random_state=42)
    y_pred = grid_search.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if predict_rating == 'yes':
        user_id = input("Enter User ID: ")
        ISBN = input("Enter ISBN: ")
        user_data = users[users['User-ID'] == int(user_id)]
        book_data = books[books['ISBN'] == ISBN]
        input_data = pd.concat([user_data, book_data], axis=1).reindex(columns=features[:-1])  # Exclude target variable
        prediction = grid_search.predict(input_data)
        print(f"The predicted rating tier for this book by the user is {prediction[0]}.")

if __name__ == "__main__":
    model_all_ratings(method, predict_rating, users, books, matrix, features)
