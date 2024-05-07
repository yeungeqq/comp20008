import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
users = pd.read_csv('datasets/BX-Users-processed.csv')
books = pd.read_csv('datasets/BX-Books-processed.csv')
ratings = pd.read_csv('datasets/BX-Ratings-processed.csv')
features = pd.read_csv('datasets/features.csv')
features = features['Features'].to_list()
features.append('Book-Rating')
matrix = pd.merge(ratings, users, on='User-ID', how='inner')
matrix = pd.merge(matrix, books, on='ISBN', how='inner')
matrix = matrix.fillna('NA')
matrix = matrix[features]

def predict_ratings(user_id, ISBN, users=users, books=books, matrix=matrix, features=features):
    # Ensure ISBN is treated as string
    books['ISBN'] = books['ISBN'].astype(str)

    # Split features and target
    target = matrix['Book-Rating']
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

    # Define the model pipeline
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

    # Retrieve user and book attributes
    user = users[users['User-ID'] == user_id]
    book = books[books['ISBN'] == ISBN]
    user_attributes = user.columns.values.tolist()
    book_attributes = book.columns.values.tolist()
    final_user_attb = [attb for attb in user_attributes if attb in features]
    final_book_attb = [attb for attb in book_attributes if attb in features]

    # Combine user and book attributes for prediction
    new_data = user[final_user_attb].values.tolist()[0] + book[final_book_attb].values.tolist()[0]
    new_data = pd.DataFrame({
        'User-City': [new_data[0]],
        'User-State': [new_data[1]],
        'User-Country': [new_data[2]],
        'User-Age': [new_data[3]],
        'Book-Author': [new_data[4]]
    })

    # Predict the labels for the new data
    predicted_labels = model.predict(new_data)
    print(f"The predicted rating for this book is {predicted_labels[0]} by the user.")
    return predicted_labels

predict_ratings(78, '0671021001')
