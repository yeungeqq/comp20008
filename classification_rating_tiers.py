import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import sys

# Load data
users = pd.read_csv('datasets/BX-Users-processed.csv')
books = pd.read_csv('datasets/BX-Books-processed.csv')
features = pd.read_csv('features.csv')
features = features['Features'].to_list()
features.append('Book-Rating-Tier')
matrix = pd.read_csv('datasets/discretizedData.csv')
matrix = matrix.fillna('NA')
matrix = matrix[features]

DT = 'dt'
KNN = 'knn'
method = None
YES = 'y'
NO = 'n'
predict_rating = None

try:
    method = sys.argv[1]
    predict_rating = sys.argv[2]
except:
    print("Please enter inputs.")

if method == DT: print('You selected decision tree as the algorithm.')
elif method == KNN: print('You selected K-nearest neighbour as the algorithm.')

def visualiseConfusionMatrix(cm):
    labels = np.array([["True Positive", "False Negative"], ["False Positive", "True Negative"]])  # Adjust based on your specific confusion matrix

    # Calculate normalization to add text color contrast
    norm_cm = cm / cm.max()

    # change legends
    legends = np.array([["High", "Not High"], ["Not High", "High"]]) 

    # Plotting
    plt.figure(2, figsize=(10, 7), clear=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, fmt='d', cmap='Blues', ax=ax, xticklabels=legends[0], yticklabels=legends[0])  # fmt='d' for integer formatting
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    if method == DT:
        ax.set_title('Decision Tree Classification Model Confusion Matrix')
    elif method == KNN:
        ax.set_title('K-nearest Neighbours Classification Model Confusion Matrix')

    # Loop over data dimensions and create text annotations.
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            text_color = "white" if norm_cm[i, j] > 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f"{labels[i, j]}\n({cm[i, j]})",
                    ha='center', va='center', color=text_color, fontweight='bold')
    plt.savefig(f"results/confusion-matrix-tiers-{method}.png")
    plt.show()

def model_rating_tiers(method, predict_rating, users=users, books=books, matrix=matrix, features=features):

    if predict_rating == YES:
        # Prompt user to enter user ID and ISBN
        user_id = input("Enter User ID: ")
        ISBN = input("Enter ISBN: ")

        # Ensure ISBN is treated as string
        books['ISBN'] = books['ISBN'].astype(str)

        # Retrieve user and book attributes
        user = users[users['User-ID'] == int(user_id)]
        book = books[books['ISBN'] == ISBN]
        user_attributes = user.columns.values.tolist()
        book_attributes = book.columns.values.tolist()
        final_user_attb = [attb for attb in user_attributes if attb in features]
        final_book_attb = [attb for attb in book_attributes if attb in features]

    # Split features and target
    target = matrix['Book-Rating-Tier']
    values = matrix.drop(columns=['Book-Rating-Tier'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(values, target, test_size=0.1, random_state=42, shuffle=True)

    # Define which columns are categorical
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Define the model pipeline
    if method == DT:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ])
    elif method == KNN:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())
        ])
    else:
        print('Please enter what method to use: dt or knn.')
        return

    # Perform k-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Mean cross-validation accuracy: {np.mean(scores)}")

    # Fit the model on the entire dataset
    model.fit(X_train, y_train)

    # Generate classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    visualiseConfusionMatrix(cm)

    if predict_rating == YES:
        # Combine user and book attributes for prediction
        new_data = user[final_user_attb].values.tolist()[0] + book[final_book_attb].values.tolist()[0]
        new_data_dict = {feat: [val] for feat, val in zip(features, new_data)}
        new_data_df = pd.DataFrame(new_data_dict)
        predicted_label = model.predict(new_data_df)
        # Predict the labels for the new data
        print(f"The predicted rating for this book is {predicted_label[0]} by the user.")


model_rating_tiers(method, predict_rating)
