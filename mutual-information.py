from itertools import product, combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def generate_combinations(list_A, list_B, max_length_A=None, max_length_B=None):
    final_combinations = []

    # Determine maximum lengths for combinations from each list
    if max_length_A is None:
        max_length_A = len(list_A)
    if max_length_B is None:
        max_length_B = len(list_B)

    # Generate combinations with varying lengths from each list
    for i in range(1, min(max_length_A, len(list_A)) + 1):
        for j in range(1, min(max_length_B, len(list_B)) + 1):
            combinations_A = combinations(list_A, i)
            combinations_B = combinations(list_B, j)
            
            # Combine combinations of list A with combinations of list B
            for combination_A in combinations_A:
                for combination_B in combinations_B:
                    # Merge two combinations
                    merged_combination = list(combination_A) + list(combination_B)

                    # Remove duplicates and check if it has at least one item from each list
                    unique_combination = list(set(merged_combination))
                    if len(unique_combination) == len(set(combination_A)) + len(set(combination_B)):
                        final_combinations.append(unique_combination)

    return final_combinations

def calculate_mutual_information(X, y):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    X_encoded = np.array([label_encoder.fit_transform(col) for col in X.T]).T

    # Calculate mutual information
    mutual_info = mutual_info_classif(X_encoded, y)

    return mutual_info

def calculate_one_mutual_information(X, y):
    # Encode concatenated strings into numerical labels
    label_encoder = LabelEncoder()
    X_concatenated = np.array(['_'.join(row) for row in X])
    X_encoded = label_encoder.fit_transform(X_concatenated)

    # Calculate mutual information
    mutual_info = mutual_info_classif(X_encoded.reshape(-1, 1), y)

    # Return the mutual information score
    return mutual_info[0]  # Return the first (and only) element


users = pd.read_csv('datasets/BX-Users-processed.csv')
books = pd.read_csv('datasets/BX-Books-processed.csv')
ratings = pd.read_csv('datasets/BX-Ratings-processed.csv')
matrix = pd.merge(ratings, users, on='User-ID', how='inner')
matrix = pd.merge(matrix, books, on='ISBN', how='inner')
matrix = matrix.fillna('NA')

user_attributes = users.columns.to_list()
user_attributes.remove('User-ID')
books_attributes = books.columns.to_list()
books_attributes.remove('ISBN')
rating_label = matrix['Book-Rating'].to_list()

print(user_attributes)
print(books_attributes)

attributes_combinations = generate_combinations(user_attributes, books_attributes)
print("Mutual Informtion for all possible books and users attributes combinations:")

highest_mi = 0
features = None
for combination in attributes_combinations:
    X = np.array(matrix[combination].values.tolist())
    y = np.array(rating_label)
    mi = calculate_one_mutual_information(X, y)
    if mi > highest_mi:
        highest_mi = mi
        features = combination
    print(f"The MI of {combination} is {mi}.")
print(f"Features with the highest MI to ratings is {features} and the score is {highest_mi}")