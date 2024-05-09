from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
np.random.seed(42)

def generate_combinations(list_A, list_B):
    final_combinations = []

    # Generate combinations with varying lengths from each list
    for i in range(1, len(list_A) + 1):
        for j in range(1, len(list_B) + 1):
            for combination_A in combinations(list_A, i):
                for combination_B in combinations(list_B, j):
                    # Merge two combinations while preserving relative order
                    combined = combination_A + combination_B
                    final_combinations.append(combined)

    return final_combinations

def calculate_mutual_information(X, y):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    X_encoded = label_encoder.fit_transform(X)

    # Calculate mutual information
    mutual_info = mutual_info_classif(X_encoded.reshape(-1, 1), y)

    return mutual_info[0]  # Return the first (and only) element

matrix = pd.read_csv('datasets/discretizedData.csv')
matrix = matrix.fillna('NA')

user_attributes = ['User-City', 'User-State', 'User-Country', 'User-Generation', 'User-Age']
books_attributes = ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 'Publication-Era']
rating_label = matrix['Book-Rating'].to_list()

attributes_combinations = generate_combinations(user_attributes, books_attributes)
print("Mutual Informtion for all possible books and users attributes combinations:")

highest_mi = []
features = []
for combination in attributes_combinations:
    # Convert each value to string and then join them
    X = matrix[list(combination)].astype(str).apply('_'.join, axis=1)
    y = np.array(rating_label)  # Ensure y is a 1-dimensional array
    mi = calculate_mutual_information(X, y)
    print(f"The MI of {list(combination)} is {mi}.")
    if len(highest_mi) < 10 or mi > min(highest_mi):
        if len(highest_mi) >= 10:
            min_index = np.argmin(highest_mi)
            del highest_mi[min_index]
            del features[min_index]
        highest_mi.append(mi)
        features.append(list(combination))
    
# Print the top 10 MI scores and corresponding feature combinations
for i, (mi, feat) in enumerate(zip(highest_mi, features), 1):
    print(f"Top {i}: MI = {mi}, Features = {feat}")

# find the minimal common features
common_features = set(features[0])
for comb in features[1:]:
    common_features = common_features.intersection(comb)
top_features = pd.DataFrame({'Features': list(common_features)})
top_features.to_csv('features.csv', index=False)