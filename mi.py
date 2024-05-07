from itertools import product
from sklearn.metrics import mutual_info_score
import pandas as pd

def generate_combinations(list1, list2):
    # Generate all possible combinations within each input list
    combinations_list1 = [list(combo) for r in range(1, len(list1)+1) for combo in product(list1, repeat=r)]
    combinations_list2 = [list(combo) for r in range(1, len(list2)+1) for combo in product(list2, repeat=r)]
    
    # Combine combinations from both lists to create the final list of combinations
    final_combinations = []
    for combo1 in combinations_list1:
        for combo2 in combinations_list2:
            # Combine and convert to set to remove duplicates
            combined_combo = set(combo1 + combo2)
            # Convert back to list and sort for consistent ordering
            final_combinations.append(sorted(map(str, combined_combo)))
    
    # Remove duplicates
    unique_combinations = [list(comb) for comb in set(map(tuple, final_combinations))]
    
    return unique_combinations

users = pd.read_csv('')
books = pd.read_csv('')
ratings = pd.read_csv('')
matrix = pd.merge(ratings, users, on='User-ID', how='inner')
matrix = pd.merge(matrix, books, on='ISBN', how='inner')

user_attributes = users.columns.to_list().remove('User-ID')
books_attributes = books.columns.to_list().remove('ISBN')
rating_label = matrix['Book-Rating']

attributes_combinations = generate_combinations(user_attributes, books_attributes)
print("Mutual Informtion for all possible books and users attributes combinations:")
for combination in attributes_combinations:
    mi = mutual_info_score(matrix['combination'], rating_label)
    print(f"The MI of {combination} is {mi}.")
