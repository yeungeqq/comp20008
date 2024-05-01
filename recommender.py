import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
import sys

ratings = pd.read_csv("BX-Ratings.csv")
ratings['ISBN'] = ratings['ISBN'].astype("string")

def recommend(user_id, ratings, n):
    reader = Reader(line_format='user item rating', rating_scale = (1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    sim_options = {
        "name": "cosine",
        "user_based": False,
    }
    algo = KNNWithMeans(sim_options=sim_options)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)

    books = ratings['ISBN'].unique()
    predition = {'book': [], 'predicted_ratings': []}
    for book in books:
        predition['book'].append(book)
        predition['predicted_ratings'].append(algo.predict(user_id, book).est)
    recommended_books = pd.DataFrame(predition).nlargest(n, 'predicted_ratings')['book']

    return list(recommended_books)

recommendations = recommend(sys.argv[1], ratings, int(sys.argv[2]))
print(f"Here are the top {sys.argv[2]} recommended books for {sys.argv[1]}: ")
for book in recommendations:
    print(book)