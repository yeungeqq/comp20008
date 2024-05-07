from preprocessing import users, ratings, books, booksAndRatings, books_ratings_users, all_rating_data
import pandas as pd
import string

matrix = books_ratings_users[['User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age']].drop_duplicates()

old_book = books_ratings_users[books_ratings_users['Publication-Era']=='1920-1998'][['User-ID', 'Book-Rating']]
old_book = old_book.rename(columns={'Book-Rating': 'Old-Book-Rating'})
new_book = books_ratings_users[books_ratings_users['Publication-Era']=='1999-2010'][['User-ID', 'Book-Rating']]
new_book = new_book.rename(columns={'Book-Rating': 'New-Book-Rating'})

matrix = pd.merge(matrix, old_book, on='User-ID', how='inner')
matrix = pd.merge(matrix, new_book, on='User-ID', how='inner')

discreteAges={
  "teenager": 12,
  "youth": 16,
  "genz": 19,
  "millenial": 28,
  "genx": 44,
  "boomer+": 60,
  "twilight": 103,
}
discreteAgeLabels = list(discreteAges.keys())
discreteAgeValues = list(discreteAges.values())
matrix['User-Age'] = pd.cut(matrix['User-Age'], discreteAgeValues, labels=discreteAgeLabels[:-1])
matrix = matrix.rename(columns={"User-Age": "User-Generation"})
matrix.to_csv('datasets/matrix.csv')

rating_from_generation = all_rating_data.groupby(['ISBN', 'User-Generation'])['Book-Rating'].mean().reset_index()
rating_from_generation['Book-Rating'] = rating_from_generation['Book-Rating'].fillna(0)
rating_from_generation.to_csv('datasets/rating_from_generation.csv')


favourite_generation_for_book = rating_from_generation.loc[rating_from_generation.groupby('ISBN')['Book-Rating'].idxmax()]
favourite_generation_for_book = favourite_generation_for_book.rename(columns={"User-Generation": "Favourite-Generation"})
favourite_generation_for_book.to_csv('datasets/favourite_rating.csv')

new_matrix = pd.merge(books, favourite_generation_for_book[['ISBN', 'Favourite-Generation']], on='ISBN', how='inner')
new_matrix = new_matrix.drop(columns=['Year-Of-Publication', 'ISBN'])
new_matrix['Book-Title'] = new_matrix['Book-Title'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
new_matrix['Book-Title'] = new_matrix['Book-Title'].str.split().str.len()
new_matrix = new_matrix.rename(columns={"Book-Title": "Book-Title-Length"})
new_matrix.to_csv('datasets/new_matrix.csv')
