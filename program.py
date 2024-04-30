import numpy as np
import pandas as pd
import sklearn
import requests
from scipy.sparse import csr_matrix

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing should happen here
"""
# data scaling: remove users with age below 18 years and above 100 years
users['User-Age'] = users['User-Age'].str.extract('(\d+)', expand=False)
users['User-Age'] = pd.to_numeric(users['User-Age'], errors='coerce')
users.dropna(subset=['User-Age'], inplace=True)
users = users[(users['User-Age'] >= 18) & (users['User-Age'] <= 100)]

# function to get published year from ISBN provided
def get_published_year(isbn):
    url = "https://www.googleapis.com/books/v1/volumes?q=isbn:8845229041&key=AIzaSyDpbLUFFLUn10GEf7LVc6OkevdIk1INHIE"
    page = requests.get(url)
    data = page.json()

    try:
        published_year = data['items'][0]['volumeInfo']['publishedDate']
        return int(published_year)
    except (KeyError, IndexError):
        return None

# data enrichment: filling in missing year of publication
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
for index, row in books.iterrows():
    year = int(row['Year-Of-Publication'])
    isbn = row['ISBN']
    if year == 0:
        missing_year = get_published_year(isbn)
        if missing_year:
            books.at[index, 'Year-Of-Publication'] = int(missing_year)

# get the number of ratings, unique books, and unique users
number_of_books = len(ratings['ISBN'].unique())
number_of_users = len(ratings['User-ID'].unique())
number_of_ratings = len(ratings)

# fuction to create user-item matrix
def create_matrix(number_of_books, number_of_users, ratings):
    # get the unique value of book ISBN and user id
    book_index = ratings['ISBN'].unique()
    user_index = ratings['User-ID'].unique()
    # create and return the user-item matrix
    matrix = csr_matrix((ratings["Book-Rating"], (book_index, user_index)), shape=(number_of_books, number_of_users))
    return matrix

# use k-nearest neighbors to find out books that similar to the entered book (passed as an parameter in this function)
def book_suggestions(ISBN, matrix, k):
    similar_books = []
    book_vec = matrix[ISBN]
    k+=1
    # use cosine similarity to calculated the similarity score of books based on user ratings
    kNN = sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm="brute", metric='cosine')
    kNN.fit(matrix)
    book_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(book_vec, return_distance=False)

    # append books to the similar books list
    for i in range(0,k):
        n = neighbour.item(i)
        similar_books.append(n)
    # remove the original book
    similar_books.pop(0)
    return similar_books

matrix = create_matrix(number_of_books, number_of_users, ratings)
similar_books = book_suggestions(684867621, matrix, 10)
book_titles = dict(zip(books['ISBN'], books['Book-Title']))

print("Here is the suggested books for you:\n")
for i in similar_books:
    print(book_titles[i])