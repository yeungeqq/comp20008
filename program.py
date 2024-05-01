import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import requests
from scipy.sparse import csr_matrix

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing
"""

def preprocessUsers(users):
    # Preprocess users and age data
    # Remove all NaN or empty user ages
    users.dropna(subset=['User-Age'], inplace=True)
    # Remove trailing spaces
    users['User-Age'] = users['User-Age'].str.strip()
    # Remove all non-numeric symbols
    users['User-Age'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    # Alternatively
    # users['User-Age'] = users['User-Age'].str.extract('(\d+)', expand=False)
    # Convert the values to numeric
    users['User-Age'] = pd.to_numeric(users['User-Age'])
    # Remove outliers and abnormal data
    # Data scaling: remove users with age below 18 years and above 100 years
    # Only retain users above 12 years of age (lower end gen-z) and below 103 years of age (upper end boomers+).
    users = users[(users['User-Age'] >= 12) & (users['User-Age'] < 103)]
    return users

def createAgeGraphs(users):
    # Use a box plot to determine the upper limit and remove outliers to ensure there’s no invalid age data.
    plt.figure(1, clear=True)
    plt.boxplot(users['User-Age'])
    plt.ylabel("Age (years)")
    plt.title("User Ages")
    plt.show()
    plt.savefig("graphs/user_ages.png", format="png")

    # Discretize the users into bins
    # Domain knowledge bins - by generation (see notes.md)
    discreteAges=[12, 16, 19, 28, 44, 60, 103]

    # plt.clf()
    plt.figure(2, clear=True)
    plt.hist(users['User-Age'], bins=discreteAges)
    # plt.hist(pd.cut(users['User-Age'], bins, labels=["gen-z", "millennials", "gen x", "Boomers II", "Boomers I", "Post War", "WWII"]))
    plt.xlabel("Generation")
    plt.ylabel("Quantity")
    plt.title("Users per generation")
    plt.show()
    plt.savefig("graphs/users-per-generation.png")

    # Discretize the users into bins
    # Equal-width bins - by decade
    # plt.clf()
    plt.figure(3, clear=True)
    plt.hist(users['User-Age'], bins=10, range=(12,103))
    plt.xlabel("Decade")
    plt.ylabel("Quantity")
    plt.title("Users per decade")
    plt.show()
    plt.savefig("graphs/users-per-decade.png")
    return

users = preprocessUsers(users)
createAgeGraphs(users)

# Preprocess book ratings
def preprocessRatings(ratings):

    # Check column types
    ## print(ratings.dtypes)
    # Remove all NaN or empty rows
    ratings.dropna(subset=['Book-Rating'], inplace=True)
    # Only retain ratings 1 or more.
    ratings = ratings[ratings['Book-Rating'] >= 1]
    # Only retain ratings10 or less.
    ratings = ratings[ratings['Book-Rating'] <= 10]

    # Confirm that all rating ISBNs are valid (have a matching book in the book list)
    ratings = pd.merge(books['ISBN'], ratings, on='ISBN', how='inner')

    return ratings

def createRatingsGraphs(ratings):
    # Draw a histogram of rating scores
    plt.figure(4, clear=True)
    plt.hist(ratings['Book-Rating'], bins=10, range=(1,11))
    plt.xlabel("Score")
    plt.ylabel("Quantity")
    plt.title("Quantity of scores")
    plt.show()
    plt.savefig("graphs/frequency-of-scores.png")
    return

ratings = preprocessRatings(ratings)
createRatingsGraphs(ratings)

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

def preprocessBooks(books):
    # Preprocess books
    # Remove empty publishing years
    books.dropna(subset=['Year-Of-Publication'], inplace=True)
    # Check that the years are within a valid range
    # data enrichment: filling in missing year of publication
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    for index, row in books.iterrows():
        year = int(row['Year-Of-Publication'])
        isbn = row['ISBN']
        if year == 0:
            missing_year = get_published_year(isbn)
            if missing_year:
                books.at[index, 'Year-Of-Publication'] = int(missing_year)
    return books

books = preprocessBooks(books)


# Create a dataframe that is connected on user id and book ISBN which has
# - the users age
# - the books publishing year
# - the rating

pd.set_option('display.max_columns', None)

# booksCondensed = books['ISBN', 'Year-Of-Publication']
booksCondensed = pd.DataFrame({'ISBN': books['ISBN'], 'Year-Of-Publication': books['Year-Of-Publication']})
bookJoin = pd.merge(booksCondensed, ratings, on=['ISBN', 'ISBN'], how='inner')
usersCondensed = pd.DataFrame({'User-ID': users['User-ID'], 'User-Age':  users['User-Age']})
dataJoined = pd.merge(usersCondensed, bookJoin, on=['User-ID', 'User-ID'], how='inner')

# 3D scatterplot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataJoined['User-Age'], dataJoined['Year-Of-Publication'], dataJoined['Book-Rating'])

ax.set_xlabel('User Age')
ax.set_ylabel('Year of publication')
ax.set_zlabel('Rating')

plt.show()

# Scatter plot with histograms
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py

# Other graph solutions
# https://python-graph-gallery.com/134-how-to-avoid-overplotting-with-python/



# get the number of ratings, unique books, and unique users
number_of_books = len(ratings['ISBN'].unique())
number_of_users = len(ratings['User-ID'].unique())
number_of_ratings = len(ratings)

# fuction to create user-item matrix
def create_matrix(number_of_books, number_of_users, ratings):
    # get the unique value of book ISBN and user id
    book_index = [i for i in ratings['ISBN']]
    user_index = [i for i in ratings['User-ID']]
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
