import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import requests
from scipy.sparse import csr_matrix
import re
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing
"""
def preprocessUsers(users):
    # Preprocess users and age data
    users.dropna(subset=['User-Country'], inplace=True)
    # Remove all NaN or empty user ages
    users.dropna(subset=['User-Age'], inplace=True)
    # Remove trailing spaces
    users['User-Age'] = users['User-Age'].str.strip()
    # Remove all non-numeric symbols
    users['User-Age'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    # Alternatively : users['User-Age'] = users['User-Age'].str.extract('(\d+)', expand=False)
    # Convert the values to numeric
    users['User-Age'] = pd.to_numeric(users['User-Age'])
    # Data scaling: only retain users above 12 (lower end gen-z) and below 103 (upper end boomers+) years of age.
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
    url = "https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn + "&key=AIzaSyDpbLUFFLUn10GEf7LVc6OkevdIk1INHIE"
    page = requests.get(url)
    data = page.json()

    try:
        published_year = data['items'][0]['volumeInfo']['publishedDate']
        # The publishedDate is not always a year, it occassionally has the month and the month and day
        # eg: 1994-01 or 1984-01-04
        # Need to sanitise the date to be only a year inorder to convert it to an int 
        published_year_sanitised = re.search(r'\d+', published_year).group()
        return int(published_year_sanitised)
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
    # Remove any remaining books published in the zero year or published in the ~ future ~
    books = books[(books['Year-Of-Publication'] > 0) & (books['Year-Of-Publication'] <= 2024) ]
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

print(dataJoined.head(10))


"""
Draw a 3D scatter plot
"""
def draw_3D_scatterplot(title, xLabel, x, yLabel, y, zLabel, z, zValueColours):
    colours = [zValueColours[val] for val in z]
    
    labels = list(zValueColours.keys())
    print(labels)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(x, y, z, marker='o', c=colours, s=1, alpha=0.5)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)

    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="legend")
    ax.add_artist(legend1)

    plt.show()


# create a 3D scatterplot 
zValueColours = {
    1: [1, 0, 0, 0.5],    # Red
    2: [1, 0.5, 0, 0.5],  # Orange
    3: [1, 1, 0, 0.5],    # Yellow
    4: [0.5, 1, 0, 0.5],  # Lime
    5: [0, 1, 0, 0.5],    # Green
    6: [0, 1, 1, 0.5],    # Cyan
    7: [0, 0.5, 1, 0.5],  # Sky Blue
    8: [0, 0, 1, 0.5],    # Deep Blue
    9: [0.5, 0, 1, 0.5],  # Purple
    10: [1, 0, 1, 0.5],    # Magenta
}

draw_3D_scatterplot("Scatterplot 1", 'User Age', dataJoined['User-Age'], 'Year of publication', dataJoined['Year-Of-Publication'], 'Rating', dataJoined['Book-Rating'], zValueColours)


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




"""
Create a dataframe containing all attributes
"""
booksAndRatings = pd.merge(books, ratings, on=['ISBN', 'ISBN'], how='inner')
allData = pd.merge(users, booksAndRatings, on=['User-ID', 'User-ID'], how='inner')
# Columns: 'User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age', 'ISB', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 'Book-Rating',
# Rows: a rating (given by a single user for a single book)
print(allData.tail(10))


"""
Discretize the age, publication year and ratings
"""
discretizedData = allData

# Discretise ratings into bins
ratingBins = [0, 4, 7, 10]
discretizedData['Book-Rating'] = pd.cut(discretizedData['Book-Rating'], ratingBins, labels=["poor", "okay", "good"])
discretizedData = discretizedData.rename(columns={"Book-Rating": "Book-Rating-Tier"})

# Create bins by decade for the book's year of publication
floorMinYear = int(math.floor(discretizedData['Year-Of-Publication'].min() / 10.0)) * 10
ceilMaxYear = int(math.ceil(discretizedData['Year-Of-Publication'].max() / 10.0)) * 10
yearsRange = ceilMaxYear - floorMinYear
numberOfDecades = round(yearsRange / 10)
yearBins = list(range(floorMinYear, ceilMaxYear + 1, 10))
# Discretize the years of publications into equal-width bins by decade
discretizedData['Year-Of-Publication'] = pd.cut(discretizedData['Year-Of-Publication'], yearBins, labels=yearBins[:-1])
discretizedData = discretizedData.rename(columns={"Year-Of-Publication": "Decade-Of-Publication"})

# Discretize the users into bins
# Domain knowledge bins - by generation (see notes.md)
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
discretizedData['User-Age'] = pd.cut(discretizedData['User-Age'], discreteAgeValues, labels=discreteAgeLabels[:-1])
discretizedData = discretizedData.rename(columns={"User-Age": "User-Generation"})

# Create a 3D scatterplot
zValueColours = {
    "poor": [1, 0, 0, 0.5],
    "okay": [1, 1, 0, 0.5],
    "good": [0, 1, 0, 0.5],
}

draw_3D_scatterplot("Scatterplot 1", 'User Generation', discretizedData['User-Generation'], 'Decade of publication', discretizedData['Decade-Of-Publication'], 'Rating', discretizedData['Book-Rating-Tier'], zValueColours)


"""
Classification
Given a collection of records (training set), each record contains a set of attributes, one class label.
Find a predictive model for class label as a function of the values of other attributes.
"""

# Separate the data into train and test set
# train set size 80%, test set size 20%
train, test = train_test_split(discretizedData, test_size=0.2, random_state=42, shuffle=True)

# save the train and test file
train.to_csv('datasets/train.csv', index=False)
test.to_csv('datasets/text.csv', index=False)

# Create a training model
numericData = allData
"""
OPT 1: Decision Tree
"""
# Change string values into numerical values
unique_countries = users['User-Country'].unique()
countries = {value: idx for idx, value in enumerate(unique_countries)}
print(countries)
numericData['User-Country'] = numericData['User-Country'].map(countries)

unique_publishers = books['Book-Publisher'].unique()
publishers = {value: idx for idx, value in enumerate(unique_publishers)}
print(publishers)
numericData['Book-Publisher'] = numericData['Book-Publisher'].map(publishers)

# Separate the feature columns from the target column.
# Target variable (discrete) [y] : rating
# columns [x1, x2...]: attributes / predictors
# Create a predictive model [f(x...)]
attributes = ['User-Age', 'User-Country', 'Book-Publisher', 'Year-Of-Publication']

X = numericData[attributes]
y = numericData['Book-Rating']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=attributes)

"""
OPT 2: k Nearest Neighbour
"""

