import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
import sklearn
import requests
import Levenshtein
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from variables import countries

TEST_SIZE = 0.2
RANDOM_STATE = 20008

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Initial data pre-processing on 1) Users, 2) Ratings, and 3) Books
"""

# 1) Users
def preprocessUserCountries(users):
    # remove the """
    users.loc[:, 'User-Country'] = users['User-Country'].str.strip()
    users.loc[:, 'User-Country'] = users['User-Country'].replace(to_replace='["]+', value='', regex=True)

    # create a mapping dictionary
    country_mapping = {}
    for country in countries:
        # determine the final 'map_to' target country name if 'map_to' exists
        final_country_name = country["country"]
        if "map_to" in country:
            # find the target country entry that matches the 'map_to' value
            target_country = next((item for item in countries if item["country"] == country["map_to"]), None)
            if target_country:
                final_country_name = target_country["country"]

    # all the provided data is in lowercase
    for country in countries:
        country_mapping[country["country"].lower()] = country["country"]
        country_mapping[country["full_name"].lower()] = country["country"]
        country_mapping[country["iso_code_2"].lower()] = country["country"]
        country_mapping[country["iso_code_3"].lower()] = country["country"]

    # normalize the 'User-Country' entries - invalid countries will be set to na
    users.loc[:, 'User-Country'] = users['User-Country'].str.lower().map(country_mapping)
    
    # optionally remove invalid (na) countries - eliminates 795 rows (48299 to 47504)
    users.loc[:, 'User-Country'] = users.dropna(subset=['User-Country'])
    return users

def preprocessUsers(users):
    '''Pre-process users' age and country data'''

    # Age
    users.dropna(subset=['User-Age'])
    users['User-Age'] = users['User-Age'].str.strip()
    users['User-Age'] = users['User-Age'].replace(to_replace='[^0-9]+', value='', regex=True)
    users['User-Age'] = pd.to_numeric(users['User-Age'])
    # only retain users above 12 (lower end gen-z) and below 103 (upper end boomers+) years of age.
    users = users[(users['User-Age'] >= 12) & (users['User-Age'] < 103)]

    # Country
    users = preprocessUserCountries(users)

    return users

users = preprocessUsers(users)

# 2) Ratings
def preprocessRatings(ratings):
    '''Pre-process ratings data'''

    # Book Rating
    ratings.dropna(subset=['Book-Rating'], inplace=True)
    # only retain ratings 1 or more.
    ratings = ratings[ratings['Book-Rating'] >= 1]
    # only retain ratings10 or less.
    ratings = ratings[ratings['Book-Rating'] <= 10]

    # ISBN
    # confirm that all rating ISBNs are valid (have a matching book in the book list)
    ratings = pd.merge(books['ISBN'], ratings, on='ISBN', how='inner')

    return ratings

ratings = preprocessRatings(ratings)

# 3) Books
def fuzzyMatching(books, attribute):
    '''Perform Fuzzy Wuzzy string matching so that duplicate authors and publishers
    aren't repeated and can be recorded in one unique form'''

    if attribute == "Book-Publisher":
        unique_data = books['Book-Publisher'].unique()
    elif attribute == "Book-Author":
        unique_data = books['Book-Author'].unique()
    
    for index, row in books.iterrows():
        outcome = process.extract(row[attribute], unique_data)
        # out of the top 2 high scoring strings, pick the one with shortest length
        min_index = 0
        len_str = len(outcome[0][0])
        # outcome is by default sorted in descending order by their score
        for i in range(1,2):
            if len(outcome[i][0]) < len_str:
                len_str = len(outcome[i][0])
                min_index = i
        books.at[index, attribute] = outcome[min_index][0]

    return books

def getPublishedYear(isbn):
    url = "https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn + "&key=AIzaSyAJzrffw8L22ZhjKU3f1k8igWejFN5AhVg"
    page = requests.get(url)
    data = page.json()

    try:
        published_year = data['items'][0]['volumeInfo']['publishedDate']
        # the publishedDate is not always a year, it occassionally has the month and the month and day
        # e.g. 1994-01 or 1984-01-04
        # need to sanitise the date to be only a year inorder to convert it to an int 
        published_year_sanitised = re.search(r'\d+', published_year).group()
        return int(published_year_sanitised)
    except (KeyError, IndexError):
        return None

def preprocessBooks(books):
    '''Pre-process books' author, year of publication, and publisher data'''
    
    # Author
    books['Book-Author'] = books['Book-Author'].replace(to_replace='[^A-Za-z\s.]+', value='', regex=True)
    books['Book-Author'] = books['Book-Author'].str.title()

    # Year of Publication
    books.dropna(subset=['Year-Of-Publication'], inplace=True)
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    # data enrichment: filling in missing year of publication
    for index, row in books.iterrows():
        year = int(row['Year-Of-Publication'])
        isbn = row['ISBN']
        if year == 0:
            missing_year = getPublishedYear(isbn)
            if missing_year:
                books.at[index, 'Year-Of-Publication'] = int(missing_year)
    books = books[(books['Year-Of-Publication'] > 0) & (books['Year-Of-Publication'] <= 2024)]

    # Publisher
    books.loc[:, 'Book-Publisher'] = books['Book-Publisher'].replace(to_replace='[^A-Za-z\s.&]+', value='', regex=True)
    books.loc[:, 'Book-Publisher'] = books['Book-Publisher'].str.title()
    books = fuzzyMatching(books, "Book-Publisher")

    return books

books = preprocessBooks(books)

books.to_csv('Fuzzy-Publishers.csv', index=False)

"""
Feature Selection: determine which attributes of users and books affect the 
book ratings significantly
"""

# X = ratings
# X = pd.merge(X, users[['User-ID', 'User-Country', 'User-Age']], on='User-ID', how='inner')
# X = pd.get_dummies(X, columns=['User-Country'])
# y = X['Book-Rating']
# X = X.drop('Book-Rating', axis=1)
# X = X.drop('ISBN', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# model = RandomForestRegressor(n_estimators=100)
# model.fit(X_train, y_train)

# importances = pd.Series(model.feature_importances_, index=X_train.columns)
# importances.sort_values(ascending=False).plot(kind='bar')
# plt.title("Feature Importance")
# plt.show()

"""
Create a dataframe containing all attributes
"""
booksAndRatings = pd.merge(ratings, books, on=['ISBN'], how='inner')
booksAndRatings_ = booksAndRatings.groupby(['User-ID', 'Publication-Era'])['Book-Rating'].mean().reset_index()
books_ratings_users = pd.merge(booksAndRatings_[['User-ID', 'Book-Rating', 'Publication-Era']], users,
                               on='User-ID', how='inner')

### added by eq
matrix = books_ratings_users[['User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age']].drop_duplicates()

old_book = books_ratings_users[books_ratings_users['Publication-Era']=='1920-1998'][['User-ID', 'Book-Rating']]
old_book = old_book.rename(columns={'Book-Rating': 'Old-Book-Rating'})
new_book = books_ratings_users[books_ratings_users['Publication-Era']=='1999-2010'][['User-ID', 'Book-Rating']]
new_book = new_book.rename(columns={'Book-Rating': 'New-Book-Rating'})

matrix = pd.merge(matrix, old_book, on='User-ID', how='inner')
matrix = pd.merge(matrix, new_book, on='User-ID', how='inner')
### added by eq

allData = pd.merge(users, booksAndRatings, on=['User-ID', 'User-ID'], how='inner')
# Columns: 'User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age', 'ISB', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 'Book-Rating',
# Rows: a rating (given by a single user for a single book)
# print(allData.tail(10))

# print("allData.to_csv")
allData.to_csv('datasets/combinedData.csv', index=False)

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
yearBins = [floorMinYear, 2000, ceilMaxYear]
# Discretize the years of publications into equal-width bins by decade
discretizedData['Year-Of-Publication'] = pd.cut(discretizedData['Year-Of-Publication'], yearBins, labels=yearBins[:-1])
discretizedData = discretizedData.rename(columns={"Year-Of-Publication": "Publication-Era"})

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

### added by eq
matrix['User-Age'] = pd.cut(matrix['User-Age'], discreteAgeValues, labels=discreteAgeLabels[:-1])
matrix = matrix.rename(columns={"User-Age": "User-Generation"})
matrix.to_csv('datasets/matrix.csv')
### added by eq

# print(books_ratings_users.head(20))

# Create a 3D scatterplot
zValueColours = {
    "poor": [1, 0, 0, 0.5],
    "okay": [1, 1, 0, 0.5],
    "good": [0, 1, 0, 0.5],
}

# draw_3D_scatterplot("Scatterplot 1", 'User Generation', discretizedData['User-Generation'], 'Decade of publication', discretizedData['Publication-Era'], 'Rating', discretizedData['Book-Rating-Tier'], zValueColours)

# Output the discretized data sheet
print("discretizedData.to_csv")
discretizedData.to_csv('datasets/discretizedData.csv', index=False)