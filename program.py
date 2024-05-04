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

def groupYears(books):
    # Group year of publication of books into 20-year periods
    bins = [1940, 1959, 1979, 1999, 2024]
    labels = ['1940-1959', '1960-1979', '1980-1999', '2000-2024']
    books['Publication-Era'] = pd.cut(books['Year-Of-Publication'], bins=bins, labels=labels, right=True)
    return books

# Preprocess books
def preprocessBooks(books):
    # Remove empty publishing years
    books.dropna(subset=['Year-Of-Publication'], inplace=True)
    # Check that the years are within a valid range
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    # Remove any remaining books published in the zero year or published in the ~ future ~
    books = books[(books['Year-Of-Publication'] > 0) & (books['Year-Of-Publication'] <= 2024)]
    # Group the year of publication into 20-year period intervals
    books = groupYears(books)
    return books

books = preprocessBooks(books)
print(books)

"""
Create a dataframe that is connected on user id and book ISBN which has
- the users age
- the books publishing year
- the rating
"""

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



"""
Create a dataframe containing all attributes
"""
booksAndRatings = pd.merge(books, ratings, on=['ISBN', 'ISBN'], how='inner')
allData = pd.merge(users, booksAndRatings, on=['User-ID', 'User-ID'], how='inner')
# Columns: 'User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age', 'ISB', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 'Book-Rating',
# Rows: a rating (given by a single user for a single book)
print(allData.tail(10))

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

# Create a 3D scatterplot
zValueColours = {
    "poor": [1, 0, 0, 0.5],
    "okay": [1, 1, 0, 0.5],
    "good": [0, 1, 0, 0.5],
}

# draw_3D_scatterplot("Scatterplot 1", 'User Generation', discretizedData['User-Generation'], 'Decade of publication', discretizedData['Publication-Era'], 'Rating', discretizedData['Book-Rating-Tier'], zValueColours)


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

