import math
import pandas as pd
import matplotlib.pyplot as plt
from variables import countries

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing
"""

def preprocessUserCountries(users):
    # Remove the """
    users['User-Country'] = users['User-Country'].str.strip()
    users['User-Country'].replace(to_replace='["]+', value='', inplace=True, regex=True)
    users['User-Country'] = users['User-Country'].str.strip()

    # Create a mapping dictionary
    country_mapping = {}
    for country in countries:
        # Determine the final 'map_to' target country name if 'map_to' exists
        final_country_name = country["country"]
        if "map_to" in country:
            # Find the target country entry that matches the 'map_to' value
            target_country = next((item for item in countries if item["country"] == country["map_to"]), None)
            if target_country:
                final_country_name = target_country["country"]

    # All the provided data is in lowercase
    for country in countries:
        country_mapping[country["country"].lower()] = country["country"]
        country_mapping[country["full_name"].lower()] = country["country"]
        country_mapping[country["iso_code_2"].lower()] = country["country"]
        country_mapping[country["iso_code_3"].lower()] = country["country"]

    # Normalize the 'User-country' entries - invalid countries will be set to na
    users['User-Country'] = users['User-Country'].str.lower().map(country_mapping)
    
    # Optionally remove invalid (na) countries - eliminates 795 rows (48299 to 47504)
    users.dropna(subset=['User-Country'], inplace=True)
    users.dropna(subset=['User-Country'], inplace=True)
    return users
    

def preprocessUsers(users):
    print("preprocessUsers")
    users = preprocessUserCountries(users)
    # Preprocess users and age data
    # Remove all NaN or empty user ages
    ### users.dropna(subset=['User-Age'], inplace=True)
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
    print("createAgeGraphs")
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
# createAgeGraphs(users)
# print("users.to_csv")
users.to_csv('datasets/BX-Users-processed.csv', index=False)

# Preprocess book ratings
def preprocessRatings(ratings):
    print("preprocessRatings")

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
    print("createRatingsGraphs")
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
# createRatingsGraphs(ratings)
print("ratings.to_csv")
ratings.to_csv('datasets/BX-Ratings-processed.csv', index=False)

def groupYears(books):
    print("groupYears")
    # Group year of publication of books into 20-year periods
    bins = [1919, 1998, 2010]
    labels = ['1920-1998', '1999-2010']
    books['Publication-Era'] = pd.cut(books['Year-Of-Publication'], bins=bins, labels=labels, right=True)
    return books

# Preprocess books
def preprocessBooks(books):
    print("preprocessBooks")
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
# print("books.to_csv")
books.to_csv('datasets/BX-Books-processed.csv', index=False)
# print(books)

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


