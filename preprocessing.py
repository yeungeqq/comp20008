import pandas as pd
import re
import requests
from fuzzywuzzy import process
from variables import countries

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing: Users
"""
def preprocessUserCountries(users):
    # Remove trailing and preceeding whitespace
    users['User-Country'] = users['User-Country'].str.strip()

    # remove the """ and any space characters between the """ and the content
    users['User-Country'].replace(to_replace='["]+', value='', inplace=True, regex=True)
    users['User-Country'] = users['User-Country'].str.strip()

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
    users['User-Country'] = users['User-Country'].str.lower().map(country_mapping)

    # optionally remove invalid (na) countries - eliminates 795 rows (48299 to 47504)
    users.dropna(subset=['User-Country'], inplace=True)

    return users

def preprocessUsers(users):
    # Age
    # Remove all NaN or empty user ages
    users.dropna(subset=['User-Age'], inplace=True)
    users['User-Age'] = users['User-Age'].str.strip()
    # Remove all non-numeric symbols
    users['User-Age'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    # Convert the values to numeric
    users['User-Age'] = pd.to_numeric(users['User-Age'])
    # only retain users above 12 (lower end gen-z) and below 103 (upper end boomers+) years of age.
    users = users[(users['User-Age'] >= 12) & (users['User-Age'] < 103)]

    ### Country
    users = preprocessUserCountries(users)

    return users

users = preprocessUsers(users)
users.to_csv('datasets/BX-Users-processed.csv', index=False)


"""
Data pre-processing: Ratings
"""
def preprocessRatings(ratings):
    # Book Rating
    # Remove all NaN or empty rows
    ratings.dropna(subset=['Book-Rating'], inplace=True)
    # only retain ratings 1 or more.
    ratings = ratings[ratings['Book-Rating'] >= 1]
    # only retain ratings 10 or less.
    ratings = ratings[ratings['Book-Rating'] <= 10]

    # ISBN
    # confirm that all rating ISBNs are valid (have a matching book in the book list)
    ratings = pd.merge(books['ISBN'], ratings, on='ISBN', how='inner')

    return ratings

ratings = preprocessRatings(ratings)
ratings.to_csv('datasets/BX-Ratings-processed.csv', index=False)

"""
Data pre-processing: Books
"""
def fuzzyMatching(books, attribute):
    '''Perform Fuzzy Wuzzy string matching so that duplicate authors and publishers
    aren't repeated and can be recorded in one unique form'''

    if attribute == "Book-Publisher":
        unique_data = books['Book-Publisher'].unique()
    elif attribute == "Book-Author":
        unique_data = books['Book-Author'].unique()

    for index, row in books.iterrows():
        curr_data = row[attribute]
        outcome = process.extract(curr_data, unique_data)
        # out of the top 2 high scoring strings, pick the one with shortest length
        min_index = 0
        len_str = len(outcome[0][0])
        # outcome is by default sorted in descending order by their score
        for i in range(1,3):
            if len(outcome[i][0]) < len_str:
                if re.search(outcome[i][0], curr_data):
                    # only change data if the searched string is a substring of curr_data
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
    # Year of Publication
    books.dropna(subset=['Year-Of-Publication'], inplace=True)
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)

    # Data enrichment: filling in missing year of publication
    for index, row in books.iterrows():
        year = int(row['Year-Of-Publication'])
        isbn = row['ISBN']
        if year == 0:
            missing_year = getPublishedYear(isbn)
            if missing_year:
                books.at[index, 'Year-Of-Publication'] = int(missing_year)

    # Filter out books with an invalid year of publication
    books = books[(books['Year-Of-Publication'] > 0) & (books['Year-Of-Publication'] <= 2024)]

    # Author
    books['Book-Author'] = books['Book-Author'].replace(to_replace='[^A-Za-z\s.]+', value='', regex=True)
    books['Book-Author'] = books['Book-Author'].str.title()

    # Publisher
    books['Book-Publisher'] = books['Book-Publisher'].replace(to_replace='[^A-Za-z\s.&]+', value='', regex=True)
    books['Book-Publisher'] = books['Book-Publisher'].str.title()
    books = fuzzyMatching(books, "Book-Publisher")

    return books

books = preprocessBooks(books)
books.to_csv('datasets/BX-Books-processed.csv', index=False)


"""
Create a dataframe containing all attributes
    Columns: 'User-ID', 'User-City', 'User-State', 'User-Country', 'User-Age', 'ISB', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Publisher', 'Book-Rating',
    Rows: a "rating", including all data from a single user and for a single book
"""
booksAndRatings = pd.merge(ratings, books, on=['ISBN', 'ISBN'], how='inner')
allData = pd.merge(users, booksAndRatings, on=['User-ID', 'User-ID'], how='inner')

# Write out a combined CVS
allData.to_csv('datasets/combinedData.csv', index=False)
# Please note this is the rawest, processed data.
# For discretisation, please see discretisation.py
