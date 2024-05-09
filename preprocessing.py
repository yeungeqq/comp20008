import pandas as pd
import requests
import re
import numpy as np
from fuzzywuzzy import process
from variables import countries
from unidecode import unidecode
import credentials

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books.csv")
users = pd.read_csv("BX-Users.csv")
ratings = pd.read_csv("BX-Ratings.csv")

"""
Data pre-processing: Users
"""
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
    users.loc[(users['User-Age'] >= 12) & (users['User-Age'] < 103)]

    ### Country
    # Remove trailing and preceeding whitespace
    users['User-Country'] = users['User-Country'].str.strip()
    users['User-Country'].replace(to_replace='["]+', value='', inplace=True, regex=True)
    # remove the """ and any space characters between the """ and the content
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

        country_mapping[country["country"].lower()] = final_country_name
        country_mapping[country["full_name"].lower()] = final_country_name
        country_mapping[country["iso_code_2"].lower()] = final_country_name
        country_mapping[country["iso_code_3"].lower()] = final_country_name

    # normalize the 'User-Country' entries - invalid countries will be set to na
    users['User-Country'] = users['User-Country'].str.lower().map(country_mapping)

    # optionally remove invalid (na) countries - eliminates 795 rows (48299 to 47504)
    users.dropna(subset=['User-Country'], inplace=True)

    ### Cities
    users['User-City'] = users['User-City'].astype(str)
    users['User-City'] = users['User-City'].str.strip()
    users['User-City'] = users['User-City'].apply(unidecode)
    users['User-City'] = users['User-City'].replace(to_replace='[^A-Za-z\s.]+', value='', regex=True)
    users['User-City'] = users['User-City'].str.title()

    ### States
    users['User-State'] = users['User-State'].astype(str)
    users['User-State'] = users['User-State'].str.strip()
    users['User-State'] = users['User-State'].apply(unidecode)
    users['User-State'] = users['User-State'].replace(to_replace='[^A-Za-z\s.]+', value='', regex=True)
    users['User-State'] = users['User-State'].str.title()

    return users

users = preprocessUsers(users)
users.to_csv('datasets/BX-Users-processed.csv', index=False)

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
    url = "https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn + "&key=" + credentials.googleBooksApi
    page = requests.get(url)
    data = page.json()

    # Need to be able to monitor if Google's API accepts the request
    # Avoid exhausting quota of 1000 per day — check the file output
    with open("googleApiFetch.txt", "a") as fileFetch:

        try:
            if data['items']:
                book_info = data['items'][0]['volumeInfo']
                # the publishedDate is not always a year, it occassionally has the month and the month and day
                # e.g. 1994-01 or 1984-01-04
                # need to sanitise the date to be only a year inorder to convert it to an int
                published_year = book_info.get('publishedDate', '').split('-')[0]
                return int(published_year)
        except Exception:
            print(f"Failed to fetch book data: {isbn}", file=fileFetch)
            return None

def preprocessBooks(unprocessedbooks):
    processedBooks = unprocessedbooks.copy()

    # Year of Publication
    processedBooks.dropna(subset=['Year-Of-Publication'], inplace=True)
    processedBooks['Year-Of-Publication'] = processedBooks['Year-Of-Publication'].astype(int)

    # Data enrichment: filling in missing year of publication
    for index, row in books.iterrows():
        year = int(row['Year-Of-Publication'])
        isbn = row['ISBN']
        if year == 0:
            missing_year = getPublishedYear(isbn)
            if missing_year:
                books.at[index, 'Year-Of-Publication'] = int(missing_year)

    # Filter out books with an invalid year of publication
    processedBooks = processedBooks.loc[(processedBooks['Year-Of-Publication'] > 0) & (processedBooks['Year-Of-Publication'] <= 2024)]


    # Author
    processedBooks['Book-Author'] = processedBooks['Book-Author'].replace(to_replace='[^A-Za-z\s.]+', value='', regex=True)
    # Step 1: Ensure there is a space after each period (handles multiple initials)
    processedBooks['Book-Author'] = processedBooks['Book-Author'].apply(lambda x: re.sub(r'\.(?=[A-Za-z])', '. ', x))
    # Step 2: Remove excess spaces (if any) after periods
    processedBooks['Book-Author'] = processedBooks['Book-Author'].apply(lambda x: re.sub(r'\.\s+', '. ', x))
    processedBooks['Book-Author'] = processedBooks['Book-Author'].str.title()
    processedBooks = fuzzyMatching(processedBooks, "Book-Publisher")

    # Output the fuzzy wuzzy results for authors for evaluation
    uniqueAuthors = processedBooks['Book-Author'].unique()
    uniqueAuthors = np.sort(uniqueAuthors)
    file = open("subsets/unique-authors.txt", "w+")
    file.write(str(np.array2string(uniqueAuthors, threshold = np.inf)))
    file.close()

    # Publisher
    processedBooks['Book-Publisher'] = processedBooks['Book-Publisher'].replace(to_replace='[^A-Za-z\s.&]+', value='', regex=True)
    processedBooks['Book-Publisher'] = processedBooks['Book-Publisher'].str.title()
    processedBooks = fuzzyMatching(processedBooks, "Book-Publisher")

    # Output the fuzzy wuzzy results for publishers for evaluation
    uniquePublishers = processedBooks['Book-Publisher'].unique()
    uniquePublishers = np.sort(uniquePublishers)
    file = open("subsets/unique-publishers.txt", "w+")
    file.write(str(np.array2string(uniquePublishers, threshold = np.inf)))
    file.close()

    return processedBooks

books = preprocessBooks(books)
books.to_csv('datasets/BX-Books-processed.csv', index=False)

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
    # confirm that all rating users are valid (have a matching user id in the user list)
    ratings = pd.merge(users['User-ID'], ratings, on='User-ID', how='inner')

    return ratings

ratings = preprocessRatings(ratings)
ratings.to_csv('datasets/BX-Ratings-processed.csv', index=False)

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
