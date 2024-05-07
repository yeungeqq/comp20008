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

def fetchBookDetails(isbn):
    url = "https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn + "&key=AIzaSyDT0-rFRG-uYlxey21gXGwc8fRMnigMSAU"
    page = requests.get(url)
    data = page.json()

    # Initialize a dictionary with keys corresponding to your dataframe columns
    book_details = {
        'ISBN': isbn,
        'Book-Author': None,
        'Year-Of-Publication': None,
        'Book-Publisher': None
    }

    with open("googleApiFetch.txt", "a") as fileFetch:

        try:
            if data['items']:
                book_info = data['items'][0]['volumeInfo']
                book_details['Book-Author'] = book_info.get('authors', [None])[0]
                # the publishedDate is not always a year, it occassionally has the month and the month and day
                # e.g. 1994-01 or 1984-01-04
                # need to sanitise the date to be only a year inorder to convert it to an int
                book_details['Year-Of-Publication'] = book_info.get('publishedDate', '').split('-')[0]
                book_details['Book-Publisher'] = book_info.get('publisher')

            return book_details
        except Exception:
            print(f"Failed to fetch book data: {isbn}", file=fileFetch)
            return {}  # Return an empty dictionary on failure



def preprocessBooks(unprocessedbooks):
    processedBooks = unprocessedbooks.copy()

    # Year of Publication
    processedBooks.dropna(subset=['Year-Of-Publication'], inplace=True)

    # Author
    processedBooks['Book-Author'].replace(to_replace='[^A-Za-z\s.]+', value='', inplace=True, regex=True)
    processedBooks['Book-Author'] = processedBooks['Book-Author'].str.title()

    # Publisher
    processedBooks['Book-Publisher'] = processedBooks['Book-Publisher'].replace(to_replace='[^A-Za-z\s.&]+', value='', regex=True)
    processedBooks['Book-Publisher'] = processedBooks['Book-Publisher'].str.title()

    # Data enrichment: filling in missing or incorrect data
    for index, row in processedBooks.iterrows():
        api_book_details = fetchBookDetails(row['ISBN'])

        fileComparisons = open("googleApiComparison.txt", "a")

        if api_book_details is not None:
            # Check each column for mismatches and update if necessary
            if 'Year-Of-Publication' in api_book_details and api_book_details['Year-Of-Publication'] and str(row['Year-Of-Publication']).strip() != str(api_book_details['Year-Of-Publication']).strip():
                print(f"Updating ISBN {row['ISBN']} column Year-Of-Publication from {row['Year-Of-Publication']} to {api_book_details['Year-Of-Publication']}", file=fileComparisons)
                processedBooks.at[index, 'Year-Of-Publication'] = api_book_details['Year-Of-Publication']

            if 'Book-Author' in api_book_details and  api_book_details['Book-Author'] and str(row['Book-Author']).strip() != str(api_book_details['Book-Author']).strip():
                print(f"Updating ISBN {row['ISBN']} column Book-Author from {row['Book-Author']} to {api_book_details['Book-Author']}", file=fileComparisons)
                processedBooks.at[index, 'Book-Author'] = api_book_details['Book-Author']

            if 'Book-Publisher' in api_book_details and api_book_details['Book-Publisher'] and str(row['Book-Publisher']).strip() != str(api_book_details['Book-Publisher']).strip():
                print(f"Updating ISBN {row['ISBN']} column Book-Author from {row['Book-Publisher']} to {api_book_details['Book-Publisher']}", file=fileComparisons)
                processedBooks.at[index, 'Book-Publisher'] = api_book_details['Book-Publisher']

        fileComparisons.close()

    # Filter out books with an invalid year of publication
    processedBooks['Year-Of-Publication'] = processedBooks['Year-Of-Publication'].astype(int)
    processedBooks = processedBooks.loc[(processedBooks['Year-Of-Publication'] > 0) & (processedBooks['Year-Of-Publication'] <= 2024)]

    return processedBooks

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
