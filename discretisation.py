import pandas as pd
import math

"""
Discretize: Age
"""
discretizedData = pd.read_csv("datasets/combinedData.csv")

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

# Create an aditional column for generation
new_column = 'User-Generation'
insert_location = 4
discretizedData.insert(insert_location, new_column, pd.cut(discretizedData['User-Age'], discreteAgeValues, labels=discreteAgeLabels[:-1]))

# Optionally: can override and rename the age column
# discretizedData['User-Age'] = pd.cut(discretizedData['User-Age'], discreteAgeValues, labels=discreteAgeLabels[:-1])
# discretizedData = discretizedData.rename(columns={"User-Age": "User-Generation"})


"""
Discretize: Ratings
"""
ratingBins = [0, 6, 10]
discretizedData['Book-Rating-Tier'] = pd.cut(discretizedData['Book-Rating'], ratingBins, labels=["Not High", "High"])

# Optionally: can override and rename the age column
# discretizedData['Book-Rating'] = pd.cut(discretizedData['Book-Rating'], ratingBins, labels=["poor", "okay", "good"])
# discretizedData = discretizedData.rename(columns={"Book-Rating": "Book-Rating-Tier"})


"""
Discretize: Publication year
"""
# Create bins by decade for the book's year of publication
floorMinYear = int(math.floor(discretizedData['Year-Of-Publication'].min() / 10.0)) * 10
ceilMaxYear = int(math.ceil(discretizedData['Year-Of-Publication'].max() / 10.0)) * 10

# Group year of publication of books into 20-year periods
yearBins = [floorMinYear, 1940, 1960, 1980, 2000, ceilMaxYear]
labels = ['before 1940s', '1940s-1960s', '1960s-1980s', '1980s-2000s', '>=2000']
# Discretize the years of publications into equal-width bins by decade
discretizedData['Publication-Era'] = pd.cut(discretizedData['Year-Of-Publication'], bins=yearBins, labels=labels)


"""
Output discretized data
"""
# Output the discretized data sheet
discretizedData.to_csv('datasets/discretizedData.csv', index=False)


