import pandas as pd
import matplotlib.pyplot as plt

# read files and store in pandas dataframe
books = pd.read_csv("BX-Books-processed.csv")
users = pd.read_csv("datasets/BX-Users-processed.csv")
ratings = pd.read_csv("datasets/BX-Ratings-processed.csv")

"""
Draw user data graphs
"""

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

createAgeGraphs(users)

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

createRatingsGraphs(ratings)


discretizedData = pd.read_csv("datasets/discretizedData.csv")

# Create a 3D scatterplot
zValueColours = {
    "poor": [1, 0, 0, 0.5],
    "okay": [1, 1, 0, 0.5],
    "good": [0, 1, 0, 0.5],
}

# draw_3D_scatterplot("Scatterplot 1", 'User Generation', discretizedData['User-Generation'], 'Decade of publication', discretizedData['Publication-Era'], 'Rating', discretizedData['Book-Rating-Tier'], zValueColours)



