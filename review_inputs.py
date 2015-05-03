__author__ = 'qb'

import re
import pandas as pd
from nltk.stem import *
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords   # Import the stop word list
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML tags
    #
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters such as numbers and punctuations
    #
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    #
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Use Snowball Stemmer to reduce the words to original form (using "english")
    stemmer = SnowballStemmer("porter")
    lmtzr = WordNetLemmatizer()
    lemmatized_word= []
    for word in meaningful_words:
        lemmatized_word.append(lmtzr.lemmatize(word))
    # 7. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join( lemmatized_word ))

def rew_raw_file_to_word(file_path):
    train = pd.read_csv(file_path, header=0,
                    delimiter="\t", quoting=3)
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    print "Cleaning and parsing the training set movie reviews...\n"
    file = open("newfile.txt", "w")
    for i in xrange(0, num_reviews):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % ( i+1, num_reviews )
        new_word_list = review_to_words( train["review"][i] )
        file.write(new_word_list)
        clean_train_reviews.append(new_word_list)
    file.close()
    return clean_train_reviews

if __name__ == "__main__":
    wordList = rew_raw_file_to_word("labeledTrainData.tsv")
    print wordList


