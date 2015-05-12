# Compare several sentiment analysis models

import os
import pandas as pd
import numpy as np 
import nltk.data
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

#Create ROC curve
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

# ****** Define functions to create average word vectors
#
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review {0} of {1}".format(counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews
    
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

if __name__ == '__main__':

	# Data setup: read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    
    # However, since the test provided from Kaggle does not the sentiment, we'll use a subset of train set as test set
    #sample_test = train.ix[np.random.choice(train.index, len(train)/10)]
    msk = np.random.rand(len(train)) <= 0.8
    #train_sample, test_sample = train_test_split(train, test_size = 0.2)
    train_sample = train[msk]
    test_sample = train[~msk]
    y_test = test_sample["sentiment"]
    # Using bag-of-words model
    print("------------- Bag of Words model ------------")

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print("Cleaning and parsing the training set movie reviews...\n")
    for review in train_sample["review"]:
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review, True)))
    # ****** Create a bag of words from the training set
    #
    print("Creating the bag of words...\n")
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train_sample["sentiment"] )
    
    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []

    print ("Cleaning and parsing the test set movie reviews...\n")
    """
    for i in range(0,len(sample_test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(sample_test["review"][i], True)))
    """
    for review in test_sample["review"]:
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review,True)))
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print ("Predicting test labels...\n")
    #result = forest.predict(test_data_features)
    print ("RandomForestClassifier with Bag-of_Words result: {0}".format(forest.score(test_data_features, y_test)))

    pred_probas_forest = forest.predict_proba(test_data_features)[:,1]
    
    fpr,tpr,_ = roc_curve(y_test, pred_probas_forest)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()

    print("------------- Word Vector model ------------")
    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    print ("Parsing sentences from training set")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print ("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 500    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)
    
    # ****** Create average vectors for the training and test sets
    #
    print ("Creating average feature vecs for training reviews")

    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train_sample), model, num_features )

    print ("Creating average feature vecs for test reviews")

    testDataVecs = getAvgFeatureVecs( getCleanReviews(test_sample), model, num_features )

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    #forest = RandomForestClassifier( n_estimators = 100 )
    forest = RandomForestClassifier(n_estimators = 100)
    
    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, train_sample["sentiment"] )
    print ("RandomForestClassifier with Word2Vec result: {0}".format(forest.score(testDataVecs, y_test)))
    # Test & extract results
    #result = forest.predict( testDataVecs )
    
    pred_probas_forest = forest.predict_proba(testDataVecs)[:,1]
    
    fpr,tpr,_ = roc_curve(y_test, pred_probas_forest)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()


    print("------------- Paragraph Vector model ------------")
                
    print ("Parsing sentences from training set")
    index = 0
    x_train =[]
    for review in train_sample["review"]:
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=False)
        label = str('TRAIN_'+str(index))
        x_train.append(LabeledSentence(parsed_sentences,[label]))
        index += 1
    y_train = train['sentiment']
        
    print(len(x_train))
    
    print ("Parsing sentences from unlabeled training set")
    index =0
    x_unlabeled = []
    for review in unlabeled_train["review"]:
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=False)
        label = str('UNL_TRAIN_'+str(index))    
        x_unlabeled.append(LabeledSentence(parsed_sentences,[label]))
        index += 1
        
    print("Parsing sentences for sample test")
    index = 0
    x_test = []
    for review in test_sample["review"]:
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=False)
        label = str('TEST_'+str(index))      
        x_test.append(LabeledSentence(parsed_sentences,[label]))
        index += 1  
    
    # Initialize and train the model (this will take some time)
    print ("Training Doc2Vec model...")
    model_dm = Doc2Vec(min_count=1, window=context, size=num_features, sample=downsampling, negative=5, workers=num_workers)
    model_dm.build_vocab(np.concatenate((x_train,x_unlabeled,x_test)))    
    
    #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    print("Training train and unlabeled set..")    
    all_train_reviews = np.concatenate((x_train, x_unlabeled))
    for epoch in range(10):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        
    x_test = np.array(x_test)
    
    print ("Creating average feature vecs for training reviews")
    #trainDataVecs = getVecs(model_dm, x_train, num_features)
    trainDataVecs_Doc2 = getAvgFeatureVecs(getCleanReviews(train_sample),model_dm,num_features)
    #trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model_dm, num_features )

    print ("Creating average feature vecs for test reviews")
    #testDataVecs = getVecs(model_dm, x_test, num_features)
    testDataVecs_Doc2 = getAvgFeatureVecs(getCleanReviews(test_sample),model_dm,num_features)
    
    forest = RandomForestClassifier(n_estimators = 100)
    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs_Doc2, train_sample["sentiment"]  )
    print ("RandomForestClassifier with Paragraph Vector result: {0}".format(forest.score(testDataVecs_Doc2, y_test)))
    
    # random forest classifier result
    pred_probas_forest = forest.predict_proba(testDataVecs_Doc2)[:,1]
    
    fpr,tpr,_ = roc_curve(y_test, pred_probas_forest)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()
