# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:07:08 2015

@author: donghochoi
"""
import os
import pandas as pd
import numpy as np 
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.ensemble import RandomForestClassifier
import logging

from sklearn.linear_model import SGDClassifier

#Create ROC curve
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews
    
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

if __name__ == "__main__":
    
    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    
    print ("Parsing sentences from training set")
    index = 0
    x_train =[]
    for review in train["review"]:
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=True)
        #sentences += parsed_sentences
        #labels.append(str('sent_'+str(index)))
        label = str('TRAIN_'+str(index))
        x_train.append(LabeledSentence(parsed_sentences,[label]))
        index += 1
    y_train = train['sentiment']
        
    print(len(x_train))
    
    print ("Parsing sentences from unlabeled training set")
    index =0
    x_unlabeled = []
    for review in unlabeled_train["review"]:
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=True)
        #sentences += parsed_sentences
        #labels.append(str('sent_'+str(index)))
        label = str('UNL_TRAIN_'+str(index))    
        x_unlabeled.append(LabeledSentence(parsed_sentences,[label]))
        index += 1
    """
    print ("Parsing sentences from test set")
    index =0
    x_test = []
    for review in test["review"]:
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=True)
        #sentences += parsed_sentences
        #labels.append(str('sent_'+str(index)))
        label = str('UNL_TRAIN_'+str(index))      
        x_test.append(LabeledSentence(parsed_sentences,[label]))
        index += 1
    """
    print("Parsing sentences for sample test")
    index = 0
    x_test = []
    for review in train["review"]:
        #sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        parsed_sentences = KaggleWord2VecUtility.review_to_wordlist(review,remove_stopwords=True)
        #sentences += parsed_sentences
        #labels.append(str('sent_'+str(index)))
        label = str('UNL_TRAIN_'+str(index))      
        x_test.append(LabeledSentence(parsed_sentences,[label]))
        index += 1  
        if index==1000: 
            break
    y_test = y_train[:1000]
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Training Doc2Vec model...")
    model_dm = Doc2Vec(min_count=1, window=10, size=num_features, sample=1e-3, negative=5, workers=3)
    model_dm.build_vocab(np.concatenate((x_train,x_unlabeled,x_test)))    
    
    #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    print("Training train and unlabeled set..")    
    all_train_reviews = np.concatenate((x_train, x_unlabeled))
    for epoch in range(10):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        
    x_test = np.array(x_test)
    
    print("Training test set..")    
    for epoch in range(10):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
    
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    #model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()   

    # ****** Create average vectors for the training and test sets
    #
    print ("Creating average feature vecs for training reviews")
    trainDataVecs = getVecs(model_dm, x_train, num_features)
    #trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model_dm, num_features )

    print ("Creating average feature vecs for test reviews")
    testDataVecs = getVecs(model_dm, x_test, num_features)
    #testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model_dm, num_features )
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(trainDataVecs, y_train)

    print ("'Test Accuracy: {0}".format(lr.score(testDataVecs, y_test)))    
    
    

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier( n_estimators = 100 )

    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, y_train )
    print ("RandomForestClassifier result: {0}".format(forest.score(testDataVecs, y_test)))
    # Test & extract results
    #result = forest.predict( testDataVecs )
    
    # lr graph
    pred_probas = lr.predict_proba(testDataVecs)[:,1]
    
    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()
    
    # random forest classifier result
    pred_probas_forest = forest.predict_proba(testDataVecs)[:,1]
    
    fpr,tpr,_ = roc_curve(y_test, pred_probas_forest)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()
    """
    # Write the test results
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    
    output.to_csv( "Doc2Vec_Vectors.csv", index=False, quoting=3 )
    print ("Wrote Doc2Vec_Vectors.csv")
    """    
