"""
Built and modified from the base tutorial code from the Kaggle Competition, by Angela Chapman.

"""

import os
import pickle
import logging
import nltk.data
import numpy as np
import pandas as pd
import sarcasm_classifier
from time import time
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from sklearn.linear_model import SGDClassifier
from gensim.models.doc2vec import LabeledSentence
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility

def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def getLabeledSentences(reviews, prefix, skip=0, limit=0, dispose_percent=0):
    labels = []
    index = 0
    if limit == 0:
      for review in reviews["review"][skip:]:
          index += 1
          labels.append(LabeledSentence(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=False, dispose_percent=dispose_percent),[prefix+str(index)]))
    else:
      for review in reviews["review"][skip:limit]:
          index += 1
          labels.append(LabeledSentence(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=False, dispose_percent=dispose_percent),[prefix+str(index)]))
    return labels

classifier_filename = "data/dumps/doc2vec_model.pkl"
num_features = 300       # Word vector dimensionality
min_word_count = 40      # Minimum word count
num_workers = 4          # Number of threads to run in parallel
context = 10             # Context window size
downsampling = 1e-3      # Downsample setting for frequent words
local_test_size = 2000   # Number of training reviews to reserve for local evaluation
percent_disposal = 0.25  # Amount of each review to throw away (see note in KaggleWord2VecUtility.py)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
t0 = time()

#=======================================================================================================
#  Read and format data
#=======================================================================================================

train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('data/testData.tsv', header=0, delimiter="\t", quoting=3)

model = None
if os.path.isfile(classifier_filename):
  model = Doc2Vec.load(classifier_filename)
else:
  unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3)

  # Verify the number of reviews that were read (100,000 in total)
  print ("Read {0} labeled train reviews, {1} test reviews, and {2} unlabeled reviews\n".format(len(train["review"][local_test_size:]), test["review"].size, unlabeled_train["review"].size ))

  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #Used for tokenizing paragraphs into individual sentences.

  print ("Parsing sentences from training set")
  train_sentences = getLabeledSentences(train, 'TRAIN_', skip=local_test_size)

  print ("Parsing sentences from unlabeled set")
  unlabeled_sentences = getLabeledSentences(unlabeled_train, 'UNL_TRAIN_')

  print ("Parsing sentences from test set")
  test_sentences = getLabeledSentences(test, 'TEST_')

  #=======================================================================================================
  #  Train the model
  #=======================================================================================================

  # Initialize and train the model (this will take some time)
  print ("Training Doc2Vec model...")
  model_dm = Doc2Vec(min_count=min_word_count, window=context, size=num_features, sample=downsampling, negative=5, workers=num_workers)
  model_dm.build_vocab(np.concatenate((train_sentences, unlabeled_sentences)))

  #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
  all_train_reviews = np.concatenate((train_sentences, unlabeled_sentences))
  for _ in range(10):
      perm = np.random.permutation(all_train_reviews.shape[0])
      model_dm.train(all_train_reviews[perm])   

  model_dm.init_sims(replace=True) #Unloads everything from memory not related to querying the models
  model_dm.save(classifier_filename)
  del train_sentences
  del unlabeled_sentences

trainDataVecs = None
if os.path.isfile('data/dumps/trainDataVecs.pkl'):
  dump = open('data/dumps/trainDataVecs.pkl', 'rb')
  trainDataVecs = pickle.load(dump)
  dump.close()
else:
  print ("Creating average feature vecs for training reviews")
  trainDataVecs = getAvgFeatureVecs(getCleanReviews(train, skip=local_test_size, dispose_percent=percent_disposal), model, num_features)
  dump = open('data/dumps/trainDataVecs.pkl', 'wb')
  pickle.dump(trainDataVecs, dump)
  dump.close()

testDataVecs = None
if os.path.isfile('data/dumps/testDataVecs.pkl'):
  dump = open('data/dumps/testDataVecs.pkl', 'rb')
  testDataVecs = pickle.load(dump)
  dump.close()
else:
  print ("Creating average feature vecs for test reviews")
  testDataVecs = getAvgFeatureVecs(getCleanReviews(test, dispose_percent=percent_disposal), model, num_features)
  dump = open('data/dumps/testDataVecs.pkl', 'wb')
  pickle.dump(testDataVecs, dump)
  dump.close()

# Fit a random forest to the training data, using 100 trees
print ("Fitting a random forest to labeled training data...")
forest = RandomForestClassifier( n_estimators = 100 )
forest = forest.fit(trainDataVecs, train["sentiment"][local_test_size:])

# Test & extract results
result = forest.predict(testDataVecs)
del trainDataVecs
del testDataVecs

#=======================================================================================================
#  Sarcasm detection
#=======================================================================================================

print("Modifying final predictions based on sarcasm disambiguation... (%0.2fs)" % (time() - t0))
# Initialize the sarcasm classifier (and/or train it if the classifier does not exist)
sarc_classifier = sarcasm_classifier.Classifier()
if os.path.isfile("data/dumps/sarcasm_classifier.pkl"):
  print("Found pickled sarcasm classifier. Loading it...")
  sarc_classifier.load_pickle('data/dumps/sarcasm_classifier.pkl')
else:
  sarcasm_file_data = pd.read_csv('data/sarcasm/five_labels_plus_stars.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Type','File'])
  train_reviews = []

  for i in range(len(sarcasm_file_data)):
    f = None
    if sarcasm_file_data['Type'][i] == 'regular':
      f = open('data/sarcasm/Regular/'+sarcasm_file_data['File'][i]+'.txt', 'r', encoding='latin-1')
    else:
      f = open('data/sarcasm/Ironic/'+sarcasm_file_data['File'][i]+'.txt', 'r', encoding='latin-1')

    review_found = False
    review = ""
    for line in f:
      if line.strip() == "</REVIEW>":
        break
      if review_found:
        review += line.strip()
      if line.strip() == "<REVIEW>":
        review_found = True
    f.close()
    train_reviews.append(review)

  sarc_classifier.fit_transform(train_reviews, sarcasm_file_data['Type'], dispose_percent=percent_disposal)
  sarc_classifier.save_pickle('data/dumps/sarcasm_classifier.pkl') 

# Modify predictions based on percieved sarcasm:
#   -If the sentiment classifier predicts positive sentiment, but the sarcasm classifier detects sarcasm, assume negative sentiment
def apply_sarcasm(sentiment_predictions, sarcasm_predictions):
  modified_predictions = []

  for i in range(len(sentiment_predictions)):
    if sentiment_predictions[i] == 1 and sarcasm_predictions[i] == "ironic":
      modified_predictions.append(0)
    else:
      modified_predictions.append(sentiment_predictions[i])

  return modified_predictions

'''sarcasm_predictions = []
for i in range(len(test["review"])):
    if result[i] == 0:
      sarcasm_predictions.append("regular")
    else:
      sarcasm_predictions.append(sarc_classifier.predict(test["review"][i], dispose_percent=percent_disposal))

    if i % 100 == 0:
      print("Sarcasm predictions so far:", i, "(%0.2fs)" % (time() - t0))
result = apply_sarcasm(result, sarcasm_predictions)
del sarcasm_predictions'''

#=======================================================================================================
#  Local evaluation
#=======================================================================================================
print("Performing local evaluation... (%0.2fs)" % (time() - t0))
if local_test_size > 0:
  sarcasm_predictions = []
  evaluation_predictions = []

  evalTrainDataVecs = getAvgFeatureVecs(getCleanReviews(train, limit=local_test_size, dispose_percent=percent_disposal), model, num_features)
  eval_result = forest.predict(evalTrainDataVecs)
  for i in range(local_test_size):
    if eval_result[i] == 0:
      sarcasm_predictions.append("regular")
    else:
      sarcasm_predictions.append(sarc_classifier.predict(train["review"][i], dispose_percent=percent_disposal))

    if i % 100 == 0:
      print("Sarcasm predictions so far:", i, "(%0.2fs)" % (time() - t0))
  new_eval_result = apply_sarcasm(eval_result, sarcasm_predictions)

  num_correct = 0       # Items whose sentiment was correctly classified in the evaluation results
  num_correct_old = 0   # Items whose sentiment was correctly classified without using sarcasm
  sarc_fixed = 0        # Items that were incorrect, but fixed to the correct sentiment by sarcasm disambigution
  sarc_damaged = 0      # Items that were correct, but were changed to the wrong sentiment by sarcasm disambiguation
  sarc_missed = 0       # Items that were incorrect, but weren't changed by sarcasm disambiguation
  for i in range(local_test_size):
    if new_eval_result[i] == train['sentiment'][i]:
      num_correct += 1
    if eval_result[i] == train['sentiment'][i]:
      num_correct_old += 1
    if eval_result[i] != train['sentiment'][i] and new_eval_result[i] == train['sentiment'][i]:
      sarc_fixed += 1
    if eval_result[i] == train['sentiment'][i] and new_eval_result[i] != train['sentiment'][i]:
      sarc_damaged += 1
    if new_eval_result[i] != train['sentiment'][i] and new_eval_result[i] == 1:
      sarc_missed += 1

  print("Correctly classified:", num_correct, "out of", local_test_size)
  print("Correctly classified without sarcasm:", num_correct_old, "out of", local_test_size)
  print("Fixed by sarcasm:", sarc_fixed)
  print("Broken by sarcasm:", sarc_damaged)
  print("Missed by sarcasm:", sarc_missed)

#=======================================================================================================
#  Output
#=======================================================================================================
'''output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_Sarcasm_AverageVectors.csv", index=False, quoting=3 )
print ("Wrote Word2Vec_Sarcasm_AverageVectors.csv (%0.2fs)" % (time() - t0))'''
