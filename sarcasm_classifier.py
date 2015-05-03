"""
A classifier for detecting if a sentence appears to be written in a sarcastic tone or not.
Trained primarily on feature vectors based on prevelance of sentence patterns -- those formulated by 'pattern_extractor.py' and 'pattern_analysis.py'

This choice of features is based on (Davidov et.al 2010) "Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon"

@author andrew
"""

import pandas as pd
import nltk.data
import pickle
from time import time
from pattern_functions import load_list_from_file, list_to_dict, sentence_to_patterns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from KaggleWord2VecUtility import KaggleWord2VecUtility

# Returns counts that are not normalized
def get_misc_features(tok_sentence):
	features = dict()
	features['exc_count'] = 0
	features['que_count'] = 0
	features['quo_count'] = 0
	features['cap_count'] = 0

	for word in tok_sentence:
		if word.isupper():
			features['cap_count'] += 1
		elif word == "!":
			features['exc_count'] += 1
		elif word == "?":
			features['que_count'] += 1
		elif word == '"':
			features['quo_count'] += 1

	return features

class Classifier():

	def __init__(self, tokenizer):
		self.cws = list_to_dict(load_list_from_file('data/sarcasm/CWs.txt'))
		self.hfws = list_to_dict(load_list_from_file('data/sarcasm/HFWs.txt'))
		self.pattern_data = pd.read_csv('data/sarcasm/indicative_of_sarcasm.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern'])
		self.pattern_data = self.pattern_data.append(pd.read_csv('data/sarcasm/indicative_of_nonsarcasm.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern']), ignore_index=True)
		self.tok_patterns = []
		for pattern in self.pattern_data['Pattern']:
			self.tok_patterns.append(pattern.lower().split("_")[1:])

		self.alpha = 0.1 #Weighted parameter for "sparse" pattern matches
		self.gamma = 0.1 #Weighted parameter for "incomplete" pattern matches
		self.tokenizer = tokenizer

	#Returns a value between 0 & 1, indicating the similarity between the two input patterns
	#pattern should be the 'input' pattern, and comp_pattern should be a pattern from the self.pattern_data database
	def get_pattern_match(self, pattern, comp_pattern):
		consecutive_queue = list(comp_pattern)
		hfw_matched = False #In the case of an incomplete match, only count the incomplete match if at least one HFW was matched.

		for item in pattern:
			if item == consecutive_queue[0]:
				if item != "cw":
					hfw_matched = True
				consecutive_queue.pop(0)
				if len(consecutive_queue) == 0:
					break

		if len(consecutive_queue) == 0: #All consecutive patterns matched
			if len(pattern) == len(comp_pattern): 
				return 1			#Exact match
			else:
				return self.alpha	#Sparse match

		if len(consecutive_queue) == len(comp_pattern) or not hfw_matched:
			return 0				#No match
		else:
			return self.gamma*((len(comp_pattern)-len(consecutive_queue))/len(pattern))	#Incomplete match

	def get_sentence_features(self, tok_sentence):
		feature_vectors = dict()
		sent_pats = sentence_to_patterns(tok_sentence, self.hfws, self.cws)

		#Pattern features
		for i in range(len(self.tok_patterns)):
			pattern = self.pattern_data['Pattern'][i]
			feature_vectors[pattern] = 0
			for sent_pattern in sent_pats:
				match = self.get_pattern_match(sent_pattern.lower().split("_")[1:], self.tok_patterns[i])
				if match > feature_vectors[pattern]:
					feature_vectors[pattern] = match #Store the best pattern match found

		#Other features
		misc_features = get_misc_features(tok_sentence)
		feature_vectors['exc_count'] = misc_features['exc_count']/(self.max_exclamations*self.max_avg)
		feature_vectors['que_count'] = misc_features['que_count']/(self.max_questions*self.max_avg)
		feature_vectors['quo_count'] = misc_features['quo_count']/(self.max_quotations*self.max_avg)
		feature_vectors['cap_count'] = misc_features['cap_count']/(self.max_allcaps*self.max_avg)
		#feature_vectors['word_count'] = len(tok_sentence)/(self.max_sent_length*self.max_avg)

		return feature_vectors

	#Train the classifier.
	def fit_transform(self, train_reviews, train_labels):
		print("Training sarcasm classifier...")
		print("Number of reviews:", len(train_reviews))
		t0 = time()
		count_so_far = 0

		#Find maximum occuring values of certain features (used later for normalization)
		#self.max_sent_length = 1
		self.max_exclamations = 1
		self.max_questions = 1
		self.max_quotations = 1
		self.max_allcaps = 1

		print("Gathering features (part 1 of 2)...")
		for review in train_reviews:
			count_so_far += 1
			if count_so_far % 100 == 0:
				print(count_so_far, "reviews processed (%0.2fs)" % (time() - t0))
			for tok_sent in KaggleWord2VecUtility.review_to_sentences(review, self.tokenizer, case_sensitive=True):
				misc_features = get_misc_features(tok_sent)

				#self.max_sent_length = max(self.max_sent_length, len(tok_sent))
				self.max_exclamations = max(self.max_exclamations, misc_features['exc_count'])
				self.max_questions = max(self.max_questions, misc_features['que_count'])
				self.max_quotations = max(self.max_quotations, misc_features['quo_count'])
				self.max_allcaps = max(self.max_allcaps, misc_features['cap_count'])

		self.max_avg = (self.max_exclamations+self.max_questions+self.max_quotations+self.max_allcaps)/4

		#Compile features of training sentences
		print("Gathering features (part 2 of 2)...")
		count_so_far = 0
		train_data = []
		train_data_labels = []
		for i in range(len(train_reviews)):
			count_so_far += 1
			if count_so_far % 50 == 0:
				print(count_so_far, "reviews processed (%0.2fs)" % (time() - t0))
			for sent in KaggleWord2VecUtility.review_to_sentences(train_reviews[i], self.tokenizer):
				train_data.append(self.get_sentence_features(sent))
				train_data_labels.append(train_labels[i])

		self.classifier = Pipeline([('vect', DictVectorizer()),										
	                      			('clf', SGDClassifier(n_jobs=-1))])	
		self.classifier.fit_transform(train_data, train_data_labels)
		print("Done training sarcasm classifier.")

	def load_pickle(self, filename):
		dump = open(filename, 'rb')
		self.classifier = pickle.load(dump)
		dump.close()
		dump = open(filename+"v", 'rb')
		persistent_vars = pickle.load(dump)
		self.max_exclamations = persistent_vars['max_exc'] 
		self.max_questions = persistent_vars['max_que'] 
		self.max_quotations = persistent_vars['max_quo']
		self.max_allcaps = persistent_vars['max_cap']	
		self.max_avg = (self.max_exclamations+self.max_questions+self.max_quotations+self.max_allcaps)/4	
		dump.close()

	def save_pickle(self, filename):
		dump = open(filename, 'wb')
		pickle.dump(self.classifier, dump)
		dump.close()	
		dump = open(filename+"v", 'wb')
		persistent_vars = dict()
		persistent_vars['max_exc'] = self.max_exclamations
		persistent_vars['max_que'] = self.max_questions
		persistent_vars['max_quo'] = self.max_quotations
		persistent_vars['max_cap'] = self.max_allcaps
		pickle.dump(persistent_vars, dump)
		dump.close()	

	def predict_sent(self, sent):
		tok_sentence = KaggleWord2VecUtility.review_to_wordlist(sent, case_sensitive=True)
		sent_feats = self.get_sentence_features(tok_sentence)
		print(sent_feats)
		return self.classifier.predict(sent_feats)[0]

	def predict_review(self, review):
		return None
		
