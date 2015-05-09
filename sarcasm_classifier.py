"""
A classifier for detecting if a sentence appears to be written in a sarcastic tone or not.
Trained primarily on feature vectors based on prevelance of sentence patterns -- those formulated by 'pattern_extractor.py' and 'pattern_analysis.py'

This choice of features is based on (Davidov et.al 2010) "Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon"

@author andrew
"""

import pickle
import pandas as pd
import nltk.data
from time import time, sleep
from pattern_functions import load_list_from_file, list_to_dict, sentence_to_patterns
from sklearn.linear_model import SGDClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from project_settings import pattern_alpha, pattern_gamma, sarcasm_thres, sarcasm_confidence

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

	def __init__(self):
		self.cws = list_to_dict(load_list_from_file('data/sarcasm/CWs.txt'))
		self.hfws = list_to_dict(load_list_from_file('data/sarcasm/HFWs.txt'))
		self.pattern_data = pd.read_csv('data/sarcasm/indicative_of_sarcasm.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern'])
		self.pattern_data = self.pattern_data.append(pd.read_csv('data/sarcasm/indicative_of_nonsarcasm.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern']), ignore_index=True)
		self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		self.tok_patterns = []
		for pattern in self.pattern_data['Pattern']:
			self.tok_patterns.append(pattern.split("_")[1:])

		self.alpha = pattern_alpha #Weighted parameter for "sparse" pattern matches
		self.gamma = pattern_gamma #Weighted parameter for "incomplete" pattern matches

	def is_cw(self, word):
		if self.cws.get(word) != None:
			return True
		elif self.hfws.get(word) == None:
			return True
		return False

	#Returns a value between 0 & 1, indicating the best similarity between patterns in the sentence, and the comparative pattern.
	def get_pattern_match(self, tok_sentence, comp_pattern):

		#First pass: Check for exact and/or incomplete matches
		# Incomplete match is one where only a subset of the consecutive pattern elements occur in the sentence.
		match_index = 0
		sparse_index = 0
		best_match = 0
		found_hfw = False #An incomplete match must at the very least match 1 HFW.

		for word in tok_sentence:
			mword = word.lower()
			if comp_pattern[match_index] == "CW":
				if self.is_cw(word.lower()):
					mword = "CW"

			if mword == comp_pattern[match_index]:
				if mword != "CW":
					found_hfw = True
				match_index += 1
				if match_index > best_match:
					best_match = match_index

				if match_index == len(comp_pattern):
					return 1 #Exact match
			elif match_index != 0:
				mword = word.lower()
				if comp_pattern[0] == "CW":
					if self.is_cw(word.lower()):
						mword = "CW"

				if mword == comp_pattern[0]:
					match_index = 1
				else:
					match_index = 0

			if sparse_index < len(comp_pattern):
				mword = word.lower()
				if comp_pattern[sparse_index] == "CW":
					if self.is_cw(mword):
						mword = "CW"
				if mword == comp_pattern[sparse_index]:
					sparse_index += 1

		if best_match == 0:
			return 0 #No match

		if sparse_index == len(comp_pattern):
			#Second pass: There could be a valid sparse match, search for it
			# Sparse match is one where all of the pattern elements appear in the sentence, but other non-pattern elements are padded in-between.
			for i in range(len(tok_sentence)):
				match_index = 0
				for j in range(i,min(len(tok_sentence),i+12)): #Patterns can't have more than 6 HFWs & 6 CWs, so can never be longer than 12 words
					mword = tok_sentence[j].lower()
					if comp_pattern[match_index] == "CW":
						if self.is_cw(mword):
							mword = "CW"

					if mword == comp_pattern[match_index]:
						match_index += 1
						if match_index == len(comp_pattern):
							return self.alpha					#Sparse Match

		if found_hfw:
			return self.gamma*(best_match/len(comp_pattern))	#Incomplete match
		else:
			return 0

	def get_sentence_features(self, tok_sentence):
		features = list()

		#Pattern features
		for i in range(len(self.tok_patterns)):
			features.append(self.get_pattern_match(tok_sentence, self.tok_patterns[i]))

		#Other features
		misc_features = get_misc_features(tok_sentence)
		features.append(misc_features['exc_count']/(self.max_exclamations*self.max_avg))
		features.append(misc_features['que_count']/(self.max_questions*self.max_avg))
		features.append(misc_features['quo_count']/(self.max_quotations*self.max_avg))
		features.append(misc_features['cap_count']/(self.max_allcaps*self.max_avg))
		features.append(len(tok_sentence)/self.max_sent_length)

		return features

	#Train the classifier.
	def fit_transform(self, train_reviews, train_labels, dispose_percent=(0,0)):
		print("Training sarcasm classifier...")
		print("Number of reviews:", len(train_reviews))
		t0 = time()
		count_so_far = 0

		#Find maximum occuring values of certain features (used later for normalization)
		self.max_sent_length = 1
		self.max_exclamations = 1
		self.max_questions = 1
		self.max_quotations = 1
		self.max_allcaps = 1

		print("Gathering features (part 1 of 2)...")
		for review in train_reviews:
			count_so_far += 1
			if count_so_far % 100 == 0:
				print(count_so_far, "reviews processed (%0.2fs)" % (time() - t0))
			
			tok_review = KaggleWord2VecUtility.review_to_sentences(review, self.tokenizer, case_sensitive=True, dispose_percent=dispose_percent)
			for sent in tok_review:
				misc_features = get_misc_features(sent)

				self.max_sent_length = max(self.max_sent_length, len(sent))
				self.max_exclamations = max(self.max_exclamations, misc_features['exc_count'])
				self.max_questions = max(self.max_questions, misc_features['que_count'])
				self.max_quotations = max(self.max_quotations, misc_features['quo_count'])
				self.max_allcaps = max(self.max_allcaps, misc_features['cap_count'])

		self.max_avg = (self.max_exclamations+self.max_questions+self.max_quotations+self.max_allcaps)/4

		#Compile features of training sentences
		print("Gathering features (part 2 of 2)...")
		count_so_far = 0
		train_data = list()
		train_data_labels = list()
		for i in range(len(train_reviews)):
			count_so_far += 1
			if count_so_far % 100 == 0:
				print(count_so_far, "reviews processed (%0.2fs)" % (time() - t0))
			tok_review = KaggleWord2VecUtility.review_to_sentences(train_reviews[i], self.tokenizer, case_sensitive=True, dispose_percent=dispose_percent)
			for sent in tok_review:
				train_data.append(self.get_sentence_features(sent))
				train_data_labels.append(train_labels[i])

		self.classifier = SGDClassifier(n_jobs=-1, loss='log', class_weight="auto", shuffle=True, n_iter=100) #n_iter is high because we only have ~10000 training samples
		self.classifier.fit(train_data, train_data_labels)
		print("Done training sarcasm classifier.")
		print(self.classifier.classes_)

	#Load the classifier from file
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
		self.max_sent_length = persistent_vars['max_len']
		self.max_avg = (self.max_exclamations+self.max_questions+self.max_quotations+self.max_allcaps)/4	
		dump.close()

	#Save the classifier to file
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
		persistent_vars['max_len'] = self.max_sent_length
		pickle.dump(persistent_vars, dump)
		dump.close()	

	#Determine whether the inputted sentence is sarcastic or not.
	def predict_sentence(self, tok_sent):
		sent_feats = [self.get_sentence_features(tok_sent)]
		return self.classifier.predict_proba(sent_feats)[0]
		
	#Determine whether the inputted review is sarcastic or not
	def predict_review(self, review, dispose_percent=(0,0)):
		tok_sents = KaggleWord2VecUtility.review_to_sentences(review, self.tokenizer, case_sensitive=True, dispose_percent=dispose_percent)

		num_sarcastic = 0
		num_regular = 0
		for sent in tok_sents:
			prediction = self.predict_sentence(sent)
			for i in range(2):
				if prediction[i] > sarcasm_confidence:
					if self.classifier.classes_[i] == 'ironic':
						num_sarcastic += 1
					else:
						num_regular += 1

		if num_regular == 0 and num_sarcastic == 0:
			return 'regular'

		#More than 'sarcasm_thres' percent of sentences must be classified sarcastic, for the review to be classified sarcastic.
		if num_sarcastic > num_regular*sarcasm_thres:
			return 'ironic'
		return 'regular'

