import pandas as pd
from pattern_functions import load_list_from_file, list_to_dict, sentence_to_patterns
from nltk import word_tokenize

# Returns counts that are not normalized
def get_misc_features(tok_sentence):
	features = dict()
	features['exc_count'] = 0
	features['que_count'] = 0
	features['quo_count'] = 0
	features['cap_count'] = 0

	for word in tok_sent:
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

	#Returns a value between 0 & 1, indicating the similarity between the two input patterns
	#pattern should be the 'input' pattern, and comp_pattern should be a pattern from the self.pattern_data database
	def get_pattern_match(self, pattern, comp_pattern):
		consecutive_queue = comp_pattern
		consecutive_queue.reverse()
		next_consecutive = consecutive_queue.pop()

		for item in pattern:
			if item == next_consecutive:
				if len(consecutive_queue) == 0:
					break
				else:
					next_consecutive = consecutive_queue.pop()

		if len(consecutive_queue) == 0: #All consecutive patterns matched
			if len(pattern) == len(comp_pattern): 
				return 1			#Exact match
			else:
				return self.alpha	#Sparse match

		if len(consecutive_queue) == len(comp_pattern):
			return 0				#No match
		else:
			return self.gamma*((len(comp_pattern)-len(consecutive_queue))/len(pattern))	#Incomplete match

	def get_sentence_features(self, tok_sentence):
		feature_vectors = dict()
		sent_pats = sentence_to_patterns(tok_sentence, self.hfws, self.cws)

		#Pattern features
		for pattern in self.pattern_data['Pattern']:
			feature_vectors[pattern] = 0
			for sent_pattern in sent_pats:
				match = get_pattern_match(sent_pattern, pattern)
				if match > feature_vectors[pattern]:
					feature_vectors[pattern] = match #Store the best pattern match found

		#Other features
		misc_features = get_misc_features(tok_sentence)
		feature_vectors['exc_count'] = misc_features['exc_count']/(self.max_exclamations*self.max_avg)
		feature_vectors['que_count'] = misc_features['que_count']/(self.max_questions*self.max_avg)
		feature_vectors['quo_count'] = misc_features['quo_count']/(self.max_quotations*self.max_avg)
		feature_vectors['cap_count'] = misc_features['cap_count']/(self.max_allcaps*self.max_avg)
		feature_vectors['word_count'] = len(tok_sentence)/(self.max_sent_length*self.max_avg)

		return feature_vectors

	def __init__(self, train_sents, train_labels):
		self.cws = list_to_dict(load_list_from_file('data/sarcasm/CWs.txt'))
		self.hfws = list_to_dict(load_list_from_file('data/sarcasm/HFWs.txt'))
		self.pattern_data = pd.read_csv('data/sarcasm/indicative_of_sarcasm.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern'])

		self.alpha = 0.1 #Weighted parameter for "sparse" pattern matches
		self.gamma = 0.1 #Weighted parameter for "incomplete" pattern matches

		#Find maximum occuring values of certain features (used later for normalization)
		self.max_sent_length = 0
		self.max_exclamations = 0
		self.max_questions = 0
		self.max_quotations = 0
		self.max_allcaps = 0

		for sent in train_sents:
			tok_sent = word_tokenize(sent)
			misc_features = get_misc_features(tok_sent)

			self.max_sent_length = max(self.max_sent_length, len(tok_sent))
			self.max_exclamations = max(self.max_exclamations, misc_features['exc_count'])
			self.max_questions = max(self.max_questions, misc_features['que_count'])
			self.max_quotations = max(self.max_quotations, misc_features['quo_count'])
			self.max_allcaps = max(self.max_allcaps, misc_features['cap_count'])

		self.max_avg = (self.max_sent_length+self.max_exclamations+self.max_questions+self.max_quotations+self.max_allcaps)/5

		#Compile features of training sentences
		train_data = []
		for sent in train_sents:
			train_data.append(get_sentence_features(sent))

		self.classifier = 

	def predict(self, tok_sentence):
		sent_feats = get_sentence_features(tok_sentence)
		
