import nltk.data
import pandas as pd
from time import time
from nltk import word_tokenize
from KaggleWord2VecUtility import KaggleWord2VecUtility

PAT_THRES = 500      # Pattern must occur at least this many times per million words to bother considering it relevant.
HFW_THRES = 100      # Term must occur at least this many times per million words to be considered 'high frequency'.
CW_THRES  = 1000     # Term must occur at most this many times per million words to be considered a 'content word'.
THRES_PER = 1000000  # Threshold values above are treated under the ratio "x per y", where this value is y.
t0 = time()

#=======================================================================================
#  Load data
#=======================================================================================
print("Loading data to extract patterns from...")

extract_data = pd.read_csv('data/labeledTrainData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review'])
extract_data = extract_data.append(pd.read_csv('data/testData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review']), ignore_index=True)
extract_data = extract_data.append(pd.read_csv('data/unlabeledTrainData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review']), ignore_index=True)

print(str(extract_data['review'].size)+" reviews loaded.")

#=======================================================================================
#  Determine HFWs/CWs
#=======================================================================================
print("Now, determining which words are HFWs, and which are CWs...")

num_words = 0
progress_count = 0
word_counts = dict()
hfw_dict = dict()
cw_dict  = dict()

for review in extract_data['review']:
	progress_count += 1
	if progress_count % 2500 == 0:
		print("Reviews processed so far: "+str(progress_count)+" (%0.3fs)" % (time() - t0))

	for word in word_tokenize(review):
		num_words += 1
		if word_counts.get(word.lower()) == None:
			word_counts[word.lower()] = 1
		else:
			word_counts[word.lower()] += 1

print("Total word count: "+str(num_words))
print("Total unique words: "+str(len(word_counts.items())))

for _ in range(len(word_counts.items())):
	word_count = word_counts.popitem()

	if word_count[1] >= HFW_THRES*(num_words/THRES_PER):
		hfw_dict[word_count[0]] = 1
	if word_count[1] <= CW_THRES*(num_words/THRES_PER):
		cw_dict[word_count[0]] = 1

#=======================================================================================
#  Extract patterns
#=======================================================================================
def sentence_to_patterns(tok_sentence, hfws, cws):
	return_patterns = []
	for i in range(len(tok_sentence)):
		return_patterns += pattern_recurse(tok_sentence, hfws, cws, i, "", 0, 0)

	return return_patterns

#Each pattern must have between 2-6 HFWs, and between 1-6 CWs
def pattern_recurse(tok_sentence, hfws, cws, ind, cur_pattern, num_hfw, num_cw):
	if ind >= len(tok_sentence) or num_hfw > 6 or num_cw > 6:
		return []

	pat = cur_pattern
	pat_so_far = []

	if hfws.get(tok_sentence[ind].lower()) != None:
		if num_hfw >= 1 and num_cw >= 1:
			pat_so_far.append(cur_pattern+"_HFW")
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_HFW", num_hfw+1, num_cw)

	if cws.get(tok_sentence[ind].lower()) != None:
		if num_hfw >= 2:
			pat_so_far.append(cur_pattern+"_"+tok_sentence[ind].lower())
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_"+tok_sentence[ind].lower(), num_hfw, num_cw+1)	

	return pat_so_far

print("Now, extracting patterns...")

patterns = dict()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress_count = 0

for review in extract_data['review']:
	progress_count += 1
	if progress_count % 500 == 0:
		print("Reviews processed so far: "+str(progress_count)+" (%0.3fs)" % (time() - t0))

	for sentence in KaggleWord2VecUtility.review_to_sentences(review, sentence_tokenizer):
		sent_patterns = sentence_to_patterns(sentence, hfw_dict, cw_dict)
		for pattern in sent_patterns:
			if patterns.get(pattern) == None:
				patterns[pattern] = 1
			else:
				patterns[pattern] += 1

#=======================================================================================
#  Output patterns
#=======================================================================================
print("Writing patterns to file...")
f = open("extracted_patterns.txt", 'w')

for pattern in sorted(patterns, key=patterns.get, reverse=True):
	pattern_count = patterns[pattern]
	if pattern_count >= PAT_THRES*(num_words/THRES_PER):
		f.write(str(pattern_count)+":"+pattern+"\n")
f.close()


