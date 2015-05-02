"""
A mass extraction of (notable) sentence patterns from the entire collection of 100,000 IMDB reviews.
Pattern extraction based on (Davidov et.al 2010) "Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon"

@author andrew
"""

import nltk.data
import os
import pandas as pd
from time import time
from nltk import word_tokenize
from pattern_functions import sentence_to_patterns, ord_to_pattern, pattern_to_ord
from KaggleWord2VecUtility import KaggleWord2VecUtility

THRES_PER = 300000 
PAT_THRES = 50     # Pattern must occur at least this many times per THRES_PER words to bother considering it relevant.
HFW_THRES = 30      # Term must occur at least this many times per THRES_PER words to be considered 'high frequency'.
CW_THRES  = 300     # Term must occur at most this many times per THRES_PER words to be considered a 'content word'.
t0 = time()

#=======================================================================================
#  Load data
#=======================================================================================
print("Loading data to extract patterns from...")

extract_data = pd.read_csv('data/unlabeledTrainData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review'])
extract_data = extract_data.append(pd.read_csv('data/testData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review']), ignore_index=True)
extract_data = extract_data.append(pd.read_csv('data/labeledTrainData.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['review']), ignore_index=True)

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
		print("Reviews processed so far: "+str(progress_count)+" (%0.2fs)" % (time() - t0))

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
print("Now, extracting patterns...")

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress_count = 0
word_dump_log = 0
patterns = dict()

for review in extract_data['review']:
	progress_count += 1
	if progress_count % 500 == 0:
		print("Reviews processed so far: "+str(progress_count)+" (%0.2fs)" % (time() - t0))

	for sentence in KaggleWord2VecUtility.review_to_sentences(review, sentence_tokenizer):
		word_dump_log += len(sentence)
		sent_patterns = sentence_to_patterns(sentence, hfw_dict, cw_dict)
		for pattern in sent_patterns:
			if patterns.get(pattern) == None:
				patterns[pattern] = 1
			else:
				patterns[pattern] += 1
		
		#Have to work-around this with disk IO, unfortunately, because storing all patterns to an in-memory dict requires too much RAM.
		#Dump all patterns to file every several hundred thousand words or so...
		if word_dump_log > THRES_PER:
			print("Pausing to dump patterns to disk...")
			word_dump_log = 0
			for _ in range(len(patterns.items())):
				out_pattern = patterns.popitem()

				if out_pattern[1] >= PAT_THRES:
					old_count = 0
					ord_pat = pattern_to_ord(out_pattern[0]) #Have to convert symbols to their ASCII codes, because many symbols are not allowed in file names

					if os.path.isfile("pattern_temp/"+ord_pat):
						f = open("pattern_temp/"+ord_pat, 'r')
						old_count = int(f.readline())
						f.close()

					f = open("pattern_temp/"+ord_pat, 'w')
					f.write(str(old_count+out_pattern[1]))
					f.close()		
			print("Done. (%0.2fs)" % (time() - t0))	

#=======================================================================================
#  Output patterns
#=======================================================================================
print("Gathering final patterns from disk... (%0.2fs)" % (time() - t0))

patterns = dict()
for ord_pattern in os.listdir("pattern_temp"):
	pattern = ord_to_pattern(ord_pattern)
	f = open("pattern_temp/"+ord_pattern, 'r')
	patterns[pattern] = int(f.readline())
	f.close()
	os.remove("pattern_temp/"+ord_pattern)

print("Writing patterns to file... (%0.2fs)" % (time() - t0))
f = open("data/sarcasm/extracted_patterns.tsv", 'w')
f.write("Count\tPattern\n")
for pattern in sorted(patterns, key=patterns.get, reverse=True):
	pattern_count = patterns[pattern]
	f.write(str(pattern_count)+"\t"+pattern+"\n")
f.close()

f = open("data/sarcasm/HFWs.txt", 'wb')
for _ in range(len(hfw_dict.items())):
	hfw = hfw_dict.popitem()
	f.write(bytes(hfw[0]+"\n", 'utf-8'))
f.close()

f = open("data/sarcasm/CWs.txt", 'wb')
for _ in range(len(cw_dict.items())):
	cw = cw_dict.popitem()
	f.write(bytes(cw[0]+"\n", 'utf-8'))
f.close()

print("All done! (%0.2fs)" % (time() - t0))

