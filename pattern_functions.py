"""
Utility functions dealing with patterns extractracted from sentences.
Pattern extraction based on (Davidov et.al 2010) "Semi-Supervised Recognition of Sarcastic Sentences in Twitter and Amazon"

@author: andrew
"""

#Creates a list, where each item in the list is a stripped line of the file
def load_list_from_file(filename):
	list_data = []
	f = open(filename, 'r', encoding='utf-8')
	for line in f:
		list_data.append(line.strip())
	f.close()
	return list_data

#Converts a list into a sparse dictionary representation where each item in the list has value '1'.
def list_to_dict(list_in):
	ret_dict = dict()
	for item in list_in:
		ret_dict[item] = 1
	return ret_dict

#Converts symbol/punctuation characters in a pattern into an ordinal representation (for situations where symbols are not allowed, such as filenames)
def pattern_to_ord(pattern):
	split_pattern = pattern.split("_")
	for i in range(len(split_pattern)):
		if not split_pattern[i].isalpha() and not split_pattern[i].isdigit() and len(split_pattern[i]) == 1:
			split_pattern[i] = "o"+str(ord(split_pattern[i]))
	return '_'.join(split_pattern)

#Converts an ordinal-converted pattern (such as from pattern_to_ord), back into the pattern's correct symbolic representation
def ord_to_pattern(ord_pattern):
	split_pattern = ord_pattern.split("_")
	for i in range(len(split_pattern)):
		if split_pattern[i][0] == 'o' and split_pattern[i][1:].isdigit():
			split_pattern[i] = chr(int(split_pattern[i][1:]))
	return '_'.join(split_pattern)

#Finds all patterns contained in a sentence.
def sentence_to_patterns(tok_sentence, hfws, cws):
	return_patterns = []
	for i in range(len(tok_sentence)):
		return_patterns += pattern_recurse(tok_sentence, hfws, cws, i, "", 0, 0)

	return return_patterns

#Recursive subroutine in sentence_to_patterns() to extract patterns from a sentence.
def pattern_recurse(tok_sentence, hfws, cws, ind, cur_pattern, num_hfw, num_cw):
	#A pattern must have between 2-6 HFWs, and between 1-6 CWs to be considered valid
	if ind >= len(tok_sentence) or num_hfw > 6 or num_cw > 6:
		return []

	pat = cur_pattern
	pat_so_far = []

	if hfws.get(tok_sentence[ind].lower()) != None:
		if num_hfw >= 1 and num_cw >= 1:
			pat_so_far.append(cur_pattern+"_"+tok_sentence[ind].lower())
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_"+tok_sentence[ind].lower(), num_hfw+1, num_cw)

	if cws.get(tok_sentence[ind].lower()) != None:
		if num_hfw >= 2:
			pat_so_far.append(cur_pattern+"_CW")
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_CW", num_hfw, num_cw+1)	

	return pat_so_far