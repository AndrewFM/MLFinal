def load_list_from_file(filename):
	list_data = []
	f = open(filename, 'r', encoding='utf-8')
	for line in f:
		list_data.append(line.strip())
	f.close()
	return list_data

def list_to_dict(list_in):
	ret_dict = dict()
	for item in list_in:
		ret_dict[item] = 1
	return ret_dict


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
			pat_so_far.append(cur_pattern+"_"+tok_sentence[ind].lower())
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_"+tok_sentence[ind].lower(), num_hfw+1, num_cw)

	if cws.get(tok_sentence[ind].lower()) != None:
		if num_hfw >= 2:
			pat_so_far.append(cur_pattern+"_CW")
		pat_so_far += pattern_recurse(tok_sentence, hfws, cws, ind+1, cur_pattern+"_CW", num_hfw, num_cw+1)	

	return pat_so_far