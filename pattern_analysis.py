import pandas as pd
from pattern_functions import load_list_from_file, list_to_dict, sentence_to_patterns
from nltk import word_tokenize

print("Loading data...")
cws = list_to_dict(load_list_from_file('data/sarcasm/CWs.txt'))
hfws = list_to_dict(load_list_from_file('data/sarcasm/HFWs.txt'))

sarcasm_pattern_count = dict()
regular_pattern_count = dict()

sarcasm_file_data = pd.read_csv('data/sarcasm/five_labels_plus_stars.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Type','File'])
sarcasm_text_data = pd.read_csv('data/sarcasm/sarcasm_lines.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Text'])
pattern_data = pd.read_csv('data/sarcasm/extracted_patterns.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Pattern'])

num_sarcastic_words = 0
num_regular_words = 0

#Sarcasm pattern counts
print("Counting sarcastic pattern matches...")
for text in sarcasm_text_data['Text']:
	tok_text = word_tokenize(text)
	num_sarcastic_words += len(tok_text)
	sent_pats = sentence_to_patterns(tok_text, hfws, cws)

	for pattern in sent_pats:
		if pattern in pattern_data['Pattern'].values:
			if sarcasm_pattern_count.get(pattern) == None:
				sarcasm_pattern_count[pattern] = 1
			else:
				sarcasm_pattern_count[pattern] += 1

#Regular pattern counts
print("Counting non-sarcastic pattern matches...")
for i in range(len(sarcasm_file_data)):
	if sarcasm_file_data['Type'][i] == 'regular':
		f = open('data/sarcasm/Regular/'+sarcasm_file_data['File'][i]+'.txt', 'r', encoding='latin-1')
		found_review = False
		for line in f:
			if found_review:
				tok_text = word_tokenize(line.strip())
				num_regular_words += len(tok_text)
				sent_pats = sentence_to_patterns(tok_text, hfws, cws)
				for pattern in sent_pats:
					if pattern in pattern_data['Pattern'].values:
						if regular_pattern_count.get(pattern) == None:
							regular_pattern_count[pattern] = 1
						else:
							regular_pattern_count[pattern] += 1
				break
			if line.strip() == '<REVIEW>':
				found_review = True

#Find patterns that seem indicative of a certain speech type
def format_decimal(num):
	return "{0:.6f}".format(num)

print("Finding notable patterns...")

f_sar = open('data/sarcasm/indicative_of_sarcasm.tsv', 'w')
f_sar.write("Ratio\tPattern\n")
f_reg = open('data/sarcasm/indicative_of_nonsarcasm.tsv', 'w')
f_reg.write("Ratio\tPattern\n")

for pattern in pattern_data['Pattern']:
	sarcastic_count = 0
	regular_count = 0

	if sarcasm_pattern_count.get(pattern) != None:
		sarcastic_count = sarcasm_pattern_count[pattern]/num_sarcastic_words
	if regular_pattern_count.get(pattern) != None:
		regular_count = regular_pattern_count[pattern]/num_regular_words

	if regular_count*3 < sarcastic_count:
		f_sar.write(format_decimal(sarcastic_count)+":"+format_decimal(regular_count)+"\t"+pattern+"\n")
	if sarcastic_count*3 < regular_count:
		f_reg.write(format_decimal(regular_count)+":"+format_decimal(sarcastic_count)+"\t"+pattern+"\n")

f_sar.close()
f_reg.close()

print("All done!")