"""
A small test program to evaluate the quality of the sarcasm classifier.

@author andrew
"""

from KaggleWord2VecUtility import KaggleWord2VecUtility
import nltk.data
import sarcasm_classifier

sarcasm_file_data = pd.read_csv('data/sarcasm/five_labels_plus_stars.tsv', delimiter="\t", quoting=3, quotechar='"', usecols=['Type','File'])
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
train_sents = []
train_labels = []
for i in range(len(sarcasm_file_data)):
	f = None
	if sarcasm_file_data['Type'] == 'regular':
		f = open('data/sarcasm/Regular/'+sarcasm_file_data['File'][i]+'.txt', 'r', encoding='latin-1')
	else:
		f = open('data/sarcasm/Ironic/'+sarcasm_file_data['File'][i]+'.txt', 'r', encoding='latin-1')

	review_found = False
	for line in f:
		if line.strip() == "</REVIEW>"
			break
		if review_found:
			for sent in KaggleWord2VecUtility.review_to_sentences(line, sentence_tokenizer)
				train_sents.append(sent)
				train_labels.append(sarcasm_file_data['Type'])
		if line.strip() == "<REVIEW>":
			review_found = True

	f.close()

classifier = sarcasm_classifier.Classifier(train_sents, train_labels)
