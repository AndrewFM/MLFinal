"""
A small test program to evaluate the quality of the sarcasm classifier.

@author andrew
"""

import nltk.data
import numpy
import os
import pandas as pd
import sarcasm_classifier

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

classifier = sarcasm_classifier.Classifier(nltk.data.load('tokenizers/punkt/english.pickle'))
if os.path.isfile("data/dumps/sarcasm_classifier.pkl"):
	print("Found pickled sarcasm classifier. Loading it...")
	classifier.load_pickle('data/dumps/sarcasm_classifier.pkl')
else:
	classifier.fit_transform(train_reviews, sarcasm_file_data['Type'])
	classifier.save_pickle('data/dumps/sarcasm_classifier.pkl')	

while(True):
	user_sent = input("Enter a sentence:")
	print("I think your sentence is", classifier.predict_sent(user_sent))
