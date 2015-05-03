# -*- coding: utf-8 -*-
"""
@author: donghochoi
"""
import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False, remove_punctuation=False, case_sensitive=False):
        cleaned_text = BeautifulSoup(review).get_text() #Remove HTML

        words = cleaned_text.lower()
        if case_sensitive:
            words = cleaned_text
        words = words.replace("\\", "")
        words = words.replace("/", "")
        words = words.replace("--", "-")
        words = nltk.word_tokenize(words)
        words = [word.replace("''", '"').replace("``", '"') for word in words]

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        if remove_punctuation:
            words = [w for w in words if w.isalpha() or w.isdigit()]

        return words

    # Take an entire review in raw-text/raw-string form, and convert it into a list of individual sentences.
    # A good tokenizer to use is nltk.data.load('tokenizers/punkt/english.pickle')
    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False, remove_punctuation=False, case_sensitive=False):
        raw_sentences = tokenizer.tokenize(review.strip())

        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                #The return value will actually be a collection of tokenized sentences
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, remove_stopwords, remove_punctuation, case_sensitive))

        return sentences