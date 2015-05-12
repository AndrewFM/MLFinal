'python ModelComparison.py' runs comparisons between the bag-of-words, Word2Vec, and paragraph vector models. 

'python sentiment_classifier_word2vec.py' runs the sentiment analysis code containing word2vec + the sarcasm classifier, giving accuracy results with and without sarcasm disambiguation. The sarcasm classification was never ported to the main paragraph vector file above, since we were unable to achieve good results with the sarcasm classifier as a standalone. 

'python pattern_extractor.py' compiles a list of frequent sentence patterns from the IMDB reviews. This takes about an hour to run, and has been done already. (Output: data/sarcasm/extracted_patterns.tsv, as well as HFWs/CWs.txt)

'python pattern_analyser.py' finds the best patterns from pattern_extractor's list, by comparing against the sarcasm corpus. This takes around 10-30 minutes to run, and has been done already. (Output: data/sarcasm/indicative_of_sarcasm.tsv, and indicative_of_nonsarcasm.tsv) 
