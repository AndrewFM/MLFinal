# Important parameters used by different parts of the project.

num_features = 300       # Word vector dimensionality
min_word_count = 40      # A word must appear at least this many times in the corpus to be included in the word vector dictionary
num_workers = 4          # Number of threads to run in parallel
context = 10             # Context window size
downsampling = 1e-3      # Downsample setting for frequent words
local_test_size = 2000   # Number of training reviews to reserve for local evaluation

indicitave_pattern_thres = 2 # A pattern must appear this many times more frequently in sarcastic text than non-sarcastic, and vice verse.
THRES_PER = 200000 
PAT_THRES = 40      # Pattern must occur at least this many times per THRES_PER words to bother considering it relevant.
HFW_THRES = 20      # Term must occur at least this many times per THRES_PER words to be considered 'high frequency'.
CW_THRES  = 200     # Term must occur at most this many times per THRES_PER words to be considered a 'content word'.

pattern_alpha = 0.1 # Weighting parameter for sparse pattern matches in sarcasm feature-set.
pattern_gamma = 0.1 # Weighting parameter for incomplete pattern matches in sarcasm feature-set.

sarcasm_thres = 0.2 # At least this % of sentences in a review must be classified as sarcastic for the entire review to be considered sarcastic. 
sarcasm_confidence = 0.75 # The classifier must be at least this confident about the sarcastic nature of the sentence to make a determination.

# Many sarcastic reviews start out with positive sentiment, which devolve into negative sentiment later in the review
#    IE: - I'm a huge fan of this series... but this movie was just terrible
#        - To be fair, this movie had a great cast of actors... but not even they could save this awful narrative.
#
# Thus, how about just throwing away some percentage of the intro?
percent_disposal = (0.2, 3)  #Throw away either [0]% of the intro away, or [1] sentences, whichever is smaller.
