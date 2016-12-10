import numpy as np
import pandas as pd
import re
import string
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords # Import the stop word list
#from tqdm import tqdm, tqdm_pandas
from sklearn.feature_extraction.text import CountVectorizer
import random
from functools import partial
import nltk
from sklearn.decomposition import RandomizedPCA
import cPickle
import sys

# warning this will need to be run on a machine with a shit ton of memory!



num_components = int(sys.argv[1])


#num_components = 20


overallFrame = pd.read_pickle('overallTrainingData')

pattern = '''([\w-]+)'''
pattern2 = '''(?u)\\b\\w\\w+\\b'''
#pattern3 = '''(?u)\\b\\w+\\w-w\\b'''



#vectorizer = CountVectorizer(min_df=1)
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
#vectorizer = CountVectorizer(analyzer=partial(nltk.regexp_tokenize, pattern=pattern) )
vectorizer = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
       dtype=np.float32, encoding='utf-8', input='content',
       lowercase=True, max_df=1.0, max_features=None, min_df=1,
       ngram_range=(1, 1), preprocessor=None, stop_words=None,
       strip_accents=None, token_pattern=pattern2,
       tokenizer=None, vocabulary=None)

corpus = overallFrame['content_clean'].tolist()
X = vectorizer.fit_transform(corpus)
#vectorizer = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#        dtype=np.float32, encoding='utf-8', input='content',
#        lowercase=True, max_df=1.0, max_features=None, min_df=1,
#        ngram_range=(1, 1), preprocessor=None, stop_words=None,
#        strip_accents=None, token_pattern=pattern3,
#        tokenizer=None, vocabulary=None)

pattern = '''([\w-]+)'''
vectorizer = CountVectorizer(analyzer=partial(nltk.regexp_tokenize, pattern=pattern) )

corpus = overallFrame['tags_clean'].tolist()
Y = vectorizer.fit_transform(corpus)


X_dense = X[1:2000,:].toarray()

pca = RandomizedPCA(n_components=num_components)

pca.fit(X_dense)

cPickle.dump(pca, open('pca_model.p', 'wb')) 