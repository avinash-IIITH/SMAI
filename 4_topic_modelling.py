# Read the captions + words for each image.
# Make one string for each image.
# Tokenize
# Remove stopwords and '.'

import json
from nltk import word_tokenize
from nltk.corpus import stopwords
import gensim
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

fp = open('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/dataset_corpus_train.json','r')
data = json.load(fp)
data_corpus = []
for i in data.keys():
    val = data[i]
    captions = val['caption']
    words_str = ''
    words = val['words']
    for word in words:
        words_str = words_str + ' ' + word
    temp_doc = captions + ' ' + words_str
    data_corpus.append(temp_doc)

stop_words = set(stopwords.words('english')).union('.')
stop_words_str = [str(i).lower() for i in stop_words]

tok_corpus = []
for i in data_corpus:
    temp_tokens = [j.lower() for j in word_tokenize(i) if j not in stop_words_str]
    tok_corpus.append(temp_tokens)

id2word = gensim.corpora.Dictionary.load_from_text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/train_gensim_dict.txt')
corpus = [id2word.doc2bow(text) for text in tok_corpus]
gensim.corpora.MmCorpus.serialize('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/corpus_train.mm', corpus)
