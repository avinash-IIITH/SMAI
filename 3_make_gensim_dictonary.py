import gensim
import json


#fp = open('./val_dict.json','r')
fp = open('./train_dict.json','r')
list_of_words = json.load(fp)
dictionary_str = ''
for i in list_of_words:
    dictionary_str = dictionary_str + ' ' + str(i).lower()

gensim_dict = gensim.corpora.dictionary.Dictionary([dictionary_str.split()])
#gensim_dict = gensim.corpora.dictionary.Dictionary(["hi this is testing. hi this is not testing".split(), "This is new document this".split()])
gensim_dict.save_as_text('./train_gensim_dict.txt')
