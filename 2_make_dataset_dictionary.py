import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import sys
sys.path.insert(0, '/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/COCO-Text')
import coco_text
from nltk.corpus import stopwords
import json

# This script is to make dictionary from all the
# words in the dataset. Thus this include both 
# training and validation images from COCO-Text.

dataset_dictionary_all = []
stop_words = set(stopwords.words('english'))
stop_words_str = [str(i).lower() for i in stop_words]
stop_words_post_space = [str(i) + " " for i in stop_words_str]
stop_words_pre_space = [" " + str(i) for i in stop_words_str]
print str(stop_words_str)
cntr = 0
coco_txt = coco_text.COCO_Text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset/COCO_Text.json')

# First make on Training set.
im_ids_train = coco_txt.getImgIds(imgIds=coco_txt.train)  # I think it should be coco_txt.train not coco_txt.val.. Modifying it
#im_ids_train = coco_txt.getImgIds(imgIds=coco_txt.val)
for i in im_ids_train:
    ann_ids_txt = coco_txt.getAnnIds(imgIds = i)
    ann_txt = coco_txt.loadAnns(ann_ids_txt)
    for j in ann_txt:
        try:
            if j['utf8_string'] != '':
                try:
                    given_word = str(j['utf8_string']).lower()
                    if ((given_word not in dataset_dictionary_all) and (given_word not in stop_words_str) and (given_word not in stop_words_post_space) and (given_word not in stop_words_pre_space)):
                        cntr += 1
                        dataset_dictionary_all.append(given_word)
                except UnicodeEncodeError:
                    pass
        except KeyError:
            pass

print str(cntr)
fp = open('train_dict.json','w')  # changed name from val_dict.json
#fp = open('val_dict.json','w')
json.dump(dataset_dictionary_all,fp)
