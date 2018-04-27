from pycocotools.coco import COCO
from nltk.corpus import stopwords
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

# This is the structure of the dataset_corpus
# dataset_corups = {}
# Key = Image_id
# Value = {"image_url","captions"}

dataset_corpus = {}

# Take all stopwords.
stop_words = set(stopwords.words('english'))
stop_words_str = [str(i).lower() for i in stop_words]
stop_words_post_space = [str(i) + " " for i in stop_words_str]
stop_words_pre_space = [" " + str(i) for i in stop_words_str]
print(str(stop_words_str))
coco_txt = coco_text.COCO_Text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset/COCO_Text.json')
annFile= '/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset/captions_train2014.json'
coco_ms = COCO(annFile)
cntr = 0
# First make on Training set.
im_ids_train = coco_txt.getImgIds(imgIds=coco_txt.train)
for i in im_ids_train:
    cntr = cntr + 1
    ann_ids_txt = coco_txt.getAnnIds(imgIds = i)
    ann_txt = coco_txt.loadAnns(ann_ids_txt)

    for j in ann_txt:
        words = []
        try:
            if j['utf8_string'] != '':
                try:
                    given_word = str(j['utf8_string']).lower()
                    words.append(given_word)
                except UnicodeEncodeError:
                    pass
        except KeyError:
            pass
    captions = ''
    ann_ids_ms = coco_ms.getAnnIds(imgIds = i)
    ann_ms = coco_ms.loadAnns(ann_ids_ms)
    for j in ann_ms:
        try:
            if j['caption'] != '':
                captions = captions + ' ' + j['caption']
        except KeyError:
            pass
    im = coco_ms.loadImgs(i)
    temp_dict = {}
    temp_dict['caption'] = captions
    temp_dict['words'] = words
    temp_dict['url'] = im[0]['file_name']
    dataset_corpus[str(i)] = temp_dict

print("counter : " + str(cntr))
fp = open('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/dataset_corpus_train.json','w')
json.dump(dataset_corpus,fp)