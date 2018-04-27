from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gensim
import json
import os
import sys
import logging
sys.path.insert(0, '/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/COCO-Text')
import coco_text

# Include Tensorflow deploy script in this
# script only.
#################################################################################################
##################################### Tensorflow imports ########################################
#################################################################################################
import sys
import json

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
##################################################################################################
################################# Tensorflow global variables ####################################
##################################################################################################

OUTPUT_TENSOR_NAME = 'final_result:0'
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
global graph
global final_tensor
global jpeg_data_tensor
global resized_image_tensor
global sess
global num_topics
###################################################################################################
#################################### Tensorflow functions #########################################
###################################################################################################
def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
    """
    with tf.Session() as sess:
        model_filename = '/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset_dict/output_graph.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            final_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    OUTPUT_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, final_tensor, jpeg_data_tensor, resized_input_tensor

def run_graph_on_image(sess, image_data, image_data_tensor,bottleneck_tensor):
    """Runs inference on an image to extract the final layer.
      Args:
        sess: Current active TensorFlow Session.
        image_data: Numpy array of image data.
        image_data_tensor: Input data layer in the graph.
        bottleneck_tensor: Layer before the final softmax.
      Returns:
        Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def make_cnn_query(image_url):
    global graph
    global final_tensor
    global jpeg_data_tensor
    global resized_image_tensor
    global sess
    image_data = gfile.FastGFile(image_url, 'rb').read()
    final_values = run_graph_on_image(sess, image_data, jpeg_data_tensor, final_tensor)
    return (final_values)


###################################################################################################
############################## Tensorflow Initialize / main function ##############################
###################################################################################################
def tensorflow_init():
    global final_tensor
    global jpeg_data_tensor
    global resized_image_tensor
    global sess
    graph, final_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())
    sess = tf.Session()

###################################################################################################
##################################### End Tensorflow ##############################################
###################################################################################################
global id2word
global lda_model

def main():
    global id2word
    global lda_model
    global num_topics

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        
    # Initialize tensorflow params.
    tensorflow_init()
    # Write the results in the .txt file.
    f = open('results_50.txt','w')
    # First, read the dictionary and make a dictionary.
    id2word = gensim.corpora.Dictionary.load_from_text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/train_gensim_dict.txt')
    
    # Specify the number of topics present in the LDA_model.
    num_topics = 50

    # Initiate COCO-Text API instance.
    coco_txt = coco_text.COCO_Text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset/COCO_Text.json')
    im_ids_txt = coco_txt.getImgIds(imgIds=coco_txt.train)
    
    # Image base_url.
    base_url_image = '/home/avinash/Desktop/SMAI/images/train2014/'

    # Parse the dictionary and Pre-process.
    iterator = id2word.iteritems()
    keys = id2word.keys()
    dict_natural = {}
    for i in range(0,len(keys)):
        temp_item = iterator.next()
        try:
            dict_natural[str(temp_item[1]).lower()] = []
        except:
            pass
    
    # Now, read the LDA model built on Natural dictionary.
    lda_model = gensim.models.ldamodel.LdaModel.load('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_50.lda', mmap='r')
    
    # Read, all the Validation set captions.
    # Combine each of them and make a document
    # Project the document on the LDA model.
    # Please verify with the CNN validation labels.
    
    # NOTE :The validation set of CNN is used here as 
    # Evaluation set for word-ranking.
    eval_file = open('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/dataset_corpus_train.json','r')
    eval_json = json.load(eval_file)
    
    # Get all the topics only once.
    # YOU FETCH ONLY ONCE B-).
    # I'm bond! James Bond B-).

    # Iterate over each instance of eval set.
    counter_image = 0
    counter_word_instance = 0
    for ind in im_ids_txt:
        # Keep printing the counter values.
        counter_image += 1
        if counter_image == 5000:
            break
        #print ("Total 20K : " + str(counter_image))
        words_present = []
        try:
            value = eval_json[str(ind)]
        except:
            continue
        captions = value['caption']
        url = value['url']
        final_url = base_url_image + str(url)
        ann_ids_txt = coco_txt.getAnnIds(imgIds = ind)
        ann_txt = coco_txt.loadAnns(ann_ids_txt)
        for j in ann_txt:
            try:
                if j['utf8_string'] != '':
                    counter_word_instance += 1
                    words_present.append(j['utf8_string'])
            except KeyError:
                pass
        if not words_present:
            pass
        else:
            # For a given image now we have prob distribution
            # and words_present in the image. Now, we need to
            # make all ranked dictionary and make inference 
            # for each and everyword.
            # REST ALL THE SHIT GOES HERE.
            try:
                query_out = make_query(final_url)
            except:
                continue

            for topic in query_out:
                topic_words = [x for x in lda_model.show_topic(topic[0], topn=len(dict_natural.keys()))]
                for word in topic_words:
                    try:
                        p_topic = topic[1]
                        p_word = word[1]
                        p_final = p_topic*p_word
                        dict_natural[str(word[0])].append(p_final)
                    except:
                        pass
            # Choose the highest values for each of the words.
            topic_word_ranks = []
            for j in dict_natural.keys():
                try:
                    topic_word_ranks.append((str(j),sum(dict_natural[str(j)])))
                except:
                    pass
            # Sort the dictionary and make final ranking prediction.
            final_ranking = sorted(topic_word_ranks, key=lambda x:x[1],reverse=True)
            final_ranks = []
            for i in final_ranking:
                final_ranks.append(i[0].lower())
            text_ranks = []
            for text in words_present:
                try:
                    text_ranks.append(str(text) + " : " + str(final_ranks.index(str(text.lower()))))
                except ValueError:
                    try:
                        text_ranks.append(str(text) + " : NA")
                    except:
                        text_ranks.append("Unicode error")
            f.write(str(text_ranks) + "\n")
            f.write("Image url : " + str(url) + "\n")
            f.write("Total words : " + str(len(final_ranks)) + "\n")
            f.write("---------------------------------------\n")
            print (str(text_ranks))
            print ("Image url : " + str(url) + "\n")
            print ("Total words : " + str(len(final_ranks)))
            print ("---------------------------------------")
        # At the end make sure to make each word probs zero.
        # For next iteration.
        for i in dict_natural.keys():
            dict_natural[i] = []
    print ("counter_word_instances : " + str(counter_word_instance))
    f.close()
def make_query(image_url):
    global id2word
    global lda_model
    global num_topics
    # In make_query, get the output from the CNN.
    script_query = make_cnn_query(image_url)
    #script_query = os.system("python " + str(script_url) + " " + str(image_url))
    query_out = []
    for ind in range(0,num_topics):
        temp = (ind, script_query[ind])
        query_out.append(temp)
    top_vals = sorted(query_out, key=lambda x:x[1], reverse=True)
    return top_vals

# Call the main function.
main()
