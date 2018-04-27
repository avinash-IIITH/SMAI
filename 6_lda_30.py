import gensim
import json
import sys
import logging
sys.path.insert(0, '/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/COCO-Text')
import coco_text


# This script is for evlation of LDA model
# which is built on Dictionary made from 


global id2word
global lda_model


def main():
    global id2word
    global lda_model
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Write the results in the .txt file.
    f = open('results_train_30.txt','w')

    # First, read the dictionary and make a dictionary.
    id2word = gensim.corpora.Dictionary.load_from_text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/train_gensim_dict.txt')
    
    # Specify the number of topics present in the LDA_model.
    num_topics = 30

    # Initiate COCO-Text API instance.
    coco_txt = coco_text.COCO_Text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/Dataset/COCO_Text.json')
    im_ids_txt = coco_txt.getImgIds(imgIds=coco_txt.train)

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
    
    # Now, read the LDA model built on Dataset_dictionary.
    lda_model = gensim.models.ldamodel.LdaModel.load('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_30.lda', mmap='r')
    
    # Read, all the Validation set captions.
    # Combine each of them and make a document
    # Project the document on the LDA model.
    # Please verify with the CNN validation labels.
    
    # NOTE :The validation set of CNN is used here as 
    # Evaluation set for word-ranking
    # .
    eval_file = open('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/dataset_corpus_train.json', 'r')
    eval_json = json.load(eval_file)

    # Iterate over each instance of eval set.
    counter_image = 0
    counter_word_instance = 0
    for ind in im_ids_txt:
        # Keep printing the counter values.
        counter_image += 1
        print "Total 20K : " + str(counter_image)
        words_present = []
        value = eval_json[str(ind)]

        captions = value['caption']
        temp_document = ''
        for doc in captions:
            temp_document = temp_document + " " + str(doc)
        query_out = make_query(temp_document)
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
            for topic in query_out[0:num_topics]:
                topic_words = [x for x in lda_model.show_topic(topic[0], topn=len(dict_natural))]
                for j in topic_words:
                    try:
                        p_topic = topic[1]
                        p_word = j[1]
                        p_final = p_topic*p_word
                        dict_natural[str(j[0])].append(p_final)
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
            f.write("Total words : " + str(len(final_ranks)) + "\n")
            f.write("---------------------------------------\n")
            print str(text_ranks)
            print "Total words : " + str(len(final_ranks))
            print "---------------------------------------"
        # At the end make sure to make each word probs zero.
        # For next iteration.
        for i in dict_natural.keys():
            dict_natural[i] = []
    print "counter_word_instances : " + str(counter_word_instance)
    f.close()
def make_query(query):
    global id2word
    global lda_model

    vec_bow = id2word.doc2bow(query.lower().split())
    query_out = lda_model[vec_bow]
    print str(query_out)
    return query_out

main()
