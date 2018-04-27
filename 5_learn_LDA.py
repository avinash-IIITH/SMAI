import gensim
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/train_gensim_dict.txt')
mm = gensim.corpora.MmCorpus('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/corpus_train.mm')

#lda_model_400 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 400, workers=3)
#lda_model_400.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_400.lda')

#lda_model_200 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 200, workers=3)
#lda_model_200.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_200.lda')

lda_model_100 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 100, workers=3)
lda_model_100.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_100.lda')

lda_model_50 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 50, workers=3)
print(lda_model_50.print_topics(num_topics=3, num_words=3))

lda_model_50.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_50.lda')

#lda_model_500 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 500, workers=3)
#lda_model_500.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_500.lda')

#lda_model_1000 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 1000, workers=3)
#lda_model_1000.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_1000.lda')

lda_model_80 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 80, workers=3)
lda_model_80.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_80.lda')

lda_model_30 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 30, workers=3)
lda_model_30.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_30.lda')

lda_model_10 = gensim.models.ldamulticore.LdaMulticore(corpus = mm, id2word=id2word ,num_topics = 10, workers=3)
lda_model_10.save('/home/avinash/Desktop/SMAI/DynamicLexiconGenerationCVC/DynamicLexiconGeneration/lda_model_10.lda')
