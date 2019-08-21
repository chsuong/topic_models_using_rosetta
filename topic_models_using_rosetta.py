
import os
import pandas as pd

DATA = '/Users/clarahsuong/topic_models_using_rosetta/data'
RAW = os.path.join(DATA, 'raw')
PROCESSED = os.path.join(DATA, 'processed')
SPARSE = os.path.join(DATA, 'sparse')
sfile_path = os.path.join(SPARSE, 'doc_tokens.vw')
filtered_sfile_path = os.path.join(PROCESSED, 'doc_tokens-filtered.vw')
sff_path = os.path.join(PROCESSED, 'sff.pkl')
​

from rosetta import TextFileStreamer, TokenizerBasic

text_streamer = TextFileStreamer(text_base_path=RAW, file_type='*', 
                                           tokenizer=TokenizerBasic())
​

from rosetta.text import streamers

stream = text_streamer.info_stream()
stream.__next__()
text = stream.__next__()['text']
print(text)
text_streamer.tokenizer.text_to_token_list(text)
token_stream.__next__()[:10]


from rosetta.text import text_processors, filefilter, streamers, vw_helpers

#create the VW format file 
my_tokenizer = text_processors.TokenizerBasic()
stream = streamers.TextFileStreamer(text_base_path=RAW, tokenizer=my_tokenizer)
stream.to_vw(sfile_path, n_jobs=-1, raise_on_bad_id=False)

### somewhere here run on your command line (stick with 5 passes or so...)
#cd data/processed
#rm -f *cache
#vw --lda 20 --cache_file doc_tokens.cache --passes 5 -p prediction.dat --readable_model topics.dat --bit_precision 16 --lda_D 975 --lda_rho 0.1 --lda_alpha 1 ../sparse/doc_tokens.vw

#load the sparse file 
formatter = text_processors.VWFormatter()
sff = text_processors.SFileFilter(formatter)
sff.load_sfile(sfile_path)

#remove "gaps" in the sequence of numbers (ids)
sff.compactify()
sff.save(PROCESSED + '/sff_basic.pkl')
sff.to_frame().sort_values(by='doc_fraction', ascending=False).head(10)

#use the LDAResults class from rosetta to convert back to readable, python friendly formats
lda = vw_helpers.LDAResults(PROCESSED + '/topics.dat', 
                            PROCESSED + '/prediction.dat', PROCESSED + '/sff_basic.pkl')

#look at some of the words
topic_words = lda.pr_token_g_topic.loc[:,'topic_12'].sort_values(ascending=False).index[:10]
lda.sfile_frame.loc[topic_words]

#look at the the first topic
a_topic = lda.pr_token_g_topic.T.loc['topic_00'].copy()
a_topic.sort_values(ascending=False)
a_topic[:10]

##look at first document's topic weights
lda.pr_topic_g_doc.T.iloc[[0]].plot(kind='bar', figsize=(12,7),
                                   title = 'First Document Topic Weights')

#or at the average topic probabilties 
import random
r = lambda: random.randint(0,255)
my_colors = ['#%02X%02X%02X' % (r(),r(),r()) for i in range(20)]
#my_colors = 'rgbkymc'
lda.pr_topic_g_doc.mean(axis=1).plot(kind='bar', figsize=(12,7), color=my_colors,
                                     title='Average Topic Probabilities')