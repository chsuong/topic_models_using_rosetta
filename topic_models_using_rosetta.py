
import os

#DATA is the directory where the documents are stored as text files on your machine
DATA = '/Users/clarahsuong/processing/topic_model_general/data'
RAW = os.path.join(DATA, 'raw')
PROCESSED = os.path.join(DATA, 'processed')
SPARSE = os.path.join(DATA, 'sparse')

sfile_path = os.path.join(SPARSE, 'doc_tokens.vw')
filtered_sfile_path = os.path.join(PROCESSED, 'doc_tokens-filtered.vw')
sff_path = os.path.join(PROCESSED, 'sff.pkl')

text_files = [f for f in os.listdir(RAW) if f.endswith('.txt')]
text_files

from rosetta.text import filefilter

def simple_file_streamer(base_path):
    paths = filefilter.get_paths(base_path, get_iter=True)
    for p in paths:
        if p.endswith('.txt'):
            with open(p) as f:
                text = f.read()
                yield(text)

simple_stream = simple_file_streamer(RAW)
simple_stream.__next__()


from rosetta import TextFileStreamer, TokenizerBasic
text_streamer = TextFileStreamer(text_base_path=RAW, file_type='*.txt', 
                                           tokenizer=TokenizerBasic())

from rosetta.text import streamers
stream = text_streamer.info_stream()
stream.__next__()

text = stream.__next__()['text']
print(text)

import nltk
nltk.word_tokenize(text)
text_streamer_nltk = TextFileStreamer(text_base_path=RAW, file_type='*.txt',  
                                      tokenizer_func=nltk.word_tokenize)

stream_nltk = text_streamer_nltk.token_stream()
stream_nltk.__next__()[:10]

#Install vowpal-wabbit via CLI with the command "brew install vowpal-wabbit"
#Convert the text file to a vw file with the command 
from rosetta.text import text_processors, filefilter, streamers, vw_helpers
my_tokenizer = text_processors.TokenizerBasic()
stream = streamers.TextFileStreamer(text_base_path=RAW, tokenizer=my_tokenizer)
stream.to_vw(sfile_path, n_jobs=-1, raise_on_bad_id=False)


from rosetta.text.text_processors import SFileFilter, VWFormatter
sff = SFileFilter(VWFormatter())
sff.load_sfile(sfile_path)
df = sff.to_frame()
df.head()
df.describe()

#sff.filter_extremes(doc_freq_min=5, doc_fraction_max=0.8)
sff.compactify()
sff.save(sff_path)
sff.save('sff_file.pkl')
sff.filter_sfile('/Users/clarahsuong/processing/topic_model_general/data/sparse/doc_tokens.vw', '/Users/clarahsuong/processing/topic_model_general/data/sparse/doc_tokens_filtered.vw')
sff.to_frame().sort_index(by='doc_fraction', ascending=False).head(10)




lda = vw_helpers.LDAResults(PROCESSED + '/topics.dat', 
                            PROCESSED + '/prediction.dat', PROCESSED + '/sff.pkl')

topic_words = lda.pr_token_g_topic.loc[:,'topic_12'].sort_values(ascending=False).index[:10]
lda.sfile_frame.loc[topic_words]

a_topic = lda.pr_token_g_topic.T.loc['topic_00'].copy()
a_topic.sort_values(ascending=False)
a_topic[:10]
