import io
import json
import re
from gensim import corpora
import pickle
import spacy
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import ssl
import gensim
import time
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

time_start = time.time()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# spacy.load('en')
parser = English()
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

en_stop = set(nltk.corpus.stopwords.words('english'))

print("Stop:", en_stop, len(en_stop))

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def get_lemma(word):
    lemma = wn.morphy(word)
    # print("lemma", lemma)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    # print("lemma2:", WordNetLemmatizer().lemmatize(word))
    return WordNetLemmatizer().lemmatize(word)


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    # print("parsing:", tokens)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


text_data = []
count = 0
with io.open('data/tweet2.json', encoding='utf-8') as f:
    for i in f:
        line = json.loads(i)['text']
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', line)
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('#([^\s]+)', '', text)
        text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]', ' ', text)
        text = re.sub('[\d]', '', text)
        text = re.sub('[^\x01-\x7F]', '', text)
        text = text.replace(".", '')
        text = text.replace("'", ' ')
        text = text.replace("\"", ' ')
        text = text.replace("\x9d", ' ').replace("\x8c", ' ')
        text = text.replace("\xa0", ' ')
        text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace(
            "\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f", ' ').replace("\x91\x8d", ' ')
        text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\xf0", ' ').replace('\xf0x9f',
                                                                                                  '').replace(
            "\x9f\x91\x8d", ' ').replace("\x87\xba\x87\xb8", ' ')
        text = text.replace("\xe2\x80\x94", ' ').replace("\x9d\xa4", ' ').replace("\x96\x91", ' ').replace(
            "\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8", ' ')
        text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace(
            "\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
        text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab",
                                                                                                           " ").replace(
            "\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
        print(text)
        print(ne_chunk(pos_tag(word_tokenize(text)), binary=True))
