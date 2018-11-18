import io
import json
import re
import pandas as pd
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
import operator
import nltk
import spacy
from spacy import displacy
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

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



def ner_help(text):
    text = nltk.word_tokenize(text)
    text = nltk.pos_tag(text)
    return text


global_entity = {}
text_data = []
count = 0
per = []
loc = []
with io.open('data/tweet2Hijack.json', encoding='utf-8') as f:
    for i in f:
        if count!=10000:
            line = json.loads(i)['full_text']
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', line)
            text = re.sub('@[^\s]+', '', text)
            text = re.sub('#([^\s]+)', '', text)
            text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]', ' ', text)
            text = re.sub('[\d]', '', text)
            text = re.sub('[^\x01-\x7F]', '', text)
            text = re.sub('^RT', '', text)
            text = re.sub( '\s+', ' ', text).strip()
            text = text.replace(".", '')
            text = text.replace("'", ' ')
            text = text.replace("\"", ' ')
            text = text.replace("\n", ' ')
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
            # ne_tree = ne_chunk(pos_tag(word_tokenize(text)), binary=True)
            # iob_tag = tree2conlltags(ne_tree)
            # print(iob_tag)
            # res = ner_help(text)
            # pattern = 'NP: {<DT>?<JJ>*<NN>}'
            # cp = nltk.RegexpParser(pattern)
            # cs = cp.parse(res)
            # iob_tagged = tree2conlltags(cs)
            # # pprint(iob_tagged)
            ne_tree = ne_chunk(pos_tag(word_tokenize(text)))
            nlp = spacy.load('en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0')
            doc = nlp(text)
            # labels = [x.label_ for x in doc.ents]
            # Counter(labels)
            # pprint([(X.text, X.label_) for X in doc.ents])

            for X in doc.ents:
                if X.label_ == "PERSON" and X.text != " ":
                    per.append(X.text)
                elif X.label_ == "GPE"  and (X.text != " "):
                    loc.append(X.text)
            count += 1
            print(count)
        else:
            break



x = dict(Counter(loc))
df1 = pd.DataFrame(columns=["Country", "Mentions"])
data1 = {}
j = 0
for n, m in x.items():
    data1["Country"] = n
    data1["Mentions"] = m
    df1.loc[j] = pd.Series(data1)
    j += 1
    data1 = {}
df1.to_csv("data/2016_Hijack_locations.csv")

y = dict(Counter(per))
df2 = pd.DataFrame(columns=["Entity", "Mentions", "Year"])
data2 = {}
j = 0
for n, m in y.items():
    data2["Entity"] = n
    data2["Mentions"] = m
    data2["Year"] = 2016
    df2.loc[j] = pd.Series(data2)
    j += 1
    data2 = {}
df2.to_csv("data/2016_Hijack_entity.csv")

print("DONE!!!!")


