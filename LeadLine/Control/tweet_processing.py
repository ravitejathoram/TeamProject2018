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
en_stop = set(nltk.corpus.stopwords.words('english'))
print("Stop:", en_stop, len(en_stop))
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
data_files = ["2015-cyclone-pam.json", "2015-germanwings-crash.json", "2015-parisattacks.json",
              "2016-brexit.json", "2016-irish-ge16.json", "2016-lahoreblast.json",
               "2017-iraq-earthquake.json"]
model = 0
for i in data_files:
    model += 1
    name = i.split('-')[1]
    text_data = []
    count = 0
    with io.open('data/'+i, encoding='utf-8') as f:
        for i in f:
            line = json.loads(i)['full_text']
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', line)
            text = re.sub('@[^\s]+','', text)
            text = re.sub('#([^\s]+)', '', text)
            text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
            text = re.sub('[\d]','', text)
            text = re.sub('[^\x01-\x7F]', '', text)
            text = text.replace(".", '')
            text = text.replace("'", ' ')
            text = text.replace("\"", ' ')
            text = text.replace("\x9d",' ').replace("\x8c",' ')
            text = text.replace("\xa0",' ')
            text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
            text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')
            text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
            text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
            text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")

            tokens = prepare_text_for_lda(text)
            text_data.append(tokens)
            count += 1
            print(count)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    # pickle.dump(corpus, open('output/pickle/corpus_'+name+'.pkl', 'wb'))
    # dictionary.save('output/model_hurricane/dictionary.gensim')
    NUM_TOPICS = 25
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)
    ldamodel.save('output/model'+model+'/model.gensim')
    topics = ldamodel.print_topics(num_words=25)
    for topic in topics:
        print(topic)
    print("done and total tweets processed:" ,count)
print("--- %s seconds ---" % (time.time() - time_start))

