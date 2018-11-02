import pickle
import gensim
temp = {}
def explore_topic(lda_model, topic_number, topn, output=True):
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if term in temp.keys():
            temp[term] += frequency
        else:
            temp[term] = frequency
    return terms

num_topics = 10
dictionary = gensim.corpora.Dictionary.load('output/model_hurricane/dictionary.gensim')
corpus = pickle.load(open('output/model_hurricane/corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('output/model_hurricane/model.gensim')
temp = {}

topic_summaries = []
for i in range(num_topics):
    tmp = explore_topic(lda,topic_number=i, topn=10, output=True)
temp = sorted(temp.items(), key=lambda item: item[1], reverse=True)
print("Top 10 words and their Frequency")
for i in temp[:10]:
    print(u'{:20} {:.3f}'.format(i[0], round(i[1], 3)))

