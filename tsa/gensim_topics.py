import gensim
from gensim.utils import simple_preprocess  # as tokenize
from tsa.data.sb5b import links

import logging
logger = logging.getLogger(__name__)


def links_gensim():
    endpoints = links.read()

    # the median length for the 6269 contentful endpoints currently in the database is 3217 characters
    maxlen = 10000

    corpus_strings = (endpoint.content[:maxlen] for endpoint in endpoints)
    # a list of lists of tokens. we need to make a dictionary and then encode these docs into bow,
    # so a generator is not possible
    corpus_tokens = [simple_preprocess(doc) for doc in corpus_strings]

    ## remove stopwords:
    # corpus_tokens = (token for token in corpus_tokens if token not in stopwords)

    ## remove hapaxlegomena:
    # from collections import Counter
    # counts = Counter(token for tokens in corpus_tokens for token in tokens)
    # hapax_legomena = set(word for word, count in counts.iteritems() if count == 1)
    # corpus_tokens = (token for token in corpus_tokens if token not in hapax_legomena)

    # plain bag of words
    dictionary_model = gensim.corpora.Dictionary(corpus_tokens)
    # corpus_bow = dictionary_model[corpus_tokens] # does this work? nope.
    corpus_bow = [dictionary_model.doc2bow(doc) for doc in corpus_tokens]

    # tf-idf transform
    # tfidf_model = gensim.models.TfidfModel(corpus_tokens)
    tfidf_model = gensim.models.TfidfModel(id2word=dictionary_model, dictionary=dictionary_model)
    corpus_tfidf = tfidf_model[corpus_bow]

    # okay, corpus is ready
    lda_model = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary_model, num_topics=10, passes=1)
    # hdp = gensim.models.HdpModel(corpus_tfidf, id2word=dictionary_model)

    # Initialize a transformation (Latent Semantic Indexing with 200 latent dimensions).
    # lsi = gensim.models.LsiModel(tfidf_corpus, num_topics=50, id2word=corpus.dictionary)
    # lsi.print_topics()
    # Convert another corpus to the latent space and index it.
    # index = similarities.MatrixSimilarity(lsi[another_corpus])
    # determine similarity of a query document against each document in the index
    # sims = index[query]

    tokens_per_topic = 15
    print
    for topic_i in range(lda_model.num_topics):
        topic = lda_model.show_topic(topic_i, topn=tokens_per_topic)
        # tokens = [dictionary[int(key)] for _, key in lda.show_topic(i)]
        ratios, tokens = zip(*topic)
        print 'Topic %d (%0.4f > ratio > %0.4f):' % (topic_i, ratios[0], ratios[-1])
        print ' ', ', '.join(tokens)

links_gensim()
