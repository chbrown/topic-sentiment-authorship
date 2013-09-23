from itertools import islice
from collections import defaultdict
import gensim
from gensim.utils import simple_preprocess  # as tokenize
from tsa.data.sb5b import links

import logging
logger = logging.getLogger(__name__)


def links_gensim():
    endpoints = links.read()

    # debug, sort of
    endpoints = list(endpoints)

    # the median length for the 6269 contentful endpoints currently in the database is 3217 characters
    maxlen = 10000

    # a list of strings
    corpus_strings = (endpoint.content[:maxlen] for endpoint in endpoints)

    # a list of lists of strings. we need to make a dictionary and then encode these docs into bow,
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
    # dictionary = gensim.corpora.Dictionary(corpus_tokens)
    # corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_tokens]
    dictionary = gensim.corpora.Dictionary()
    corpus_bow = [dictionary.doc2bow(doc, allow_update=True) for doc in corpus_tokens]

    # tf-idf transform
    # tfidf_model = gensim.models.TfidfModel(corpus_tokens)
    tfidf_model = gensim.models.TfidfModel(id2word=dictionary, dictionary=dictionary)
    corpus_tfidf = tfidf_model[corpus_bow]

    # okay, corpus is ready
    num_topics = 5
    topic_model = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics, passes=1)
    # topic_model = gensim.models.HdpModel(corpus_tfidf, id2word=dictionary)

    # Initialize a transformation (Latent Semantic Indexing with 200 latent dimensions).
    # lsi = gensim.models.LsiModel(tfidf_corpus, num_topics=50, id2word=corpus.dictionary)
    # lsi.print_topics()
    # Convert another corpus to the latent space and index it.
    # index = similarities.MatrixSimilarity(lsi[another_corpus])
    # determine similarity of a query document against each document in the index
    # sims = index[query]

    tokens_per_topic = 15
    print
    for topic_i in range(topic_model.num_topics):
        topic = topic_model.show_topic(topic_i, topn=tokens_per_topic)
        # tokens = [dictionary[int(key)] for _, key in lda.show_topic(i)]
        ratios, tokens = zip(*topic)
        print 'Topic %d (%0.4f > ratio > %0.4f):' % (topic_i, ratios[0], ratios[-1])
        print ' ', ', '.join(tokens)

    print 'analyzing total topic alignments:'
    topic_indices = range(num_topics)
    topics_sums = defaultdict(int)
    topics_count_firsts = defaultdict(int)
    topics_count_seconds = defaultdict(int)
    corpus_topics = topic_model[corpus_tfidf]
    for doc_topics in corpus_topics:
        for topic_index, topic_ratio in doc_topics:
            topics_sums[topic_index] += topic_ratio
        doc_topics_sorted = sorted(doc_topics, key=lambda tup: tup[1], reverse=True)

        topics_count_firsts[doc_topics_sorted[0][0]] += 1
        if len(doc_topics_sorted) > 1:
            topics_count_seconds[doc_topics_sorted[1][0]] += 1

    for topic_index in topic_indices:
        print 'Topic %d: count[1]=%d, count[2]=%d, sum=%0.2f' % (topic_index,
            topics_count_firsts[topic_index], topics_count_seconds[topic_index], topics_sums[topic_index])

    print 'previewing 10 endpoints'
    # look at only the first 10 endpoint.
    # yes, we're testing on our training data, but it's for a good cause.
    for endpoint in islice(endpoints, 10):
        doc_tokens = simple_preprocess(endpoint.content)
        doc_bow = dictionary.doc2bow(doc_tokens)
        doc_tfidf = tfidf_model[doc_bow]

        doc_topics = topic_model[doc_tfidf]
        # doc_topics is now a list of (topic_index, ratio) tuples, i.e.,
        #   doc_topics.sum(_._2) == 1
        doc_topics_sorted = sorted(doc_topics, key=lambda tup: tup[1], reverse=True)
        print
        print '-'*80
        print
        print 'top', doc_topics_sorted[0]
        print 'all', doc_topics
        print
        print endpoint.content


links_gensim()
