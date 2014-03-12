'''
Feature functions

Each feature function should take a single argument, the data, and maybe some keywords,
and return a tuple: (values (np.array), feature_names ([str]))

    N = len(data)  # or data.shape[0]
    k = len(feature_names)  # varies based on data
    values.shape = (N, k)

'''
import re
import itertools
import numpy as np
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from lexicons import Liwc, Anew, Afinn

from tsa import logging
logger = logging.getLogger(__name__)


def ngrams(documents, min_df=0.01, max_df=0.99, ngram_max=2):
    '''
    ngram features

    documents should be an iterable of strings (not tokenized)
    '''
    # ngram_range: All values of n such that min_n <= n <= max_n will be used.
    # min_df = 10   : ignore terms that occur less often than in 10 different documents
    # max_df =  0.99: ignore terms that occur in greater than 99% of document
    logger.debug('Extracting ngram features')
    # default token_pattern=u'(?u)\b\w\w+\b'
    vectorizer = CountVectorizer(token_pattern=ur'\b\S+\b',
        ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df)
    values = vectorizer.fit_transform(documents)
    return (values, vectorizer.get_feature_names())


def cooccurrences(documents, min_df=0.01, max_df=0.99):
    '''
    Co-occurrences. Right now, just pairs.

    documents should be an iterable of strings (not tokenized)
    '''
    logger.debug('Extracting co-occurrences features')

    def tokenizer(document):
        # itertools.combinations, compared to itertools.permutations, will
        # return a list of sets, rather than tuples, all 2-long
        tokens = re.findall(ur'(?u)\b\w\w+\b', document)
        for pair in itertools.combinations(tokens, 2):
            yield '-'.join(pair)
        return

    vectorizer = CountVectorizer(tokenizer=tokenizer,
        ngram_range=(1, 1), min_df=min_df, max_df=max_df)
    values = vectorizer.fit_transform(documents)
    return (values, vectorizer.get_feature_names())


def hashtags(documents, min_df=1):
    logger.debug('Extracting hashtags features')
    # sort of like:
    # counter = Counter()
    # for document in documents:
    #     hashtags = text.hashtags(document)
    #     counter.update(hashtags)
    # regex flags=re.UNICODE
    vectorizer = CountVectorizer(token_pattern=ur'#\w+', lowercase=True, min_df=min_df)
    values = vectorizer.fit_transform(documents)
    return (values, vectorizer.get_feature_names())


def liwc(documents):
    logger.debug('Extracting LIWC features')
    lexicon = Liwc()
    corpus_liwc_categories = [lexicon.read_document(document) for document in documents]
    # the identity analyzer means that each document is a list (or generator) of relevant tokens already
    #   tokenizer or analyzer could be liwc.read_document, perhaps, I'm not sure. More clear like this.
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    values = vectorizer.fit_transform(corpus_liwc_categories)
    return (values, vectorizer.get_feature_names())


def anew(documents):
    logger.debug('Extracting ANEW features')
    lexicon = Anew()

    X = np.zeros((len(documents), 3))
    feature_names = ('anew_pleasure', 'anew_arousal', 'anew_dominance')
    for row, document in enumerate(documents):
        arr = np.array(list(lexicon.read_document(document)))
        # X[row, :] = arr.sum(axis=0)
        X[row, :] = arr.mean(axis=0)

    return (X, feature_names)

def afinn(documents):
    logger.debug('Extracting AFINN features')
    lexicon = Afinn()

    X = np.zeros((len(documents), 1))
    feature_names = ('afinn', )
    for row, document in enumerate(documents):
        token_scores = np.array(list(lexicon.read_document(document)))
        # X[row, :] = token_scores.mean()
        X[row, :] = token_scores.sum()

    return (X, feature_names)
