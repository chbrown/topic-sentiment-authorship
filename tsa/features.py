'''
Feature functions

Each feature function should take a single argument, the data, and maybe some keywords,
and return a tuple: (values (np.array), feature_names ([str]))

    N = len(data)  # or data.shape[0]
    k = len(feature_names)  # varies based on data
    values.shape = (N, k)

'''
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from lexicons import Liwc

from tsa import logging
logger = logging.getLogger(__name__)


def ngrams(documents, min_df=0.01, max_df=0.99, ngram_max=2):
    '''
    ngram features

    documents should be an iterable of strings (not tokenized)
    '''
    # min_df = 10   : ignore terms that occur less often than in 10 different documents
    # max_df =  0.99: ignore terms that occur in greater than 99% of document
    logger.debug('Extracting ngram features')
    vectorizer = CountVectorizer(token_pattern=ur'\b\S+\b',
        ngram_range=(1, ngram_max), min_df=min_df, max_df=max_df)
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
