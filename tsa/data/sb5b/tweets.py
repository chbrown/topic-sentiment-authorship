import os
from datetime import datetime

import numpy as np
from tsa.lib import cache, html  # , itertools
from tsa.science import features
from tsa.science.corpora import MulticlassCorpus
from tsa import logging
logger = logging.getLogger(__name__)


def read_MulticlassCorpus(labeled_only=False):
    # look into caching with np.load and/or stdlib's pickle
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html
    if labeled_only:
        @cache.decorate('/tmp/tsa-corpora-sb5-tweets-labeled-only.pickle')
        def cached_read():
            for tweet in read():
                if tweet['Label'] != 'NA':
                    yield tweet
    else:
        cached_read = cache.wrap(read, '/tmp/tsa-corpora-sb5-tweets-all.pickle')

    # cached_read is cached, so it will return a list, not an iterator, even if it looks like a generator
    tweets = cached_read()

    # do label filtering AFTER caching
    # if limits is set, it'll be something like: dict(Against=2500, For=2500)
    # if you wanted to just select certain labels, you could use something like dict(Against=1e9, For=1e9)
    # if limits is not None:
    #     quota = itertools.Quota(**limits)
    #     tweets = list(quota.filter(tweets, keyfunc=lambda tweet: tweet['Label']))

    # FWIW, sorted always returns a list
    tweets = sorted(tweets, key=lambda tweet: tweet['TweetTime'])
    corpus = MulticlassCorpus(tweets)
    corpus.apply_labelfunc(lambda tweet: tweet['Label'])
    corpus.times = np.array([tweet['TweetTime'] for tweet in corpus.data]).astype('datetime64[s]')
    # this corpus has one special attribute: corpus.times
    return corpus
