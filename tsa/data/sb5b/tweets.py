import os
from datetime import datetime

import numpy as np
from tsa.lib import cache, tabular, html  # , itertools
from tsa.lib.datetime_extra import utc
from tsa.science import features
from tsa.science.corpora import MulticlassCorpus
from tsa import logging
logger = logging.getLogger(__name__)


xlsx_filepath = '%s/ohio/sb5-b.xlsx' % os.getenv('CORPORA', '.')
label_keys = ['For', 'Against', 'Neutral', 'Broken Link', 'Not Applicable']


def read():
    '''Yields dicts with at least 'Labels' and 'Tweet' fields.'''
    for row in tabular.read_xlsx(xlsx_filepath):
        # ignore invalid tweets (i.e., the header row)
        if row['Tweet'] == 'Tweet' and row['Author'] == 'Author' and row['TweetID'] == 'TweetID':
            logger.silly('Ignoring invalid tweet: %r', row)
        else:
            for label_key in label_keys:
                row[label_key] = bool(row[label_key])

            row['Labels'] = [label_key for label_key in label_keys if row[label_key]]
            row['Label'] = (row['Labels'] + ['NA'])[0]
            row['Tweet'] = html.unescape(row['Tweet'])
            # assert that all naive datetimes are actually timezone aware (and UTC)
            tweet_time = row['TweetTime']
            if isinstance(tweet_time, datetime):
                row['TweetTime'] = tweet_time.replace(tzinfo=utc)

            yield row


def read_MulticlassCorpus(labeled_only=False, ngram_max=1, min_df=0.001, max_df=0.95):
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

    # tweets is now what we want to limit it to
    y_raw = np.array([tweet['Label'] for tweet in tweets])
    corpus = MulticlassCorpus(y_raw)
    corpus.tweets = tweets

    times = [tweet['TweetTime'] for tweet in tweets]
    corpus.times = np.array(times).astype('datetime64[s]')

    corpus.documents = [tweet['Tweet'] for tweet in tweets]
    corpus.apply_features(corpus.documents, features.ngrams,
        ngram_max=ngram_max, min_df=min_df, max_df=max_df)
    # this corpus has the following additional attributes:
    #   tweets
    #   documents
    #   times
    logger.debug('MulticlassCorpus created: %s', corpus.X.shape)
    return corpus
