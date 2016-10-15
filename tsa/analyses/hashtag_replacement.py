import IPython
from collections import Counter

import numpy as np

from viz import terminal
from viz.geom import hist

import pandas as pd

from sklearn import cross_validation, metrics
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from tsa.lib import cache
from tsa.lib.itertools import Quota
from tsa.science.summarization import explore_mispredictions, explore_uncertainty, metrics_summary
from tsa.science.text import hashtags

import logging
logger = logging.getLogger(__name__)

# logger.critical('sys.getdefaultencoding: %s', sys.getdefaultencoding())
# logger.critical('sys.stdout.encoding: %s', sys.stdout.encoding)


def read_tweets(ttv2_file):
    '''yield TTV2 objects'''
    from twilight.lib import tweets
    for line in ttv2_file:
        yield tweets.TTV2.from_line(line)


def read_tweets_hashtagged(ttv2_file, replacement=''):
    '''yield (hashtag, text) tuples, where `text` has had `hashtag` removed from it'''
    for tweet in read_tweets(ttv2_file):
        for hashtag in hashtags(tweet.text):
            yield hashtag, tweet.text.replace(hashtag, replacement)
            # text = tweet.text.lower()
            # if '#obama' in text and '#bieber' in text:
            #     yield 'BOTH', text
            # elif '#obama' in text:
            #     yield 'OBAMA', re.sub('#?obama', '', text, flags=re.I)
            # elif '#bieber' in text:
            #     yield 'BIEBER', re.sub('#?bieber', '', text, flags=re.I)
            # else:
            #     yield 'NEITHER', text


# :p is picked up by a custom formatter which sanitizes filepaths (removes slashes)
@cache.decorate('/tmp/top_{k}_hashtags-{ttv2_filepath:p}.pickle')
def top_k_hashtags(ttv2_filepath=None, k=10):
    '''
    return k-long list of hashtags
    '''
    with open(ttv2_filepath) as ttv2_file:
        hashtags = (hashtag for hashtag, _ in read_tweets_hashtagged(ttv2_file))
        counter = Counter(hashtags)
    # this is cached, so return a list
    return [hashtag for hashtag, _ in counter.most_common(k)]


def read_hashtags_as_labels(n_hashtags, per_hashtag):
    '''
    yield (hashtag, text) pairs (at most n_hashtags*per_hashtag of them)
    '''
    ttv2_filepath = '/Users/chbrown/corpora/twitter/hashtags_bieber_obama_2013-11-0x.ttv2'
    labels = top_k_hashtags(ttv2_filepath=ttv2_filepath, k=n_hashtags)
    logger.info('top %d hashtags: %s', n_hashtags, labels)
    label_counts = dict.fromkeys(labels, per_hashtag)

    # for line in sys.stdin:
    with open(ttv2_filepath) as ttv2_file:
        # for tweet in read_tweets(ttv2_file):
        # tweet_hashtags = list(hashtags(tweet.text))
        quota = Quota(**label_counts)
        hashtag_text_pairs = read_tweets_hashtagged(ttv2_file)
        for hashtag, text in quota.filter(hashtag_text_pairs):
            yield hashtag, text


def given_labels():
    n_hashtags = 20
    per_hashtag = 1000
    data = read_hashtags_as_labels(n_hashtags, per_hashtag)
    # data is now a 10*1000-long list of (hashtag, text) tuples
    labels, texts = zip(*data)
    label_names = list(set(labels))
    # label_ids is something like {'#bieber': 0, '#obama': 1}
    label_ids = dict((label_name, label_index) for label_index, label_name in enumerate(label_names))

    y = np.array([label_ids[label] for label in labels])
    count_vectorizer = CountVectorizer(min_df=2, max_df=0.99, ngram_range=(1, 1), token_pattern=ur'\b\S+\b')
    corpus_count_vectors = count_vectorizer.fit_transform(texts)
    dimension_names = np.array(count_vectorizer.get_feature_names())
    X = corpus_count_vectors.toarray()

    logger.info('X.shape=%s, y.shape=%s', X.shape, y.shape)

    n_folds = 10
    logger.info('n_folds=%d', n_folds)
    for k, (train_indices, test_indices) in enumerate(cross_validation.KFold(y.size, n_folds, shuffle=True)):
        train_X, test_X = X[train_indices], X[test_indices]
        train_y, test_y = y[train_indices], y[test_indices]
        logger.debug('k=%d; %d train, %d test.', k, len(train_indices), len(test_indices))

        model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        pred_probabilities = model.predict_proba(test_X)

        logger.info('Overall %s; log loss: %0.4f',
            metrics_summary(test_y, pred_y), metrics.log_loss(test_y, pred_probabilities))

        # logger.info('classification_report')
        # print metrics.classification_report(test_y, pred_y, target_names=label_names)
        logger.info('explore_confusion_matrix')
        explore_confusion_matrix(test_y, pred_y, label_names)
        # logger.info('explore_labels')
        # explore_labels(label_names, dimension_names, model.coef_)
        # logger.info('explore_mispredictions')
        # explore_mispredictions(test_X, test_y, model, test_indices, label_names, texts)
        # logger.info('explore_uncertainty')
        # explore_uncertainty(test_X, test_y, model)


def explore_confusion_matrix(test_y, pred_y, label_names):
    # across the left are the true classes
    # along the top are what model classified them as
    # so for a single row, the things that are not in that row's diagonal are what are most conflated with that row's hashtag
    confusion_counts = metrics.confusion_matrix(test_y, pred_y)
    print pd.DataFrame(confusion_counts, index=label_names, columns=label_names)


def explore_labels(label_names, dimension_names, coef_):
    # coefficient analysis:
    # in the multi-class case:
    # model.coef_ has <n_labels> rows, and <n_dimensions> columns
    for label_index, label_name in enumerate(label_names):
        logger.info('label %d=%s, coefficient:', label_index, label_name)
        label_coef = coef_[label_index, :]
        hist(label_coef, range=(-2, 2))
        ranked_dimensions = label_coef.argsort()
        print 'dimensions of extreme coefficients:', dimension_names[ranked_dimensions[margins(5)]]
