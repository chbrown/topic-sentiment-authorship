import IPython
import os
import time
from datetime import datetime
# from datetime import timedelta

from tsa import logging
from viz import terminal, format
from viz.geom import hist

# logging.basicConfig(format='%(levelname)-8s %(asctime)14s (%(name)s): %(message)s', level=17)
# logging.WARNING = 30
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logger.level = 10  # SILLY < 10 <= DEBUG
# logger.critical('logger init: root.level = %s, logger.level = %s', logging.root.level, logger.level)

import numpy as np
np.set_printoptions(edgeitems=25, threshold=100, linewidth=terminal.width())
# import scipy
from scipy import sparse
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 10
pd.options.display.width = terminal.width()


from lexicons import Liwc

from collections import Counter
from sklearn import cross_validation
# from sklearn import metrics
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
# from sklearn import qda
from sklearn import ensemble
# from sklearn import cluster
from sklearn import decomposition
# from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import chi2, f_classif, f_regression
logger.silly('loaded sklearn')

from tsa import numpy_ext as npx
from tsa.lib import cache, tabular, itertools
from tsa.scikit import metrics_dict, explore_topics
# from tsa.scikit import explore_mispredictions, explore_uncertainty

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
plt.rcParams['interactive'] = True
plt.rcParams['axes.grid'] = True

# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True

def fig_path(name, index=0):
    dirpath = os.path.expanduser('~/Dropbox/ut/qp/figures-qp-2')
    base, ext = os.path.splitext(name)
    filename = base + ('-%02d' % index if index > 0 else '') + ext
    filepath = os.path.join(dirpath, filename)
    if os.path.exists(filepath):
        return fig_path(name, index + 1)
    logger.info('Using filepath: %r', filepath)
    return filepath


def clear():
    plt.cla()
    plt.axes(aspect='auto')
    # plt.axis('tight')
    # plt.tight_layout()
    plt.margins(0.025, tight=True)


def links_scikit():
    # index_range = range(len(dictionary.index2token))
    # docmatrix = np.matrix([[bag_of_tfidf.get(index, 0) for index in index_range] for bag_of_tfidf in tfidf_documents])
    # for i, bag_of_tfidf in enumerate(tfidf_documents):
    #     index_scores = sorted(bag_of_tfidf.items(), key=lambda x: -x[1])
    #     doc = ['%s=%0.4f' % (dictionary.index2token[index], score) for index, score in index_scores]
    #     print i, '\t'.join(doc)

    # U, s, V = np.linalg.svd(docmatrix, full_matrices=False)
    # print docmatrix.shape, '->', U.shape, s.shape, V.shape
    maxlen = 10000

    import tsa.data.sb5b.links
    endpoints = tsa.data.sb5b.links.read(limit=1000)
    corpus_strings = (endpoint.content[:maxlen] for endpoint in endpoints)

    count_vectorizer = CountVectorizer(min_df=2, max_df=0.95)
    counts = count_vectorizer.fit_transform(corpus_strings)
    # count_vectorizer.vocabulary_ is a dict from strings to ints

    tfidf_transformer = TfidfTransformer()
    tfidf_counts = tfidf_transformer.fit_transform(counts)

    # count_vectors_index_tokens is a list (map from ints to strings)
    count_vectors_index_tokens = count_vectorizer.get_feature_names()

    # eg., 1000 documents, 2118-long vocabulary (with brown from the top)
    print count_vectors_index_tokens.shape

    pca = decomposition.PCA(2)
    doc_pca_data = pca.fit_transform(tfidf_counts.toarray())

    print doc_pca_data

    # target_vocab = ['man', 'men', 'woman', 'women', 'dog', 'cat', 'today', 'yesterday']
    # for token_type_i, coords in enumerate(vocab_pca_data):
    #     token = vector_tokens[token_type_i]
    #     if token in target_vocab:
    #         x, y = coords
    #         print "%s,%.12f,%.12f" % (token, x, y)


@cache.decorate('/tmp/sklearn_models-read_sb5b_tweets-min={per_label}.pickle')
def read_tweets(labels=None, per_label=2500):
    # ensure a For/Against 50/50 split with per_label per label
    #   the different classes will be in order, by the way
    label_groups = dict((label, []) for label in labels)
    import tsa.data.sb5b.tweets
    for tweet in tsa.data.sb5b.tweets.read():
        if tweet['Label'] in label_groups:
            label_groups[tweet['Label']].append(tweet)
            # if we have at least `minimum` in each class, we're done.
            if all(len(label_group) >= per_label for label_group in label_groups.values()):
                break
    return [tweet for label_group in label_groups.values() for tweet in label_group[:per_label]]


class ClassificationCorpus(object):
    '''
    structure to keep related corpus-specific objects all together.

    '''
    X = None
    y = None
    label_names = []
    label_ids = []
    dimension_names = []

    def __init__(self, labels=None):
        self.X = np.array([])
        if labels is not None:
            label_names = list(set(labels))
            self.set_label_names(label_names)
            self.y = np.array([self.label_ids[label] for label in labels])
        else:
            self.y = np.array([])

    def set_label_names(self, label_names):
        self.label_names = label_names
        self.label_ids = dict((label, index) for index, label in enumerate(label_names))

    def __iter__(self):
        '''Allow unpacking like:

        X, y, label_names, label_ids, dimension_names = corpus
        '''
        yield self.X
        yield self.y
        yield self.label_names
        yield self.label_ids
        yield self.dimension_names

    def add_ngram_features(self, documents, min_df=10, max_df=0.99, ngram_max=2):
        '''
        ngram features

        documents should be a list/iterable of strings (not tokenized)
        '''
        # min_df = 10   : ignore terms that occur less often than in 10 different documents
        # max_df =  0.99: ignore terms that occur in greater than 99% of document
        count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, ngram_max), token_pattern=ur'\b\S+\b')
        corpus_count_vectors = count_vectorizer.fit_transform(documents)
        # hstack doesn't work with sparse arrays, so make it dense
        #   also, some models require dense arrays -- and
        #   count_vectorizer.transform gives sparse output normally
        # only merge non-empty matrices
        self.X = sparse.hstack([X for X in [self.X, corpus_count_vectors] if X.size > 0])

        # get the BOW vocabulary (and make it an np.array so we can index into it easily)
        self.dimension_names = np.concatenate((self.dimension_names, np.array(count_vectorizer.get_feature_names())))

    def add_liwc_features(self, documents):
        liwc = Liwc()
        corpus_liwc_categories = [liwc.read_document(document) for document in documents]
        # the identity analyzer means that each document is a list (or generator) of relevant tokens already
        #   tokenizer or analyzer could be liwc.read_document, perhaps, I'm not sure. More clear like this.
        liwc_category_vectorizer = CountVectorizer(analyzer=lambda x: x)
        corpus_liwc_category_vectors = liwc_category_vectorizer.fit_transform(corpus_liwc_categories)

        self.X = sparse.hstack([X for X in [self.X, corpus_liwc_category_vectors] if X.size > 0])
        self.dimension_names = np.concatenate((self.dimension_names, np.array(liwc_category_vectorizer.get_feature_names())))

    @classmethod
    def sb5_equal(cls, label_names=['Against', 'For'], per_label=2500):
        @cache.decorate('/tmp/tsa-corpora-sb5-tweets-min={per_label}.pickle')
        def cached_read(label_names=label_names, per_label=per_label):
            label_counts = dict.fromkeys(label_names, per_label)
            quota = itertools.Quota(**label_counts)
            from tsa.data.sb5b.tweets import read
            tweets = quota.filter(read(), keyfunc=lambda tweet: tweet['Label'])
            # the sort is not really necessary, I think
            # but sorted always returns a list
            return sorted(tweets, key=lambda tweet: tweet['TweetTime'])

        corpus = cls()
        corpus.set_label_names(label_names)

        # read is cached, returns a list (not an iterator)
        tweets = cached_read(label_names=label_names, per_label=per_label)

        # resolve named labels to integers
        corpus.y = np.array([corpus.label_ids[tweet['Label']] for tweet in tweets])
        corpus.X = np.array([])

        documents = [tweet['Tweet'] for tweet in tweets]

        logger.debug('Adding ngram features')
        corpus.add_ngram_features(documents, min_df=5, max_df=0.95, ngram_max=1)

        logger.silly('Not adding liwc features')
        # corpus.add_liwc_features(documents)

        # look into caching with np.load and/or stdlib's pickle
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html

        logger.debug('ClassificationCorpus.sb5_equal: read %d tweets', len(documents))
        return corpus

    @classmethod
    def sb5_all(cls):
        @cache.decorate('/tmp/tsa-corpora-sb5-tweets-all.pickle')
        def read():
            from tsa.data.sb5b.tweets import read
            return list(read())

        # read is cached, returns a list (not an iterator)
        tweets = read()
        # hack - ignore invalid data
        tweets = [tweet for tweet in tweets if isinstance(tweet['TweetTime'], datetime)]

        # resolve named labels to integers in the __init__ function
        corpus = cls([tweet['Label'] for tweet in tweets])
        corpus.tweets = tweets

        documents = [tweet['Tweet'] for tweet in tweets]

        logger.info('Adding ngram features')
        corpus.add_ngram_features(documents)

        logger.info('Not adding liwc features')
        # corpus.add_liwc_features(documents)

        logger.debug('ClassificationCorpus.sb5_all: read %d tweets', len(documents))

        return corpus


def datetime_to_yyyymmdd(x, *args, **kw):
    return x.strftime('%Y-%m-%d')


def epoch_to_yyyymmdd(x, *args, **kw):
    return datetime_to_yyyymmdd(datetime.fromtimestamp(x))


def datetime_to_seconds(x):
    # returns None if x is not a datetime instance
    try:
        return int(time.mktime(x.timetuple()))
    except AttributeError:
        return None


def mdate_formatter(num, pos=None):
    return datetime_to_yyyymmdd(mdates.num2date(num))


'''Numpy axes:

>>> grades_by_age = np.array([
    [98, 14],
    [92, 15],
    [87, 13],
    [93, 14]])
>>> grades_by_age.mean(axis=0)
array([ 92.5,  14. ])
>>> grades_by_age.mean(axis=1)
array([ 56. ,  53.5,  50. ,  53.5])

When our rows are observations, usually axis=0 is the only thing that makes sense.
This is because each cell has much more in common with the rest of the column than the rest of the row.

axis=0: apply function to each column in turn
axis=1: apply function to each row in turn

This is also helpful: http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html
'''


def bootstrap():
    corpus = ClassificationCorpus.sb5_equal()
    X, y, label_names, label_ids, dimension_names = corpus
    # make X sliceable:
    X = X.tocsr()

    logger.debug('bootstrap(): X.shape = %s, y.shape = %s', X.shape, y.shape)
    # -> X.shape: (5000, 2009), y.shape: (5000,)

    # presumably, specifying "count" speeds things up
    # coefs = np.fromiter(bootstrap_coefs(folds), count=K)

    # hstack: Stack arrays in sequence horizontally (column wise).
    # vstack: Stack arrays in sequence vertically (row wise).

    K = 2000
    # X.shape[1] == len(dimension_names)
    # coefs = np.empty((K, X.shape[1]))
    # each row in coefs represents the results from a single bootstrap run
    coefs = np.zeros((K, X.shape[1]))

    # KFold(y.size, 10, shuffle=True)
    # cross_validation.Bootstrap(n, n_iter=3, train_size=0.5, test_size=None)
    # bootstrap samples with replacement

    # folds = cross_validation.Bootstrap(y.size, n_iter=K, train_size=0.9999)
    folds = npx.bootstrap(y.size, n_iter=K, proportion=0.1)

    for fold, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        # logger.silly('Fold %d/%d', fold + 1, K)
        train_indices_counter = Counter(train_indices)
        # repeats = sum(1 for _, count in train_indices_counter.items() if count > 1)
        # logger.debug('%d/%d of random sample are repeats', repeats, len(train_indices))
        model = linear_model.LogisticRegression(penalty='l2')
        model.fit(X[train_indices, :], y[train_indices])
        # from the docs: If the number of objects in the selection tuple is less than N ,
        #   then : is assumed for any subsequent dimensions
        # i.e., coefs[fold, :] == coefs[fold, ]
        coefs[fold, :] = model.coef_.ravel()

    qmargins = [0, 5, 10, 50, 90, 95, 100]

    print 'coefs_means'
    coefs_means = np.mean(coefs, axis=0)
    hist(coefs_means)
    format.quantiles(coefs_means, qs=qmargins)

    print 'coefs_std_deviations'
    coefs_std_deviations = np.std(coefs, axis=0)
    hist(coefs_std_deviations)
    format.quantiles(coefs_std_deviations, qs=qmargins)

    print 'coefs_variances'
    coefs_variances = np.var(coefs, axis=0)
    hist(coefs_variances)
    format.quantiles(coefs_variances, qs=qmargins)

    cumulative_coefs_means = npx.mean_accumulate(coefs, axis=0)
    cumulative_coefs_variances = npx.var_accumulate(coefs, axis=0)
    # cumulative_coefs_variances.shape = (1000, 2009)

    # Looking at the extremes
    def sample_table(array, group_size=25):
        ordering = array.argsort()
        group_names = ['smallest', 'median', 'largest']
        headers = [cell for group_name in group_names for cell in [group_name + '-v', group_name + '-k']]
        groups_indices = [
            npx.head_indices(array, group_size),
            npx.median_indices(array, group_size),
            npx.tail_indices(array, group_size)]
        printer = tabular.Printer(headers=headers, FS=' & ', RS='\\\\\n')
        printer.write_strings(printer.headers)
        for row in range(group_size):
            row_dict = dict()
            for group_name, group_indices in zip(group_names, groups_indices):
                indices = ordering[group_indices]
                row_dict[group_name + '-k'] = dimension_names[indices][row]
                row_dict[group_name + '-v'] = array[indices][row]
            printer.write(row_dict)

    print 'means'
    sample_table(coefs_means, group_size=25)

    # print '- means:'
    # print 'head:',   dimension_names[means_ordering[npx.head_indices  (means_ordering, 25)]]
    # print 'median:', dimension_names[means_ordering[npx.median_indices(means_ordering, 25)]]
    # print 'tail:',   dimension_names[means_ordering[npx.tail_indices  (means_ordering, 25)]]

    print 'stdevs'
    sample_table(coefs_std_deviations, group_size=25)

    # print '- vars:'
    # print 'smallest:',   dimension_names[variances_ordering[npx.head_indices  (variances_ordering, 25)]]
    # print 'averagest:',  dimension_names[variances_ordering[npx.median_indices(variances_ordering, 25)]]
    # print 'largest:',    dimension_names[variances_ordering[npx.tail_indices  (variances_ordering, 25)]]

    # dimension reduction
    # f_regression help:
    #   http://stackoverflow.com/questions/15484011/scikit-learn-feature-selection-for-regression-data
    # other nice ML variable selection help:
    #   http://www.quora.com/What-are-some-feature-selection-methods-for-SVMs
    #   http://www.quora.com/What-are-some-feature-selection-methods
    ## train_chi2_stats, train_chi2_pval = chi2(train_X, train_y)
    ## train_classif_F, train_classif_pval = f_classif(train_X, train_y)
    # train_F, train_pval = f_regression(X[train_indices, :], y[train_indices])
    # train_pval.shape = (4729,)
    # ranked_dimensions = np.argsort(train_pval)
    # ranked_names = dimension_names[np.argsort(train_pval)]
    plt.scatter(coefs_means, coefs_variances, alpha=0.2)
    plt.title('Coefficient statistics after %d-iteration bootstrap' % K)
    plt.xlabel('means')
    plt.ylabel('variances')
    plt.savefig(fig_path('coefficient-scatter-%d-bootstrap.pdf' % K))


    IPython.embed(); raise SystemExit(111)


    # find the dimensions of the least and most variance
    indices = npx.edge_indices(ordering, 25)
    # indices = np.random.choice(ordering.size, 50)
    subset = cumulative_coefs_variances[:, ordering[indices]]
    # subset.shape = 40 columns, K=1000 rows
    plt.plot(subset)
    plt.title('Coefficient variances converging across a %d-iteration bootstrap\n(25 highest and 25 lowest variances)' % subset.shape[0])
    plt.ylim(-0.05, 0.375)
    plt.savefig(fig_path('cumulative-variances-%d-bootstrap.pdf' % subset.shape[0]))

    plt.cla()
    ordering = coefs_means.argsort()
    middle = ordering.size // 2
    indices = npx.edge_and_median_indices(0, 25) + range(middle - 12, middle + 13) + range(-25, 0)
    subset = cumulative_coefs_means[:, ordering[indices]]
    plt.plot(subset)
    plt.title('Coefficient means converging across a %d-iteration bootstrap\n(75 of the lowest / nearest-average / highest means)' % subset.shape[0])
    plt.savefig(fig_path('cumulative-means-%d-bootstrap.pdf' % subset.shape[0]))


def to_gensim(array):
    # convert a csr corpus to what gensim wants: a list of list of tuples
    mat = sparse.csr_matrix(array)
    return [zip(row.indices, row.data) for row in mat]

import gensim

def build_topic_model(X, dimension_names, tfidf_transform=True, num_topics=5):
    if tfidf_transform:
        if sparse.issparse(X):
            X = X.toarray()
        X = npx.tfidf(X)

    corpus = to_gensim(X)

    vocab = dict(enumerate(dimension_names))
    topic_model = gensim.models.LdaModel(corpus,
        id2word=vocab, num_topics=num_topics, passes=1)
    return topic_model


def topics(num_topics=5):
    corpus = ClassificationCorpus.sb5_equal()
    X, y, label_names, label_ids, dimension_names = corpus
    # make X sliceable:
    # X = X.tocsr()
    X = X.toarray()

    logger.debug('topics(): X.shape = %s, y.shape = %s', X.shape, y.shape)
    # X.todok().items() returns a list of ((row, col), value) tuples

    build_topic_model
    for label_i, label_name in enumerate(label_names):
        sub_X = X[y == label_i, :]

        colsums = sub_X.sum(axis=0)
        # sub_dims is a list of the indices of columns that have nonzero occurrences:
        sub_dims = colsums.nonzero()[0]

        topic_model = build_topic_model(sub_X[:, sub_dims], dimension_names[sub_dims],
            tfidf_transform=True, num_topics=num_topics)

        print '---'
        print label_i, ':', label_name
        explore_topics(topic_model)

    IPython.embed(); raise SystemExit(45)


main = topics


def explore_coefs(coefs):
    # sample_cov = np.cov(coefs) # is this anything?
    coefs_cov = np.cov(coefs, rowvar=0)
    plt.imshow(coefs_cov)
    # w, v = np.linalg.eig(coefs_cov)
    u, s, v = np.linalg.svd(coefs_cov)

    # reorder least-to-biggest
    rowsums = np.sum(coefs_cov, axis=0)
    # colsums = np.sum(coefs_cov, axis=1)
    # rowsums == colsums, obviously
    ordering = np.argsort(rowsums)
    coefs_cov_reordered = coefs_cov[ordering, :][:, ordering]
    # coefs_cov_2 = coefs_cov[:, :]
    log_coefs_cov_reordered = np.log(coefs_cov_reordered)
    plt.imshow(log_coefs_cov_reordered)
    plt.imshow(log_coefs_cov_reordered[0:500, 0:500])

    coefs_corrcoef = np.corrcoef(coefs, rowvar=0)

    ordering = np.argsort(np.sum(coefs_corrcoef, axis=0))
    coefs_corrcoef_reordered = coefs_corrcoef[ordering, :][:, ordering]
    plt.imshow(coefs_corrcoef_reordered)

    # dimension_names[ordering]
    from scipy.cluster.hierarchy import linkage, dendrogram
    # Y = scipy.spatial.distance.pdist(X, 'correlation')  # not 'seuclidean'
    Z = linkage(X, 'single', 'correlation')
    dendrogram(Z, color_threshold=0)
    # sklearn.cluster.Ward


def label_proportions():
    corpus = ClassificationCorpus.sb5_all()
    X, y, label_names, label_ids, dimension_names = corpus

    logger.debug('X.shape: %s, y.shape: %s', X.shape, y.shape)

    # the COO sparse matrix format (which is what X will probably be in) is not sliceable
    # but the CSR format is, and it's still sparse and very to quick to convert from COO
    X = X.tocsr()

    # folds = cross_validation.KFold(y.size, 10, shuffle=True)
    # for fold_index, (train_indices, test_indices) in enumerate(folds):
    # test_X, test_y = X[test_indices], y[test_indices]
    # train_X, train_y = X[train_indices], y[train_indices]

    # full training set
    all_labels_mask = (y == label_ids['Against']) | (y == label_ids['For'])
    all_labels_indices = npx.bool_mask_to_indices(all_labels_mask)

    # balanced training set
    per_label = 2500
    against_indices = npx.bool_mask_to_indices(y == label_ids['Against'])
    against_selection = np.random.choice(against_indices, per_label, replace=False)
    for_indices = npx.bool_mask_to_indices(y == label_ids['For'])
    for_selection = np.random.choice(for_indices, per_label, replace=False)
    balanced_labels_indices = np.concatenate((against_selection, for_selection))
    # balanced_labels_mask = bool_mask(balanced_labels_indices, y.size)

    for train_indices, train_name in [(balanced_labels_indices, 'balanced-labels'), (all_labels_indices, 'all-labels')]:
        model = linear_model.LogisticRegression(penalty='l2')
        model.fit(X[train_indices, :], y[train_indices])

        print
        print '# model trained on "%s" (%d samples)' % (train_name, train_indices.size)
        # now predict using the model just trained

        balanced_labels_pred = model.predict(X[balanced_labels_indices, :])
        print 'predictions on balanced-labels:', npx.table(balanced_labels_pred, label_names)
        print '  against / total:', (balanced_labels_pred == label_ids['Against']).mean()

        all_labels_pred = model.predict(X[all_labels_indices, :])
        print 'predictions on all-labels:', npx.table(all_labels_pred, label_names)
        print '  against / total:', (all_labels_pred == label_ids['Against']).mean()

        all_pred = model.predict(X)
        print 'predictions on everything:', npx.table(all_pred, label_names)
        print '  against / total:', (all_pred == label_ids['Against']).mean()


    # y.size = train_mask.size = 106703

    # valid_dates = dates[dates != np.datetime64()]
    # epochs = [datetime_to_seconds(tweet['TweetTime']) for tweet in corpus.tweets]

    # np.empty(y.size)


    train_indices, train_mask = all_labels_indices, all_labels_mask
    # fig.autofmt_xdate()

    # plt.hist(dtseconds[predictions == label_ids['Against']],
    #     bins=bins, alpha=0.75, color='blue')
    # plt.hist(dtseconds[predictions == label_ids['For']],
    #     bins=bins, alpha=0.75, color='red')

    # plt.clf()
    # plt.cla()
    # plt.close()
    #

    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    # plt.subplot(nrows, ncols, plot_number)
    # np.max(tweet_times).astype('datetime64[M]')

    # ax.autofmt_xdate()
    # ax.xaxis.pad
    # quarters = np.arange(np.min(datetimes).astype('datetime64[M]'),
    #     np.max(datetimes).astype('datetime64[M]'), np.timedelta64(3, 'M'))
    # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    # [D] forces it to step at day-lengths
    # bins = np.arange(np.min(dates), np.max(dates), dtype='datetime64[D]')
    # day_bins = np.arange(np.min(tweet_times), np.max(tweet_times), np.timedelta64(1, 'D'))

    tweet_times_datetime64 = np.array([npx.datetime64(tweet['TweetTime']) for tweet in corpus.tweets])
    tweet_times = tweet_times_datetime64.astype(datetime)

    first, last = np.min(tweet_times_datetime64), np.max(tweet_times_datetime64)

    # bin by week
    bins_datetime64 = np.arange(first, last, np.timedelta64(7, 'D'))
    bins = bins_datetime64.astype(datetime)

    quarters_datetime64 = datespace(first, last, 3, 'M')
    quarters = quarters_datetime64.astype(datetime)

    # fig = plt.figure()
    # fig = plt.gcf()
    # fig.set_size_inches(8, 4)
    # plt.axes(aspect='auto')
    # plt.axis('tight')
    # plt.margins(0.025, tight=True)

    IPython.embed(); raise SystemExit(43)

    plt.close('all')
    plt.title('For/Against manual annotations')
    # fig.suptitle('Annotations', fontsize=18)
    # plt.gcf().set_size_inches(8, 4)
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(mdate_formatter))

    ax = plt.subplot(211)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mdate_formatter))
    # for: republican = red
    # against: democrat = blue
    plt.hist(mdates.date2num(tweet_times[(y == label_ids['Against']) & train_mask]),
        bins=mdates.date2num(bins), color='blue', label='Against')
    plt.hist(mdates.date2num(tweet_times[(y == label_ids['For']) & train_mask]),
        bins=mdates.date2num(bins), color='red', label='For')
    plt.xticks(quarters)
    plt.legend()

    # plt.tight_layout()

    # plt.close('all')
    plt.title('For/Against predictions on all data')

    ax = plt.subplot(212)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mdate_formatter))
    plt.hist(mdates.date2num(tweet_times[pred == label_ids['Against']]),
        bins=mdates.date2num(bins), color='blue', label='Against')
    plt.hist(mdates.date2num(tweet_times[pred == label_ids['For']]),
        bins=mdates.date2num(bins), color='red', label='For')
    plt.xticks(quarters)
    plt.legend()

    # plt.savefig('plot-all.pdf')

    IPython.embed(); raise SystemExit(44)

    # plt.hist(dtseconds[(pred == label_ids['Against']) & ~train_mask],
    #     bins=bins, alpha=0.75, color='blue')
    # plt.hist(dtseconds[(pred == label_ids['For']) & ~train_mask],
    #     bins=bins, alpha=0.75, color='red')


def explore_texts(corpus):
    texts = [tweet['Tweet'].lower() for tweet in corpus.tweets]
    containing = lambda xs, y: [x for x in xs if y in x]
    ncontaining = lambda xs, y: sum(1 for x in xs if y in x)

    # lambda
    hashtag_sb5_texts = ncontaining(texts, '#sb5')
    # hashtag_issue2_texts = [text for text in texts if ]
    either_texts = [text for text in texts if 'issue2' in text or 'sb5' in text]
    neither_texts = [text for text in texts if 'issue2' not in text and 'sb5' not in text]

    keywords = ['issue2', 'sb5', 'weareohio']

    def find(texts, keywords):
        for text in texts:
            if any(keyword in text for keyword in keywords):
                yield text

    def skip(texts, keywords):
        for text in texts:
            if not any(keyword in text for keyword in keywords):
                yield text

    # print len(list(find(texts, ['issue2', 'sb5', 'weareohio'])))
    for text in skip(texts, keywords):
        print text

    # any_texts = map(lambda text: any([(keyword in text) ]), texts)
     # [text for text in texts if ]


def tweets_scikit():
    corpus = ClassificationCorpus.sb5()
    X, y, label_names, label_ids, dimension_names = corpus

    logger.debug('X.shape: %s, y.shape: %s', X.shape, y.shape)

    # prepend the liwc categories with a percent sign
    # liwc_names = ['%' + category for category in category_vectorizer.get_feature_names()]
    # dimension_names = np.hstack((dimension_names, liwc_names))

    # TF-IDF was actually performing worse, last time I compared to normal counts.
    # tfidf_transformer = TfidfTransformer()
    # corpus_tfidf_vectors = tfidf_transformer.fit_transform(corpus_count_vectors)

    # k_means = cluster.KMeans(n_clusters=2)
    # k_means.fit(X)

    # for label, tweets in itertools.groupby(tweets_sorted, label_item_value):
    #     print label, len(list(tweets))
    # viz.geom.hist([time.mktime(tweet['TweetTime'].timetuple()) for tweet in tweets])

    # y_X_iterator = ((tweet['Label'], tweet) for for tweet in tsa.data.sb5b.tweets.read())

    # def tabulate_edges(ordering, edgeitems, *data):
    #     # matrix is a (N,) of floats
    #     # data is a list of (N,) columns of other output values to accompany the extreme coefficients
    #     # example call:
    #     #   tabulate_edges(model.coef_.ravel(), zip(*(dimension_names, sums)))
    #     # np.sort sorts ascending -- from most negative to most positive
    #     indices = np.argsort(ordering)
    #     for coefficient, row in zip(*(ordering[edge_indices], data[edge_indices, :])):
    #         # print 'row', row.shape, list(row), [item for item in row]
    #         print ','.join([str(coefficient)] + list(row))
    #         yield coefficient,
    # ravel just reduces the 1-by-M matrix to a M-vector
    # print 'tabulating edges:'
    # dimension_sums is the total number of each token that occurs in the corpus
    # (btw: dimensions == columns)
    # dimension_sums = X.sum(axis=0)
    # Note: np.column_stack(...) == np.vstack(...).T
    # summarized_data = np.column_stack((dimension_names, dimension_sums))
    # tabulate_edges(model.coef_.ravel(), 25, dimension_names, dimension_sums)
    # model_coefficients = model.coef_.ravel()
    # indices = np.argsort(model_coefficients)

    # model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)
    # model.fit(X, y)
    # model_df = pd.DataFrame({
    #     # ravel flattens, but doesn't copy
    #     'coef': model.coef_.ravel(),
    #     'label': dimension_names,
    #     'count': X.sum(axis=0),
    # })

    # edges = model_df.sort('coef').iloc[margins(30)]
    # print 'edges.coef', repr(edges.coef)
    # x_ticks = np.arange(edges.coef.min(), edges.coef.max(), dtype=int)
    # print gg.ggplot(gg.aes(x='coef', y='count'), data=edges) + \
    #     gg.geom_bar(stat='identity', width=0.1) + \
    #     gg.scale_x_continuous('Coefficients', labels=list(x_ticks), breaks=list(x_ticks))

    # gg.geom_point(weight='coef')
    # print edges.describe()
    printer = tabular.Printer()

    # def evaluate(model, train_indices, test_indices):
    def run(sklearn_model, train_indices, test_indices, **results):
        started = time.time()
        # requires globals: label_names, X, y
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]
        sklearn_model.fit(train_X, train_y)
        # predict using the model just trained
        pred_y = sklearn_model.predict(test_X)
        results.update(**metrics_dict(test_y, pred_y))
        # probabilistic models can give us log loss
        # pred_probabilities = model.predict_proba(test_X)
        # results['log_loss'] = metrics.log_loss(test_y, pred_probabilities)
        # linear coefficients give us a reasonable measure of sparsity
        # results['sparsity'] = np.mean(model.coef_.ravel() == 0)
        results['elapsed'] = (time.time() - started)
        printer.write(results)

    # loop over the n_folds
    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    proportions = [0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 1.0]
    models = [
        ('logistic_regression-L1', linear_model.LogisticRegression(penalty='l1', dual=False)),
        ('logistic_regression-L2', linear_model.LogisticRegression(penalty='l2', dual=False)),
        ('logistic_regression-L2-C100', linear_model.LogisticRegression(penalty='l2', C=100.0)),
        ('sgd', linear_model.SGDClassifier()),
        ('linear-svc-L2', svm.LinearSVC(penalty='l2')),
        ('linear-svc-L1', svm.LinearSVC(penalty='l1', dual=False)),
        ('random-forest', ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
            max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
            bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
            min_density=None, compute_importances=None)),
        ('naivebayes', naive_bayes.MultinomialNB()),
        ('knn-10', neighbors.KNeighborsClassifier(10)),
        # ('neuralnet', neural_network.BernoulliRBM()),
        # ('qda', qda.QDA()),
        # ('knn-3', neighbors.KNeighborsClassifier(3)),
        # ('sgd-log-elasticnet', linear_model.SGDClassifier(loss='log', penalty='elasticnet')),
        # ('linear-regression', linear_model.LinearRegression(normalize=True)),
        # ('svm-svc', svm.SVC()),
        # ('adaboost-50', ensemble.AdaBoostClassifier(n_estimators=50)),
    ]

    logging.root.level = 40
    # logger.level = 40
    # Look into cross_validation.StratifiedKFold
    # data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size=0.20)
    # in KFold, if shuffle=False, we look at a sliding window for the test sets, starting at the left
    for fold_index, (train_indices, test_indices) in enumerate(folds):

        model = linear_model.LogisticRegression(penalty='l2')
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]
        model.fit(train_X, train_y)

        coefs = model.coef_.ravel()
        expcoefs = np.exp(coefs)

        IPython.embed()

        for proportion in proportions:
            size = int(train_indices.size * proportion)
            train_indices_subset = np.random.choice(train_indices, size=size, replace=False)
            for model, sklearn_model in models:
                results = dict(fold=fold_index, model=model, proportion=proportion,
                    train=len(train_indices_subset), test=len(test_indices))
                run(sklearn_model, train_indices_subset, test_indices, **results)

            # logger.info('Overall %s; log loss: %0.4f; sparsity: %0.4f')
            # logger.info('k=%d, proportion=%.2f; %d train, %d test, results: %s',
                # k, proportion, len(train_indices_subset),, results)

            # print 'Accuracy: %0.5f, F1: %0.5f' % (
            #     metrics.accuracy_score(test_y, pred_y),
            #     metrics.f1_score(test_y, pred_y))
            # print 'confusion:\n', metrics.confusion_matrix(test_y, pred_y)
            # print 'report:\n', metrics.classification_report(test_y, pred_y, target_names=label_names)

            # logger.info('explore_mispredictions')
            # explore_mispredictions(test_X, test_y, model, test_indices, label_names, corpus_strings)
            # logger.info('explore_uncertainty')
            # explore_uncertainty(test_X, test_y, model)

        # train_F_hmean = scipy.stats.hmean(train_F[train_F > 0])
        # print 'train_F_hmean', train_F_hmean
        # neg_train_pval_hmean = scipy.stats.hmean(1 - train_pval[train_pval > 0])
        # print '-train_pval_hmean', neg_train_pval_hmean

        # print corpus_types[np.argsort(model.coef_)]
        # the mean of a list of booleans returns the percentage of trues
        # logger.info('Sparsity: {sparsity:.2%}'.format(sparsity=sparsity))

        # train_X.shape shrinkage:: (4500, 18884) -> (4500, 100)
        # train_X = train_X[:, ranked_dimensions[:top_k]]
        # train_X.shape shrinkage: (500, 18884) -> (500, 100)
        # test_X = test_X[:, ranked_dimensions[:top_k]]

        # train_X, test_X = X[train_indices], X[test_indices]
        # train_y, test_y = y[train_indices], y[test_indices]

        # nice L1 vs. L2 norm tutorial: http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html

        # if k == 9:
        #     print '!!! randomizing predictions'
        #     pred_y = [random.choice((0, 1)) for _ in pred_y]


if __name__ == '__main__':
    main()
