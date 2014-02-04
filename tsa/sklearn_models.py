# -*- coding: utf-8 -*-
import IPython
import os
import time
from datetime import datetime
# from datetime import timedelta

from tsa import logging
import viz
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
import scipy
from scipy import sparse
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 10
pd.options.display.width = terminal.width()


from collections import Counter
from sklearn import cross_validation
from sklearn import metrics
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
# from sklearn import qda
from sklearn import ensemble
# from sklearn import cluster
from sklearn import decomposition
# from sklearn import neural_network
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import chi2, f_classif, f_regression
logger.silly('loaded sklearn')

import gensim

from tsa import numpy_ext as npx, features
from tsa.corpora import MulticlassCorpus
from tsa.lib import cache, tabular, itertools
from tsa.lib.timer import Timer
from tsa.scikit import metrics_dict, explore_topics
# from tsa.scikit import explore_mispredictions, explore_uncertainty

import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
plt.rcParams['interactive'] = True
plt.rcParams['axes.grid'] = True

qmargins = [0, 5, 10, 50, 90, 95, 100]

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


def styles():
    # can distinguish about six different colors from rainbox
    colors = colormap.rainbow(np.linspace(1, 0, 6))
    # and a few different linestyles
    linestyles = ['-', ':', '--', '-.']
    # and linewidth
    linewidths = [1, 2, 3]
    for linewidth in linewidths:
        for linestyle in linestyles:
            for color in colors:
                yield dict(linewidth=linewidth, linestyle=linestyle, color=color)


def binned_timeseries(times, values, time_units_per_bin=1, time_unit='D', statistic='mean'):
    first, last = npx.bounds(times)
    bins = npx.datespace(first, last, time_units_per_bin, time_unit).astype('datetime64[s]')
    # valid "statistic" strings: mean, median, count, sum
    result = scipy.stats.binned_statistic(times.astype(float), values,
        statistic=statistic, bins=bins.astype(float))
    bin_statistics, bin_edges, bin_number = result
    # the resulting statistic is 1 shorter than the bins
    # return the left edges of the bins, and the resulting statistics
    return bins[:-1], bin_statistics


def read_sb5b_MulticlassCorpus(limits=None, sort=True):
    # @cache.decorate('/tmp/tsa-corpora-sb5-tweets-min={per_label}.pickle')
    # look into caching with np.load and/or stdlib's pickle
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html
    @cache.decorate('/tmp/tsa-corpora-sb5-tweets-all.pickle')
    def cached_read():
        # cached_read is cached, so it will return a list, not an iterator, even if it looks like a generator
        import tsa.data.sb5b.tweets
        for tweet in tsa.data.sb5b.tweets.read():
            # hack: filter out invalid data here
            if isinstance(tweet['TweetTime'], datetime):
                yield tweet
            else:
                # there's just one invalid tweet - I'm pretty it's the header row
                logger.debug('Ignoring invalid tweet: %r', tweet)

    tweets = cached_read()

    # do label filtering AFTER caching
    # if limits is set, it'll be something like: dict(Against=2500, For=2500)
    # if you wanted to just select certain labels, you could use something like dict(Against=1e9, For=1e9)
    if limits is not None:
        quota = itertools.Quota(**limits)
        tweets = list(quota.filter(tweets, keyfunc=lambda tweet: tweet['Label']))

    # FWIW, sorted always returns a list
    if sort:
        tweets = sorted(tweets, key=lambda tweet: tweet['TweetTime'])

    # tweets is now what we want to limit it to
    y_raw = np.array([tweet['Label'] for tweet in tweets])
    corpus = MulticlassCorpus(y_raw)
    corpus.tweets = tweets

    times = [tweet['TweetTime'] for tweet in tweets]
    corpus.times = np.array(times).astype('datetime64[s]')

    documents = [tweet['Tweet'] for tweet in tweets]
    corpus.apply_features(documents, features.ngrams,
        ngram_max=1, min_df=0.001, max_df=0.95)
    logger.debug('MulticlassCorpus created: %s', corpus.X.shape)
    # this corpus has the following additional attributes:
    #   tweets
    #   times
    return corpus


def sample_table(feature_names, array, group_size=25):
    # Looking at the extremes
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
            row_dict[group_name + '-k'] = feature_names[indices][row]
            row_dict[group_name + '-v'] = array[indices][row]
        printer.write(row_dict)


def bootstrap_model(X, y, n_iter=100, proportion=0.5):
    # each row in coefs represents the results from a single bootstrap run
    coefs = np.zeros((n_iter, X.shape[1]))
    folds = npx.bootstrap(y.size, n_iter=n_iter, proportion=proportion)
    for fold, (train_indices, _) in itertools.sig_enumerate(folds, logger=logger):
        # repeats = sum(1 for _, count in Counter(train_indices).items() if count > 1)
        # logger.debug('%d/%d of random sample are repeats', repeats, len(train_indices))
        model = linear_model.LogisticRegression(penalty='l2', fit_intercept=False)
        model.fit(X[train_indices, :], y[train_indices])
        # IPython.embed(); raise SystemExit(91)
        coefs[fold, :] = model.coef_.ravel()
    return coefs


def confidence():
    corpus = read_sb5b_MulticlassCorpus(sort=True)
    X, y = corpus
    X = X.tocsr()

    labeled_mask = (y == corpus.labels['Against']) | (y == corpus.labels['For'])
    labeled_indices = npx.bool_mask_to_indices(labeled_mask)
    n_iter = 100
    coefs = bootstrap_model(X[labeled_indices], y[labeled_indices],
        n_iter=n_iter, proportion=0.5)
    coefs_means = np.mean(coefs, axis=0)
    # coefs_variances = np.var(coefs, axis=0)

    '''
    what a normally fitted model (but with fit_intercept = False) looks like:
    model.coef_.shape
    >>> (1, 1736)
    model.classes_
    >>> array([0, 3])
    model.intercept_
    >>> 0.0
    '''
    # model = linear_model.LogisticRegression(penalty='l2')
    # model.coef_ = coefs_means.reshape(1, -1)
    # model.intercept_ = 0.0
    # model.classes_ = np.array([0, 3])
    # model.set_params(classes_=np.array([0, 3]))
    # nope, doesn't work like that.

    test_X = X[labeled_indices, :]
    test_y = y[labeled_indices]
    classes = np.array([corpus.labels['Against'], corpus.labels['For']])

    bootstrap_transformed = test_X.dot(coefs_means)
    class_1_probabilities = npx.logistic(bootstrap_transformed)
    bootstrap_pred_probabilities = np.column_stack((
        1 - class_1_probabilities,
        class_1_probabilities))
    bootstrap_pred_y = classes[(bootstrap_transformed < 0).astype(int)]


    # okay, we have a distribution like this:
    # -15.76[                 ▁▁▁▁▁▁▂▃▅▆▇▉▇▆▄▃▂▁       ]13.81
    # and we want to compare the tails with the norm, i.e., extremes
    # like, if we exclude the middle 50%, does our accuracy increase?
    # n = transformed.size
    # middle_50_indices = np.abs(transformed).argsort()[:n/2]
    # -4.2331864[ ▁▁ ▁▁▁▁▁▁▁▁▁▁▁▂▂▃▄▅▅▆▆▇▇▉]4.24129606
    # ordered_indices = transformed.argsort()
    # middle_50_indices = ordered_indices[range(n/4, 3*n/4)]
    # 1.651[▄▃▄▄▅▅▄▅▆▅▅▆▆▆▆▇▇▆▇▇▇▇▆▉▆▇▆▆▅▅]5.5102
    quantiles = np.percentile(bootstrap_transformed, range(0, 100, 25))
    bins = np.digitize(bootstrap_transformed, quantiles)
    # npx.table(bins - 1)
    extreme_50_indices = (bins == 1) | (bins == 4)
    # hist(transformed[extreme_50_indices])
    # -15.762301[         ▁▁▁▁▂▂▂▂▄▁   ▉▇▄▂▁    ]13.8187724
    middle_50_indices = (bins == 2) | (bins == 3)
    # hist(transformed[middle_50_indices])
    # 1.6521[▃▃▄▃▅▄▅▅▅▅▆▅▇▆▇▇▆▇▆▉▆▆▇▆▆▆▅▅]5.5088
    # print (gold_y == bootstrap_pred_y).mean()
    print 'Bootstrap overall accuracy: %.4f' % metrics.accuracy_score(test_y, bootstrap_pred_y)
    print 'Accuracy over middle: %.4f' % metrics.accuracy_score(
        test_y[middle_50_indices], bootstrap_pred_y[middle_50_indices])
    print 'Accuracy over extremes: %.4f' % metrics.accuracy_score(
        test_y[extreme_50_indices], bootstrap_pred_y[extreme_50_indices])

    biased_model = linear_model.LogisticRegression(
        penalty='l2', fit_intercept=False)
    biased_model.fit(test_X, test_y)
    biased_pred_probabilities = biased_model.predict_proba(test_X)
    # pred_probabilities has as many columns as there are classes
    # and each row sums to 1
    # biased_transformed = test_X.dot(biased_model.coef_.ravel())

    # pred_y = model.predict(test_X)

    IPython.embed(); raise SystemExit(91)

    print 'bootstrap LogLoss:', metrics.log_loss(test_y, bootstrap_pred_probabilities)
    print 'biased LogLoss:', metrics.log_loss(test_y, biased_pred_probabilities)
    # probabilistic models, like LogReg, can give us log loss

    # Positive class probabilities are computed as
    # 1. / (1. + np.exp(-self.decision_function(X)));

    # def logit(x):
    #     return np.log(x / (1 - x))
    # probs = npx.logistic(biased_transformed)
    # pred_probabilities_manual = np.column_stack((1 - probs, probs))

    # linear coefficients give us a reasonable measure of sparsity
    # results['sparsity'] = np.mean(model.coef_.ravel() == 0)

    # logger.info('explore_mispredictions')
    # explore_mispredictions(test_X, test_y, model, test_indices, label_names, corpus_strings)
    # logger.info('explore_uncertainty')
    # explore_uncertainty(test_X, test_y, model)


def standard():
    corpus = read_sb5b_MulticlassCorpus(sort=True)
    X, y = corpus
    # ya get some weird things if you leave X in CSC format,
    # particularly if you index it with a boolmask
    X = X.tocsr()

    labeled_mask = (y == corpus.labels['Against']) | (y == corpus.labels['For'])
    labeled_indices = npx.bool_mask_to_indices(labeled_mask)
    n_iter = 100
    coefs = bootstrap_model(X[labeled_indices], y[labeled_indices],
        n_iter=n_iter, proportion=0.5)
    coefs_means = np.mean(coefs, axis=0)
    coefs_variances = np.var(coefs, axis=0)

    print 'coefs_means'
    hist(coefs_means)
    format.quantiles(coefs_means, qs=qmargins)
    # sample_table(coefs_means, group_size=25)

    print 'coefs_variances'
    hist(coefs_variances)
    format.quantiles(coefs_variances, qs=qmargins)
    # sample_table(coefs_variances, group_size=25)

    plt.scatter(coefs_means, coefs_variances, alpha=0.2)
    plt.title('Coefficient statistics after %d-iteration bootstrap' % n_iter)
    plt.xlabel('means')
    plt.ylabel('variances')

    # model = linear_model.RandomizedLogisticRegression()
    # model.fit(X[labeled_indices], y[labeled_indices])

    IPython.embed(); raise SystemExit(91)

    random_lr_coefs = model.coef_.ravel()

    # plt.scatter(

    # folds = cross_validation.KFold(y.size, 10, shuffle=True)
    # for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
    #     test_X, test_y = X[test_indices], y[test_indices]
    #     train_X, train_y = X[train_indices], y[train_indices]
    #     model.fit(train_X, train_y)


    # cumulative_coefs_means = npx.mean_accumulate(coefs, axis=0)
    # cumulative_coefs_variances = npx.var_accumulate(coefs, axis=0)
    # cumulative_coefs_variances.shape = (1000, 2009)

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

    # balance training set:
    # per_label = 2500
    # against_indices = npx.bool_mask_to_indices(y == label_ids['Against'])
    # against_selection = np.random.choice(against_indices, per_label, replace=False)
    # for_indices = npx.bool_mask_to_indices(y == label_ids['For'])
    # for_selection = np.random.choice(for_indices, per_label, replace=False)
    # balanced_labels_indices = np.concatenate((against_selection, for_selection))

    # extreme_features = set(np.abs(coefs_means).argsort()[::-1][:50]) | set(coefs_variances.argsort()[::-1][:50])
    # for feature in extreme_features:
    #     plt.annotate(corpus.feature_names[feature], xy=(coefs_means[feature], coefs_variances[feature]))

    # plt.savefig(fig_path('coefficient-scatter-%d-bootstrap.pdf' % K))

    # model.intercept_

    IPython.embed(); raise SystemExit(111)

    # minimum, maximum = npx.bounds(times)
    # npx.datespace(minimum, maximum, 7, 'D')

    features = np.arange(X.shape[1])
    bin_variance = np.zeros(X.shape[1])
    print 'when binned by day, what features have the greatest variance?'
    for feature in features:
        counts = X[:, feature].toarray().ravel()
        bin_edges, bin_values = binned_timeseries(times, counts, 7, 'D')
        bin_variance[feature] = np.nanvar(bin_values)

    hist(bin_variance)

    # X.sum(axis=0)
    # totals is the total number of times each word has been seen in the corpus
    totals = np.sum(X.toarray(), axis=0).ravel()
    order = totals.argsort()[::-1]

    # plt.hist(totals)
    # from bokeh.plotting import *

    plt.cla()
    plt.scatter(totals, coefs_means, alpha=0.3)
    selection = order[:10]
    print 'tops:', corpus.feature_names[selection]

    # convention: order is most extreme first
    order = np.abs(coefs_means).argsort()[::-1]
    # order = np.abs(coefs_variances).argsort()[::-1]
    # most_extreme_features = order[-10:]

    selection = order[:12]
    # selection = order[-10:]
    print 'selected features:', corpus.feature_names[selection]


    # plt.plot(a)
    # plt.plot(smooth(a, 20, .75))
    # .reshape((1,-1))
    window = 7
    alpha = .5

    def smooth_days(corpus, selection, window=7, alpha=.5, time_units_per_bin=1, time_unit='D'):
        style_iter = styles()
        plt.cla()
        axes = plt.gca()
        for feature in selection:
            # toarray() because X is sparse, ravel to make it one-dimension
            counts = corpus.X[:, feature].toarray().ravel()
            bin_edges, bin_values = binned_timeseries(corpus.times, counts, time_units_per_bin, time_unit)
            smoothed_bin_values = npx.exponential_decay(bin_values, window=window, alpha=alpha)
            style_kwargs = style_iter.next()
            # plt.plot(bin_edges, bin_values,
            #     drawstyle='steps-post', **style_kwargs)
            plt.plot(bin_edges, smoothed_bin_values, label=corpus.feature_names[feature], **style_kwargs)
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(datetime64_formatter))
        plt.legend(loc='top left')


    # , bbox_to_anchor=(1, 0.5)
    # plt.legend(bbox_to_anchor=(0, 1))
    # box = axes.get_position()
    # axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    smooth_days(corpus, selection, 7, .5, 1, 'D')
    smooth_days(corpus, selection, 1, 1, 7, 'D')
    plt.vlines(np.array(npx.bounds(corpus.times[labeled_indices])).astype(float), *plt.ylim())

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


    # Look into cross_validation.StratifiedKFold
    # data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size=0.20)

            # logger.info('Overall %s; log loss: %0.4f; sparsity: %0.4f')
            # logger.info('k=%d, proportion=%.2f; %d train, %d test, results: %s',
                # k, proportion, len(train_indices_subset),, results)

            # print 'Accuracy: %0.5f, F1: %0.5f' % (
            #     metrics.accuracy_score(test_y, pred_y),
            #     metrics.f1_score(test_y, pred_y))
            # print 'confusion:\n', metrics.confusion_matrix(test_y, pred_y)
            # print 'report:\n', metrics.classification_report(test_y, pred_y, target_names=label_names)

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

main = confidence

def perceptron():
    # corpus = read_sb5b_MulticlassCorpus(sort=False, limits=dict(For=2500, Against=2500))
    corpus = read_sb5b_MulticlassCorpus(sort=False, limits=dict(For=1e9, Against=1e9))
    X, y = corpus
    # make X sliceable:
    X = X.tocsr()
    print 'label table:', dict(npx.table(y, corpus.classes))

    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        # Percepton penalty : None, l2 or l1 or elasticnet
        # model = linear_model.Perceptron(penalty='l1')
        # LogisticRegression
        model = linear_model.LogisticRegression(penalty='l2')
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]
        model.fit(train_X, train_y)

        pred_y = model.predict(test_X)
        result = metrics_dict(test_y, pred_y)
        print 'Accuracy:', result['accuracy']
        coefs = model.coef_.ravel()
        # mid_coefs = coefs[np.abs(coefs) < 1]

        hist(coefs)
        format.quantiles(coefs, qs=qmargins)
        print 'sparsity (fraction of coef == 0)', (coefs == 0).mean()
        print

    IPython.embed(); raise SystemExit(101)



def to_gensim(array):
    # convert a csr corpus to what gensim wants: a list of list of tuples
    mat = sparse.csr_matrix(array)
    return [zip(row.indices, row.data) for row in mat]


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

    first, last = npx.bounds(tweet_times_datetime64)

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


def grid():
    corpus = read_sb5b_MulticlassCorpus(sort=False, limits=dict(For=1e9, Against=1e9))
    X, y = corpus
    # make X sliceable:
    X = X.tocsr()

    logger.debug('X.shape: %s, y.shape: %s', X.shape, y.shape)
    timer = Timer()
    printer = tabular.Printer()

    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    proportions = [0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 1.0]
    models = [
        ('logistic_regression-L1', linear_model.LogisticRegression(penalty='l1', dual=False)),
        ('logistic_regression-L2', linear_model.LogisticRegression(penalty='l2', dual=False)),
        ('logistic_regression-L2-C100', linear_model.LogisticRegression(penalty='l2', C=100.0)),
        # ('randomized_logistic_regression', linear_model.RandomizedLogisticRegression()),
        # ('sgd', linear_model.SGDClassifier()),
        ('perceptron-L1', linear_model.Perceptron(penalty='l1')),
        ('perceptron-L2', linear_model.Perceptron(penalty='l2')),
        # ('linear-svc-L2', svm.LinearSVC(penalty='l2')),
        # ('linear-svc-L1', svm.LinearSVC(penalty='l1', dual=False)),
        # ('random-forest', ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
        #     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
        #     min_density=None, compute_importances=None)),
        # ('naivebayes', naive_bayes.MultinomialNB()),
        # ('knn-10', neighbors.KNeighborsClassifier(10)),
        # ('neuralnet', neural_network.BernoulliRBM()),
        # ('qda', qda.QDA()),
        # ('knn-3', neighbors.KNeighborsClassifier(3)),
        # ('sgd-log-elasticnet', linear_model.SGDClassifier(loss='log', penalty='elasticnet')),
        # ('linear-regression', linear_model.LinearRegression(normalize=True)),
        # ('svm-svc', svm.SVC()),
        # ('adaboost-50', ensemble.AdaBoostClassifier(n_estimators=50)),
    ]

    # in KFold, if shuffle=False, we look at a sliding window for the test sets, starting at the left
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        # for each fold
        for proportion in proportions:
            # for each proportion
            size = int(train_indices.size * proportion)
            train_indices_subset = np.random.choice(train_indices, size=size, replace=False)
            for model_name, model in models:
                # for each model
                test_X, test_y = X[test_indices], y[test_indices]
                train_X, train_y = X[train_indices], y[train_indices]
                with timer:
                    # fit and predict
                    model.fit(train_X, train_y)
                    pred_y = model.predict(test_X)

                results = metrics_dict(test_y, pred_y)

                results.update(
                    fold=fold_index,
                    model=model_name,
                    proportion=proportion,
                    train=len(train_indices_subset),
                    test=len(test_indices),
                    elapsed=timer.elapsed,
                )

                printer.write(results)



if __name__ == '__main__':
    main()
