# -*- coding: utf-8 -*-
# universals:
import IPython
import numpy as np
import pandas as pd
from tsa.science import numpy_ext as npx

import viz
from viz.format import quantiles
from viz.geom import hist

# from collections import Counter
from sklearn import metrics
from sklearn import linear_model


from tsa.data.sb5b.tweets import read_MulticlassCorpus as read_sb5b_MulticlassCorpus
from tsa.lib import itertools
from tsa.science import timeseries
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.science.plot import plt
# import matplotlib.pyplot as plt

logger.info('%s finished with imports', __file__)


def explore_coefs(X, coefs):
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


def errors(analysis_options):
    corpus = read_sb5b_MulticlassCorpus()
    print '--loaded corpus--'
    # labeled_mask = (corpus.y == corpus.labels['For']) | (corpus.y == corpus.labels['Against'])

    # balance training set:
    # per_label = min(Counter(corpus.y[labeled_mask]).values())
    for_indices = corpus.indices[corpus.y == corpus.labels['For']]
    against_indices = corpus.indices[corpus.y == corpus.labels['Against']]
    per_label = min(against_indices.size, for_indices.size)
    for_selection = np.random.choice(for_indices, per_label, replace=False)
    against_selection = np.random.choice(against_indices, per_label, replace=False)
    balanced_indices = np.concatenate((for_selection, against_selection))

    X, y = corpus.X[balanced_indices, :], corpus.y[balanced_indices]
    tweets = [corpus.tweets[index] for index in balanced_indices]
    # don't use corpus past this point

    n_iter = 100
    bootstrap_coefs = bootstrap_model(X, y, n_iter=n_iter, proportion=0.5)
    coefs_means = np.mean(bootstrap_coefs, axis=0)
    coefs_variances = np.var(bootstrap_coefs, axis=0)
    classes = np.array([corpus.labels['For'], corpus.labels['Against']])

    # coefs_adjusted = coefs_means / (coefs_variances + 1)**2
    coefs_adjusted = coefs_means / (coefs_variances + 1)
    bootstrap_transformed = X.dot(coefs_adjusted)
    # bootstrap_transformed = X.dot(coefs_means)
    # order = np.argsort(bootstrap_transformed)
    class_1_probabilities = npx.logistic(bootstrap_transformed)
    bootstrap_pred_probabilities = np.column_stack((
        1 - class_1_probabilities,
        class_1_probabilities))
    bootstrap_pred_y = classes[np.argmax(bootstrap_pred_probabilities, axis=1)]

    print 'Bootstrap overall accuracy: %.4f' % metrics.accuracy_score(y, bootstrap_pred_y)

    binned_accuracy(bootstrap_transformed, y, bootstrap_pred_y)


    errors_mask = bootstrap_pred_y != y
    logger.info('The model mis-predicted %d out of a total of %d', errors_mask.sum(), y.size)

    bounds = npx.bounds(bootstrap_transformed)
    print 'Transforms of mispredictions'
    hist(bootstrap_transformed[errors_mask], bounds)
    # quantiles(bootstrap_transformed[errors_mask])
    print 'Transforms of correct predictions'
    hist(bootstrap_transformed[~errors_mask], bounds)
    # quantiles(bootstrap_transformed[~errors_mask])

    IPython.embed() or exit()

    errors_indices = corpus.indices[errors_mask]
    # pred_pairs = np.column_stack((bootstrap_pred_y, y))
    # Counter(zip(bootstrap_pred_y, y))

    confusion_matrix = metrics.confusion_matrix(y, bootstrap_pred_y, range(len(corpus.classes)))
    rownames = ['(Correct) ' + name for name in corpus.classes]
    print pd.DataFrame(confusion_matrix, index=rownames, columns=corpus.classes)

    '''
    Negative, in this case, means "For"
    i.e., bootstrap_transformed[84] = -16.36, which shows the following SUPER "For"-SB5 tweet:
        RT @GOHPBlog: ICYMI: Say 'YES' to Ohio jobs. Say 'YES' to #Issue2 RT @BetterOhio: What Issue 2 means for Ohio Jobs: http://t.co/HJ8sL4l8 - #YesOn2
    positive, means "Against", here's bootstrap_transformed[2905] = 10.62:
        RT @ProgressOhio: Ohio Issue 2: 30 TV Stations Pull Misleading Anti-Union Ad [UPDATE] http://t.co/BUxxH3yz #p2 #1U #SB5 #Issue2 #WeAreOhio #StandUpOh #NoOn2
    '''


    def flt(x):
        return '%.2f' % x

    # randomization shouldn't hurt
    selected_indices = corpus.indices[errors_mask]
    np.random.shuffle(selected_indices)
    for error_index in selected_indices[:50]:
        print
        print 'Predicted %r, should be %r' % (
            corpus.classes[bootstrap_pred_y[error_index]],
            corpus.classes[y[error_index]])
        print tweets[error_index]['Tweet']


        x = X[error_index].toarray().ravel()
        # could also just use the .indices of a CSR matrix
        nonzero = x > 0
        # total = coefs_means.dot(x)  # = sum(values)
        x_names = corpus.feature_names[nonzero]
        x_means = x[nonzero] * coefs_means[nonzero]
        x_variances = x[nonzero] * coefs_variances[nonzero]
        reordering = np.argsort(x_means)
        # [('', '='), ('total', ])
        print viz.gloss.gloss(
            [('', 'means  ', 'vars')] + zip(
                x_names[reordering],
                map(flt, x_means[reordering]),
                map(flt, x_variances[reordering]),
            ) +
            [('', '%.2f' % sum(x_means), '%.2f' % sum(x_variances))]
        )

    # what is it that differentiates the hard-to-classify tweets near the middle vs.
    # the tweets near the middle that it gets right?

    # biased_model = linear_model.LogisticRegression(fit_intercept=False)
    # biased_model.fit(X, y)
    # biased_pred_y = biased_model.predict(X)
    # print 'Biased overall accuracy: %.4f' % metrics.accuracy_score(y, biased_pred_y)


def logreg_accuracy(X, y, train_indices):
    if len(set(y[train_indices])) == 1:
        logger.debug('skipping test since only one label is available')
        return 0

    # penalty='l1', fit_intercept=False, C=0.01
    model = linear_model.LogisticRegression(penalty='l2')
    model.fit(X[train_indices], y[train_indices])
    pred_y = model.predict(X)
    return metrics.accuracy_score(y, pred_y)


def oracle(analysis_options):
    # so incestuous
    corpus = read_sb5b_MulticlassCorpus(labeled_only=True)

    # if limits is not None:
    #     quota = itertools.Quota(**limits)
    #     tweets = list(quota.filter(tweets, keyfunc=lambda tweet: tweet['Label']))
    # labeled_mask = (corpus.y == corpus.labels['Against']) | (corpus.y == corpus.labels['For'])
    # labeled_indices = corpus.indices[labeled_mask]

    per_label = 500
    for_indices = np.random.choice(corpus.indices[corpus.y == corpus.labels['For']], per_label, replace=False)
    against_indices = np.random.choice(corpus.indices[corpus.y == corpus.labels['Against']], per_label, replace=False)
    selected_indices = np.concatenate((for_indices, against_indices))

    # shake it up!
    np.random.shuffle(selected_indices)

    X = corpus.X.toarray()[selected_indices]
    y = corpus.y[selected_indices]


    all_indices = npx.indices(y)
    # used_mask initialized to all False
    used_mask = all_indices == -1


    # def add_3(x):
        # return x + 3
    # np.fromfunction(

    for i, _ in itertools.sig_enumerate(range(100), logger=logger):
        print 'Iteration #%d' % i
        # for each i in the top 100 training examples
        # split indices into:
        #   1) the ones we've picked
        #   2) the others
        # best_indices = np.array(best_data)
        # candidate_mask = ~used_mask
        # for each candidate index
        current_indices = all_indices[used_mask]

        def accuracy_with_index(index):
            # uses current_indices from outside scope
            # add the given to the already used indices and test it out
            candidate_indices = np.append(current_indices, index)
            accuracy = logreg_accuracy(X, y, candidate_indices)
            print candidate_indices, '=', accuracy
            return accuracy

        candidate_indices = all_indices[~used_mask]
        print 'testing out %d indices' % candidate_indices.size
        accuracies = map(accuracy_with_index, candidate_indices)

        df = pd.DataFrame({'added_index': candidate_indices, 'accuracy': accuracies})
        # df = pd.DataFrame(index=candidate_indices, data={'accuracy': accuracies})
        margins = range(0, 10) + range(len(df) - 10, len(df))
        # pd.options.display.max_rows = 20
        # pd.options.display.show_dimensions = False
        print df.ix[margins]
        # print df.tail()
        # table = np.array((candidate_indices, np.array(accuracies)), dtype=[('added_index', int), ('accuracy', float)])
        # print np.column_stack((candidate_indices, accuracies))

        best_index = candidate_indices[np.argmax(accuracies)]
        used_mask[best_index] = True

        # penalty='l1', fit_intercept=False, C=0.01

        if i > 10:
            # IPython.embed()
            used_indices = all_indices[used_mask]
            top_tweets = np.array(corpus.tweets)[selected_indices][used_indices]
            for tweet in top_tweets:
                print tweet['Label'], tweet['Tweet']




def binned_accuracy(values, true_y, pred_y, n_bins=10):
    # values should be the real-valued products of features and coefficients
    bounds = npx.bounds(values)
    # this argsort / order could be parameterized (along with the message)
    order = np.argsort(values)
    print 'Bin from lowest (0) to highest (%d)' % (n_bins - 1)
    bins = n_bins * npx.indices(order) / order.size
    for bin_i in range(n_bins):
        indices = order[bins == bin_i]
        hist(values[indices], bounds)
        mean = values[indices].mean()
        print 'Accuracy over bin %d (N=%d, mean=%0.3f): %.4f' % (
            bin_i, indices.size, mean,
            metrics.accuracy_score(true_y[indices], pred_y[indices]))


def confidence(analysis_options):
    corpus = read_sb5b_MulticlassCorpus()
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
    classes = np.array([corpus.labels['For'], corpus.labels['Against']])

    bootstrap_transformed = test_X.dot(coefs_means)
    class_1_probabilities = npx.logistic(bootstrap_transformed)
    bootstrap_pred_probabilities = np.column_stack((
        1 - class_1_probabilities,
        class_1_probabilities))
    # bootstrap_pred_y_old = classes[(bootstrap_transformed > 0).astype(int)]
    # we find the prediction by lining up the classes and picking one of them
    # according to which column is the max
    bootstrap_pred_y = classes[np.argmax(bootstrap_pred_probabilities, axis=1)]


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
    # percentiles = np.percentile(bootstrap_transformed, range(0, 100, 25))
    # bins = np.digitize(bootstrap_transformed, percentiles)
    # set(bins) == {1, 2, 3, 4}
    # npx.table(bins - 1)
    # extreme_50_indices = (bins == 1) | (bins == 4)
    # hist(transformed[extreme_50_indices])
    # -15.762301[         ▁▁▁▁▂▂▂▂▄▁   ▉▇▄▂▁    ]13.8187724
    # middle_50_indices = (bins == 2) | (bins == 3)
    # hist(transformed[middle_50_indices])
    # 1.6521[▃▃▄▃▅▄▅▅▅▅▆▅▇▆▇▇▆▇▆▉▆▆▇▆▆▆▅▅]5.5088
    # print (gold_y == bootstrap_pred_y).mean()
    # print 'Bootstrap overall accuracy: %.4f' % metrics.accuracy_score(test_y, bootstrap_pred_y)
    # print 'Accuracy over middle: %.4f' % metrics.accuracy_score(
    #     test_y[middle_50_indices], bootstrap_pred_y[middle_50_indices])
    # print 'Accuracy over extremes: %.4f' % metrics.accuracy_score(
    #     test_y[extreme_50_indices], bootstrap_pred_y[extreme_50_indices])

    biased_model = linear_model.LogisticRegression(
        penalty='l2', fit_intercept=False)
    biased_model.fit(test_X, test_y)
    biased_pred_probabilities = biased_model.predict_proba(test_X)
    # biased_pred_y = biased_model.predict(test_X)
    # pred_probabilities has as many columns as there are classes
    # and each row sums to 1
    biased_pred_y = classes[np.argmax(biased_pred_probabilities, axis=1)]

    biased_transformed = test_X.dot(biased_model.coef_.ravel())
    hist(biased_transformed)

    # we want to get at more than just a 50/50 extreme/middle split
    # order goes from most extreme to least

    # percentiles = np.percentile(biased_transformed, range(0, 100, 25))
    # bins = np.digitize(biased_transformed, percentiles)
    # extreme_50_indices = (bins == 1) | (bins == 4)
    # middle_50_indices = (bins == 2) | (bins == 3)
    print 'Biased overall accuracy: %.4f' % metrics.accuracy_score(test_y, biased_pred_y)
    # np.linspace(0, 10, )
    # for i in range:
    IPython.embed(); raise SystemExit(91)



    bounds = npx.bounds(biased_transformed)

    # print 'Bin from most extreme (0) to least extreme (%d)' % (nbins - 1)
    # order = np.argsort(-np.abs(biased_transformed))

    nbins = 10
    print 'Bin from lowest (0) to highest (%d)' % (nbins - 1)
    order = np.argsort(biased_transformed)
    bins = nbins * npx.indices(order) / order.size
    for bin_i in set(bins):
        indices = order[bins == bin_i]
        hist(biased_transformed[indices], bounds)
        transform_mean = biased_transformed[indices].mean()
        print 'Accuracy over bin %d (N=%d, mean=%0.3f): %.4f' % (
            bin_i, indices.size, transform_mean,
            metrics.accuracy_score(test_y[indices], biased_pred_y[indices]))



    percentiles = np.percentile(biased_transformed, range(0, 100, 25))
    bins = np.digitize(biased_transformed, percentiles)

    for bin_i in range(1, 5):
        indices = npx.indices(biased_transformed)[bins == bin_i]
        hist(biased_transformed[indices], bounds)
        transform_mean = biased_transformed[indices].mean()
        print 'Accuracy over bin %d (N=%d, mean=%0.3f): %.4f' % (
            bin_i, indices.size, transform_mean,
            metrics.accuracy_score(test_y[indices], biased_pred_y[indices]))

    extreme_50_indices = (bins == 1) | (bins == 4)
    middle_50_indices = (bins == 2) | (bins == 3)
    print 'Accuracy over middle: %.4f' % metrics.accuracy_score(
        test_y[middle_50_indices], biased_pred_y[middle_50_indices])
    print 'Accuracy over extremes: %.4f' % metrics.accuracy_score(
        test_y[extreme_50_indices], biased_pred_y[extreme_50_indices])





    # hist(transformed[middle_50_indices])
    # 1.6521[▃▃▄▃▅▄▅▅▅▅▆▅▇▆▇▇▆▇▆▉▆▆▇▆▆▆▅▅]5.5088
    # print (gold_y == bootstrap_pred_y).mean()
    print 'Bootstrap overall accuracy: %.4f' % metrics.accuracy_score(test_y, bootstrap_pred_y)
    print 'Accuracy over middle: %.4f' % metrics.accuracy_score(
        test_y[middle_50_indices], bootstrap_pred_y[middle_50_indices])
    print 'Accuracy over extremes: %.4f' % metrics.accuracy_score(
        test_y[extreme_50_indices], bootstrap_pred_y[extreme_50_indices])



    # bokeh quickstart
    # import bokeh.plotting as bp
    # bp.output_server('tsa')
    # # bp.output_file('boring.html')
    # x = np.linspace(-2*np.pi, 2*np.pi, 100)
    # y = np.cos(x)
    # bp.scatter(x, y, marker="square", color="blue")
    # bp.show()


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


def standard(analysis_options):
    corpus = read_sb5b_MulticlassCorpus()
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
    quantiles(coefs_means, qs=qmargins)
    # sample_table(coefs_means, group_size=25)

    print 'coefs_variances'
    hist(coefs_variances)
    quantiles(coefs_variances, qs=qmargins)
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
        bin_edges, bin_values = timeseries.binned_timeseries(times, counts, 7, 'D')
        bin_variance[feature] = np.nanvar(bin_values)

    hist(bin_variance)

    # X.sum(axis=0)
    # totals is the total number of times each word has been seen in the corpus
    totals = np.sum(X.toarray(), axis=0).ravel()
    order = totals.argsort()[::-1]

    # plt.hist(totals)

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
            bin_edges, bin_values = timeseries.binned_timeseries(
                corpus.times, counts, time_units_per_bin, time_unit)
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
