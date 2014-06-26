# -*- coding: utf-8 -*-
import IPython
import scipy
import numpy as np
import pandas as pd
from tsa.science import numpy_ext as npx

import viz
from viz.format import quantiles
from viz.geom import hist

from sklearn import metrics, cross_validation
from sklearn import linear_model
from sklearn import naive_bayes

from tsa import stdout, stderr
from tsa.lib.itertools import sig_enumerate
from tsa.models import Source, Document, create_session
from tsa.science import features, models, timeseries
from tsa.science.corpora import MulticlassCorpus
from tsa.science.plot import plt, figure_path, distinct_styles, ticker
from tsa.science.summarization import metrics_dict, metrics_summary

from tsa import logging
logger = logging.getLogger(__name__)


def flt(x):
    return '%.2f' % x



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



def sb5_confidence(analysis_options):
    session = create_session()
    sb5b_documents = session.query(Document).join(Source).\
        filter(Source.name == 'sb5b').all()
    corpus = MulticlassCorpus(sb5b_documents)
    corpus.apply_labelfunc(lambda doc: doc.label or 'Unlabeled')
    corpus.extract_features(lambda doc: 1, features.intercept)
    corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)

    polar_classes = [corpus.class_lookup[label] for label in ['For', 'Against']]
    polar_indices = np.in1d(corpus.y, polar_classes)
    labeled_corpus = corpus.subset(polar_indices)
    # unlabeled_corpus = corpus.subset(corpus.y == corpus.class_lookup['Unlabeled'])

    penalty = 'l2'

    # we want to compare the confidence of the bootstrap on the things it
    # gets wrong vs. a straight logistic regression

    folds = cross_validation.StratifiedShuffleSplit(labeled_corpus.y, test_size=0.1, n_iter=20)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_corpus = labeled_corpus.subset(train_indices)
        test_corpus = labeled_corpus.subset(test_indices)

        logreg_model = linear_model.LogisticRegression(fit_intercept=False, penalty=penalty)
        logreg_model.fit(train_corpus.X, train_corpus.y)
        logreg_pred_y = logreg_model.predict(test_corpus.X)
        logreg_pred_proba = logreg_model.predict_proba(test_corpus.X)

        bootstrap_model = models.Bootstrap(
            linear_model.LogisticRegression, fit_intercept=False, penalty=penalty)
        bootstrap_model.fit(train_corpus.X, train_corpus.y, n_iter=200, proportion=0.5)

        bootstrap_pred_y = bootstrap_model.predict(test_corpus.X)
        bootstrap_pred_proba = bootstrap_model.predict_proba(test_corpus.X)
        # bootstrap_pred_proba.sum(axis=1) == [1, 1, 1, ...]

        # hmean penalizes extremes; hmean [0.5, 5.0] is 0.5, hmean [0.1, 0.9] is very low

        # plt.cla()
        # log reg
        print 'logreg accuracy {:.2%}'.format(
            metrics.accuracy_score(test_corpus.y, logreg_pred_y))
        print 'histogram of logreg proba hmean on misclassifications'
        logreg_proba_hmean = npx.hmean(
            logreg_pred_proba[test_corpus.y != logreg_pred_y], axis=1)
        hist(logreg_proba_hmean)
        print 'logreg max pred mean', logreg_pred_proba.max(axis=1).mean()
        # print logreg_pred_proba.mean(axis=0)
        # plt.figure(0)
        # plt.hist(logreg_pred_proba)

        # bootstrap
        print 'bootstrap accuracy {:.2%}'.format(
            metrics.accuracy_score(test_corpus.y, bootstrap_pred_y))
        print 'histogram of bootstrap proba hmean on misclassifications'
        bootstrap_proba_hmean = npx.hmean(
            bootstrap_pred_proba[test_corpus.y != bootstrap_pred_y], axis=1)
        hist(bootstrap_proba_hmean)
        print 'bootstrap max pred mean', bootstrap_pred_proba.max(axis=1).mean()
        # plt.figure(1)
        # plt.hist(bootstrap_pred_proba)

        # bootstrap_mean_coef = np.mean(bootstrap_model.coefs_, axis=0)
        # bootstrap_var_coef = np.var(bootstrap_model.coefs_, axis=0)

    IPython.embed()
    # doesn't work:
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.hist(xyz_pred_proba)



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

    for i, _ in sig_enumerate(range(100), logger=logger):
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

    biased_model = linear_model.LogisticRegression(penalty='l2', fit_intercept=False)
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

    # plt.savefig(figure_path('coefficient-scatter-%d-bootstrap.pdf' % K))

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
    plt.savefig(figure_path('cumulative-variances-%d-bootstrap.pdf' % subset.shape[0]))

    plt.cla()
    ordering = coefs_means.argsort()
    middle = ordering.size // 2
    indices = npx.edge_and_median_indices(0, 25) + range(middle - 12, middle + 13) + range(-25, 0)
    subset = cumulative_coefs_means[:, ordering[indices]]
    plt.plot(subset)
    plt.title('Coefficient means converging across a %d-iteration bootstrap\n(75 of the lowest / nearest-average / highest means)' % subset.shape[0])
    plt.savefig(figure_path('cumulative-means-%d-bootstrap.pdf' % subset.shape[0]))


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
