# -*- coding: utf-8 -*-
# universals:
import IPython
import numpy as np
from tsa.science import numpy_ext as npx

from datetime import datetime

from viz.format import quantiles
from viz.geom import hist

from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
# from sklearn.feature_selection import chi2, f_classif, f_regression

from tsa import logging
logger = logging.getLogger(__name__)

from tsa.lib import tabular, itertools
from tsa.lib.timer import Timer
from tsa.science.summarization import metrics_dict  # explore_mispredictions, explore_uncertainty

from tsa.data.sb5b.tweets import read_MulticlassCorpus as read_sb5b_MulticlassCorpus



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



def perceptron(analysis_options):
    # corpus = read_sb5b_MulticlassCorpus(sort=False, limits=dict(For=2500, Against=2500))
    corpus = read_sb5b_MulticlassCorpus()
    X, y = corpus
    X = X.tocsr()  # make X sliceable
    logger.info('label table: %r', dict(npx.table(y, corpus.classes)))

    mask = (y == corpus.labels['Against']) | (y == corpus.labels['For'])
    # npx.bool_mask_to_indices(labeled_mask)
    labeled_indices = npx.indices(y)[mask]
    X, y = X[labeled_indices], y[labeled_indices]

    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        # Percepton penalty : None, l2 or l1 or elasticnet
        model = linear_model.Perceptron(penalty='l1')
        # model = linear_model.LogisticRegression(penalty='l2')
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]
        model.fit(train_X, train_y)

        pred_y = model.predict(test_X)
        result = metrics_dict(test_y, pred_y)
        print 'Accuracy:', result['accuracy']
        coefs = model.coef_.ravel()
        # mid_coefs = coefs[np.abs(coefs) < 1]

        hist(coefs)
        quantiles(coefs, qs=qmargins)
        print 'sparsity (fraction of coef == 0)', (coefs == 0).mean()
        print



def simple(analysis_options):
    corpus = read_sb5b_MulticlassCorpus(labeled_only=True, ngram_max=2, min_df=1, max_df=1.0)

    from sklearn import naive_bayes

    for_mask = corpus.y == corpus.labels['For']
    against_mask = corpus.y == corpus.labels['Against']
    selected_indices = npx.bool_mask_to_indices(for_mask | against_mask)
    # selected_indices = npx.balance(for_mask, against_mask)
    X = corpus.X[selected_indices]
    y = corpus.y[selected_indices]

    X = X.tocsr()  # make X sliceable
    print 'X =', X.shape, 'y =', y.shape

    accuracies = []
    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        # Percepton penalty : None, l2 or l1 or elasticnet
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]

        model = linear_model.LogisticRegression(penalty='l2')
        # model = linear_model.LogisticRegression(penalty='l1')
        # model = linear_model.LogisticRegression(penalty='l2', dual=True)
        # model = naive_bayes.MultinomialNB()
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, pred_y)
        accuracies.append(accuracy)
        print '[%d] accuracy = %.4f' % (fold_index, accuracy)
    print 'mean accuracy = %.4f' % (np.mean(accuracies))


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

def grid(analysis_options):
    corpus = read_sb5b_MulticlassCorpus(sort=False, limits=dict(For=1e9, Against=1e9))
    X, y = corpus
    # make X sliceable
    X = X.tocsr()

    logger.debug('X.shape: %s, y.shape: %s', X.shape, y.shape)
    timer = Timer()
    printer = tabular.Printer()

    # from sklearn import cluster
    # from sklearn import decomposition
    # from sklearn import ensemble
    # from sklearn import linear_model
    # from sklearn import naive_bayes
    # from sklearn import neighbors
    # from sklearn import neural_network
    # from sklearn import qda
    # from sklearn import svm

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
