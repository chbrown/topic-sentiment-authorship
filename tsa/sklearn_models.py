import IPython
import logging
from viz import terminal

# logging.basicConfig(format='%(levelname)-8s %(asctime)14s (%(name)s): %(message)s', level=17)
logger = logging.getLogger(__name__)
logger.level = 1
# logger.critical('logger init: root.level = %s, logger.level = %s', logging.root.level, logger.level)

import numpy as np
np.set_printoptions(edgeitems=25, threshold=500, linewidth=1000)
logger.debug('loaded numpy')
# import scipy
# logger.debug('loaded scipy')
import pandas as pd
pd.options.display.max_rows = 200
pd.options.display.max_columns = 10
pd.options.display.width = terminal.width()
logger.debug('loaded pandas')
import ggplot as gg
logger.debug('loaded ggplot')

# from viz.geom import hist
# logger.debug('loaded viz')

from lexicons import Liwc
logger.debug('loaded lexicons')

from sklearn import cross_validation, metrics
from sklearn import linear_model
# from sklearn import naive_bayes
# from sklearn import neighbors
# from sklearn import svm
# from sklearn import ensemble
# from sklearn import cluster
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif, f_regression
logger.debug('loaded sklearn')

from tsa.lib import cache
from tsa.scikit import explore_mispredictions, explore_uncertainty, margins

# from twilight.lib import tweets


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


def tweets_scikit():
    label_names = ['Against', 'For']
    label_ids = dict((label, index) for index, label in enumerate(label_names))

    # tweets_sorted = sorted(tweets, key=label_item_value)
    # for label, tweets in itertools.groupby(tweets_sorted, label_item_value):
    #     print label, len(list(tweets))
    # Against 10842
    # Broken Link 36
    # For 2785
    # NA 92320
    # Neutral 149
    # Not Applicable 571

    # read_tweets is cached
    data = [(tweet['Label'], tweet['Tweet']) for tweet in read_tweets(labels=label_names, per_label=2500)]
    labels, documents = zip(*data)
    logger.debug('Done reading tweets')

    # resolve named labels to integers
    y = np.array([label_ids[label] for label in labels])

    # ngram features
    count_vectorizer = CountVectorizer(min_df=2, max_df=0.99, ngram_range=(1, 2), token_pattern=ur'\b\S+\b')
    corpus_count_vectors = count_vectorizer.fit_transform(documents)
    # some models require dense arrays (count_vectorizer.transform gives sparse output normally)
    X = corpus_count_vectors.toarray()
    # get the BOW vocabulary
    dimension_names = np.array(count_vectorizer.get_feature_names())

    # liwc features
    liwc = Liwc()
    corpus_categories = [liwc.read_document(document) for document in documents]
    # the identity analyzer means that each document is a list (or generator) of relevant tokens already
    category_vectorizer = CountVectorizer(analyzer=lambda x: x)
    corpus_category_vectors = category_vectorizer.fit_transform(corpus_categories)
    logger.info('Not using liwc features')
    # X = np.hstack((X, corpus_category_vectors.toarray()))
    # prepend the liwc categories with a percent sign
    # liwc_names = ['%' + category for category in category_vectorizer.get_feature_names()]
    # dimension_names = np.hstack((dimension_names, liwc_names))

    # TF-IDF was actually performing worse, last time I compared to normal counts.
    # tfidf_transformer = TfidfTransformer()
    # corpus_tfidf_vectors = tfidf_transformer.fit_transform(corpus_count_vectors)

    logger.debug('X.shape: %s, y.shape: %s', X.shape, y.shape)
    # k_means = cluster.KMeans(n_clusters=2)
    # k_means.fit(X)

    # def tabulate_edges(ordering, edgeitems, *data):
    #     # matrix is a (N,) of floats
    #     # data is a list of (N,) columns of other output values to accompany the extreme coefficients
    #     # example call:
    #     #   tabulate_edges(model.coef_.ravel(), zip(*(dimension_names, sums)))
    #     # np.sort sorts ascending -- from most negative to most positive
    #     indices = np.argsort(ordering)
    #     edge_indices = np.concatenate((indices[:edgeitems], indices[-edgeitems:]))
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
    # edge_indices = np.concatenate((indices[:edgeitems], indices[-edgeitems:]))

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

    # IPython.embed()

    # loop over the n_folds
    n_folds = 10
    for k, (train_indices, test_indices) in enumerate(cross_validation.KFold(y.size, n_folds, shuffle=True)):
        # data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size=0.20, random_state=42)
        test_X, test_y = X[test_indices], y[test_indices]
        train_X = X[train_indices]
        train_y = y[train_indices]

        proportions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        for proportion in proportions:
            np.random.choice(




        logger.info('k=%d; %d train, %d test.', k, len(train_indices), len(test_indices))

        # dimension reduction
        # f_regression help:
        #   http://stackoverflow.com/questions/15484011/scikit-learn-feature-selection-for-regression-data
        # other nice ML variable selection help:
        #   http://www.quora.com/What-are-some-feature-selection-methods-for-SVMs
        #   http://www.quora.com/What-are-some-feature-selection-methods
        ## train_chi2_stats, train_chi2_pval = chi2(train_X, train_y)
        ## train_classif_F, train_classif_pval = f_classif(train_X, train_y)
        # train_F, train_pval = f_regression(train_X, train_y)
        # train_pval.shape = (4729,)
        # ranked_dimensions = np.argsort(train_pval)
        # ranked_names = dimension_names[np.argsort(train_pval)]
        # top_k = 100

        # train_X.shape shrinkage:: (4500, 18884) -> (4500, 100)
        # train_X = train_X[:, ranked_dimensions[:top_k]]
        # train_X.shape shrinkage: (500, 18884) -> (500, 100)
        # test_X = test_X[:, ranked_dimensions[:top_k]]

        # train_X, test_X = X[train_indices], X[test_indices]
        # train_y, test_y = y[train_indices], y[test_indices]

        # train and predict
        model = linear_model.LogisticRegression(penalty='l1')
        # model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)
        # nice L1 vs. L2 norm tutorial: http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
        # model = linear_model.SGDClassifier(loss='log', penalty='elasticnet')
        # model = linear_model.LinearRegression(normalize=True)
        # model = svm.LinearSVC(penalty='l2', dual=False, C=2.0)
        # model = svm.SVC(probability=True)
        # model = svm.LinearSVC(penalty='l2', dual=False, tol=0.0001, C=1.0)
        # model = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
        #     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
        #     min_density=None, compute_importances=None)
        # model = naive_bayes.MultinomialNB()
        # model = neighbors.KNeighborsClassifier(10)
        # linear_model.LogisticRegression()
        model.fit(train_X, train_y)
        logger.info('Model.classes_: %s', ', '.join(label_names[class_] for class_ in model.classes_))

        # print ranked_names[np.argsort(model.coef_)]
        # train_F_hmean = scipy.stats.hmean(train_F[train_F > 0])
        # print 'train_F_hmean', train_F_hmean
        # neg_train_pval_hmean = scipy.stats.hmean(1 - train_pval[train_pval > 0])
        # print '-train_pval_hmean', neg_train_pval_hmean

        # coef_sort = np.argsort(model.coef_)
        # print corpus_types[np.argsort(model.coef_)]
        print 'coef_', model.coef_
        # the mean of a list of booleans returns the percentage of trues
        sparsity = np.mean(model.coef_.ravel() == 0)
        print 'sparsity', sparsity

        # predict using the model just trained
        pred_y = model.predict(test_X)
        print 'Accuracy: %0.5f, F1: %0.5f' % (
            metrics.accuracy_score(test_y, pred_y),
            metrics.f1_score(test_y, pred_y))
        # print 'confusion:\n', metrics.confusion_matrix(test_y, pred_y)
        print 'report:\n', metrics.classification_report(test_y, pred_y, target_names=label_names)

        logger.info('explore_mispredictions')
        explore_mispredictions(test_X, test_y, model, test_indices, label_names, corpus_strings)
        logger.info('explore_uncertainty')
        explore_uncertainty(test_X, test_y, model)

        # if k == 9:
        #     print '!!! randomizing predictions'
        #     pred_y = [random.choice((0, 1)) for _ in pred_y]


if __name__ == '__main__':
    tweets_scikit()
