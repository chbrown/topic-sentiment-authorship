import re
from collections import defaultdict

from viz import histogram, terminal
from viz.histogram import hist

from lexicons import liwc

import numpy as np
np.set_printoptions(edgeitems=20, threshold=500, linewidth=terminal.width())
import scipy

from sklearn import cross_validation, metrics
from sklearn import linear_model, naive_bayes, neighbors, svm, ensemble, cluster, decomposition
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif, f_regression

from tsa.data.sb5b import links as sb5b_links, tweets as sb5b_tweets
from tsa.lib.cache import pickleable

import IPython

# import random
import logging
logger = logging.getLogger(__name__)


def unique(document):
    # TODO: unique doesn't really belong here, but doesn't quite merit its own module
    seen = {}
    features = []
    for token in document:
        features.append(['UNIQUE'] if token not in seen else [])
        seen[token] = 1
    return features


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

    endpoints = sb5b_links.read(limit=1000)
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


def tweets_scikit():
    @pickleable('read_tweets-min=%(minimum)d.pickle')
    def read_tweets(labels=None, minimum=2500):
        label_groups = dict((label, []) for label in labels)
        for tweet in sb5b_tweets.read_cached():
            if tweet['Label'] in label_groups:
                label_groups[tweet['Label']].append(tweet)

                # if we have at least `minimum` in each class, we're done.
                if all(len(label_group) >= minimum for label_group in label_groups.values()):
                    break

        # ensure a For/Against 50/50 split with [:minimum] slice
        #   the different classes will be in order, by the way
        return [tweet for label_group in label_groups.values() for tweet in label_group[:minimum]]

    labels = ['Against', 'For']
    label_ids = dict((label, index) for index, label in enumerate(labels))
    tweets = read_tweets(labels=labels, minimum=2500)

    print 'Done reading tweets'

    # tweets_sorted = sorted(tweets, key=label_item_value)
    # for label, tweets in itertools.groupby(tweets_sorted, label_item_value):
    #     print label, len(list(tweets))
    # Against 10842
    # Broken Link 36
    # For 2785
    # NA 92320
    # Neutral 149
    # Not Applicable 571

    N = len(tweets)
    K_folds = 10

    # calculate labels, y
    corpus_labels = [tweet['Label'] for tweet in tweets]
    y = np.array([label_ids[corpus_label] for corpus_label in corpus_labels])

    # calculate data, X
    corpus_strings = [tweet['Tweet'] for tweet in tweets]
    count_vectorizer = CountVectorizer(min_df=2, max_df=0.99, ngram_range=(1, 2),
        token_pattern=ur'\b\S+\b')
    corpus_count_vectors = count_vectorizer.fit_transform(corpus_strings)
    # liwc analysis:
    corpus_categories = [liwc.text_categories(document) for document in corpus_strings]
    category_vectorizer = CountVectorizer(analyzer=lambda x: x)
    corpus_category_vectors = category_vectorizer.fit_transform(corpus_categories)
    # get the BOW vocabulary

    # TF-IDF was actually performing worse, last time I compared to normal counts.
    # tfidf_transformer = TfidfTransformer()
    # corpus_tfidf_vectors = tfidf_transformer.fit_transform(corpus_count_vectors)

    # some models require dense arrays (count_vectorizer.transform gives sparse output normally)
    X = np.hstack((corpus_count_vectors.toarray(), corpus_category_vectors.toarray()))
    # X = scipy.sparse.hstack((corpus_count_vectors, corpus_category_vectors)).toarray()
    print 'X.shape', X.shape

    dimension_names = np.array(
        count_vectorizer.get_feature_names() +
        # prepend the liwc categories with a percent sign
        ['%' + category for category in category_vectorizer.get_feature_names()])
    # categories = np.array(count_vectorizer.get_feature_names())

    # k_means = cluster.KMeans(n_clusters=2)
    # k_means.fit(X)

    for k, (train_indices, test_indices) in enumerate(cross_validation.KFold(N, K_folds, shuffle=True)):
        # data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size=0.20, random_state=42)
        train_X, test_X = X[train_indices], X[test_indices]
        train_y, test_y = y[train_indices], y[test_indices]

        logger.info('k=%d; %d train, %d test.', k, len(train_indices), len(test_indices))

        # dimension reduction
        # f_regression help:
        #   http://stackoverflow.com/questions/15484011/scikit-learn-feature-selection-for-regression-data
        # other nice ML variable selection help:
        #   http://www.quora.com/What-are-some-feature-selection-methods-for-SVMs
        #   http://www.quora.com/What-are-some-feature-selection-methods
        # train_chi2_stats, train_chi2_pval = chi2(train_X, train_y)
        # train_classif_F, train_classif_pval = f_classif(train_X, train_y)
        train_F, train_pval = f_regression(train_X, train_y)
        # train_pval.shape = (4729,)
        ranked_dimensions = np.argsort(train_pval)
        ranked_names = dimension_names[np.argsort(train_pval)]
        top_k = 100

        # train_X.shape: (4500, 18884) -> (4500, 100)
        train_X = train_X[:, ranked_dimensions[:top_k]]
        # train_X.shape: (500, 18884) -> (500, 100)
        test_X = test_X[:, ranked_dimensions[:top_k]]

        IPython.embed()

        # train_X, test_X = X[train_indices], X[test_indices]
        # train_y, test_y = y[train_indices], y[test_indices]

        # train and predict
        model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)
        # model = linear_model.LogisticRegression(penalty='l1')
        # model = linear_model.LinearRegression(normalize=True)
        # model = svm.LinearSVC(penalty='l2', dual=False, C=2.0)
        # model = svm.SVC(probability=True)
        # model = svm.LinearSVC(penalty='l2', dual=False, tol=0.0001, C=1.0)
        # model = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
        #     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
        #     min_density=None, compute_importances=None)
        # model = linear_model.SGDClassifier(loss='log')
        # model = naive_bayes.MultinomialNB()
        # neighbors.KNeighborsClassifier(3)
        # linear_model.LogisticRegression()
        model.fit(train_X, train_y)
        logger.info('Model.classes_: %s', ', '.join(labels[class_] for class_ in model.classes_))

        # classif_selector = SelectKBest(f_classif, k=100)
        # classif_selector.fit(train_X, train_y)
        # print 'classif_selector', classif_selector

        # regression_selector = SelectKBest(f_regression, k=100)
        # regression_selector.fit(train_X, train_y)
        # print 'regression_selector', regression_selector

        # train_F_hmean = scipy.stats.hmean(train_F[train_F > 0])
        # print 'train_F_hmean', train_F_hmean
        # neg_train_pval_hmean = scipy.stats.hmean(1 - train_pval[train_pval > 0])
        # print '-train_pval_hmean', neg_train_pval_hmean

        # regression_selector = SelectKBest(f_regression, k=100)
        # print 'regression_selector', regression_selector
        # top_k_corpus_types = dimension_names[regression_selector.get_support()]
        # print 'coef_', model.coef_
        # coef_sort = np.argsort(model.coef_)
        # print corpus_types[np.argsort(model.coef_)]

        # predict using the model just trained
        # pred_y = model.predict(test_X)

        pred_certainty = np.repeat(np.nan, test_y.shape)
        print 'pred_certainty.shape', pred_certainty.shape
        if hasattr(model, 'predict_proba'):
            pred_probabilities = model.predict_proba(test_X)
            # predicts_proba returns N rows, each C-long, where C is the number of labels
            # with this, we can use np.array.argmax to get the class names we would have gotten with model.predict()
            # axis=0 will give us the max for each column (not very useful)
            # axis=1 will give us the max for each row (what we want)
            # find best guess (same as model.predict(...), I think)
            pred_y = pred_probabilities.argmax(axis=1)
            # pred_certainty now ranges between 0 and 1,
            #   a pred_certainty of 1 means the prediction probabilities were extreme,
            #                       0 means they were near 0.5 each
            # hmean takes the harmonic mean of its arguments
            if (pred_probabilities > 0).all():
                pred_certainty = 1 - 2 * scipy.stats.hmean(pred_probabilities, axis=1)
        else:
            logger.info('predict_proba is unavailable for this model: %s', model)
            pred_y = model.predict(test_X)

        # if k == 9:
        #     print '!!! randomizing predictions'
        #     pred_y = [random.choice((0, 1)) for _ in pred_y]
        # logger.info('Mispredictions')

        certainties = defaultdict(list)
        for test_i, (gold, pred, certainty) in enumerate(zip(test_y, pred_y, pred_certainty)):
            correct = gold == pred
            correct_name = 'Correct' if correct else 'Incorrect'

            certainties[correct_name].append(certainty)

            if not correct:
                y_i = test_indices[test_i]
                print 'test %d => y %d' % (test_i, y_i)
                print 'gold (%s=%s) != pred (%s=%s)' % (gold, labels[gold], pred, labels[pred])
                print 'certainty: %0.4f' % certainty
                print 'Document#%d: [%s] %s' % (y_i, corpus_labels[y_i], corpus_strings[y_i])
                print
                # print 'vec', corpus_count_vectors[y_i]

        if np.isnan(pred_certainty).any():
            print 'certainty is unavailable'
        else:
            print '*: certainty mean=%0.5f' % np.mean(pred_certainty)
            histogram.hist(pred_certainty, range=(0, 1))
            for certainty_name, certainty_values in certainties.items():
                print '%s: certainty mean=%0.5f' % (certainty_name, np.mean(certainty_values))
                histogram.hist(certainty_values, range=(0, 1))

        # evaluate
        print 'Accuracy: %0.5f, F1: %0.5f' % (
            metrics.accuracy_score(test_y, pred_y),
            metrics.f1_score(test_y, pred_y))
        # print 'confusion:\n', metrics.confusion_matrix(test_y, pred_y)
        print 'report:\n', metrics.classification_report(test_y, pred_y, target_names=labels)


tweets_scikit()
