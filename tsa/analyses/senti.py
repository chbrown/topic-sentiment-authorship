# -*- coding: utf-8 -*-
import numpy as np

import iter8

from sklearn import cross_validation
from sklearn import naive_bayes
from tsa.science.summarization import metrics_dict

from tsa import logging
logger = logging.getLogger(__name__)

# from tsa.science.plot import plt


def rottentomatoes(analysis_options):
    import tsa.data.rt_polaritydata
    corpus = tsa.data.rt_polaritydata.read_MulticlassCorpus()

    indices = corpus.indices.copy()
    np.random.shuffle(indices)

    from scipy import sparse
    if sparse.issparse(corpus.X):
        X = corpus.X.tocsr()[indices]
        # X = corpus.X.toarray()[indices]
    else:
        X = corpus.X[indices]
    y = corpus.y[indices]

    # from tsa.lib import tabular
    # printer = tabular.Printer()

    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    for fold_index, (train_indices, test_indices) in iter8.sig_enumerate(folds, logger=logger):
        test_X, test_y = X[test_indices], y[test_indices]
        train_X, train_y = X[train_indices], y[train_indices]

        # model = linear_model.LogisticRegression(penalty='l2', C=1)
        # model = svm.LinearSVC(penalty='l2', )
        # model = linear_model.SGDClassifier()
        # model = neural_network.BernoulliRBM()
        model = naive_bayes.MultinomialNB()
        # model = naive_bayes.GaussianNB() # Whoa, incredibly slow
        # model = naive_bayes.BernoulliNB()

        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        # coefs = model.coef_.ravel()

        result = metrics_dict(test_y, pred_y)
        # print 'Prediction result:', result
        print('Prediction accuracy:%s' % result['accuracy'])

        # print metrics.accuracy_score(y, pred_y)

    # exit(IPython.embed())
