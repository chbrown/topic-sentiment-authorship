# -*- coding: utf-8 -*-
import os
import IPython
import numpy as np
# import pandas as pd
from tsa.science import numpy_ext as npx

import viz
from viz.format import quantiles
from viz.geom import hist

# from collections import Counter
from sklearn import metrics
from sklearn import cross_validation
from sklearn import linear_model
from tsa.science.corpora import MulticlassCorpus
from tsa.science import features
from tsa.science.summarization import metrics_dict

from tsa.lib import itertools
from tsa import logging
logger = logging.getLogger(__name__)

# from tsa.science.plot import plt
# import matplotlib.pyplot as plt


def read_MulticlassCorpus(labeled_only=False):
    dirpath = os.path.expanduser('~/corpora-public/bopang_lillianlee/rt-polaritydata/')

    def read_file(filename, label):
        with open(os.path.join(dirpath, filename)) as fd:
            for line in fd:
                yield (label, line)

    data = list(read_file('rt-polarity.neg.utf8', 'neg')) + list(read_file('rt-polarity.pos.utf8', 'pos'))
    # data is now a list of label:string-document:string tuples

    labels, documents = zip(*data)
    corpus = MulticlassCorpus(labels)
    corpus.documents = documents

    corpus.apply_features(documents, features.ngrams,
        # ngram_max=2, min_df=0.001, max_df=0.95)
        ngram_max=2, min_df=1, max_df=1.0)
    # corpus.apply_features(documents, features.liwc)
    # corpus.apply_features(documents, features.afinn)
    # corpus.apply_features(documents, features.anew)
    logger.debug('rt-polaritydata MulticlassCorpus created: %s', corpus.X.shape)
    return corpus


def rottentomatoes(analysis_options):
    corpus = read_MulticlassCorpus()

    indices = corpus.indices.copy()
    np.random.shuffle(indices)

    from scipy import sparse
    if sparse.issparse(corpus.X):
        X = corpus.X.tocsr()[indices]
        # X = corpus.X.toarray()[indices]
    else:
        X = corpus.X[indices]
    y = corpus.y[indices]


    from sklearn import svm
    from sklearn import neural_network
    from sklearn import naive_bayes

    folds = cross_validation.KFold(y.size, 10, shuffle=True)
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
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
        print 'Prediction accuracy:', result['accuracy']

        # print metrics.accuracy_score(y, pred_y)

    # exit(IPython.embed())
