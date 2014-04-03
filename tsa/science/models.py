import numpy as np
from tsa.science import numpy_ext as npx
from tsa.lib import itertools
# from sklearn import linear_model

from tsa import logging
logger = logging.getLogger(__name__)


class Bootstrap(object):
    def __init__(self, ClassificationModel, **model_args):
        self.ClassificationModel = ClassificationModel
        # penalty='l2', C=1.0):
        self.model_args = model_args

    def fit(self, X, y, n_iter=100, proportion=0.5):
        # the parameters to this method are specific to the Bootstrap process
        n_features = X.shape[1]
        # each row in coefs represents the results from a single bootstrap run
        self.coefs_ = np.zeros((n_iter, n_features))
        self.classes_ = np.unique(y)
        folds = npx.bootstrap(y.size, n_iter=n_iter, proportion=proportion)
        for fold, (train_indices, _) in itertools.sig_enumerate(folds, logger=logger):
            model = self.ClassificationModel(**self.model_args)
            model.fit(X[train_indices, :], y[train_indices])
            self.coefs_[fold, :] = model.coef_.ravel()

    def predict(self, X):
        # we find the prediction by lining up the classes and picking one of them
        # according to which column is the max
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        coefs_means = np.mean(self.coefs_, axis=0)
        projections = X.dot(coefs_means)
        # probabilities refers to the first class in self.classes_, I think,
        # and this model only accounts for binary classification (see ravel() in fit method)
        # projections_variance = projections.var(axis=1)
        projections_logistic = npx.logistic(projections)
        return np.column_stack((1 - projections_logistic, projections_logistic))