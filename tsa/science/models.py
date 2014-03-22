import numpy as np
from tsa.science import numpy_ext as npx
from tsa.lib import itertools
from sklearn import linear_model

from tsa import logging
logger = logging.getLogger(__name__)


def bootstrap(X, y, n_iter=100, proportion=0.5):
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
