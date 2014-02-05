import numpy as np
from scipy import sparse


class MulticlassCorpus(object):
    '''
    Structure to keep related corpus-specific objects all together.

    scikit-learn terminology:
        Multiclass: one label per document
        Multilabel: multiple labels per document

    tsa terminology:
        `label` (str):  human name of a specific class, e.g., "For", "Againt", "Undecided"
        `class` (int):  index representation of a label

    A MulticlassCorpus instance has the following attributes:
        `X` (np.array):             (N, k) matrix of numbers (usually, floats)
                                    N rows: documents in corpus
                                    k columns: features
        `y` (np.array):             (N,) vector of ints
        `labels` (dict):            mapping from label  (str) to class_ (int)
        `classes` (np.array):       mapping from class_ (int) to label  (str)
                                    the indices of classes matches numpy's model.classes_
        `feature_names` (np.array): (k,) vector of strings, e.g., 'RT', '#weareohio', 'vote no'

    '''
    def __init__(self, y_raw=None):
        # y_raw is an N-long np.array of strings
        # calculate unique classes using set()
        self.classes = np.array(list(set(y_raw)))
        # reverse and zip with labels
        self.labels = dict((label, class_) for class_, label in enumerate(self.classes))
        # maybe self.labels should be labelize?
        labelize = np.vectorize(self.labels.__getitem__)

        # maybe coerce y to np.array here?
        self.y = labelize(y_raw)
        self.X = np.array([])
        self.feature_names = np.array([])

    def apply_features(self, documents, feature_function, **feature_function_kwargs):
        '''
        e.g.,
        from tsa import features
        corpus.apply_features(documents, features.ngrams, ngram_max=1)
        '''
        X, feature_names = feature_function(documents, **feature_function_kwargs)
        # incorporate / merge X
        if self.X.size == 0:
            # only merge non-empty matrices
            self.X = X
        elif sparse.issparse(self.X) or sparse.issparse(X):
            self.X = sparse.hstack((self.X, X))
        else:
            self.X = np.hstack((self.X, X))
        # merge feature_names
        self.feature_names = np.concatenate((self.feature_names, feature_names))

    def __iter__(self):
        '''
        Allow unpacking like:

        X, y = corpus
        '''
        yield self.X
        yield self.y
        # yield self.labels
        # yield self.classes
        # yield self.feature_names
