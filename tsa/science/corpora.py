import numpy as np
from scipy import sparse

from tsa import logging
logger = logging.getLogger(__name__)


class MulticlassCorpus(object):
    '''
    Structure to keep related corpus-specific objects all together.

    scikit-learn terminology:
        Multiclass: one label per document
        Multilabel: multiple labels per document

    tsa terminology:
        label (str):  human name of a specific class, e.g., "For", "Againt", "Undecided"
        class (int):  index representation of a label

    A MulticlassCorpus instance has the following attributes:
        data (np.array):          (N,) Array of objects, usually.
        X (np.array):             (N, k) matrix of numbers (usually, floats)
                                    N rows: documents in corpus
                                    k columns: features
        y (np.array):             (N,) vector of ints
        labels (np.array[str]):   Basically just a list of strings
                                  Acts as a mapping from class_ (int) to label (str),
        class_lookup (dict):      Map from label (str) to class_ (int)
        feature_names (np.array): (k,) vector of strings, e.g., 'RT', '#weareohio', 'vote no'

    '''
    def __init__(self, data):
        self.data = np.array(data)

        # labels is a unique list of labels, one for each value of self.y
        self.labels = np.array([])

        # data-related variables, zeroed out (empty)
        self.X = np.array([[]])
        self.feature_names = np.array([])

        # logger.debug('MulticlassCorpus created (N = %d)', len(self))


    def __len__(self):
        return len(self.data)


    def __repr__(self):
        return '<MulticlassCorpus X = {:s}, y = {:s}>'.format(self.X.shape, self.y.shape)


    def apply_labelfunc(self, labelfunc):
        '''
        Updates self.labels and sets self.y based on self.data and the given label getter function.
        '''
        # y_labels is an N-long np.array of strings
        y_labels = list(map(labelfunc, self.data))
        # calculate unique classes using set()
        new_labels = set(y_labels) - set(self.labels)
        self.labels = np.concatenate((self.labels, list(new_labels)))
        # class_lookup is a map from label strings to class numbers
        self.class_lookup = dict((label, i) for i, label in enumerate(self.labels))
        # class_lookup_func = np.vectorize(self.labels.__getitem__)
        self.y = np.array([self.class_lookup[y_label] for y_label in y_labels])


    def extract_features(self, docfunc, feature_function, **feature_function_kwargs):
        '''
        docfunc is a function which, applied to each item in data,
        will produce a document that will then have the specified feature_function applied to it.

        e.g.,
        from tsa import features
        corpus.apply_features(documents, features.ngrams, ngram_max=1)
        '''
        documents = list(map(docfunc, self.data))
        X, feature_names = feature_function(documents, **feature_function_kwargs)
        original_ncolumns = self.X.shape[1]
        # incorporate / merge X
        if self.X.size == 0:
            # can skip merging non-empty matrices
            self.X = X
        elif sparse.issparse(self.X) or sparse.issparse(X):
            # sparse.hstack output is always COO, it seems
            self.X = sparse.hstack((self.X, X)).tocsr()
        else:
            self.X = np.hstack((self.X, X))
        # merge feature_names
        self.feature_names = np.concatenate((self.feature_names, feature_names))
        # return the indices of the added columns
        return np.arange(original_ncolumns, self.X.shape[1])


    # def extract_default_features(self, docfunc):
    #     # use the defaults: ngram_max=2, min_df=0.01, max_df=0.99
    #     self.extract_features(docfunc, features.ngrams)


    def subset(self, rows=None, features=None):
        '''
        Return new corpus, for given subset of rows.

        indices can be a boolean mask.
        '''
        n_rows, n_features = self.X.shape
        if rows is None:
            rows = np.arange(n_rows)
        if features is None:
            features = np.arange(n_features)

        corpus = MulticlassCorpus(self.data[rows])
        corpus.labels = self.labels
        corpus.class_lookup = self.class_lookup
        corpus.feature_names = self.feature_names[features]
        corpus.y = self.y[rows]
        # empty X handling could be better
        if self.X.size > 0:
            if sparse.issparse(self.X):
                # this sparse subsetting could probably be better, too
                corpus.X = self.X.tocsc()[:, features].tocsr()[rows, :]
            else:
                # well, this is awkward... [rows, features] doesn't *just work*
                corpus.X = self.X[rows, :][:, features]
        else:
            corpus.X = self.X

        return corpus
