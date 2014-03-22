import IPython
import sys
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
from tsa.science import numpy_ext as npx

from datetime import datetime

import viz
from viz.geom import hist

from sklearn import cluster
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import qda
from sklearn import svm
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
# from sklearn.feature_selection import chi2, f_classif, f_regression

from tsa import stdout, stderr
from tsa.lib import cache, tabular, itertools
from tsa.lib.timer import Timer
from tsa.science import features, models
from tsa.science.corpora import MulticlassCorpus
from tsa.science.summarization import metrics_dict  # explore_mispredictions, explore_uncertainty
from tsa.science.plot import plt, figure_path, clear, distinct_styles
from tsa.data.sb5b.tweets import read_MulticlassCorpus as read_sb5b_MulticlassCorpus
from tsa.data.rt_polaritydata import read_MulticlassCorpus as read_RT_MulticlassCorpus
from tsa.models import Source, Document, create_session

from tsa import logging
logger = logging.getLogger(__name__)


# def MulticlassCorpus_from_documents(documents):
#     documents = list(documents)
#     corpus = MulticlassCorpus(documents)
#     corpus.apply_labelfunc(lambda doc: doc.label)
#     return corpus

def database_overview(analysis_options):
    from sqlalchemy import func

    print '# database overview'

    DBSession = create_session()
    sources = DBSession.query(Source).all()
    for source in sources:
        print '## %s' % (source.name)
        n = DBSession.query(Document).filter(Document.source == source).count()
        # print 'n', DBSession.query(Document).filter(Document.source == source), n
        print 'N = %6d' % n
        labels = DBSession.query(Document.label, func.count(Document.label)).\
            filter(Document.source == source).\
            group_by(Document.label)
        for label, count in labels:
            print '    %6d %s' % (count, label)
        print


def source_documents(source_name):
    DBSession = create_session()
    return DBSession.query(Document).\
        join(Source, Source.id == Document.source_id).\
        filter(Source.name == source_name).\
        filter(Document.label != None).\
        order_by(Document.published).all()


def source_corpus(source_name):
    documents = source_documents(source_name)
    corpus = MulticlassCorpus(documents)
    corpus.apply_labelfunc(lambda doc: doc.label)
    # assume the corpus is suitably balanced
    corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)
    # corpus.extract_features(documents, features.liwc)
    # corpus.extract_features(documents, features.afinn)
    # corpus.extract_features(documents, features.anew)
    return corpus


def sb5b_source_corpus():
    # mostly like source_corpus except it selects just For/Against labels
    documents = source_documents('sb5b')
    corpus = MulticlassCorpus(documents)
    corpus.apply_labelfunc(lambda doc: doc.label)
    # balanced_indices = npx.balance(
    #     corpus.y == corpus.class_lookup['For'],
    #     corpus.y == corpus.class_lookup['Against'])
    polar_indices = (corpus.y == corpus.class_lookup['For']) | (corpus.y == corpus.class_lookup['Against'])
    corpus = corpus.subset(polar_indices)
    # ngram_max=2, min_df=0.001, max_df=0.95
    corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)
    return corpus


def sample_corpus():
    # return the corpus
    session = create_session()
    sb5b_documents = session.query(Document).join(Source).\
        filter(Source.name == 'sb5b').all()
        # filter(Document.label.in_(['For', 'Against'])).\
    sample_documents = session.query(Document).join(Source).\
        filter(Source.name == 'twitter-sample').all()
    corpus = MulticlassCorpus(sb5b_documents + sample_documents)
    corpus.apply_labelfunc(lambda doc: doc.source.name)
    corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)

    return corpus

def debate_corpus():
    session = create_session()
    documents = session.query(Document).join(Source).\
        filter(Source.name == 'debate08').\
        filter(Document.label.in_(['Positive', 'Negative'])).\
        order_by(Document.published).all()

    corpus = MulticlassCorpus(documents)
    corpus.apply_labelfunc(lambda doc: doc.label)
    corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)
    return corpus




def all_corpora():
    # yield (corpus, title) tuples
    yield sb5b_source_corpus(), 'SB-5 For/Against'
    yield source_corpus('rt-polarity'), 'Rotten Tomatoes Polarity'
    yield source_corpus('convote'), 'Congressional vote'
    yield debate_corpus(), 'Debate08'
    yield source_corpus('stanford-politeness-wikipedia'), 'Politeness on Wikipedia'
    yield source_corpus('stanford-politeness-stackexchange'), 'Politeness on StackExchange'
    yield sample_corpus(), 'In-sample / Out-of-sample'


def sb5_corpora():
    # yield sb5b_source_corpus(), 'SB-5 For/Against'
    yield sample_corpus(), 'In-sample/Out-of-sample'
    # yield source_corpus('rt-polarity'), 'Rotten Tomatoes Polarity'


def grid_plots(analysis_options):
    for corpus, title in sb5_corpora():
        print title
        grid_plot(corpus)
        plt.title(title)
        plt.savefig(figure_path('grid-plot-%s' % title))
        plt.cla()


def grid_hists(analysis_options):
    for corpus, title in all_corpora():
        grid_hist(corpus)
        plt.title(title)
        plt.savefig(figure_path('02-%s.pdf' % title))
        plt.cla()


def grid_hist(corpus):
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    # model = linear_model.RandomizedLogisticRegression(penalty='l2')
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html
    bootstrap_coefs = models.bootstrap(corpus.X, corpus.y, n_iter=100, proportion=1.0)
    coefs_means = np.mean(bootstrap_coefs, axis=0)
    coefs_variances = np.var(bootstrap_coefs, axis=0)

    logger.info('coefs_means.shape = %s, coefs_variances.shape = %s', coefs_means.shape, coefs_variances.shape)

    plt.hist(coefs_means, bins=25)
    plt.xlabel('Frequency of bootstrapped log. reg. coefficients')
    plt.gcf().set_size_inches(8, 5)
    # plt.legend(loc='best')
    # plt.ylim(.4, 1.0)


def sample_errors(analysis_options):
    corpus = sample_corpus()
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    folds = cross_validation.StratifiedShuffleSplit(corpus.y, test_size=0.5, n_iter=20)
    for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
        train_corpus = corpus.subset(train_indices)
        test_corpus = corpus.subset(test_indices)

        model = linear_model.LogisticRegression(penalty='l2')
        model.fit(train_corpus.X, train_corpus.y)
        pred_y = model.predict(test_corpus.X)
        pred_proba = model.predict_proba(test_corpus.X)

        # print metrics_dict(test_corpus.y, pred_y)
        incorrect_indices = npx.bool_mask_to_indices(pred_y != test_corpus.y)
        # misclassified_docs = test_corpus.data[incorrect]
        # misclassified_proba = pred_proba[incorrect]
        print 'number incorrect =', len(incorrect_indices)

        # for doc, prob in zip(misclassified_docs, misclassified_proba):
        coefs = model.coef_.ravel()
        for index in incorrect_indices:
            doc = test_corpus.data[index]
            x = test_corpus.X[index].toarray().ravel()
            prob = pred_proba[index]
            nonzero_features = x > 0
            x_names = test_corpus.feature_names[nonzero_features]
            x_coefs = x[nonzero_features] * coefs[nonzero_features]
            # pairs = zip(x_names, ['%.2f' % x for x in x_coefs])
            reordering = np.argsort(x_coefs)
            pairs = zip(x_names[reordering], ['%.2f' % x for x in x_coefs[reordering]])
            print
            print '--- %s ---' % test_corpus.labels[test_corpus.y[index]]
            print '%s (%s)' % (doc.document.replace('\n', ' '), doc.label)
            print dict(zip(test_corpus.labels[model.classes_], prob))
            print viz.gloss.gloss([('', 'means')] + pairs + [('SUM', '%.2f' % sum(x_coefs))])

        exit(IPython.embed())



def grid_plot(corpus):
    # make X sliceable
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    models = [
        # ('Logistic Regression (L1)', linear_model.LogisticRegression(penalty='l1')),
        ('Logistic Regression (L2)', linear_model.LogisticRegression(penalty='l2')),
        # ('logistic_regression-L2-C100', linear_model.LogisticRegression(penalty='l2', C=100.0)),
        # ('randomized_logistic_regression', linear_model.RandomizedLogisticRegression()),
        # ('SGD', linear_model.SGDClassifier()),
        # ('Perceptron', linear_model.Perceptron(penalty='l1')),
        # ('perceptron-L2', linear_model.Perceptron(penalty='l2')),
        # ('linear-svc-L2', svm.LinearSVC(penalty='l2')),
        # ('linear-svc-L1', svm.LinearSVC(penalty='l1', dual=False)),
        # ('random-forest', ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
        #     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
        #     min_density=None, compute_importances=None)),
    #('Naive Bayes', naive_bayes.MultinomialNB()),
        # ('Naive Bayes (Bernoulli)', naive_bayes.BernoulliNB()),
        # ('knn-10', neighbors.KNeighborsClassifier(10)),
        # ('neuralnet', neural_network.BernoulliRBM()),
        # ('qda', qda.QDA()),
        # ('knn-3', neighbors.KNeighborsClassifier(3)),
        # ('sgd-log-elasticnet', linear_model.SGDClassifier(loss='log', penalty='elasticnet')),
        # ('linear-regression', linear_model.LinearRegression(normalize=True)),
        # ('svm-svc', svm.SVC()),
        # ('adaboost-50', ensemble.AdaBoostClassifier(n_estimators=50)),
    ]
    train_sizes = [10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 0.9]
    # proportions = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.95]
    n_iter = 20

    # in KFold, if shuffle=False, we look at a sliding window for the test sets, starting at the left
    # folds = cross_validation.KFold(len(corpus), 10, shuffle=True)
    # folds = cross_validation.StratifiedKFold(corpus.y, 10)

    def combinations(model_tuples, train_sizes, y, n_iter):
        # thin deep loop over all combinations
        for model_name, model in model_tuples:
            for train_size in train_sizes:
                if train_size < y.size:
                    folds = cross_validation.StratifiedShuffleSplit(y, n_iter=n_iter,
                        train_size=train_size, test_size=None)
                    for train_indices, test_indices in folds:
                        yield model_name, model, corpus, train_indices, test_indices

    # printer is mostly for logging
    printer = tabular.Printer()

    def evaluate(model_name, model, corpus, train_indices, test_indices):
        train_corpus = corpus.subset(train_indices)
        test_corpus = corpus.subset(test_indices)

        with Timer() as timer:
            # fit and predict
            model.fit(train_corpus.X, train_corpus.y)
            pred_y = model.predict(test_corpus.X)

        # pos_label=test_corpus.class_lookup['For']
        results = metrics_dict(test_corpus.y, pred_y)
        results.update(
            train=len(train_corpus),
            test=len(test_corpus),
            total=len(corpus),
            elapsed=timer.elapsed,
            model=model_name,
            # proportion=proportion,
        )
        # printer.write(results)
        stdout('.')
        return results

    rows = [evaluate(*args) for args in combinations(models, train_sizes, corpus.y, n_iter)]
    stdout('\n')

    df = pd.DataFrame.from_records(rows)

    counts = np.array(Counter(corpus.y).values(), dtype=float)
    baseline = counts.max() / counts.sum()
    print 'baseline:', baseline
    for index, group in df.groupby(['model']):
        agg = group.groupby(['train']).aggregate(np.mean)
        print index  # name of model
        accuracies = agg.ix[[10, 100, 1000]].accuracy
        improvements = 1 - ((1 - accuracies) / (1 - baseline))  # == (accuracies - baseline) / (1 - baseline)
        print agg  # full cross-tabulation
        print
        print 'labels      {:d} & {:d} & {:d}'.format(*accuracies.index)
        print 'baseline    {0:.2%} & {0:.2%} & {0:.2%}'.format(baseline)
        print 'accuracy    {:.2%} & {:.2%} & {:.2%}'.format(*accuracies)
        print 'improvement {:.2%} & {:.2%} & {:.2%}'.format(*improvements)
        style = distinct_styles.next()
        agg.plot(y='accuracy', label=index, **style)

    IPython.embed()

    plt.legend(loc='best')
    # plt.legend(loc='lower right')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.ylim(.4, 1.0)
    # plt.xlim(0, 5000)
    plt.gcf().set_size_inches(8, 5)


def plot_runs(runs):
    df = pd.DataFrame.from_records(runs)
    # df_agg = df.groupby(['model', 'train']).aggregate(np.mean)
    # df_agg.plot(x='train', y='accuracy')

    for index, group in df.groupby(['model']):
        agg = group.groupby(['train']).aggregate(np.mean)
        style = distinct_styles.next()
        agg.plot(y='accuracy', label=index, **style)
