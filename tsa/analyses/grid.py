import IPython
import numpy as np
import pandas as pd
from tsa.science import numpy_ext as npx

from datetime import datetime

from viz.geom import hist

from sklearn import cluster
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
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

from tsa.lib import cache, tabular, itertools
from tsa.lib.timer import Timer
from tsa.science import features
from tsa.science.corpora import MulticlassCorpus
from tsa.science.summarization import metrics_dict  # explore_mispredictions, explore_uncertainty
from tsa.science.plot import plt, fig_path, clear, distinct_styles
from tsa.data.sb5b.tweets import read_MulticlassCorpus as read_sb5b_MulticlassCorpus
from tsa.data.rt_polaritydata import read_MulticlassCorpus as read_RT_MulticlassCorpus
from tsa.models import Source, Document, create_session

from tsa import logging
logger = logging.getLogger(__name__)


def evaluate(model, corpus, train_indices, test_indices):
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
        elapsed=timer.elapsed,
    )
    return results


def plot_runs(runs):
    df = pd.DataFrame.from_records(runs)
    # df_agg = df.groupby(['model', 'train']).aggregate(np.mean)
    # df_agg.plot(x='train', y='accuracy')

    for index, group in df.groupby(['model']):
        agg = group.groupby(['train']).aggregate(np.mean)
        style = distinct_styles.next()
        agg.plot(y='accuracy', label=index, **style)


# def MulticlassCorpus_from_documents(documents):
#     documents = list(documents)
#     corpus = MulticlassCorpus(documents)
#     corpus.apply_labelfunc(lambda doc: doc.label)
#     return corpus

def database_overview(analysis_options):
    from sqlalchemy import func

    DBSession = create_session()
    sources = DBSession.query(Source).all()
    for source in sources:
        print '# %s' % (source.name)
        n = DBSession.query(Document).filter(Document.source == source).count()
        # print 'n', DBSession.query(Document).filter(Document.source == source), n
        print 'N = %d' % n
        labels = DBSession.query(Document.label, func.count(Document.label)).\
            filter(Document.source == source).\
            group_by(Document.label)
        for label, count in labels:
            print '  %5d %s' % (count, label)
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


def sb5b_corpus():
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


def grid_multi(analysis_options):
    # sb5
    corpus = sb5b_corpus()
    grid_plot(corpus)
    plt.title('SB-5 For/Against')
    plt.savefig(fig_path('01-SB5.pdf'))

    # rt
    corpus = source_corpus('rt-polarity')
    grid_plot(corpus)
    plt.title('Rotten Tomatoes Polarity')
    plt.savefig(fig_path('01-RT.pdf'))

    # convote
    corpus = source_corpus('convote')
    grid_plot(corpus)
    plt.title('Congressional vote')
    plt.savefig(fig_path('01-convote.pdf'))

    # debate08_corpus
    corpus = source_corpus('debate08')
    grid_plot(corpus)
    plt.title('Congressional vote')
    plt.savefig(fig_path('01-convote.pdf'))

    # stanford-politeness-wikipedia
    corpus = source_corpus('stanford-politeness-wikipedia')
    grid_plot(corpus)
    plt.title('Politeness on Wikipedia')
    plt.savefig(fig_path('01-stanford-politeness-wikipedia.pdf'))

    # raise IPython.embed()


def grid_plot(corpus):
    # make X sliceable
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    printer = tabular.Printer()

    models = [
        # ('logistic_regression-L1', linear_model.LogisticRegression(penalty='l1', dual=False)),
        ('Logistic Regression', linear_model.LogisticRegression(penalty='l2')),
        # ('logistic_regression-L2-C100', linear_model.LogisticRegression(penalty='l2', C=100.0)),
        # ('randomized_logistic_regression', linear_model.RandomizedLogisticRegression()),
        ('SGD', linear_model.SGDClassifier()),
        ('Perceptron', linear_model.Perceptron(penalty='l1')),
        # ('perceptron-L2', linear_model.Perceptron(penalty='l2')),
        # ('linear-svc-L2', svm.LinearSVC(penalty='l2')),
        # ('linear-svc-L1', svm.LinearSVC(penalty='l1', dual=False)),
        # ('random-forest', ensemble.RandomForestClassifier(n_estimators=10, criterion='gini',
        #     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
        #     min_density=None, compute_importances=None)),
        ('Naive Bayes', naive_bayes.MultinomialNB()),
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
    proportions = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.95]

    rows = []
    for model_name, model in models:
        # for each model
        for proportion in proportions:
            # for each proportion
            # np.random.permutation
            # in KFold, if shuffle=False, we look at a sliding window for the test sets, starting at the left
            # folds = cross_validation.KFold(len(corpus), 10, shuffle=True)
            # folds = cross_validation.StratifiedKFold(corpus.y, 10)
            folds = cross_validation.StratifiedShuffleSplit(corpus.y, test_size=1.0 - proportion, n_iter=20)
            for fold_index, (train_indices, test_indices) in itertools.sig_enumerate(folds, logger=logger):
                # for each fold
                # size = int(train_indices.size * proportion)
                # train_indices_subset = np.random.choice(train_indices, size=size, replace=False)
                # logger.info('table(train_selection) = %s', npx.table(corpus.y[train_indices]))
                results = evaluate(model, corpus, train_indices, test_indices)
                results.update(
                    # fold=fold_index,
                    model=model_name,
                    proportion=proportion,
                )
                rows += [results]
                printer.write(results)
            # average over folds...

    plot_runs(rows)
    plt.legend(loc='lower right')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    # plt.ylim(.5, 1.0)
    # plt.xlim(0, 5000)

    plt.gcf().set_size_inches(8, 5)
