import IPython
from itertools import groupby
import numpy as np
import pandas as pd
from tsa.science import numpy_ext as npx

from collections import Counter
# from datetime import datetime

import viz
from viz.geom import hist

from sklearn import metrics, cross_validation
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
# from sklearn import cluster, decomposition, ensemble, neighbors, neural_network, qda
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
# from sklearn.feature_selection import chi2, f_classif, f_regression

from tsa import stdout, stderr
from tsa.lib import tabular, datetime_extra
from tsa.lib.timer import Timer
from tsa.models import Source, Document, create_session
from tsa.science import features, models, timeseries
from tsa.science.corpora import MulticlassCorpus
from tsa.science.plot import plt, figure_path, distinct_styles, ticker
from tsa.science.summarization import metrics_dict, metrics_summary
# from tsa.science.summarization import explore_mispredictions, explore_uncertainty

from tsa import logging
logger = logging.getLogger(__name__)


def source_corpus(source_name):
    documents = Source.from_name(source_name)
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
    documents = Source.from_name('sb5b')
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


def iter_corpora():
    # yield (corpus, title) tuples
    yield source_corpus('rt-polarity'), 'Rotten Tomatoes Polarity'
    yield sample_corpus(), 'In-sample/Out-of-sample'
    yield sb5b_source_corpus(), 'SB-5 For/Against'
    yield source_corpus('convote'), 'Congressional vote'
    yield debate_corpus(), 'Debate08'
    yield source_corpus('stanford-politeness-wikipedia'), 'Politeness on Wikipedia'
    yield source_corpus('stanford-politeness-stackexchange'), 'Politeness on StackExchange'


def grid_plots(analysis_options):
    for corpus, title in iter_corpora():
        print title
        grid_plot(corpus)
        plt.title(title)
        plt.savefig(figure_path('grid-plot-%s' % title))
        plt.cla()


def representation(analysis_options):
    corpus = sb5b_source_corpus()

    print 'Tweets per person, by label'

    for class_name in ['For', 'Against']:
        print 'Class =', class_name
        indices = corpus.y == corpus.class_lookup[class_name]
        keyfunc = lambda doc: doc.details['Author'].split()[0].lower()
        data = sorted(corpus.data[indices], key=keyfunc)
        author_groups = groupby(data, keyfunc)
        # map values . sum
        lengths = np.array([len(list(group_iter)) for author, group_iter in author_groups])
        # print 'Hist for authors with more than one tweet:'
        # hist(lengths[lengths > 1])
        print 'Average # of documents per user', lengths.mean()
        inlier_max = np.percentile(lengths, 99)
        inliers = lengths[lengths < inlier_max]
        print '  ditto excluding 99%-file ({:d}): {:.1f}'.format(
            lengths.size - inliers.size, inliers.mean())

    IPython.embed()



def corpus_sandbox(analysis_options):
    print 'Exploring SB-5 corpus'
    session = create_session()
    sb5b_documents = session.query(Document).join(Source).\
        filter(Source.name == 'sb5b').all()

    print 'Found %d documents' % len(sb5b_documents)

    rows = [dict(label=document.label, inferred=bool(document.details.get('Inferred')), source=document.details.get('Source', 'NA'))
            for document in sb5b_documents]
    df = pd.DataFrame.from_records(rows)

    # df_agg = df.groupby(['label', 'inferred'])

    # df.pivot_table(values=['label'], rows=['inferred'], aggfunc=[len])
    df.pivot_table(rows=['label', 'inferred'], aggfunc=[len])
    df.pivot_table(rows=['label', 'source'], aggfunc=[len])
    df.pivot_table(rows=['source'], aggfunc=[len])
    # df_agg.plot(x='train', y='accuracy')

    for document in sb5b_documents:
        # 'weareohio' in document.document.lower(), .document
        print document.details.get('Source'), document.label


    IPython.embed()


def sb5_extrapolate(analysis_options):
    session = create_session()
    sb5b_documents = session.query(Document).join(Source).\
        filter(Source.name == 'sb5b').all()
    full_corpus = MulticlassCorpus(sb5b_documents)
    full_corpus.apply_labelfunc(lambda doc: doc.label or 'Unlabeled')
    full_corpus.extract_features(lambda doc: 1, features.intercept)
    full_corpus.extract_features(lambda doc: doc.document, features.ngrams,
        ngram_max=2, min_df=2, max_df=1.0)

    full_corpus_times = np.array([doc.published for doc in full_corpus.data]).astype('datetime64[s]')

    polar_classes = [full_corpus.class_lookup[label] for label in ['For', 'Against']]
    polar_indices = np.in1d(full_corpus.y, polar_classes)
    # balanced_indices = npx.balance(
    #     full_corpus.y == full_corpus.class_lookup['For'],
    #     full_corpus.y == full_corpus.class_lookup['Against'])

    labeled_corpus = full_corpus.subset(polar_indices)
    labeled_times = full_corpus_times[polar_indices]
    # unlabeled_corpus = full_corpus.subset(full_corpus.y == full_corpus.class_lookup['Unlabeled'])

    # pos_label = corpus.class_lookup['For']
    penalty = 'l1'

    logreg_model = linear_model.LogisticRegression(fit_intercept=False, penalty=penalty)
    logreg_model.fit(labeled_corpus.X, labeled_corpus.y)
    # labeled_pred_y = logreg_model.predict(labeled_corpus.X)
    # print 'logreg_model'
    # print '  {:.2%} coefs == 0'.format((logreg_model.coef_ == 0).mean())
    # print '  accuracy on training set', metrics.accuracy_score(labeled_corpus.y, labeled_pred_y)

    # unlabeled_pred_y = logreg_model.predict(unlabeled_corpus.X)
    full_pred_y = logreg_model.predict(full_corpus.X)
    full_pred_proba = logreg_model.predict_proba(full_corpus.X)

    # histogram of for/against across entire period
    labeled_time_bounds = np.array(npx.bounds(labeled_times))

    from tsa.data.sb5b import notable_events
    notable_events_labels, notable_events_dates = zip(*notable_events)

    IPython.embed()

    full_pred_proba_max = full_pred_proba.max(axis=1)
    # full_pred_proba_hmean = npx.hmean(full_pred_proba, axis=1)
    plt.cla()
    styles = distinct_styles()
    time_hist('', full_corpus_times, full_pred_proba_max.reshape(-1, 1),
        statistic='mean', **styles.next())
    # plt.legend(loc='best')
    plt.title('Average certainty of prediction')
    plt.xlabel('Date')
    plt.axvspan(*labeled_time_bounds.astype(float), edgecolor='none', facecolor='g', alpha=0.05)
    plt.gcf().set_size_inches(8, 5)
    axes = plt.gca()
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(datetime_extra.datetime64_formatter))
    plt.savefig(figure_path('predict-proba-extrapolated.pdf'))



def time_hist(label, times, values,
        time_units_per_bin=2, time_unit='D', statistic='count', **style_args):
    bin_edges, bin_values = timeseries.binned_timeseries(
        times, values,
        time_units_per_bin=time_units_per_bin,
        time_unit=time_unit, statistic=statistic)
    plt.plot(bin_edges, bin_values, label=label, **style_args)



def sb5_extrapolate_labels():
    from tsa.data.sb5b import notable_events
    notable_events_labels, notable_events_dates = zip(*notable_events)


    # plt.vlines(notable_dates.astype(float), *auto_ylim)
    plt.legend(loc='best')
    plt.title('For / Against labels throughout corpus')
    plt.ylabel('Frequency')
    plt.xlabel('Date')
    plt.axvspan(*labeled_time_bounds.astype(float), edgecolor='none', facecolor='g', alpha=0.05)
    plt.gcf().set_size_inches(8, 5)
    plt.savefig(figure_path('for-against-extrapolated.pdf'))

    auto_ylim = plt.ylim()
    # auto_xlim = plt.xlim()
    # plt.vlines(np.array(notable_events_dates).astype('datetime64[s]').astype(float),
    #     *auto_ylim, colors='k')


    for i, (label, date) in enumerate(notable_events):
        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axvline
        x = date.astype('datetime64[s]').astype(float)
        plt.axvline(x, color='k')
        plt.text(x, auto_ylim[1]*(0.9 - i * 0.1), '- ' + label)



    IPython.embed()


def sb5_self_train(analysis_options):


    incestuous_model = linear_model.LogisticRegression(fit_intercept=False, penalty=penalty)
    incestuous_model.fit(unlabeled_corpus.X, unlabeled_pred_y)
    # apply model to data we know for sure
    incestuous_pred_y = incestuous_model.predict(labeled_corpus.X)
    # evaluate predictions
    # print metrics_summary(labeled_corpus.y, incestuous_pred_y)
    print 'accuracy on training set after extrapolation', metrics.accuracy_score(labeled_corpus.y, incestuous_pred_y)

    # we want to compare the confidence of the bootstrap on the things it gets wrong vs. a straight logistic regression

    bootstrap_model = models.Bootstrap(linear_model.LogisticRegression,
        fit_intercept=False, penalty=penalty, C=1.0)
    bootstrap_model.fit(labeled_corpus.X, labeled_corpus.y, n_iter=100, proportion=1.0)
    bootstrap_model.predict(labeled_corpus.X)

    bootstrap_mean_coef = np.mean(bootstrap_model.coefs_, axis=0)
    bootstrap_var_coef = np.var(bootstrap_model.coefs_, axis=0)
    print 'bootstrap_model'
    hist(bootstrap_mean_coef)
    print '  {:.2%} coefs == 0'.format((bootstrap_mean_coef == 0).mean())



def grid_hists(analysis_options):
    for corpus, corpus_name in iter_corpora():
        grid_hist(corpus)
        plt.title(corpus_name)
        plt.gcf().set_size_inches(8, 5)
        plt.savefig(figure_path('grid-hist-%s.pdf' % corpus_name))
        plt.cla()


def grid_hist(corpus):
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    # model = linear_model.RandomizedLogisticRegression(penalty='l2')
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html
    bootstrap_coefs = models.Bootstrap(corpus.X, corpus.y, n_iter=100, proportion=1.0, penalty='l1', C=1.0)
    coefs_means = np.mean(bootstrap_coefs, axis=0)
    coefs_variances = np.var(bootstrap_coefs, axis=0)

    bootstrap_coefs = bootstrap_model(X, y, n_iter=n_iter, proportion=0.5)
    fit_intercept=False, penalty=penalty, C=C

    logger.info('coefs_means.shape = %s, coefs_variances.shape = %s', coefs_means.shape, coefs_variances.shape)

    nonzero = coefs_means != 0
    substantial = np.abs(coefs_means) > 0.1
    print 'nonzero coef density = {:.2%}'.format(nonzero.mean())
    print '> 0.1 coef density = {:.2%}'.format(substantial.mean())
    means = coefs_means[nonzero]

    plt.cla()
    plt.hist(means, bins=25, normed=True)
    plt.xlabel('Frequency of (L1 Logistic Regression) bootstrapped coefficients')
    plt.xlim(-2, 2)


    raise IPython.embed()


def many_models(analysis_options):
    # recreate 10fold-multiple-models.pdf
    filepath = 'data/incremental-training-multiple-models-10folds.tsv'
    table = pd.io.parsers.read_table(filepath)

    # xtab = table.pivot_table(values=['accuracy'], rows=['model', 'train'], aggfunc=[len, np.mean])

    plt.cla()
    styles = distinct_styles()
    for model_name, model_table in table.groupby(['model']):
        xtab = model_table.pivot_table(values=['accuracy'], rows=['train'], aggfunc=[np.mean])
        plt.plot(xtab.index, xtab['mean'], label=model_name, **styles.next())

    plt.ylabel('Accuracy')
    plt.xlabel('# of tweets in training set')
    plt.legend(loc='best')

    # hrrm, kind of ugly

    IPython.embed()



def sample_errors(analysis_options):
    corpus = sample_corpus()
    corpus.X = corpus.X.tocsr()
    logger.info('X.shape = %s, y.shape = %s', corpus.X.shape, corpus.y.shape)

    folds = cross_validation.StratifiedShuffleSplit(corpus.y, test_size=0.5, n_iter=20)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
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
        ('Logistic Regression (L1)', linear_model.LogisticRegression(penalty='l1')),
        ('Logistic Regression (L2)', linear_model.LogisticRegression(penalty='l2')),
        # ('logistic_regression-L2-C100', linear_model.LogisticRegression(penalty='l2', C=100.0)),
        # ('randomized_logistic_regression', linear_model.RandomizedLogisticRegression()),
        # ('SGD', linear_model.SGDClassifier()),
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
    # df_agg = df.groupby(['model', 'train']).aggregate(np.mean)
    # df_agg.plot(x='train', y='accuracy')

    counts = np.array(Counter(corpus.y).values(), dtype=float)
    baseline = counts.max() / counts.sum()
    print 'baseline:', baseline
    styles = distinct_styles()
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
        style = styles.next()
        agg.plot(y='accuracy', label=index, **style)

    # IPython.embed()

    plt.legend(loc='best')
    # plt.legend(loc='lower right')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.ylim(.4, 1.0)
    plt.xlim(0, 10000)
    plt.gcf().set_size_inches(8, 5)
