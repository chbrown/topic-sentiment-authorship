import numpy as np
import scipy
from viz.geom import hist
import logging
logger = logging.getLogger(__name__)
logger.level = 1


from sklearn import metrics


def margins(margin):
    # returns indices of the <margin> left and <margin> right elements of an array
    # (n) -> [0, 1, ..., (n - 1), -n, -(n + 1), ..., -(n - 1)]
    # so, map(string.lowercase.__getitem__, margins(3))  ->  ['a', 'b', 'c', 'x', 'y', 'z']
    # or, alphabet_array = np.array(list(string.lowercase))
    #     alphabet_array[margins(3)]  ->  np.array(['a', 'b', 'c', 'x', 'y', 'z'])
    return range(0, margin) + range(-margin, 0)


def hmean(xs):
    if (xs > 0.0).all():
        return scipy.stats.hmean(xs)
    else:
        return np.nan


def metrics_summary(y_true, y_pred):
    return ', '.join([
        'accuracy: {accuracy:.2%}',
        'P/R: {precision:.4f}/{recall:.4f}',
        'F1: {f1:.4f}',
        # '0-1 loss: {zero_one_loss:.4f}',
    ]).format(**metrics_dict(y_true, y_pred))


def metrics_dict(y_true, y_pred):
    return dict(
        accuracy=metrics.accuracy_score(y_true, y_pred),
        precision=metrics.precision_score(y_true, y_pred),
        recall=metrics.recall_score(y_true, y_pred),
        f1=metrics.f1_score(y_true, y_pred),
        # hamming loss is only different from 0-1 loss in multi-label scenarios
        # hamming_loss=metrics.hamming_loss(y_true, y_pred),
        # jaccard_similarity is only different from the accuracy in multi-label scenarios
        # jaccard_similarity=metrics.jaccard_similarity_score(y_true, y_pred),
        # zero_one_loss is (1.0 - accuracy) in multi-label scenarios
        # zero_one_loss=metrics.zero_one_loss(y_true, y_pred),
    )


def explore_mispredictions(test_X, test_y, model, test_indices, label_names, documents):
    pred_y = model.predict(test_X)
    for document_index, gold_label, pred_label in zip(test_indices, test_y, pred_y):
        if gold_label != pred_label:
            # print 'certainty: %0.4f' % certainty
            print 'gold label (%s=%s) != predicted label (%s=%s)' % (
                gold_label, label_names[gold_label], pred_label, label_names[pred_label])
            print 'Document: %s' % documents[document_index]


def explore_uncertainty(test_X, test_y, model):
    if hasattr(model, 'predict_proba'):
        pred_probabilities = model.predict_proba(test_X)
        # predicts_proba returns N rows, each C-long, where C is the number of labels
        # hmean takes the harmonic mean of its arguments
        pred_probabilities_hmean = np.apply_along_axis(hmean, 1, pred_probabilities)
        pred_certainty = 1 - (2 * pred_probabilities_hmean)
        # pred_certainty now ranges between 0 and 1,
        #   a pred_certainty of 1 means the prediction probabilities were extreme,
        #                       0 means they were near 0.5 each

        # with this, we can use np.array.argmax to get the class names we would have gotten with model.predict()
        # axis=0 will give us the max for each column (not very useful)
        # axis=1 will give us the max for each row (what we want)
        # find best guess (same as model.predict(...), I think)
        pred_y = pred_probabilities.argmax(axis=1)

        print '*: certainty mean=%0.5f' % np.mean(pred_certainty)
        hist(pred_certainty, range=(0, 1))
        print 'correct: certainty mean=%0.5f' % np.mean(pred_certainty[pred_y == test_y])
        hist(pred_certainty[pred_y == test_y], range=(0, 1))
        print 'incorrect: certainty mean=%0.5f' % np.mean(pred_certainty[pred_y != test_y])
        hist(pred_certainty[pred_y != test_y], range=(0, 1))
    else:
        logger.info('predict_proba is unavailable for this model: %s', model)
