#!/usr/bin/env python
import os
from tsa.lib import tabular, html

import logging
logger = logging.getLogger(__name__)


xlsx_filepath = '%s/ohio/sb5-b.xlsx' % os.getenv('CORPORA', '.')
label_keys = ['For', 'Against', 'Neutral', 'Broken Link', 'Not Applicable']


def read(limit=None):
    '''Yields dicts with at least 'Labels' and 'Tweet' fields.'''
    for row in tabular.read_xlsx(xlsx_filepath, limit=limit):
        for label_key in label_keys:
            row[label_key] = bool(row[label_key])

        row['Labels'] = [label_key for label_key in label_keys if row[label_key]]
        row['Label'] = (row['Labels'] + ['NA'])[0]
        row['Tweet'] = html.unescape(row['Tweet'])

        yield row


def read_cached(limit=None):
    import cPickle as pickle
    pickle_filepath = '%s.pickled-%s' % (xlsx_filepath, limit or 'all')

    if os.path.exists(pickle_filepath):
        logger.info('Loading pickled sb5b tweets file from %s', pickle_filepath)
        pickle_file = open(pickle_filepath, 'rb')
        for item in pickle.load(pickle_file):
            yield item
    else:
        logger.info('Reading fresh sb5b tweets')
        items = list(read(limit=limit))

        logger.info('Pickling sb5b tweets to %s', pickle_filepath)
        pickle_file = open(pickle_filepath, 'wb')
        pickle.dump(items, pickle_file)
        for item in items:
            yield item
