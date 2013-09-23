#!/usr/bin/env python
import os
from tsa.lib import tabular, html

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
