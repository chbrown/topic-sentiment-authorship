#!/usr/bin/env python
import os
from datetime import tzinfo, datetime, timedelta
from tsa.lib import tabular, html

import logging
logger = logging.getLogger(__name__)


class UTC(tzinfo):
    def tzname(self, dt):
        return 'UTC'

    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def __repr__(self):
        return 'UTC[%#x]' % id(utc)

utc = UTC()

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
        # assert that all naive datetimes are actually timezone aware (and UTC)
        tweet_time = row['TweetTime']
        if isinstance(tweet_time, datetime):
            row['TweetTime'] = tweet_time.replace(tzinfo=utc)

        yield row
