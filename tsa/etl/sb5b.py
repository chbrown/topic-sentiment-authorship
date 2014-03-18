import os
from tsa.lib import tabular, html
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.models import Source, Document, create_session
DBSession = create_session()


def transform(row):
    label_keys = ['For', 'Against', 'Neutral', 'Broken Link', 'Not Applicable']

    if row['Tweet'] == 'Tweet' and row['Author'] == 'Author' and row['TweetID'] == 'TweetID':
        return None
    else:
        labels = [label_key for label_key in label_keys if bool(row[label_key])]
        label = None
        if len(labels) > 1:
            logger.error('Multiple labels (%s), using the first. %r' % (labels, row))
            label = labels[0]
        elif len(labels) == 1:
            label = labels[0]

        # assert that all naive datetimes are actually timezone aware (and UTC)
        # if isinstance(tweet_time, datetime):
        # from tsa.lib.datetime_extra import utc
        # row['TweetTime'] = tweet_time.replace(tzinfo=utc)

        details = {}
        details_keys = [
            'Random ID', 'ID', 'TweetID',
            'Author Count', 'Author',
            'Latitude', 'Longitude', 'TwitterPlace', 'TPlaceType',
            'Broken Link', 'Link Source', 'Sarcasm?', 'Inferred', 'Source']

        for key in details_keys:
            if row[key]:
                details[key] = row[key]

        return {
            'label': label,
            'document': html.unescape(row['Tweet']),
            'published': row['TweetTime'],
            'details': details,
        }

def run():
    xlsx_filepath = '%s/ohio/sb5-b.xlsx' % os.getenv('CORPORA', '.')

    source = Source(name='sb5b', filepath=xlsx_filepath)
    DBSession.add(source)
    # will fail on second attempt due to unique constraint
    DBSession.flush()

    for input_row in tabular.read_xlsx(xlsx_filepath):
        output_row = transform(input_row)

        if output_row:
            document = Document(source_id=source.id, **output_row)
            DBSession.add(document)
        else:
            # ignore invalid tweets (i.e., the header row)
            logger.debug('Ignoring row: %r', input_row)

    DBSession.commit()

if __name__ == '__main__':
    run()
