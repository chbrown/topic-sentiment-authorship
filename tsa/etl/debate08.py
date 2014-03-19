from collections import Counter
import csv
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.models import Source, Document, create_session
DBSession = create_session()

rating_codes = {
    '1': 'negative',
    '2': 'positive',
    '3': 'mixed',
    '4': 'other',
}

rating_scores = {
    '1': -1,
    '2': 1,
    '3': 0,
    '4': 0,
    '': 0,
    None: 0,
}

def transform(record):
    ratings = [record['rating.%d' % i] for i in range(1, 9)]
    # most_common_rating, most_common_rating_count = Counter(ratings).most_common(1)[0]
    # label = rating_codes[most_common_rating]
    score = sum(rating_scores[rating] for rating in ratings)
    label = 'Neutral'
    if score > 0:
        label = 'Positive'
    elif score < 0:
        label = 'Negative'

    details = {
        'tweet_id': record['tweet.id'],
        'screen_name': record['author.name'],
        'name': record['author.nickname'],
        'ratings': ratings,
        'score': score,
    }
    return {
        'label': label,
        'document': record['content'],
        'published': record['pub.date.GMT'],
        'details': details
    }


def run():
    filepath = '/Users/chbrown/corpora-public/debate08_sentiment_tweets.tsv'

    source = Source(name='debate08', filepath=filepath)
    DBSession.add(source)
    DBSession.flush()

    with open(filepath, 'rU') as fd:
        reader = csv.DictReader(fd, delimiter='\t', dialect='excel')
        for record in reader:
            document = Document(source_id=source.id, **transform(record))
            print document.__json__()
            DBSession.add(document)

    DBSession.commit()

if __name__ == '__main__':
    run()
