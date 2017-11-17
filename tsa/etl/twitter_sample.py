import os
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.models import Source, Document, create_session
DBSession = create_session()

def transform(line):
    from twilight.lib import tweets
    tweet = tweets.TTV2.from_line(line)
    details = {
        'tweet_id': tweet.id,
        'coordinates': tweet.coordinates,
        'place_id': tweet.place_id,
        'place_str': tweet.place_str,
        'retweet_count': tweet.retweet_count,
        'user_screen_name': tweet.user_screen_name,
        'user_name': tweet.user_name,
    }
    return {
        'document': tweet.text[:200],  # super long links in some of these!
        'published': tweet.created_at + 'Z',
        'details': details
    }


def run():
    sample_filepath = os.path.expanduser('~/Dropbox/ut/qp/qp-2/data/twitter-sample-en.ttv2')

    source = Source(name='twitter-sample', filepath=sample_filepath)
    DBSession.add(source)
    DBSession.flush()

    with open(sample_filepath) as lines:
        # chop off headers
        headers = next(lines)
        for line in lines:
            # no label
            document = Document(source_id=source.id, **transform(line))
            # print document.__json__()
            DBSession.add(document)

    DBSession.commit()

if __name__ == '__main__':
    run()
