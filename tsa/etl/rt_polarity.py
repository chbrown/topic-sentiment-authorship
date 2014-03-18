import os
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.models import Source, Document, create_session
DBSession = create_session()


def run():
    dirpath = os.path.expanduser('~/corpora-public/bopang_lillianlee/rt-polaritydata/')

    source = Source(name='rt-polarity', filepath=dirpath)
    DBSession.add(source)
    DBSession.flush()

    files = [
        ('rt-polarity.neg', 'neg'),
        ('rt-polarity.pos', 'pos'),
    ]

    for filename, label in files:
        filepath = os.path.join(dirpath, filename)
        with open(filepath) as fd:
            for line in fd:
                text = line.decode('cp1252').strip()
                document = Document(source_id=source.id, label=label, document=text)
                DBSession.add(document)

    DBSession.commit()

if __name__ == '__main__':
    run()
