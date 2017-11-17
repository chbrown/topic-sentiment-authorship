import csv
from tsa import logging
from tsa.models import Source, Document, create_session

logger = logging.getLogger(__name__)

DBSession = create_session()


def transform(record):
    score = float(record['Normalized Score'])
    details = {
        'Id': record['Id'],
        'Community': record['Community'],
        'Score1': record['Score1'],
        'Score2': record['Score2'],
        'Score3': record['Score3'],
        'Score4': record['Score4'],
        'Score5': record['Score5'],
        'Normalized Score': score,
    }
    return {
        'label': 'Polite' if score > 0 else 'Impolite',
        'document': record['Request'],
        'details': details
    }


def read_source(name, filepath):
    source = Source(name=name, filepath=filepath)
    DBSession.add(source)
    DBSession.flush()

    with open(filepath, 'rU') as fd:
        reader = csv.DictReader(fd, dialect='excel')
        for record in reader:
            document = Document(source_id=source.id, **transform(record))
            # print document.__json__()
            DBSession.add(document)

    DBSession.commit()

def run():
    read_source('stanford-politeness-wikipedia', '/Users/chbrown/corpora-public/Stanford_politeness_corpus/wikipedia.annotated.csv')
    read_source('stanford-politeness-stackexchange', '/Users/chbrown/corpora-public/Stanford_politeness_corpus/stack-exchange.annotated.csv')

if __name__ == '__main__':
    run()
