import os
import re
import glob
from tsa import logging
logger = logging.getLogger(__name__)

from tsa.models import Source, Document, create_session
DBSession = create_session()


def run():
    dirpath = os.path.expanduser('~/corpora-public/convote-v1.1/data_stage_one')

    source = Source(name='convote', filepath=dirpath)
    DBSession.add(source)
    DBSession.flush()

    for filepath in glob.glob(os.path.join(dirpath, '*/*.txt')):
        # filename format: ###_@@@@@@_%%%%$$$_PMV
        #            e.g., 052_400011_0327044_DON
        print(filepath)
        m = re.search(r'\d{3}_\d{6}_\d{7}_(I|D|R|X)(M|O)(Y|N).txt$', filepath)
        party, mentioned, vote = m.groups()
        label = 'For' if vote == 'Y' else 'Against'

        with open(filepath) as fd:
            text = ' '.join(line.strip() for line in fd)
            # print label, text
            document = Document(source_id=source.id, label=label, document=text)
            DBSession.add(document)

    DBSession.commit()

if __name__ == '__main__':
    run()
