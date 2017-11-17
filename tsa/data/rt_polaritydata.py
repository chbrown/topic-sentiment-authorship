import os.path
from tsa import logging
from tsa.science.corpora import MulticlassCorpus

logger = logging.getLogger(__name__)


def read_MulticlassCorpus():
    dirpath = os.path.expanduser('~/corpora-public/bopang_lillianlee/rt-polaritydata/')

    def read_file(filename, label):
        with open(os.path.join(dirpath, filename)) as fd:
            for line in fd:
                yield (label, line)

    data = list(read_file('rt-polarity.neg.utf8', 'neg')) + list(read_file('rt-polarity.pos.utf8', 'pos'))
    # data is now a list of (label:string, document:string) tuples

    corpus = MulticlassCorpus(data)
    corpus.apply_labelfunc(lambda tup: tup[0])
    # logger.debug('rt-polaritydata MulticlassCorpus created: %s', corpus.X.shape)
    return corpus
