import IPython
# import numpy as np

import subprocess
import tempfile
from sklearn import datasets

def mallet(corpus, num_topics=100):
    '''
    From Mallet's docs:

    ## Importing (http://mallet.cs.umass.edu/import.php)

    One file, one instance per line: Assume the data is in the following format:

        [URL]  [language]  [text of the page...]

    After downloading and building MALLET, change to the MALLET directory and run the following command:

        bin/mallet import-file --input /data/web/data.txt --output web.mallet

    In this case, the first token of each line (whitespace delimited, with optional comma) becomes the instance name, the second token becomes the label, and all additional text on the line is interpreted as a sequence of word tokens.

    ### SVMLight format: SVMLight-style data in the format

        target feature:value feature:value ...

    can be imported with

        bin/mallet import-svmlight --input train test --output train.mallet test.mallet

    Note that the input and output arguments can take multiple files that are processed together using the same Pipe. Note that the target and feature fields can be either indices or strings. If they are indices, note that the indices in the Mallet alphabets and indices in the file may be different, though the data is equivalent. Real valued targets are not supported.

    ## Modeling (http://mallet.cs.umass.edu/topics.php)

    bin/mallet train-topics --input topic-input.mallet --num-topics 100 --output-state topic-state.gz

    '''
    extra_stopwords = ['http', 'https', 't', 'co', 'bit', 'ly']

    # _, svmlight_tempfile_path = tempfile.mkstemp(suffix='.svmlight')
    # print 'writing svmlight file', svmlight_tempfile_path
    # datasets.dump_svmlight_file(corpus.X, corpus.y, svmlight_tempfile_path, zero_based=False)
    _, data_tempfile_path = tempfile.mkstemp(suffix='.txt')
    print 'writing {:d} documents to {:s}'.format(len(corpus.data), data_tempfile_path)
    with open(data_tempfile_path, 'w') as data_tempfile_fd:
        for datum in corpus.data:
            print >> data_tempfile_fd, datum.id, datum.label, datum.document.encode('utf-8')

    _, mallet_tempfile_path = tempfile.mkstemp(suffix='.mallet')
    _, stopwords_tempfile_path = tempfile.mkstemp(suffix='.stopwords')
    with open(stopwords_tempfile_path, 'w') as stopwords_tempfile_fd:
        stopwords_tempfile_fd.write('\n'.join(extra_stopwords))

    print 'writing mallet format to {:s}'.format(mallet_tempfile_path)
    print subprocess.check_output(['mallet', 'import-file',
        '--keep-sequence',  # required for topic modeling
        '--token-regex', r'#?\w+',
        '--remove-stopwords',
        '--extra-stopwords', stopwords_tempfile_path,
        '--input', data_tempfile_path,
        '--output', mallet_tempfile_path])

    _, state_tempfile_path = tempfile.mkstemp(suffix='.gz')
    _, topic_keys_tempfile_path = tempfile.mkstemp(suffix='.txt')
    _, doc_topics_tempfile_path = tempfile.mkstemp(suffix='.txt')
    print 'state file = {:s}, topic keys = {:s}, doc topics = {:s}'.format(
        state_tempfile_path, topic_keys_tempfile_path, doc_topics_tempfile_path)
    subprocess.Popen(['mallet', 'train-topics',
        '--input', mallet_tempfile_path,
        '--num-topics', str(num_topics),
        '--optimize-interval', '10',
        # '--optimize-burn-in', '20',
        # '--num-iterations', str(num_iterations),
        '--output-state', state_tempfile_path,
        '--output-topic-keys', topic_keys_tempfile_path,
        '--output-doc-topics', doc_topics_tempfile_path,
    ])

    IPython.embed()
