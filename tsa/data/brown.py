import os

txt_filepath = os.path.join(os.getenv('CORPORA', '.'), 'browncorpus.txt')


def read(limit=None, maxlen=10000):
    with open(txt_filepath) as fp:
        for i, line in enumerate(fp):
            if i == limit:
                break

        print([line.strip()][:maxlen])
