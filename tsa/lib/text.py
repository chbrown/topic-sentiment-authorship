import string

stoplist = set(['a', 'and', 'for', 'of', 'to', 'in', 'the', '-'])
stopwords = set(['a', 'and', 'for', 'of', 'to', 'in', 'the', '-', 'http'])
stopwords |= set(['is', 'on', 'that', 'i', 'are', 'you', 'this', 'it', 'your', 'as', 'at', 'be', '>>'])

punctuation2space = string.maketrans('".,;:!?\'/()[]', '             ')


def naive_tokenize(text):
    '''Yields lists of strings'''
    text = text.lower().translate(punctuation2space)
    yield [token for token in text.split() if token not in stopwords]
