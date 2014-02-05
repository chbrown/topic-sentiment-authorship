import re
import string

determiners = set(['a', 'an', 'the'])
conjunctions = set(['and', 'or', 'but'])
prepositions = set(['for', 'to', 'in', 'at', 'as'])
pronouns = set(['you', 'this', 'it', 'your'])
punctuation = set(['-', 'http', '>>'])
stopwords = determiners | conjunctions | prepositions | pronouns | punctuation | set(['be'])

punctuation2space = string.maketrans('".,;:!?\'/()[]', '             ')


def naive_tokenize(text):
    '''Yields lists of strings'''
    text = text.lower().translate(punctuation2space)
    yield [token for token in text.split() if token not in stopwords]


def hashtags(text, case_sensitive=False):
    '''yield every hashtag in text'''
    if not case_sensitive:
        text = text.lower()
    for match in re.finditer(ur'#\w+', text):  # , flags=re.UNICODE
        yield match.group(0)
