import re
import string

determiners = {'a', 'an', 'the'}
conjunctions = {'and', 'or', 'but'}
prepositions = {'for', 'to', 'in', 'at', 'as'}
pronouns = {'you', 'this', 'it', 'your'}
punctuation = {'-', 'http', '>>'}
stopwords = determiners | conjunctions | prepositions | pronouns | punctuation | {'be'}

punctuation2space = string.maketrans('".,;:!?\'/()[]', '             ')


def naive_tokenize(text):
    '''Yields lists of strings'''
    text = text.lower().translate(punctuation2space)
    yield [token for token in text.split() if token not in stopwords]


def hashtags(text, case_sensitive=False):
    '''yield every hashtag in text'''
    if not case_sensitive:
        # by default, it'll lowercase the input
        text = text.lower()
    for match in re.finditer(r'#\w+', text):  # , flags=re.UNICODE
        yield match.group(0)
