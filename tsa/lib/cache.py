import os
import string
import pickle

from tsa import logging
logger = logging.getLogger(__name__)
logger.level = 1


class FilepathFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        # print 'FilepathFormatter.format_field', value, format_spec
        if format_spec == 'p':
            # sanitize the filepath
            return value.lstrip('/').replace('/', '-')
        return super(FilepathFormatter, self).format_field(value, format_spec)

    # def convert_field(self, value, conversion):
    #     print 'FilepathFormatter.convert_field', value, conversion
    #     return super(FilepathFormatter, self).convert_field(value, conversion)


def hit(filepath_format, func, *args, **kwargs):
    # import inspect
    # argspec = inspect.getargspec(func)
    # if len(argspec.args) != len(argspec.defaults) or argspec.varargs is not None or argspec.keywords is not None:
    #     raise ValueError('Cannot cache a function with ....')
    filepath = FilepathFormatter().format(filepath_format, *args, **kwargs)
    if os.path.exists(filepath):
        logger.debug('CACHE HIT (reading pickled object from file: %s)', filepath)
        pickle_fp = open(filepath, 'rb')
        result = pickle.load(pickle_fp)
        logger.debug('CACHE DONE')
    else:
        logger.info('CACHE MISS (executing function)')
        result = func(*args, **kwargs)

        # ensure that result is pickleable: flatten iterators to lists
        # TODO: be smarter about this?
        # right now the only types of iterables we don't flatten to a list are lists and dicts
        if hasattr(result, '__iter__') and not isinstance(result, list) and not isinstance(result, dict):
            result = list(result)

        pickle_fp = open(filepath, 'wb')
        pickle.dump(result, pickle_fp)
        logger.debug('CACHE DONE (wrote pickled object to file: %s)', filepath)
    return result


# wrap and decorate are just composers of different ways of calling hit


def wrap(func, filepath_format):
    '''A function wrapper. If you call something like:

        get_tweets(hashtag='hcr', limit=1000):
        # does some I/O and returns a plain dict or list

    Now use this:

        get_tweets = cache.wrap(get_tweets, '/tmp/tweets-{hashtag}-{limit}.pickle')
        get_tweets(hashtag='hcr', limit=1000):

    For all we care, the function should return the same thing
    everytime it is run with any particular combination of keyword arguments.

    Only supports functions with keyword arguments.
    '''
    return decorate(filepath_format)(func)


def decorate(filepath_format):
    '''A function decorator. Use like:

    @cache.decorate('/tmp/tweets-{hashtag}-{limit}.pickle')
    def get_tweets(hashtag='hcr', limit=1000):
        #... go get some tweets and return them as a plain dict or list ...

    For all we care, the function should return the same thing
    everytime it is run with any particular combination of keyword arguments.

    Only supports functions with keyword arguments.
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            return hit(filepath_format, func, *args, **kwargs)
        return wrapper
    return decorator
