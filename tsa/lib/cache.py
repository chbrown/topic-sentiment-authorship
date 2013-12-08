import os
import string
import cPickle

import logging
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
        logger.info('Reading pickled object from file: %s', filepath)
        pickle_fp = open(filepath, 'rb')
        result = cPickle.load(pickle_fp)
    else:
        logger.info('Executing pickle-able function')
        result = func(*args, **kwargs)

        logger.info('Writing pickled object to file: %s', filepath)
        pickle_fp = open(filepath, 'wb')
        cPickle.dump(result, pickle_fp)
    return result


# wrap and decorate are just composers of different ways of calling hit


def wrap(func, filepath_format):
    '''A function wrapper. If you previously called something like:

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
