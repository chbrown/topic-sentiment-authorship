import os
import cPickle

import logging
logger = logging.getLogger(__name__)


def pickleable(file_pattern):
    '''A function helper. Use like:

    @pickle('tmp/longrunner-%(hashtag)s-%(limit)d.pyckle')
    def get_tweets(hashtag='hcr', limit=1000):
        ... go get some tweets and return them as a plain dict or list ...

    For all we care, the function should return the same thing
    everytime it is run with any particular combination of keyword arguments.

    Only supports functions with keyword arguments.
    '''
    def decorator(func):
        # print 'pickleable decorator', func
        # *args,
        def wrapper(**kw):
            # print 'pickleable wrapper', kw
            file_path = file_pattern % kw
            if os.path.exists(file_path):
                logger.info('Reading pickled object from file: %s', file_path)
                pickle_fp = open(file_path, 'rb')
                result = cPickle.load(pickle_fp)
            else:
                logger.info('Executing pickle-able function')
                result = func(**kw)

                logger.info('Writing pickled object to file: %s', file_path)
                pickle_fp = open(file_path, 'wb')
                cPickle.dump(result, pickle_fp)
            return result
        return wrapper
    return decorator
