import signal


# from operator import itemgetter
# item_zero = itemgetter(0)
# item_zero = lambda obj: obj.__getitem__[0]
item_zero = lambda tup: tup[0]


def take(iterable, n=10):
    last_index = n - 1
    # return itertools.islice(iterable, n)
    for index, item in enumerate(iterable):
        yield item
        if index == last_index:
            break


class Quota(object):
    '''
    Alternative names:
      first_of
      take_classes
      ...
    '''
    def __init__(self, **needed_keys):
        self.needed_keys = needed_keys
        self._count()

    def add(self, key):
        needed = self.needed_keys.get(key, 0)
        if needed > 0:
            self.needed_keys[key] = needed - 1
            self._count()
            return True
        return False

    def _count(self):
        self.filled = sum(self.needed_keys.values()) == 0

    def filter(self, items, keyfunc=item_zero):
        '''
        yield everything from items until this quota is filled

        `keyfunc` will be applied to each item in `items` and should return the key / class of that item.
          `keyfunc` defaults to `item[0]`
        '''
        for item in items:
            key = keyfunc(item)
            if self.add(key):
                yield item
            if self.filled:
                break
        else:
            # raise ValueError('Iterator stopped before quota was filled. Needed: %s' % self.needed_keys)
            # nevermind, don't raise an Error
            pass

    # todo: add filter_with_key which would yield (key, item) pairs


def sig_enumerate(iterable, start=0, logger=None):
    '''
    Just like the built-in enumerate(), but also respond to SIGINFO with a
    line of output to the given / default logger.
    '''
    if logger is None:
        from tsa import logging
        logger = logging.getLogger('SIGINFO')

    message = 'Iteration: -1'

    def handler(signum, frame):
        logger.info(message)

    logger.silly('enumerating... type Ctrl-T to show current iteration')

    signum = signal.SIGINFO
    old_handler = signal.signal(signum, handler)
    try:
        for i, x in enumerate(iterable, start=start):
            message = 'Iteration: %d' % i
            yield i, x
    finally:
        # put the original signal back
        signal.signal(signum, old_handler)
