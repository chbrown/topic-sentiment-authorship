import signal
import operator

item_zero = operator.itemgetter(0)


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


def sig_enumerate(seq, start=0, logger=None):
    '''
    Just like the built-in enumerate(), but also respond to SIGINFO (Ctrl-T) with a
    line of output to the given / default logger.

    Copy and pasted from iter8.generic
    '''
    if logger is None:
        import logging
        logger = logging.getLogger('SIGINFO')

    message = 'Iteration: -1'

    def handler(signum, frame):
        logger.info(message)

    logger.debug('enumerating... type Ctrl-T to show current iteration')

    signum = signal.SIGINFO
    old_handler = signal.signal(signum, handler)
    try:
        for i, x in enumerate(seq, start=start):
            message = 'Iteration: %d' % i
            yield i, x
    finally:
        # put the original signal back
        signal.signal(signum, old_handler)
