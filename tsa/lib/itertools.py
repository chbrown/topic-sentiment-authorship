def take(iterable, n=10):
    last_index = n - 1
    # return itertools.islice(iterable, n)
    for index, item in enumerate(iterable):
        yield item
        if index == last_index:
            break


class Quota(object):
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

    def filter(self, items):
        '''
        yield everything from items until this quota is filled

        Each item in items should be a tuple, the first entry of which is the key
        '''
        for item in items:
            if self.add(item[0]):
                yield item
            if self.filled:
                break
