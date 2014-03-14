import time


class Timer(object):
    '''
    import time

    with Timer() as timer:
        time.sleep(2)

    # timer.elapsed is the (floating point) number of seconds that it took
    print timer.elapsed
    '''
    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
