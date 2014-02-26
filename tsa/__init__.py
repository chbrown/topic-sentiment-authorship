import sys

logging_formats = {
    'dated': '%(levelname)s\t%(asctime)s\t%(message)s',
    'interactive': '%(levelname)-8s %(message)s',
    'debugging': '%(levelname)-8s %(message)s (%(filename)s:%(lineno)d)',
    'original': '%(levelname)-8s %(asctime)14s (%(name)s): %(message)s',
}

import logging
logging.captureWarnings(True)
logging.basicConfig(format=logging_formats['interactive'], level=logging.INFO)


# NOTSET = 0, DEBUG = 10
SILLY = 5
logging.addLevelName(SILLY, 'SILLY')


class Logger(logging.Logger):
    '''
    Mostly from Tweedr
    '''
    def silly(self, msg, *args, **kwargs):
        if self.isEnabledFor(SILLY):
            self._log(SILLY, msg, args, **kwargs)

    def notset(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.NOTSET):
            self._log(logging.NOTSET, msg, args, **kwargs)

    def __repr__(self):
        return '<%s name=%s level=%d (effective=%d) parent=%s disabled=%d>' % (self.__class__.__name__,
            self.name, self.level, self.getEffectiveLevel(), self.parent, self.disabled)


logging.setLoggerClass(Logger)


def stdout(bytes=''):
    sys.stdout.write(bytes)
    sys.stdout.flush()


def stdoutn(bytes=''):
    sys.stdout.write(bytes)
    sys.stdout.write('\n')
    sys.stdout.flush()


def stderr(bytes=''):
    sys.stderr.write(bytes)
    sys.stderr.flush()


def stderrn(bytes=''):
    sys.stderr.write(bytes)
    sys.stderr.write('\n')
    sys.stderr.flush()

# logging.root.info('%s initialized', __file__)
