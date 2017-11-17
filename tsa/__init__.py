import os
import sys
import logging

here = os.path.dirname(__file__) or os.curdir
root = os.path.dirname(os.path.abspath(here))

logging_formats = {
    'dated': '%(levelname)s\t%(asctime)s\t%(message)s',
    'interactive': '%(levelname)-8s %(message)s',
    'debugging': '%(levelname)-8s %(message)s (%(filename)s:%(lineno)d)',
    'original': '%(levelname)-8s %(asctime)14s (%(name)s): %(message)s',
}

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
        return '<{} name={} level={:d} (effective={:d}) parent={} disabled={:d}>'.format(
            self.__class__.__name__,
            self.name,
            self.level,
            self.getEffectiveLevel(),
            self.parent,
            self.disabled)


logging.setLoggerClass(Logger)


def stdout(string=''):
    sys.stdout.write(string)
    sys.stdout.flush()


def stdoutn(string=''):
    sys.stdout.write(string)
    sys.stdout.write('\n')
    sys.stdout.flush()


def stderr(string=''):
    sys.stderr.write(string)
    sys.stderr.flush()


def stderrn(string=''):
    sys.stderr.write(string)
    sys.stderr.write('\n')
    sys.stderr.flush()

# logging.root.info('%s initialized', __file__)
