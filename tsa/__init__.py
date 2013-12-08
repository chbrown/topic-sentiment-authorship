import sys

logging_formats = {
    'dated': '%(levelname)s\t%(asctime)s\t%(message)s',
    'interactive': '%(levelname)-8s %(message)s',
    'debugging': '%(levelname)-8s %(message)s (%(filename)s:%(lineno)d)'
}

import logging
logging.basicConfig(format=logging_formats['interactive'], level=logging.INFO)


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
