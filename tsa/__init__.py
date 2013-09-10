import sys


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
