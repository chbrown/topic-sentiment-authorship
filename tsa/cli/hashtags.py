import sys
from tsa.lib.text import hashtags


def main():
    for line in sys.stdin:
        for hashtag in hashtags(line):
            print hashtag
