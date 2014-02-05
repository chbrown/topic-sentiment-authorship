import sys
from tsa.science.text import hashtags


def main():
    for line in sys.stdin:
        for hashtag in hashtags(line):
            print hashtag
