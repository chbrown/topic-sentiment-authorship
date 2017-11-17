'''emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects.

Source: https://github.com/brendano/tweetmotif'''

__author__ = '''Brendan O'Connor (anyall.org, brenocon@gmail.com)'''
__version__ = '''april 2009'''

import re
import sys


# SMILEY = re.compile(r'[:=].{0,1}[\)dpD]', re.UNICODE)
# MULTITOK_SMILEY = re.compile(r' : [\)dp]', re.UNICODE)


NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'  # rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE = re.compile(
    r'(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')', re.UNICODE)
Sad_RE = re.compile(NormalEyes + NoseArea + SadMouths, re.UNICODE)

Wink_RE = re.compile(Wink + NoseArea + HappyMouths, re.UNICODE)
Tongue_RE = re.compile(NormalEyes + NoseArea + Tongue, re.UNICODE)
Other_RE = re.compile('(' + NormalEyes + '|' + Wink + ')' +
                      NoseArea + OtherMouths, re.UNICODE)

Emoticon = (
    '(' + NormalEyes + '|' + Wink + ')' +
    NoseArea +
    '(' + Tongue + '|' + OtherMouths + '|' + SadMouths + '|' + HappyMouths + ')'
)
Emoticon_RE = re.compile(Emoticon, re.UNICODE)

# Emoticon_RE = '|'.join([Happy_RE, Sad_RE, Wink_RE, Tongue_RE, Other_RE])
# Emoticon_RE = re.compile(Emoticon_RE, re.UNICODE)


def analyze_tweet(text):
    h = Happy_RE.search(text)
    s = Sad_RE.search(text)
    if h and s:
        return 'BOTH_HS'
    if h:
        return 'HAPPY'
    if s:
        return 'SAD'
    return 'NA'

    # more complex & harder, so disabled for now
    # w = Wink_RE.search(text)
    # t = Tongue_RE.search(text)
    # a = Other_RE.search(text)
    # h, w, s, t, a = [bool(x) for x in [h, w, s, t, a]]
    # if sum([h, w, s, t, a]) > 1:
    #     return 'MULTIPLE'
    # if sum([h, w, s, t, a]) == 1:
    #     if h:
    #         return 'HAPPY'
    #     if s:
    #         return 'SAD'
    #     if w:
    #         return 'WINK'
    #     if a:
    #         return 'OTHER'
    #     if t:
    #         return 'TONGUE'
    # return 'NA'


def main():
    for line in sys.stdin:
        import sane_re
        sane_re._S(line[:-1]).show_match(Emoticon_RE, numbers=False)
        # print(analyze_tweet(line.strip()), line.strip(), sep='\t')


if __name__ == '__main__':
    main()
