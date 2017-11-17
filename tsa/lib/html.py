import re
import html.parser
import justext

html_parser = html.parser.HTMLParser()
english_stopwords = justext.get_stoplist('English')
# (?:...) means do not return this group in the match
url_re = re.compile(r'((?:https?://)?[-a-z0-9]+\.[-.a-z0-9]+/\S+)', re.I)


def to_text(html_string):
    paragraphs = justext.justext(html_string, english_stopwords)
    # with justext 1.x, this used to be:
    # return = u'\n'.join(paragraph['text'] for paragraph in paragraphs if paragraph['class'] == 'good')
    return '\n'.join(paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate)


def unescape(html_string):
    return html_parser.unescape(html_string)


def extract_urls(text):
    spaced_text = text.replace('http://', ' http://')
    for url in url_re.findall(spaced_text):
        if not url.startswith('http'):
            url = 'http://' + url
        yield url


def stdin_extract_urls():
    '''Take lines from STDIN and print out any urls to STDOUT'''
    import sys
    for line in sys.stdin:
        urls = extract_urls(line)
        for url in urls:
            print(url)
