#!/usr/bin/env python
import requests
import requests.exceptions as reqexc
import socket
import urllib.parse
import sqlalchemy.exc as sqlexc
from datetime import datetime

from tsa import stdoutn
from tsa.lib import html
from tsa.models import Endpoint, create_session

import logging
logger = logging.getLogger(__name__)

whitespace_translations = dict((ord(whitespace), ' ') for whitespace in '\t\n\r')


def add_url(url, parent_id=None):
    DBSession = create_session()

    endpoint = Endpoint(url=url, parent_id=parent_id)
    DBSession.add(endpoint)

    try:
        DBSession.commit()
    except sqlexc.IntegrityError as exc:
        # simply ignore duplicates
        DBSession.rollback()
        print(exc)


def process_untried_endpoints():
    DBSession = create_session()
    # id, parent_id, url, status_code, redirect, html, content, created, accessed, timeout
    # find endpoints that aren't already fetched
    query = DBSession.query(Endpoint).\
        filter(Endpoint.status_code == None).\
        filter(Endpoint.timeout == None).\
        filter(Endpoint.error == None).\
        order_by(Endpoint.id)

    logger.info('Processing %d untried endpoints', query.count())
    while True:
        endpoint = query.first()
        if not endpoint:
            break
        print(endpoint.id, endpoint.url)

        # one of three things happens:
        try:
            # 1. set status_code
            get = requests.get(endpoint.url, allow_redirects=False, timeout=10)
            endpoint.status_code = get.status_code
            endpoint.accessed = datetime.utcnow()
            if get.status_code in [301, 302, 303]:
                endpoint.redirect = get.headers['location']
                # and add the result to the queue:
                add_url(endpoint.redirect, endpoint.id)
            else:
                endpoint.html = get.text
                # remove boilerplate from html
                endpoint.content = html.to_text(endpoint.html)

        except (socket.timeout, reqexc.Timeout):
            # 2. set endpoint.timeout
            endpoint.timeout = datetime.utcnow()
        except (reqexc.ConnectionError, reqexc.SSLError, reqexc.MissingSchema,
                reqexc.InvalidURL, reqexc.URLRequired):
            # 3. set endpoint.error
            endpoint.error = datetime.utcnow()
        except Exception:
            print(endpoint.url)
            raise

        DBSession.commit()


def tabulate(endpoints):
    stdoutn('endpoint_id\turls\tdomain\ttext')
    max_len = 65536/2 - 10
    for endpoint in endpoints:
        trail = ' -> '.join(endpoint.trail())
        domain = urllib.parse.urlparse(endpoint.url).netloc.lstrip('www.')
        text = endpoint.content.translate(whitespace_translations)

        line = '\t'.join([str(endpoint.id), trail, domain, text[:max_len]])
        stdoutn(line.encode('latin1', 'ignore'))


def analyze_content_length(endpoints):
    lengths = []
    for endpoint in endpoints:
        lengths += [len(endpoint.content)]

    # for percentile in range(
    mean = float(sum(lengths)) / float(len(lengths))
    median = sorted(lengths)[len(lengths) / 2]
    logger.info('endpoint content length: mean=%0.3f median=%0.1f', mean, median)
