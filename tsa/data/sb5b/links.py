from tsa.models import Endpoint, sessionmaker

import logging
logger = logging.getLogger(__name__)


def read(limit=None):
    '''Yields Endpoints (but only those with content)'''
    DBSession = sessionmaker()
    endpoints_with_content = DBSession.query(Endpoint).\
        filter(Endpoint.status_code == 200).\
        filter(Endpoint.content is not None).\
        filter(Endpoint.content != '').\
        order_by(Endpoint.id)
    logger.info('Found %d endpoints with content', endpoints_with_content.count())

    for endpoint in endpoints_with_content.limit(limit):
        yield endpoint
