from tsa.models import Endpoint, sessionmaker

import logging
logger = logging.getLogger(__name__)


def read(limit=None):
    '''
    Yields Endpoints (but only those with content)
    '''
    DBSession = sessionmaker()
    endpoints_with_content = DBSession.query(Endpoint).\
        filter(Endpoint.status_code == 200).\
        filter(Endpoint.content is not None).\
        filter(Endpoint.content != '').\
        order_by(Endpoint.id)
    total = endpoints_with_content.count()
    if limit:
        logger.info('Reading %d out of a total %d endpoints with content', limit, total)

    for endpoint in endpoints_with_content.limit(limit):
        yield endpoint
