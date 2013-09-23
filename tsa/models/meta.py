import os

from sqlalchemy import create_engine, MetaData

connection_string = 'postgresql+psycopg2://localhost/%s' % os.getenv('TSA_DATABASE', 'tsa')
engine = create_engine(connection_string)
metadata = MetaData(bind=engine)

from sqlalchemy.orm import sessionmaker as sessionmakerfactory
sessionmaker = sessionmakerfactory(bind=engine)


class Base(object):
    def __init__(self, **kw):
        if len(kw) > 0:
            self.update(kw)

    def __json__(self):
        return dict((k, v) for k, v in self.__dict__.items() if k != '_sa_instance_state')

    def update(self, kw):
        for k, v in kw.items():
            setattr(self, k, v)

from sqlalchemy.ext.declarative import declarative_base
Model = declarative_base(metadata=metadata, cls=Base)
