import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from tsa.lib.encoders import JSONEncoder

engine = create_engine('postgresql://localhost/%s' % os.getenv('TSA_DATABASE', 'tsa'))
metadata = MetaData(bind=engine)
sessions = sessionmaker(bind=engine)


def create_session():
    return scoped_session(sessions)


class Base(object):
    def __init__(self, **kw):
        if kw:
            self.update(kw)

    def __json__(self):
        return dict((k, v) for k, v in self.__dict__.items() if k != '_sa_instance_state')

    def update(self, kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return JSONEncoder(indent=2, sort_keys=True).encode(self.__json__())

Model = declarative_base(metadata=metadata, cls=Base)
