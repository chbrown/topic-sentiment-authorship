from sqlalchemy import Table, orm

from tsa.models.meta import Model, metadata, create_session

# DBSession = create_session()


class Endpoint(Model):
    __table__ = Table('endpoints', metadata, autoload=True)

    def trail(self, session):
        # trail includes the original url at the far right of the trail
        if self.parent_id:
            parent = Session.query(Endpoint).get(self.parent_id)
            # build to the left, so that we end up with something like:
            #     [grandparent, parent, self]
            return parent.trail() + [self.url]
        return [self.url]

    # @classmethod
    # def resolve(cls, url, fix=False):
    #     cache_key = 'tsa:links:%s' % url
    #     import redis
    #     r = redis.StrictRedis()
    #     cached = r.get(cache_key)
    #     if not cached:
    #         while True:
    #             found = DBSession.query(Endpoint).filter(Endpoint.url == url).first()
    #             # trim off the left side http://www.
    #             if found is None or found.redirect is None:
    #                 break
    #             url = found.redirect
    #         r.set(cache_key, url)
    #         cached = url
    #     if fix and not cached.startswith('http'):
    #         cached = 'http://' + cached
    #     return cached


class Source(Model):
    __table__ = Table('sources', metadata, autoload=True)

    @classmethod
    def from_name(cls, name):
        session = create_session()
        return session.query(Document).\
            join(Source, Source.id == Document.source_id).\
            filter(Source.name == name).\
            filter(Document.label != None).\
            order_by(Document.published).all()



class Document(Model):
    __table__ = Table('documents', metadata, autoload=True)

    source = orm.relationship(Source)
