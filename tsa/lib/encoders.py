import json
from datetime import datetime


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        # return super(JSONEncoder, self).default(obj)
        return obj


# encoder = JSONEncoder()
# def json(obj):
#     return encoder.encode(obj)
# c'mon, just DIY


def csv(obj):
    return ','.join(map(str, obj))
