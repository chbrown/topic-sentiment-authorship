import json
from datetime import datetime


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        return json.JSONEncoder.default(self, obj)

encoder = JSONEncoder()


def json(obj):
    return encoder.encode(obj)


def csv(obj):
    return ','.join(map(str, obj))
