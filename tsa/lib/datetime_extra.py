from datetime import datetime, timedelta, tzinfo


class UTC(tzinfo):
    def tzname(self, dt):
        return 'UTC'

    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def __repr__(self):
        return 'UTC[%#x]' % id(utc)

utc = UTC()


def datetime_to_yyyymmdd(date, *args, **kw):
    '''Used by tick formatter (thus the variable args)'''
    return date.strftime('%Y-%m-%d')


def epoch_to_yyyymmdd(x, *args, **kw):
    '''Used by tick formatter (thus the variable args)'''
    return datetime_to_yyyymmdd(datetime.fromtimestamp(x))


def datetime64_formatter(x, pos=None):
    # matplotlib converts to floats, internally, so we have to convert back out
    return datetime_to_yyyymmdd(x.astype('datetime64[s]').astype(datetime))


# import matplotlib.dates as mdates
# def mdate_formatter(num, pos=None):
#     return datetime_to_yyyymmdd(mdates.num2date(num))
