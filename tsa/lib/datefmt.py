from datetime import datetime


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
