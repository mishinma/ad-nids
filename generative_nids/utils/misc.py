
from datetime import datetime
from functools import wraps
from timeit import default_timer as timer


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = timer()
        result = f(*args, **kw)
        te = timer()
        elapsed = te - ts
        return result, elapsed
    return wrap


def yyyy_mm_dd2mmdd(dates):
    return [datetime.strptime(d, '%Y-%m-%d').strftime('%m%d') for d in dates]
