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

