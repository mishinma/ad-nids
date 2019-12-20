
from datetime import datetime
from functools import wraps
from timeit import default_timer as timer

import numpy as np


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


def int_to_roman(input):
    """ Convert an integer to a Roman numeral. """

    input = int(input)
    if not 0 < input < 4000:
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result)


def jsonify(data):

    if isinstance(data, dict):
        json_data = {k: jsonify(v) for k, v in data.items()}
    elif isinstance(data, list):
        json_data = [jsonify(v) for v in data]
    elif isinstance(data, np.ndarray):
        json_data = data.tolist()
    else:
        json_data = data

    return json_data
