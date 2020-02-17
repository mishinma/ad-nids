
from datetime import datetime
from functools import wraps
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf


def set_seed(s=0):
    np.random.seed(s)
    tf.random.set_seed(s)


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


def dd_mm_yyyy2mmdd(dates):
    return [datetime.strptime(d, '%d-%m-%Y').strftime('%m%d') for d in dates]


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


def sample_df(df, n):
    """ Sample n instances from the dataframe df. """
    if n < df.shape[0]+1:
        replace = False
    else:
        replace = True
    return df.sample(n=n, replace=replace)


def concatenate_preds(preds, other_preds):
    if preds['data'].get('feature_score') is not None:
        preds['data']['feature_score'] = np.concatenate(
            [preds['data']['feature_score'],  other_preds['data']['feature_score']]
        )
    preds['data']['instance_score'] = np.concatenate(
        [preds['data']['instance_score'], other_preds['data']['instance_score']]
    )
    preds['data']['is_outlier'] = np.concatenate(
        [preds['data']['is_outlier'], other_preds['data']['is_outlier']]
    )
    return preds


def predict_batch(od, X, batch_size=64):
    X = tf.data.Dataset.from_tensor_slices(X)
    X = X.batch(batch_size)
    pred = None
    for batch in X:
        batch_pred = od.predict(batch)
        if pred is not None:
            concatenate_preds(pred, batch_pred)
        else:
            pred = batch_pred
    return pred
