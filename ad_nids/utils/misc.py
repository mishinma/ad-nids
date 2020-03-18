
from math import floor
from datetime import datetime
from functools import wraps
from ipaddress import ip_address
from timeit import default_timer as timer

import numpy as np
import pandas as pd
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


def batch_gen(data, batch_size=1024):
    n_batch = int(np.ceil(data.shape[0] / batch_size))

    for i in range(n_batch):
        yield (data[i * batch_size:(i + 1) * batch_size])


def predict_batch(od, X, batch_size=1024):
    X = batch_gen(X, batch_size=batch_size)
    pred = None
    for batch in X:
        batch_pred = od.predict(batch)
        if pred is not None:
            concatenate_preds(pred, batch_pred)
        else:
            pred = batch_pred
    return pred


def average_results(results):

    results_ave = dict()
    results_ave['time_fit'] = np.mean(
        np.array([r['time_fit'] for r in results]).astype(float),
        axis=0
    )
    results_ave['time_score_train'] = np.mean(
        np.array([r['time_score_train'] for r in results]).astype(float),
        axis=0
    )
    results_ave['time_score_test'] = np.mean(
        np.array([r['time_score_test'] for r in results]).astype(float),
        axis=0
    )
    results_ave['train_cm'] = np.mean(
        np.array([r['train_cm'] for r in results]).astype(float),
        axis=0
    ).tolist()
    results_ave['train_prf1s'] = np.mean(
        np.array([r['train_prf1s'] for r in results]).astype(float),
        axis=0
    ).tolist()
    results_ave['test_cm'] = np.mean(
        np.array([r['test_cm'] for r in results]).astype(float),
        axis=0
    ).tolist()
    results_ave['test_prf1s'] = np.mean(
        np.array([r['test_prf1s'] for r in results]).astype(float),
        axis=0
    ).tolist()
    return results_ave


def performance_asdict(cm, prf1s):

    tn, fp, fn, tp = np.array(cm).ravel()

    p = tp + fn
    n = tn + fp

    perf = dict(
        p=p,
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=round(prf1s[0], 2),
        recall=round(prf1s[1], 2),
        f1score=round(prf1s[2], 2),
    )

    return perf


def is_valid_ip(x):

    IGNORE_IPS = ['0.0.0.0', '::', 'ff:ff:ff:ff:ff:ff']

    try:
        ip_address(x)
    except ValueError:
        return False
    if x in IGNORE_IPS:
        return False
    return True


def fair_attack_sample(data, num_sample, scenario_col='scenario'):
    uniq_scenarios = data[scenario_col].unique()
    num_sample_per_scenario = floor(num_sample / len(uniq_scenarios))

    scenario2num_sample = {s: num_sample_per_scenario for s in uniq_scenarios}
    extra_sample = num_sample - num_sample_per_scenario * len(uniq_scenarios)
    for i in range(extra_sample):
        s = list(scenario2num_sample.keys())[i]
        scenario2num_sample[s] += 1

    sample = []

    groups = data.groupby(scenario_col)
    for scenario, group in groups:
        n = scenario2num_sample[scenario]
        replace = n > group.shape[0]
        scenario_sample = group.sample(n=n, replace=replace)
        sample.append(scenario_sample)

    sample = pd.concat(sample)

    return sample
