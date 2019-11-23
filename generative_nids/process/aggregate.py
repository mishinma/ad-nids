
import multiprocessing as mp

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm


# Baseline Feature Set
# ToDo: move to another module and unit test
COMPUTE_FLOW_STATS = OrderedDict([
    ('num_f', lambda f: f.shape[0]),
    ('num_uniq_da', lambda f: f['da'].unique().shape[0]),
    ('num_uniq_dp', lambda f: f['dp'].unique().shape[0]),
    ('num_uniq_sp', lambda f: f['sp'].unique().shape[0]),
    ('entropy_da', lambda f: entropy(f['da'].value_counts())),
    ('entropy_dp', lambda f: entropy(f['dp'].value_counts())),
    ('entropy_sp', lambda f: entropy(f['sp'].value_counts())),
    ('avg_td', lambda f: np.mean(f['td'])),
    ('std_td', lambda f: np.std(f['td'])),
    ('avg_pkt', lambda f: np.mean(f['pkt'])),
    ('std_pkt', lambda f: np.std(f['pkt'])),
    ('avg_byt', lambda f: np.mean(f['byt'])),
    ('std_byt', lambda f: np.std(f['byt'])),
    ('lbl', lambda f: int(bool(np.sum(f['lbl']))))
])

FLOW_STATS_COLUMNS = ['sa', 'tws'] + list(COMPUTE_FLOW_STATS.keys())


def aggregate_extract_features(flows, freq='5T', processes=1):
    grouped = flows.groupby(['sa', pd.Grouper(key='ts', freq=freq)])
    pbar = tqdm(total=len(grouped))

    records = []

    def log_progress(result):
        records.append(result)
        pbar.update(1)

    if processes > 1:
        pool = mp.Pool(processes=processes)
        for name, grp in grouped:
            pool.apply_async(
                extract_features_wkr, ((name, grp),),
                callback=log_progress)
        pool.close()
        pool.join()
    else:
        for name, grp in grouped:
            res = extract_features_wkr((name, grp))
            log_progress(res)

    processed = pd.DataFrame.from_records(records, columns=FLOW_STATS_COLUMNS)
    return processed


def extract_features_wkr(arg):

    grp_name, flow_grp = arg

    flow_stats = []

    for f in COMPUTE_FLOW_STATS .values():
        flow_stats.append(f(flow_grp))

    record = (
        grp_name[0], grp_name[1], *flow_stats
    )

    return record

