
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from generative_nids.process.columns import FLOW_STATS, FLOW_STATS_COLUMNS


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

    for f in FLOW_STATS .values():
        flow_stats.append(f(flow_grp))

    record = (
        grp_name[0], grp_name[1], *flow_stats
    )

    return record

