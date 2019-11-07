
import multiprocessing as mp

import numpy as np
import pandas as pd

from tqdm import tqdm

FLOW_COLUMNS = [
    'ts',   # timestamp of the start of a flow
    'td',   # duration of flow
    'sa',   # src addr
    'da',   # dst addr
    'sp',   # src port
    'dp',   # dst port
    'pr',   # proto
    'pkt',  # num packets exchanged in the flow
    'byt',  # their corresponding num of bytes
    'lbl',  # 0 norm, 1 anomaly
]

FLOW_STAT_COLUMNS = [
    'sa',   # src addr
    'tws',  # time window start
    'nf',   # num flows in a window
    'sum_byt',  # sum total bytes
    'avg_byt',  # avg total bytes
    'avg_comm_t',  # avg communication time
    'n_uniq_da',  # num unique dest addr
    'n_uniq_dp',  # num unique dest port
    'freq_pr',  # most freq proto
    'lbl'
]


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
        # for name, grp in grouped.groups.items():
        for name, grp in grouped:
            res = extract_features_wkr((name, grp))
            log_progress(res)

    processed = pd.DataFrame.from_records(records, columns=FLOW_STAT_COLUMNS)
    return processed


def extract_features_wkr(arg):

    grp_name, flow_grp = arg

    # number of flows
    num_flows = flow_grp.shape[0]

    # sum of transferred bytes
    sum_tot_bytes = np.sum(flow_grp['byt'])

    # average sum of bytes per flow?
    avg_bytes_per_flow = sum_tot_bytes/num_flows

    # average communication time with each unique IP addresses
    avg_comm_time_unique = np.mean(flow_grp.groupby('da')['td'].sum())

    # num of unique dest IP addresses
    num_dest_ip = flow_grp['da'].unique().shape[0]

    # num of unique dest ports
    num_dest_port = flow_grp['dp'].unique().shape[0]

    # most freq used proto
    freq_proto = flow_grp['pr'].mode()[0]

    # label
    label = (np.sum(flow_grp['lbl']) > 0).astype(np.int)  # Normal vs anomalous

    record = (
        grp_name[0], grp_name[1], num_flows, sum_tot_bytes, avg_bytes_per_flow,
        avg_comm_time_unique, num_dest_ip, num_dest_port, freq_proto, label
    )

    return record
