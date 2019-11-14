import os
import sys

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

FLOW_COLUMNS = [
    'te',   # timestamp of the end of a flow
    'td',   # duration of flow
    'sa',   # src addr
    'da',   # dst addr
    'sp',   # src port
    'dp',   # dst port
    'pr',   # proto
    'flg',  # flags
    'fwd',  # forwarding status
    'stos', # type of service
    'pkt',  # packets exchanged in the flow
    'byt',  # their corresponding num of bytes
    'lbl'
]


def filter_flows(flows, date_ranges, on='te'):

    idx = pd.Series(np.zeros(flows.shape[0])).astype(np.bool)

    for date, hours_range in date_ranges.items():
        date_from = datetime.strptime(f'{date} {hours_range[0]}', '%Y-%m-%d %H')
        date_to = datetime.strptime(f'{date} {hours_range[1]}', '%Y-%m-%d %H')

        idx |= (flows[on] >= date_from) & (flows[on] < date_to)

    return flows[idx]


# path to downloaded week 5
data_dir = Path(sys.argv[1])

mock_dir = Path('data/ugr_mock/')
# if mock_dir.exists():
#     raise FileExistsError('Test UGR16 dataset exists')

date_ranges = {
    '2016-07-30': ('04', '06'),
    '2016-07-31': ('18', '20')
}

all_flows_path = data_dir/'uniq'/'july.week5.csv.uniqblacklistremoved'

attack_flows_path = data_dir/'july'/'week5'
attack_flows = []

# Flows per type of attack
# for flows_path in attack_flows_path.iterdir():
#
#     try:
#         flows = pd.read_csv(flows_path, header=None, names=FLOW_COLUMNS)
#     except pd.errors.EmptyDataError:
#         continue
#
#     # filter flows by time
#     flows['te'] = pd.to_datetime(flows['te'], format='%Y-%m-%d %H:%M:%S')
#     flows = filter_flows(flows, date_ranges)
#
#     attack_flows.append(flows)
#
# attack_flows = pd.concat(attack_flows)


# Now sample normal flows
all_flows = pd.read_csv(all_flows_path, header=None, names=FLOW_COLUMNS, chunksize=1000000)
for chunk in all_flows:
    first_te = datetime.strptime(chunk.iloc[0]['te'], '%Y-%m-%d %H:%M:%S')
    import pdb; pdb.set_trace()
