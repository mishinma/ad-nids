
import shutil

from pathlib import Path
from collections import defaultdict

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


def split_ugr_flows(all_flows_path, split_root_path):

    split_root_path = Path(split_root_path)
    all_flows = pd.read_csv(all_flows_path, header=None,
                        names=FLOW_COLUMNS, chunksize=1000000)

    for i, chunk in enumerate(all_flows):
        chunk['te'] = pd.to_datetime(chunk['te'], format='%Y-%m-%d %H:%M:%S')

        for tstmp, grp in chunk.resample('H', on='te'):

            if grp.empty:
                continue

            grp_date = tstmp.strftime('%Y-%m-%d')
            grp_hour = tstmp.strftime('%H')

            grp_path = split_root_path / grp_date / grp_hour
            grp_path.mkdir(parents=True, exist_ok=True)

            grp_name = f'{grp.index[0]}_{grp.index[-1]}.csv'
            grp.to_csv(grp_path/grp_name, index=None)

    # Combine chunks
    date_hour2chunk_paths = defaultdict(list)

    for chunk_path in split_root_path.glob('**/*.csv'):
        date_hour = tuple(str(chunk_path).split('/')[-3: -1])
        date_hour2chunk_paths[date_hour].append(chunk_path)

    for date_hour, chunk_paths in date_hour2chunk_paths.items():
        date, hour = date_hour
        date_hour_flows_path = split_root_path / date / f'{hour}.csv'

        date_hour_flows = pd.concat(
            [pd.read_csv(chunk_path)
             for chunk_path in chunk_paths]
        )
        date_hour_flows = date_hour_flows.sort_values(by='te')

        try:
            date_hour_flows.to_csv(date_hour_flows_path, index=None)
        except Exception as e:
            raise e
        else:
            shutil.rmtree(split_root_path / date / hour)


if __name__ == '__main__':
    root_path = Path('/home/emikmis/data/UGR16/')
    flow_path = root_path/'uniq'/'july.week5.csv.uniqblacklistremoved'
    new_root_path = Path('/home/emikmis/data/UGR16_Split/')
    split_flows(flow_path, root_path)
