
from collections import OrderedDict

import numpy as np
from scipy.stats import entropy

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

FLOW_STATS = OrderedDict([
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


FLOW_STATS_COLUMNS = ['sa', 'tws'] + list(FLOW_STATS.keys())

"""
########## CTU-13 ############
"""

FLOW2CTU_COLUMNS = {
    'ts': 'StartTime',
    'td': 'Dur',
    'sa': 'SrcAddr',
    'da': 'DstAddr',
    'sp': 'Sport',
    'dp': 'Dport',
    'pr': 'Proto',
    'pkt': 'TotPkts',
    'byt': 'TotBytes',
    'lbl': 'Label'
}
CTU2FLOW_COLUMNS = {v: k for k, v in FLOW2CTU_COLUMNS.items()}


"""
########## UGR-16 ############
"""

UGR_COLUMNS = [
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


"""
########## CIC_IDS2018 ############
"""

CIC_IDS_ORIG_COLUMNS = [
    'Protocol',
    'Timestamp',
    'Flow Duration',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'TotLen Fwd Pkts',
    'TotLen Bwd Pkts',
    'Fwd Pkt Len Max',
    'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean',
    'Fwd Pkt Len Std',
    'Bwd Pkt Len Max',
    'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std',
    'Flow Byts/s',
    'Flow Pkts/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Tot',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Tot',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'Fwd Header Len',
    'Bwd Header Len',
    'Fwd Pkts/s',
    'Bwd Pkts/s',
    'Pkt Len Min',
    'Pkt Len Max',
    'Pkt Len Mean',
    'Pkt Len Std',
    'Pkt Len Var',
    'FIN Flag Cnt',
    'SYN Flag Cnt',
    'RST Flag Cnt',
    'PSH Flag Cnt',
    'ACK Flag Cnt',
    'URG Flag Cnt',
    'CWE Flag Count',
    'ECE Flag Cnt',
    'Down/Up Ratio',
    'Pkt Size Avg',
    'Fwd Seg Size Avg',
    'Bwd Seg Size Avg',
    'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg',
    'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg',
    'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts',
    'Subflow Fwd Byts',
    'Subflow Bwd Pkts',
    'Subflow Bwd Byts',
    'Init Fwd Win Byts',
    'Init Bwd Win Byts',
    'Fwd Act Data Pkts',
    'Fwd Seg Size Min',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min',
    'Label'
]
