from collections import OrderedDict

import numpy as np
from scipy.stats import entropy


class FEATURETYPE:

    def __init__(self):
        self.type = self.__class__.__name__


class CATEGORICAL(FEATURETYPE):

    def __init__(self, values):
        self.values = values
        super(CATEGORICAL, self).__init__()


class NUMERICAL(FEATURETYPE):
    pass


class BINARY(FEATURETYPE):
    pass


class LABEL(FEATURETYPE):
    pass


TCP_FLAGS = {
    'A': 'ack',
    'S': 'syn',
    'R': 'rst',
    'P': 'psh',
    'F': 'fin',
    'U': 'urg',
    'C': 'cwe',
    'E': 'ece'
}

FLOW_COLUMNS = [
    'ts',  # timestamp of the start of a flow
    'td',  # duration of flow
    'sa',  # src addr
    'da',  # dst addr
    'sp',  # src port
    'dp',  # dst port
    'pr',  # proto
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

CTU_13_PROTOS = ['tcp', 'udp', 'icmp']

CTU_13_ORIG_COLUMN_MAPPING = {
    'StartTime': 'timestamp',
    'Dur': 'dur',
    'Proto': 'proto',
    'SrcAddr': 'src_ip',
    'Sport': 'src_port',
    'Dir': 'dir',
    'DstAddr': 'dst_ip',
    'Dport': 'dst_port',
    'State': 'state',  # transaction state
    'sTos': 'src_tos',  # source Type Of Service byte value
    'dTos': 'dst_tos',  # destination Type Of Service byte value
    'TotPkts': 'tot_pkts',
    'TotBytes': 'tot_byts',
    'SrcBytes': 'src_byts',
    'Label': 'label'
}

#  'con_state',  # transaction state CON (UDP) ?
#  'int_state',  # transaction state INT (UDP) ?
CTU_13_COLUMNS = [
    'timestamp',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'proto',  # tcp, udp, icmp or other; categorical
    'dur',
    'dir',
    'fwd_dir',  # '>' in dir
    'bwd_dir',  # '<' in dir
    'state',
    'fwd_fin_flag',
    'fwd_syn_flag',
    'fwd_rst_flag',
    'fwd_psh_flag',
    'fwd_ack_flag',
    'fwd_urg_flag',
    'fwd_cwe_flag',
    'fwd_ece_flag',
    'bwd_fin_flag',
    'bwd_syn_flag',
    'bwd_rst_flag',
    'bwd_psh_flag',
    'bwd_ack_flag',
    'bwd_urg_flag',
    'bwd_cwe_flag',
    'bwd_ece_flag',
    'src_tos',  # 1 if sTos not 0
    'dst_tos',  # 1 if dTos not 0
    'tot_pkts',
    'tot_byts',
    'src_byts',
    'scenario',
    'label',
    'target'
]

CTU_13_FWD_FLAGS_COLUMNS = [f'fwd_{f}_flag' for f in TCP_FLAGS.values()]
CTU_13_BWD_FLAGS_COLUMNS = [f'bwd_{f}_flag' for f in TCP_FLAGS.values()]

CTU_13_FEATURES = {
    'proto': CATEGORICAL(['tcp', 'udp', 'icmp', 'other']),
    'dur': NUMERICAL(),
    'fwd_dir': BINARY(),  # '>' in dir
    'bwd_dir': BINARY(),  # '<' in dir
    'fwd_fin_flag': BINARY(),
    'fwd_syn_flag': BINARY(),
    'fwd_rst_flag': BINARY(),
    'fwd_psh_flag': BINARY(),
    'fwd_ack_flag': BINARY(),
    'fwd_urg_flag': BINARY(),
    'fwd_cwe_flag': BINARY(),
    'fwd_ece_flag': BINARY(),
    'bwd_fin_flag': BINARY(),
    'bwd_syn_flag': BINARY(),
    'bwd_rst_flag': BINARY(),
    'bwd_psh_flag': BINARY(),
    'bwd_ack_flag': BINARY(),
    'bwd_urg_flag': BINARY(),
    'bwd_cwe_flag': BINARY(),
    'bwd_ece_flag': BINARY(),
    'src_tos': BINARY(),  # 1 if sTos not 0
    'dst_tos': BINARY(),  # 1 if dTos not 0
    'tot_pkts': NUMERICAL(),
    'tot_byts': NUMERICAL(),
    'src_byts': NUMERICAL(),
    'target': LABEL()
}

CTU_13_CATEGORICAL_FEATURE_MAP = {
    f: t.values for f, t in CTU_13_FEATURES.items() if t.type == 'CATEGORICAL'
}

CTU_13_NUMERICAL_FEATURES = [
    f for f, t in CTU_13_FEATURES.items() if t.type == 'NUMERICAL'
]

CTU_13_BINARY_FEATURES = [
    f for f, t in CTU_13_FEATURES.items() if t.type == 'BINARY'
]

CTU_13_META_COLUMNS = [
    'timestamp',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'proto',  # tcp, udp, icmp or other; categorical
    'dir',
    'state',
    'scenario',
    'label',
    'target'
]

# numerical:  mean, min, max, std, median
# categorical: num_unique, entropy
CTU_13_AGGR_COLUMNS = [
    'src_ip',
    'time_window_start',
    'total_cnt',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_std',
    'dur_median',
    'dst_ip_entropy',
    'dst_ip_nuniq',
    'dst_port_entropy',
    'dst_port_nuniq',
    'src_port_entropy',
    'src_port_nuniq',
    'proto_entropy',
    'proto_nuniq',
    'fwd_flag_entropy',  #
    'fwd_flag_nuniq',
    'bwd_flag_entropy',  #
    'bwd_flag_nuniq',
    'tot_pkts_mean',
    'tot_pkts_min',
    'tot_pkts_max',
    'tot_pkts_std',
    'tot_pkts_median',
    'tot_byts_mean',
    'tot_byts_min',
    'tot_byts_max',
    'tot_byts_std',
    'tot_byts_median',
    'src_byts_mean',
    'src_byts_min',
    'src_byts_max',
    'src_byts_std',
    'src_byts_median',
    'scenario',
    'target',
]

CTU_13_AGGR_FUNCTIONS = {
    'total_cnt': lambda f: f.shape[0],
    'dur_mean': lambda f: f['dur'].mean(),
    'dur_min': lambda f: f['dur'].min(),
    'dur_max': lambda f: f['dur'].max(),
    'dur_std': lambda f: f['dur'].std(),
    'dur_median': lambda f: f['dur'].median(),
    'dst_ip_entropy': lambda f: entropy(f['dst_ip'].value_counts()),
    'dst_ip_nuniq': lambda f: f['dst_ip'].unique().shape[0],
    'dst_port_entropy': lambda f: entropy(f['dst_port'].value_counts()),
    'dst_port_nuniq': lambda f: f['dst_port'].unique().shape[0],
    'src_port_entropy': lambda f: entropy(f['src_port'].value_counts()),
    'src_port_nuniq': lambda f: f['src_port'].unique().shape[0],
    'proto_entropy': lambda f: entropy(f['proto'].value_counts()),
    'proto_nuniq': lambda f: f['proto'].unique().shape[0],
    'fwd_flag_entropy': lambda f: entropy(f[CTU_13_FWD_FLAGS_COLUMNS].sum()),
    'fwd_flag_nuniq': lambda f: np.sum(f[CTU_13_FWD_FLAGS_COLUMNS].sum() > 0),
    'bwd_flag_entropy': lambda f: entropy(f[CTU_13_BWD_FLAGS_COLUMNS].sum()),
    'bwd_flag_nuniq': lambda f: np.sum(f[CTU_13_BWD_FLAGS_COLUMNS].sum() > 0),
    'tot_pkts_mean': lambda f: f['tot_pkts'].mean(),
    'tot_pkts_min': lambda f: f['tot_pkts'].min(),
    'tot_pkts_max': lambda f: f['tot_pkts'].max(),
    'tot_pkts_std': lambda f: f['tot_pkts'].std(),
    'tot_pkts_median': lambda f: f['tot_pkts'].median(),
    'tot_byts_mean': lambda f: f['tot_byts'].mean(),
    'tot_byts_min': lambda f: f['tot_byts'].min(),
    'tot_byts_max': lambda f: f['tot_byts'].max(),
    'tot_byts_std': lambda f: f['tot_byts'].std(),
    'tot_byts_median': lambda f: f['tot_byts'].median(),
    'src_byts_mean': lambda f: f['src_byts'].mean(),
    'src_byts_min': lambda f: f['src_byts'].min(),
    'src_byts_max': lambda f: f['src_byts'].max(),
    'src_byts_std': lambda f: f['src_byts'].std(),
    'src_byts_median': lambda f: f['src_byts'].median(),
    'target': lambda f: np.int(f['target'].sum() > 0)
}

CTU_13_AGGR_META_COLUMNS = [
    'src_ip',
    'time_window_start',
    'scenario',
    'target'
]

CTU_13_SCENARIO_NAMES = {
    1: 'Neris',
    2: 'Neris',
    3: 'Rbot',
    4: 'Rbot',
    5: 'Virut',
    6: 'Menti',
    7: 'Sogou',
    8: 'Murlo',
    9: 'Neris',
    10: 'Rbot',
    11: 'Rbot',
    12: 'NSIS.ay',
    13: 'Virut',
}

"""
########## UGR-16 ############
"""

UGR_COLUMNS = [
    'te',  # timestamp of the end of a flow
    'td',  # duration of flow
    'sa',  # src addr
    'da',  # dst addr
    'sp',  # src port
    'dp',  # dst port
    'pr',  # proto
    'flg',  # flags
    'fwd',  # forwarding status
    'stos',  # type of service
    'pkt',  # packets exchanged in the flow
    'byt',  # their corresponding num of bytes
    'lbl'
]

"""
########## CIC-IDS2017 ############
"""

CIC_IDS2017_COLUMN_MAPPING = {
    'Flow ID': 'flow_id',
    'Source IP': 'src_ip',
    'Source Port': 'src_port',
    'Destination IP': 'dst_ip',
    'Destination Port': 'dst_port',
    'Protocol': 'protocol',
    'Timestamp': 'timestamp',
    'Flow Duration': 'dur',
    'Total Fwd Packets': 'tot_fwd_pkts',
    'Total Backward Packets': 'tot_bwd_pkts',
    'Total Length of Fwd Packets': 'tot_len_fwd_pkts',
    'Total Length of Bwd Packets': 'tot_len_bwd_pkts',
    'Fwd Packet Length Max': 'fwd_pkt_len_max',
    'Fwd Packet Length Min': 'fwd_pkt_len_min',
    'Fwd Packet Length Mean': 'fwd_pkt_len_mean',
    'Fwd Packet Length Std': 'fwd_pkt_len_std',
    'Bwd Packet Length Max': 'bwd_pkt_len_max',
    'Bwd Packet Length Min': 'bwd_pkt_len_min',
    'Bwd Packet Length Mean': 'bwd_pkt_len_mean',
    'Bwd Packet Length Std': 'bwd_pkt_len_std',
    'Flow Bytes/s': 'flow_byts/s',
    'Flow Packets/s': 'flow_pkts/s',
    'Flow IAT Mean': 'flow_iat_mean',
    'Flow IAT Std': 'flow_iat_std',
    'Flow IAT Max': 'flow_iat_max',
    'Flow IAT Min': 'flow_iat_min',
    'Fwd IAT Total': 'fwd_iat_tot',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Fwd IAT Std': 'fwd_iat_std',
    'Fwd IAT Max': 'fwd_iat_max',
    'Fwd IAT Min': 'fwd_iat_min',
    'Bwd IAT Total': 'bwd_iat_tot',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Bwd IAT Std': 'bwd_iat_std',
    'Bwd IAT Max': 'bwd_iat_max',
    'Bwd IAT Min': 'bwd_iat_min',
    'Fwd PSH Flags': 'fwd_psh_flags',
    'Bwd PSH Flags': 'bwd_psh_flags',
    'Fwd URG Flags': 'fwd_urg_flags',
    'Bwd URG Flags': 'bwd_urg_flags',
    'Fwd Header Length': 'fwd_header_len',
    'Bwd Header Length': 'bwd_header_len',
    'Fwd Packets/s': 'fwd_pkts/s',
    'Bwd Packets/s': 'bwd_pkts/s',
    'Min Packet Length': 'pkt_len_min',
    'Max Packet Length': 'pkt_len_max',
    'Packet Length Mean': 'pkt_len_mean',
    'Packet Length Std': 'pkt_len_std',
    'Packet Length Variance': 'pkt_len_var',
    'FIN Flag Count': 'fin_flag_cnt',
    'SYN Flag Count': 'syn_flag_cnt',
    'RST Flag Count': 'rst_flag_cnt',
    'PSH Flag Count': 'psh_flag_cnt',
    'ACK Flag Count': 'ack_flag_cnt',
    'URG Flag Count': 'urg_flag_cnt',
    'CWE Flag Count': 'cwe_flag_cnt',
    'ECE Flag Count': 'ece_flag_cnt',
    'Down/Up Ratio': 'down/up_ratio',
    'Average Packet Size': 'pkt_size_avg',
    'Avg Fwd Segment Size': 'fwd_seg_size_avg',
    'Avg Bwd Segment Size': 'bwd_seg_size_avg',
    'Fwd Header Length.1': 'fwd_header_len1',  # not sure
    'Fwd Avg Bytes/Bulk': 'fwd_byts/blk_avg',
    'Fwd Avg Packets/Bulk': 'fwd_pkts/blk_avg',
    'Fwd Avg Bulk Rate': 'fwd_blk_rate_avg',
    'Bwd Avg Bytes/Bulk': 'bwd_byts/blk_avg',
    'Bwd Avg Packets/Bulk': 'bwd_pkts/blk_avg',
    'Bwd Avg Bulk Rate': 'bwd_blk_rate_avg',
    'Subflow Fwd Packets': 'subflow_fwd_pkts',
    'Subflow Fwd Bytes': 'subflow_fwd_byts',
    'Subflow Bwd Packets': 'subflow_bwd_pkts',
    'Subflow Bwd Bytes': 'subflow_bwd_byts',
    'Init_Win_bytes_forward': 'init_fwd_win_byts',
    'Init_Win_bytes_backward': 'init_bwd_win_byts',
    'act_data_pkt_fwd': 'fwd_act_data_pkts',
    'min_seg_size_forward': 'fwd_seg_size_min',
    'Active Mean': 'active_mean',
    'Active Std': 'active_std',
    'Active Max': 'active_max',
    'Active Min': 'active_min',
    'Idle Mean': 'idle_mean',
    'Idle Std': 'idle_std',
    'Idle Max': 'idle_max',
    'Idle Min': 'idle_min',
    'Label': 'label'
}

CIC_IDS2017_ATTACK_LABELS = {
    'Bot': 'botnet',
    'DDoS': 'ddos',
    'DoS GoldenEye': 'dos',
    'DoS Hulk': 'dos',
    'DoS Slowhttptest': 'dos',
    'DoS slowloris': 'dos',
    'FTP-Patator': 'brute-force',
    'Heartbleed': 'heartbleed',
    'Infiltration': 'infiltration',
    'PortScan': 'port-scan',
    'SSH-Patator': 'brute-force',
    'Web Attack  Brute Force': 'web',
    'Web Attack  Sql Injection': 'web',
    'Web Attack  XSS': 'web'
}

CIC_IDS2017_COLUMNS = list(CIC_IDS2017_COLUMN_MAPPING.values()) + \
                      ['label_orig', 'scenario', 'target']

CIC_IDS2017_PROTOCOL_MAPPING = {
    0: 'hopopt',
    17: 'udp',
    6: 'tcp',
}

CIC_IDS2017_FEATURES = {
    'protocol': CATEGORICAL(list(CIC_IDS2017_PROTOCOL_MAPPING.values())),
    'dur': NUMERICAL(),
    'tot_fwd_pkts': NUMERICAL(),
    'tot_bwd_pkts': NUMERICAL(),
    'tot_len_fwd_pkts': NUMERICAL(),
    'tot_len_bwd_pkts': NUMERICAL(),
    'fwd_pkt_len_max': NUMERICAL(),
    'fwd_pkt_len_min': NUMERICAL(),
    'fwd_pkt_len_mean': NUMERICAL(),
    'fwd_pkt_len_std': NUMERICAL(),
    'bwd_pkt_len_max': NUMERICAL(),
    'bwd_pkt_len_min': NUMERICAL(),
    'bwd_pkt_len_mean': NUMERICAL(),
    'bwd_pkt_len_std': NUMERICAL(),
    'flow_byts/s': NUMERICAL(),
    'flow_pkts/s': NUMERICAL(),
    'flow_iat_mean': NUMERICAL(),
    'flow_iat_std': NUMERICAL(),
    'flow_iat_max': NUMERICAL(),
    'flow_iat_min': NUMERICAL(),
    'fwd_iat_tot': NUMERICAL(),
    'fwd_iat_mean': NUMERICAL(),
    'fwd_iat_std': NUMERICAL(),
    'fwd_iat_max': NUMERICAL(),
    'fwd_iat_min': NUMERICAL(),
    'bwd_iat_tot': NUMERICAL(),
    'bwd_iat_mean': NUMERICAL(),
    'bwd_iat_std': NUMERICAL(),
    'bwd_iat_max': NUMERICAL(),
    'bwd_iat_min': NUMERICAL(),
    'fwd_psh_flags': NUMERICAL(),
    'bwd_psh_flags': NUMERICAL(),
    'fwd_urg_flags': NUMERICAL(),
    'bwd_urg_flags': NUMERICAL(),
    'fwd_header_len': NUMERICAL(),
    'bwd_header_len': NUMERICAL(),
    'fwd_pkts/s': NUMERICAL(),
    'bwd_pkts/s': NUMERICAL(),
    'pkt_len_min': NUMERICAL(),
    'pkt_len_max': NUMERICAL(),
    'pkt_len_mean': NUMERICAL(),
    'pkt_len_std': NUMERICAL(),
    'pkt_len_var': NUMERICAL(),
    'fin_flag_cnt': NUMERICAL(),
    'syn_flag_cnt': NUMERICAL(),
    'rst_flag_cnt': NUMERICAL(),
    'psh_flag_cnt': NUMERICAL(),
    'ack_flag_cnt': NUMERICAL(),
    'urg_flag_cnt': NUMERICAL(),
    'cwe_flag_cnt': NUMERICAL(),
    'ece_flag_cnt': NUMERICAL(),
    'down/up_ratio': NUMERICAL(),
    'pkt_size_avg': NUMERICAL(),
    'fwd_seg_size_avg': NUMERICAL(),
    'bwd_seg_size_avg': NUMERICAL(),
    'fwd_header_len1': NUMERICAL(),
    'fwd_byts/blk_avg': NUMERICAL(),
    'fwd_pkts/blk_avg': NUMERICAL(),
    'fwd_blk_rate_avg': NUMERICAL(),
    'bwd_byts/blk_avg': NUMERICAL(),
    'bwd_pkts/blk_avg': NUMERICAL(),
    'bwd_blk_rate_avg': NUMERICAL(),
    'subflow_fwd_pkts': NUMERICAL(),
    'subflow_fwd_byts': NUMERICAL(),
    'subflow_bwd_pkts': NUMERICAL(),
    'subflow_bwd_byts': NUMERICAL(),
    'init_fwd_win_byts': NUMERICAL(),
    'init_bwd_win_byts': NUMERICAL(),
    'fwd_act_data_pkts': NUMERICAL(),
    'fwd_seg_size_min': NUMERICAL(),
    'active_mean': NUMERICAL(),
    'active_std': NUMERICAL(),
    'active_max': NUMERICAL(),
    'active_min': NUMERICAL(),
    'idle_mean': NUMERICAL(),
    'idle_std': NUMERICAL(),
    'idle_max': NUMERICAL(),
    'idle_min': NUMERICAL(),
    'target': LABEL()
}

CIC_IDS2017_CATEGORICAL_FEATURE_MAP = {
    f: t.values for f, t in CIC_IDS2017_FEATURES.items() if t.type == 'CATEGORICAL'
}

CIC_IDS2017_NUMERICAL_FEATURES = [
    f for f, t in CIC_IDS2017_FEATURES.items() if t.type == 'NUMERICAL'
]

CIC_IDS2017_BINARY_FEATURES = [
    f for f, t in CIC_IDS2017_FEATURES.items() if t.type == 'BINARY'
]

CIC_IDS2017_META_COLUMNS = [
    'timestamp',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'protocol',  # tcp, udp, icmp or other; categorical
    'scenario',
    'label_orig',
    'label',
    'target'
]

# numerical:  mean, min, max, std, median
# categorical: num_unique, entropy
CIC_IDS2017_AGGR_COLUMNS = [
    'src_ip',
    'time_window_start',
    'total_cnt',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_std',
    'dur_median',
    'dst_ip_entropy',
    'dst_ip_nuniq',
    'dst_port_entropy',
    'dst_port_nuniq',
    'src_port_entropy',
    'src_port_nuniq',
    'proto_entropy',
    'proto_nuniq',
    'flag_entropy',  #
    'flag_nuniq',
    'tot_fwd_pkts_mean',
    'tot_fwd_pkts_min',
    'tot_fwd_pkts_max',
    'tot_fwd_pkts_std',
    'tot_fwd_pkts_median',
    'tot_bwd_pkts_mean',
    'tot_bwd_pkts_min',
    'tot_bwd_pkts_max',
    'tot_bwd_pkts_std',
    'tot_bwd_pkts_median',
    'tot_len_fwd_pkts_mean',
    'tot_len_fwd_pkts_min',
    'tot_len_fwd_pkts_max',
    'tot_len_fwd_pkts_std',
    'tot_len_fwd_pkts_median',
    'tot_len_bwd_pkts_mean',
    'tot_len_bwd_pkts_min',
    'tot_len_bwd_pkts_max',
    'tot_len_bwd_pkts_std',
    'tot_len_bwd_pkts_median',
    'flow_byts/s_mean',
    'flow_byts/s_min',
    'flow_byts/s_max',
    'flow_byts/s_std',
    'flow_byts/s_median',
    'flow_pkts/s_mean',
    'flow_pkts/s_min',
    'flow_pkts/s_max',
    'flow_pkts/s_std',
    'flow_pkts/s_median',
    'scenario',
    'target',
]

CIC_IDS2017_FLAGS_COLUMNS = [f'{f}_flag_cnt' for f in TCP_FLAGS.values()]

CIC_IDS2017_AGGR_FUNCTIONS = {
    'total_cnt': lambda f: f.shape[0],
    'dur_mean': lambda f: f['dur'].mean(),
    'dur_min': lambda f: f['dur'].min(),
    'dur_max': lambda f: f['dur'].max(),
    'dur_std': lambda f: f['dur'].std(),
    'dur_median': lambda f: f['dur'].median(),
    'dst_ip_entropy': lambda f: entropy(f['dst_ip'].value_counts()),
    'dst_ip_nuniq': lambda f: f['dst_ip'].unique().shape[0],
    'dst_port_entropy': lambda f: entropy(f['dst_port'].value_counts()),
    'dst_port_nuniq': lambda f: f['dst_port'].unique().shape[0],
    'src_port_entropy': lambda f: entropy(f['src_port'].value_counts()),
    'src_port_nuniq': lambda f: f['src_port'].unique().shape[0],
    'proto_entropy': lambda f: entropy(f['protocol'].value_counts()),
    'proto_nuniq': lambda f: f['protocol'].unique().shape[0],
    'flag_entropy': lambda f: entropy(f[CIC_IDS2017_FLAGS_COLUMNS].sum()),
    'flag_nuniq': lambda f: np.sum(f[CIC_IDS2017_FLAGS_COLUMNS].sum() > 0),
    'tot_fwd_pkts_mean': lambda f: f['tot_fwd_pkts'].mean(),
    'tot_fwd_pkts_min': lambda f: f['tot_fwd_pkts'].min(),
    'tot_fwd_pkts_max': lambda f: f['tot_fwd_pkts'].max(),
    'tot_fwd_pkts_std': lambda f: f['tot_fwd_pkts'].std(),
    'tot_fwd_pkts_median': lambda f: f['tot_fwd_pkts'].median(),
    'tot_bwd_pkts_mean': lambda f: f['tot_bwd_pkts'].mean(),
    'tot_bwd_pkts_min': lambda f: f['tot_bwd_pkts'].min(),
    'tot_bwd_pkts_max': lambda f: f['tot_bwd_pkts'].max(),
    'tot_bwd_pkts_std': lambda f: f['tot_bwd_pkts'].std(),
    'tot_bwd_pkts_median': lambda f: f['tot_bwd_pkts'].median(),
    'tot_len_fwd_pkts_mean': lambda f: f['tot_len_fwd_pkts'].mean(),
    'tot_len_fwd_pkts_min': lambda f: f['tot_len_fwd_pkts'].min(),
    'tot_len_fwd_pkts_max': lambda f: f['tot_len_fwd_pkts'].max(),
    'tot_len_fwd_pkts_std': lambda f: f['tot_len_fwd_pkts'].std(),
    'tot_len_fwd_pkts_median': lambda f: f['tot_len_fwd_pkts'].median(),
    'tot_len_bwd_pkts_mean': lambda f: f['tot_len_bwd_pkts'].mean(),
    'tot_len_bwd_pkts_min': lambda f: f['tot_len_bwd_pkts'].min(),
    'tot_len_bwd_pkts_max': lambda f: f['tot_len_bwd_pkts'].max(),
    'tot_len_bwd_pkts_std': lambda f: f['tot_len_bwd_pkts'].std(),
    'tot_len_bwd_pkts_median': lambda f: f['tot_len_bwd_pkts'].median(),
    'flow_byts/s_mean': lambda f: f['flow_byts/s'].mean(),
    'flow_byts/s_min': lambda f: f['flow_byts/s'].min(),
    'flow_byts/s_max': lambda f: f['flow_byts/s'].max(),
    'flow_byts/s_std': lambda f: f['flow_byts/s'].std(),
    'flow_byts/s_median': lambda f: f['flow_byts/s'].median(),
    'flow_pkts/s_mean': lambda f: f['flow_pkts/s'].mean(),
    'flow_pkts/s_min': lambda f: f['flow_pkts/s'].min(),
    'flow_pkts/s_max': lambda f: f['flow_pkts/s'].max(),
    'flow_pkts/s_std': lambda f: f['flow_pkts/s'].std(),
    'flow_pkts/s_median': lambda f: f['flow_pkts/s'].median(),
    'target': lambda f: np.int(f['target'].sum() > 0)
}

CIC_IDS2017_AGGR_META_COLUMNS = [
    'src_ip',
    'time_window_start',
    'scenario',
    'target'
]

"""
########## CSE-CIC-IDS2018 ############
"""

CIC_IDS2018_COLUMN_MAPPING = {
    'Protocol': 'protocol',
    'Timestamp': 'timestamp',
    'Flow Duration': 'flow_dur',
    'Tot Fwd Pkts': 'tot_fwd_pkts',
    'Tot Bwd Pkts': 'tot_bwd_pkts',
    'TotLen Fwd Pkts': 'tot_len_fwd_pkts',
    'TotLen Bwd Pkts': 'tot_len_bwd_pkts',
    'Fwd Pkt Len Max': 'fwd_pkt_len_max',
    'Fwd Pkt Len Min': 'fwd_pkt_len_min',
    'Fwd Pkt Len Mean': 'fwd_pkt_len_mean',
    'Fwd Pkt Len Std': 'fwd_pkt_len_std',
    'Bwd Pkt Len Max': 'bwd_pkt_len_max',
    'Bwd Pkt Len Min': 'bwd_pkt_len_min',
    'Bwd Pkt Len Mean': 'bwd_pkt_len_mean',
    'Bwd Pkt Len Std': 'bwd_pkt_len_std',
    'Flow Byts/s': 'flow_byts/s',
    'Flow Pkts/s': 'flow_pkts/s',
    'Flow IAT Mean': 'flow_iat_mean',
    'Flow IAT Std': 'flow_iat_std',
    'Flow IAT Max': 'flow_iat_max',
    'Flow IAT Min': 'flow_iat_min',
    'Fwd IAT Tot': 'fwd_iat_tot',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Fwd IAT Std': 'fwd_iat_std',
    'Fwd IAT Max': 'fwd_iat_max',
    'Fwd IAT Min': 'fwd_iat_min',
    'Bwd IAT Tot': 'bwd_iat_tot',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Bwd IAT Std': 'bwd_iat_std',
    'Bwd IAT Max': 'bwd_iat_max',
    'Bwd IAT Min': 'bwd_iat_min',
    'Fwd PSH Flags': 'fwd_psh_flags',
    'Bwd PSH Flags': 'bwd_psh_flags',
    'Fwd URG Flags': 'fwd_urg_flags',
    'Bwd URG Flags': 'bwd_urg_flags',
    'Fwd Header Len': 'fwd_header_len',
    'Bwd Header Len': 'bwd_header_len',
    'Fwd Pkts/s': 'fwd_pkts/s',
    'Bwd Pkts/s': 'bwd_pkts/s',
    'Pkt Len Min': 'pkt_len_min',
    'Pkt Len Max': 'pkt_len_max',
    'Pkt Len Mean': 'pkt_len_mean',
    'Pkt Len Std': 'pkt_len_std',
    'Pkt Len Var': 'pkt_len_var',
    'FIN Flag Cnt': 'fin_flag_cnt',
    'SYN Flag Cnt': 'syn_flag_cnt',
    'RST Flag Cnt': 'rst_flag_cnt',
    'PSH Flag Cnt': 'psh_flag_cnt',
    'ACK Flag Cnt': 'ack_flag_cnt',
    'URG Flag Cnt': 'urg_flag_cnt',
    'CWE Flag Count': 'cwe_flag_cnt',
    'ECE Flag Cnt': 'ece_flag_cnt',
    'Down/Up Ratio': 'down/up_ratio',
    'Pkt Size Avg': 'pkt_size_avg',
    'Fwd Seg Size Avg': 'fwd_seg_size_avg',
    'Bwd Seg Size Avg': 'bwd_seg_size_avg',
    'Fwd Byts/b Avg': 'fwd_byts/blk_avg',
    'Fwd Pkts/b Avg': 'fwd_pkts/blk_avg',
    'Fwd Blk Rate Avg': 'fwd_blk_rate_avg',
    'Bwd Byts/b Avg': 'bwd_byts/blk_avg',
    'Bwd Pkts/b Avg': 'bwd_pkts/blk_avg',
    'Bwd Blk Rate Avg': 'bwd_blk_rate_avg',
    'Subflow Fwd Pkts': 'subflow_fwd_pkts',
    'Subflow Fwd Byts': 'subflow_fwd_byts',
    'Subflow Bwd Pkts': 'subflow_bwd_pkts',
    'Subflow Bwd Byts': 'subflow_bwd_byts',
    'Init Fwd Win Byts': 'init_fwd_win_byts',
    'Init Bwd Win Byts': 'init_bwd_win_byts',
    'Fwd Act Data Pkts': 'fwd_act_data_pkts',
    'Fwd Seg Size Min': 'fwd_seg_size_min',
    'Active Mean': 'active_mean',
    'Active Std': 'active_std',
    'Active Max': 'active_max',
    'Active Min': 'active_min',
    'Idle Mean': 'idle_mean',
    'Idle Std': 'idle_std',
    'Idle Max': 'idle_max',
    'Idle Min': 'idle_min',
    'Label': 'label',
}

CIC_IDS2018_ATTACK_LABELS = [
    'Bot',
    'Brute Force -Web',
    'Brute Force -XSS',
    'DDOS attack-HOIC',
    'DDOS attack-LOIC-UDP',
    'DDoS attacks-LOIC-HTTP',
    'DoS attacks-GoldenEye',
    'DoS attacks-Hulk',
    'DoS attacks-SlowHTTPTest',
    'DoS attacks-Slowloris',
    'FTP-BruteForce',
    'Infilteration',
    'SQL Injection',
    'SSH-Bruteforce'
]

"""
########## IOT-23 ############
"""

IOT_23_ORIG_SCENARIO_NAME_MAPPING = {
    'CTU-IoT-Malware-Capture-33-1': (33, 'kenjiro'),
    'CTU-IoT-Malware-Capture-42-1': (42, 'trojan'),
    'CTU-IoT-Malware-Capture-48-1': (48, 'mirai'),
    'CTU-IoT-Malware-Capture-8-1': (8, 'hakai'),
    'Somfy-01': (7, 'benign'),
    'CTU-IoT-Malware-Capture-21-1': (21, 'torii'),
    'CTU-IoT-Malware-Capture-9-1': (9, 'linux'),
    'CTU-IoT-Malware-Capture-20-1': (20, 'torii'),
    'CTU-IoT-Malware-Capture-60-1': (60, 'gagfyt'),
    'CTU-IoT-Malware-Capture-39-1': (39, 'ircbot'),
    'CTU-Honeypot-Capture-5-1': (5, 'benign'),
    'CTU-IoT-Malware-Capture-52-1': (52, 'mirai'),
    'CTU-IoT-Malware-Capture-49-1': (49, 'mirai'),
    'CTU-IoT-Malware-Capture-34-1': (34, 'mirai'),
    'CTU-IoT-Malware-Capture-3-1': (3, 'muhstik'),
    'CTU-IoT-Malware-Capture-35-1': (35, 'mirai'),
    'CTU-IoT-Malware-Capture-7-1': (7, 'linux'),
    'CTU-IoT-Malware-Capture-44-1': (44, 'mirai'),
    'CTU-IoT-Malware-Capture-17-1': (17, 'kenjiro'),
    'CTU-Honeypot-Capture-4-1': (4, 'benign'),
    'CTU-IoT-Malware-Capture-1-1': (1, 'mirai'),
    'CTU-IoT-Malware-Capture-36-1': (33, 'mirai'),
    'CTU-IoT-Malware-Capture-43-1': (33, 'mirai')
}

IOT_23_ORIG_COLUMN_MAPPING = {
    'ts': 'timestamp',
    'uid': 'uid',
    'id.orig_h': 'src_ip',
    'id.orig_p': 'src_port',
    'id.resp_h': 'dst_ip',
    'id.resp_p': 'dst_port',
    'proto': 'proto',
    'service': 'service',
    'duration': 'dur',
    'orig_bytes': 'orig_bytes',
    'resp_bytes': 'resp_bytes',
    'conn_state': 'conn_state',
    'local_orig': 'local_orig',
    'local_resp': 'local_resp',
    'missed_bytes': 'missed_bytes',
    'history': 'history',
    'orig_pkts': 'orig_pkts',
    'orig_ip_bytes': 'orig_ip_bytes',
    'resp_pkts': 'resp_pkts',
    'resp_ip_bytes': 'resp_ip_bytes',
    'tunnel_parents': 'tunnel_parents',
    'label': 'label',
    'detailed-label': 'detailed_label'
}

IOT_23_HISTORY_LETTERS = ['s', 'h', 'a', 'd', 'f', 'r',
                          'c', 'g', 't', 'w', 'i', 'q']
IOT_23_PROTO_VALUES = ['tcp', 'udp', 'icmp']
IOT_23_SERVICE_VALUES = ['-', 'http', 'dns', 'ssh', 'dhcp', 'irc', 'ssl']
IOT_23_CONN_STATE_VALUES = [
    'S0', 'S1', 'SF', 'REJ',
    'S2', 'S3', 'RSTO', 'RSTR',
    'RSTOS0', 'RSTRH', 'SH',
    'SHR', 'OTH'
]

IOT_23_REPLACE_EMPTY_ZERO_FEATURES = [
    'dur', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]

IOT_23_COLUMNS = [
    'timestamp',
    'uid',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'proto',
    'service',
    'dur',
    'orig_bytes',
    'resp_bytes',
    'conn_state',
    'local_orig',
    'local_resp',
    'missed_bytes',
    'history',
    'history_empty',
    'history_dir_flipped',
    *[f'orig_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS],
    *[f'resp_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS],
    'orig_pkts',
    'orig_ip_bytes',
    'resp_pkts',
    'resp_ip_bytes',
    'tunnel_parents',
    'label',
    'detailed_label',
    'scenario',
    'target',
]

# local_orig, local_resp and tunnel parents doesn't seem to be used
IOT_23_FEATURES = {
    'proto': CATEGORICAL(IOT_23_PROTO_VALUES),
    'service': CATEGORICAL(IOT_23_SERVICE_VALUES),
    'dur': NUMERICAL(),
    'orig_bytes': NUMERICAL(),
    'resp_bytes': NUMERICAL(),
    'conn_state': CATEGORICAL(IOT_23_CONN_STATE_VALUES),
    # 'local_orig': CATEGORICAL(['T', 'F', '-']),  # True, False, Unset
    # 'local_resp': CATEGORICAL(['T', 'F', '-']),
    'missed_bytes': NUMERICAL(),
    'history_empty': BINARY(),
    'history_dir_flipped': BINARY(),
    **{f'orig_history_{l}_cnt': NUMERICAL() for l in IOT_23_HISTORY_LETTERS},
    **{f'resp_history_{l}_cnt': NUMERICAL() for l in IOT_23_HISTORY_LETTERS},
    'orig_pkts': NUMERICAL(),
    'orig_ip_bytes': NUMERICAL(),
    'resp_pkts': NUMERICAL(),
    'resp_ip_bytes': NUMERICAL(),
    # 'tunnel_parents': BINARY(),  # present or not
    'target': LABEL()
}

IOT_23_META_COLUMNS = [
    'timestamp',
    'uid',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'proto',
    'history',
    'scenario',
    'detailed_label',
    'target'
]

IOT_23_CATEGORICAL_FEATURE_MAP = {
    f: t.values for f, t in IOT_23_FEATURES.items() if t.type == 'CATEGORICAL'
}

IOT_23_NUMERICAL_FEATURES = [
    f for f, t in IOT_23_FEATURES.items() if t.type == 'NUMERICAL'
]

IOT_23_BINARY_FEATURES = [
    f for f, t in IOT_23_FEATURES.items() if t.type == 'BINARY'
]

# numerical:  mean, min, max, std, median
# categorical: num_unique, entropy
IOT_23_AGGR_COLUMNS = [
    'src_ip',
    'time_window_start',
    'total_cnt',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_std',
    'dur_median',
    'dst_ip_entropy',
    'dst_ip_nuniq',
    'dst_port_entropy',
    'dst_port_nuniq',
    'src_port_entropy',
    'src_port_nuniq',
    'proto_entropy',
    'proto_nuniq',
    'service_entropy',
    'service_nuniq',
    'orig_hist_entropy',
    'orig_hist_nuniq',
    'resp_hist_entropy',
    'resp_hist_nuniq',
    'orig_bytes_mean',
    'orig_bytes_min',
    'orig_bytes_max',
    'orig_bytes_std',
    'orig_bytes_median',
    'resp_bytes_mean',
    'resp_bytes_min',
    'resp_bytes_max',
    'resp_bytes_std',
    'resp_bytes_median',
    'conn_state_entropy',
    'conn_state_nuniq',
    'missed_bytes_mean',
    'missed_bytes_min',
    'missed_bytes_max',
    'missed_bytes_std',
    'missed_bytes_median',
    'orig_pkts_mean',
    'orig_pkts_min',
    'orig_pkts_max',
    'orig_pkts_std',
    'orig_pkts_median',
    'orig_ip_bytes_mean',
    'orig_ip_bytes_min',
    'orig_ip_bytes_max',
    'orig_ip_bytes_std',
    'orig_ip_bytes_median',
    'resp_pkts_mean',
    'resp_pkts_min',
    'resp_pkts_max',
    'resp_pkts_std',
    'resp_pkts_median',
    'resp_ip_bytes_mean',
    'resp_ip_bytes_min',
    'resp_ip_bytes_max',
    'resp_ip_bytes_std',
    'resp_ip_bytes_median',
    'scenario',
    'detailed_label',
    'target',
]


def detailed_label_aggr_fn(f):
    labels = dict(f['detailed_label'].value_counts())
    if '-' in labels:
        labels.pop('-')

    if labels:
        labels = sorted(list(labels.items()), key=lambda x: x[1])
        aggr_label = labels[0][0]
    else:
        aggr_label = '-'

    return aggr_label


IOT_23_AGGR_FUNCTIONS = {
    'total_cnt': lambda f: f.shape[0],
    'dur_mean': lambda f: f['dur'].mean(),
    'dur_min': lambda f: f['dur'].max(),
    'dur_std': lambda f: f['dur'].min(),
    'dur_max': lambda f: f['dur'].std(),
    'dur_median': lambda f: f['dur'].median(),
    'dst_ip_entropy': lambda f: entropy(f['dst_ip'].value_counts()),
    'dst_ip_nuniq': lambda f: f['dst_ip'].unique().shape[0],
    'dst_port_entropy': lambda f: entropy(f['dst_port'].value_counts()),
    'dst_port_nuniq': lambda f: f['dst_port'].unique().shape[0],
    'src_port_entropy': lambda f: entropy(f['src_port'].value_counts()),
    'src_port_nuniq': lambda f: f['src_port'].unique().shape[0],
    'proto_entropy': lambda f: entropy(f['proto'].value_counts()),
    'proto_nuniq': lambda f: f['proto'].unique().shape[0],
    'service_entropy': lambda f: entropy(f['service'].value_counts()),
    'service_nuniq': lambda f: f['service'].unique().shape[0],
    'orig_hist_entropy': lambda f: entropy(f[[f'orig_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS]].sum()),
    'orig_hist_nuniq': lambda f: np.sum(f[[f'orig_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS]].sum() > 0),
    'resp_hist_entropy': lambda f: entropy(f[[f'resp_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS]].sum()),
    'resp_hist_nuniq': lambda f: np.sum(f[[f'resp_history_{l}_cnt' for l in IOT_23_HISTORY_LETTERS]].sum() > 0),
    'orig_bytes_mean': lambda f: f['orig_bytes'].mean(),
    'orig_bytes_min': lambda f: f['orig_bytes'].min(),
    'orig_bytes_max': lambda f: f['orig_bytes'].max(),
    'orig_bytes_std': lambda f: f['orig_bytes'].std(),
    'orig_bytes_median': lambda f: f['orig_bytes'].median(),
    'resp_bytes_mean': lambda f: f['resp_bytes'].mean(),
    'resp_bytes_min': lambda f: f['resp_bytes'].min(),
    'resp_bytes_max': lambda f: f['resp_bytes'].max(),
    'resp_bytes_std': lambda f: f['resp_bytes'].std(),
    'resp_bytes_median': lambda f: f['resp_bytes'].median(),
    'conn_state_entropy': lambda f: entropy(f['conn_state'].value_counts()),
    'conn_state_nuniq': lambda f: f['conn_state'].unique().shape[0],
    'missed_bytes_mean': lambda f: f['missed_bytes'].mean(),
    'missed_bytes_min': lambda f: f['missed_bytes'].min(),
    'missed_bytes_max': lambda f: f['missed_bytes'].max(),
    'missed_bytes_std': lambda f: f['missed_bytes'].std(),
    'missed_bytes_median': lambda f: f['missed_bytes'].median(),
    'orig_pkts_mean': lambda f: f['orig_pkts'].mean(),
    'orig_pkts_min': lambda f: f['orig_pkts'].min(),
    'orig_pkts_max': lambda f: f['orig_pkts'].max(),
    'orig_pkts_std': lambda f: f['orig_pkts'].std(),
    'orig_pkts_median': lambda f: f['orig_pkts'].median(),
    'orig_ip_bytes_mean': lambda f: f['orig_ip_bytes'].mean(),
    'orig_ip_bytes_min': lambda f: f['orig_ip_bytes'].min(),
    'orig_ip_bytes_max': lambda f: f['orig_ip_bytes'].max(),
    'orig_ip_bytes_std': lambda f: f['orig_ip_bytes'].std(),
    'orig_ip_bytes_median': lambda f: f['orig_ip_bytes'].median(),
    'resp_pkts_mean': lambda f: f['resp_pkts'].mean(),
    'resp_pkts_min': lambda f: f['resp_pkts'].min(),
    'resp_pkts_max': lambda f: f['resp_pkts'].max(),
    'resp_pkts_std': lambda f: f['resp_pkts'].std(),
    'resp_pkts_median': lambda f: f['resp_pkts'].median(),
    'resp_ip_bytes_mean': lambda f: f['resp_ip_bytes'].mean(),
    'resp_ip_bytes_min': lambda f: f['resp_ip_bytes'].min(),
    'resp_ip_bytes_max': lambda f: f['resp_ip_bytes'].max(),
    'resp_ip_bytes_std': lambda f: f['resp_ip_bytes'].std(),
    'resp_ip_bytes_median': lambda f: f['resp_ip_bytes'].median(),
    'detailed_label': lambda f: detailed_label_aggr_fn(f),
    'target': lambda f: np.int(f['target'].sum() > 0),
}

IOT_23_AGGR_FEATURES = {k: NUMERICAL() for k in IOT_23_AGGR_FUNCTIONS
                        if k not in ['target', 'detailed_label']}
IOT_23_AGGR_FEATURES['target'] = LABEL()

IOT_23_AGGR_NUMERICAL_FEATURES = [f for f, t in IOT_23_AGGR_FEATURES.items()
                                  if t.type == 'NUMERICAL']

IOT_23_AGGR_META_COLUMNS = [
    'src_ip',
    'time_window_start',
    'scenario',
    'detailed_label',
    'target',
]

IOT_23_PLOT_PARAMS = {
    'target_col': 'detailed_label',
    'labels': {'-': '-',
               'C&C-HeartBeat-FileDownload': 'C&C-HeartBeat-FileDownload',
               'C&C-HeartBeat-Attack': 'C&C-HeartBeat-Attack',
               'PartOfAHorizontalPortScan': 'PartOfAHorizontalPortScan',
               'Attack': 'Attack',
               'C&C-PartOfAHorizontalPortScan': 'C&C-PartOfAHorizontalPortScan',
               'C&C': 'C&C',
               'DDoS': 'DDoS',
               'FileDownload': 'FileDownload',
               'C&C-FileDownload': 'C&C-FileDownload'},
    'mss': {'-': 1,
            'C&C-HeartBeat-FileDownload': 3,
            'C&C-HeartBeat-Attack': 3,
            'PartOfAHorizontalPortScan': 3,
            'Attack': 3,
            'C&C-PartOfAHorizontalPortScan': 3,
            'C&C': 3,
            'DDoS': 3,
            'FileDownload': 3,
            'C&C-FileDownload': 3},
    'markers': {'-': 'o',
                'C&C-HeartBeat-FileDownload': 'x',
                'C&C-HeartBeat-Attack': 'x',
                'PartOfAHorizontalPortScan': 'x',
                'Attack': 'x',
                'C&C-PartOfAHorizontalPortScan': 'x',
                'C&C': 'x',
                'DDoS': 'x',
                'FileDownload': 'x',
                'C&C-FileDownload': 'x'}}
