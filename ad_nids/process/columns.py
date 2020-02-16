
from collections import OrderedDict

import numpy as np
from scipy.stats import entropy

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
    'label',
    'target'
]

CTU_13_FEATURES = [
    'proto',  # tcp, udp, icmp or other; categorical
    'dur',
    'fwd_dir',  # '>' in dir
    'bwd_dir',  # '<' in dir
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
    'target'
]

CTU_13_CATEGORICAL_FEATURES_MAP = {
    'proto': ['tcp', 'udp', 'icmp', 'other']
}

CTU_13_BINARY_FEATURES = [
    'fwd_dir',
    'bwd_dir',
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
]


CTU_13_META = [
    'timestamp',
    'src_ip',
    'src_port',
    'dst_ip',
    'dst_port',
    'proto',  # tcp, udp, icmp or other; categorical
    'dir',
    'state',
    'label',
]

# numerical:  mean, min, max, std, median
# categorical: num_unique, entropy
CTU_13_AGGR_FEATURES = [
    'total_cnt',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_std',
    'dur_median',
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
]

CTU_13_AGGR_META = []


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
########## CIC-IDS  ############
"""


CIC_IDS_FEATURE_COLUMNS = [
    'flow_dur',
    'tot_fwd_pkts',
    'tot_bwd_pkts',
    'tot_len_fwd_pkts',
    'tot_len_bwd_pkts',
    'fwd_pkt_len_max',
    'fwd_pkt_len_min',
    'fwd_pkt_len_mean',
    'fwd_pkt_len_std',
    'bwd_pkt_len_max',
    'bwd_pkt_len_min',
    'bwd_pkt_len_mean',
    'bwd_pkt_len_std',
    'flow_byts/s',
    'flow_pkts/s',
    'flow_iat_mean',
    'flow_iat_std',
    'flow_iat_max',
    'flow_iat_min',
    'fwd_iat_tot',
    'fwd_iat_mean',
    'fwd_iat_std',
    'fwd_iat_max',
    'fwd_iat_min',
    'bwd_iat_tot',
    'bwd_iat_mean',
    'bwd_iat_std',
    'bwd_iat_max',
    'bwd_iat_min',
    'fwd_psh_flags',
    'bwd_psh_flags',
    'fwd_urg_flags',
    'bwd_urg_flags',
    'fwd_header_len',
    'bwd_header_len',
    'fwd_pkts/s',
    'bwd_pkts/s',
    'pkt_len_min',
    'pkt_len_max',
    'pkt_len_mean',
    'pkt_len_std',
    'pkt_len_var',
    'fin_flag_cnt',
    'syn_flag_cnt',
    'rst_flag_cnt',
    'psh_flag_cnt',
    'ack_flag_cnt',
    'urg_flag_cnt',
    'cwe_flag_cnt',
    'ece_flag_cnt',
    'down/up_ratio',
    'pkt_size_avg',
    'fwd_seg_size_avg',
    'bwd_seg_size_avg',
    'fwd_header_len',
    'fwd_byts/blk_avg',
    'fwd_pkts/blk_avg',
    'fwd_blk_rate_avg',
    'bwd_byts/blk_avg',
    'bwd_pkts/blk_avg',
    'bwd_blk_rate_avg',
    'subflow_fwd_pkts',
    'subflow_fwd_byts',
    'subflow_bwd_pkts',
    'subflow_bwd_byts',
    'init_fwd_win_byts',
    'init_bwd_win_byts',
    'fwd_act_data_pkts',
    'fwd_seg_size_min',
    'active_mean',
    'active_std',
    'active_max',
    'active_min',
    'idle_mean',
    'idle_std',
    'idle_max',
    'idle_min',
    'label'
]


"""
########## CSE-CIC-IDS2017 ############
"""

CIC_IDS2017_COLUMN_MAPPING = {
    'Flow ID': 'flow_id',
    'Source IP': 'src_ip',
    'Source Port': 'src_port',
    'Destination IP': 'dest_ip',
    'Destination Port': 'dest_port',
    'Protocol': 'protocol',
    'Timestamp': 'timestamp',
    'Flow Duration': 'flow_dur',
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