

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


########## CTU-13 ############

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


########## UGR-16 ############

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
