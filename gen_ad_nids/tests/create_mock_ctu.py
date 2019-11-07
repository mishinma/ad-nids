import os
import sys

import numpy as np
import pandas as pd

data_dir = sys.argv[1]

dest_dir = 'data/ctu_mock/'
if os.path.exists(dest_dir):
    raise FileExistsError('Test CTU13 dataset exists')

scenarios = ['2', '3', '9']

for scenario in scenarios:
    scenario_dir = os.path.join(data_dir, str(scenario))

    flow_files = [f for f in os.listdir(scenario_dir)
                  if os.path.splitext(f)[1] == '.binetflow']
    assert len(flow_files) == 1
    flow_file = flow_files[0]

    print(scenario, flow_file)

    flows = pd.read_csv(os.path.join(scenario_dir, flow_file))

    num_ips_sample = 3
    max_sample_size = 1000

    # Find botnet and normal ips
    botnet_idx = flows['Label'].str.contains('From-Botnet')
    normal_idx = flows['Label'].str.contains('From-Normal')
    botnet_ips = flows.loc[botnet_idx, 'SrcAddr'].unique()
    normal_ips = flows.loc[normal_idx, 'SrcAddr'].unique()

    if len(botnet_ips) > num_ips_sample:
        botnet_ips = np.random.choice(botnet_ips, num_ips_sample, replace=False)

    if len(normal_ips) > num_ips_sample:
        normal_ips = np.random.choice(normal_ips, num_ips_sample, replace=False)

    # Take flows for each ip
    sampled_ips = list(set(botnet_ips).union(set(normal_ips)))
    sampled_flows = []
    for ip in sampled_ips:
        sample = flows[flows['SrcAddr'] == ip]
        sample = sample.iloc[:max_sample_size]
        sampled_flows.append(sample)

    # Take some random flows
    sampled_flows.append(flows.iloc[:max_sample_size])

    sampled_flows = pd.concat(sampled_flows).sort_values(by='StartTime')

    scenario_dest_dir = os.path.join(dest_dir, str(scenario))
    os.makedirs(scenario_dest_dir)
    sampled_flows.to_csv(os.path.join(scenario_dest_dir, flow_file), index=None)

