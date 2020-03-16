import shutil
import logging
import os

from datetime import datetime
from collections import Counter
from subprocess import CalledProcessError, check_output
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.exception import DownloadError
from ad_nids.process.columns import IOT_23_ORIG_SCENARIO_NAME_MAPPING, IOT_23_ORIG_COLUMN_MAPPING, \
    IOT_23_HISTORY_LETTERS, IOT_23_REPLACE_EMPTY_ZERO_FEATURES, IOT_23_COLUMNS


from ad_nids.report.general import BASE

DATASET_NAME = 'IOT-23'


def download_iot23(dataset_path, separate_mirai=True):
    logging.info('Downloading the dataset')

    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True)

    mycwd = os.getcwd()
    os.chdir(dataset_path)

    try:
        check_output(
             'wget --header="Host: mcfp.felk.cvut.cz" '
             '--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" '
             '--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" '
             '--header="Accept-Language: en-US,en;q=0.9" "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz" '
             '-O "iot_23_datasets_small.tar.gz" -c --no-check-certificate', shell=True
            )
        returncode = 0
    except CalledProcessError as e:
        raise DownloadError('Could not download the dataset.'
                            ' Return code: ', e.returncode)

    check_output(['tar', '-xzvf', "iot_23_datasets_small.tar.gz"])

    for path in dataset_path.glob('**/*.log.labeled'):
        scenario = path.parent.parent.name
        idx, name = IOT_23_ORIG_SCENARIO_NAME_MAPPING[scenario]
        new_name = '{:02d}_{}.csv'.format(idx, name)
        shutil.move(path, dataset_path / new_name)
    
    shutil.rmtree(dataset_path/'opt')
    os.remove(dataset_path/"iot_23_datasets_small.tar.gz")
    os.chdir(mycwd)  # go back where you came from

    if separate_mirai:
        dataset_mirai_path = dataset_path.parent/(dataset_path.name + '_mirai')
        dataset_mirai_path.mkdir(exist_ok=True)
        for path in dataset_path.iterdir():
            if 'mirai' in path.name:
                shutil.move(path, dataset_mirai_path/path.name)
        dataset_other_path = dataset_path.parent/(dataset_path.name + '_other')
        # rename the rest to other
        shutil.move(dataset_path, dataset_other_path)

    return


def _parse_history(hist):
    orig_letters_cnt = Counter()
    resp_letters_cnt = Counter()
    is_empty = False
    is_dir_flipped = False

    for l in hist:

        if l == '-':
            is_empty = True
            break

        if l == '^':
            is_dir_flipped = True
            continue

        from_orig = l.isupper()
        l = l.lower()
        if from_orig:
            orig_letters_cnt[l] += 1
        else:
            resp_letters_cnt[l] += 1

    parsed = pd.Series({
        'history_empty': int(is_empty),
        'history_dir_flipped': int(is_dir_flipped),
        **{f'orig_history_{l}_cnt': orig_letters_cnt[l] for l in IOT_23_HISTORY_LETTERS},
        **{f'resp_history_{l}_cnt': resp_letters_cnt[l] for l in IOT_23_HISTORY_LETTERS}
    })

    return parsed


def cleanup_iot23_flows(flows):

    flows = flows.rename(columns=IOT_23_ORIG_COLUMN_MAPPING)

    flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s')
    flows = flows.sort_values(by='timestamp').reset_index(drop=True)

    # replace all empty with unset
    flows = flows.replace('(empty)', '-')

    flows['target'] = (flows['label'] == 'Malicious').astype(np.int)

    parsed_history = flows['history'].apply(_parse_history)
    flows = pd.concat([flows, parsed_history], axis=1)

    flows[IOT_23_REPLACE_EMPTY_ZERO_FEATURES] = flows[IOT_23_REPLACE_EMPTY_ZERO_FEATURES].replace('-', 0)
    flows = flows[IOT_23_COLUMNS]

    return flows


def create_mock_iot23(data_path, mock_path, num_ips_sample=3, max_sample_size=1000,
                      num_mock_scenarios=None):

    mock_path.mkdir(parents=True, exist_ok=True)
    mock_scenarios = list(data_path.iterdir())
    if num_mock_scenarios is not None:
        mock_scenarios = mock_scenarios[:num_mock_scenarios]

    for path in mock_scenarios:
        name = path.with_suffix('').name
        logging.info(f'Processing scenario {name}')
        flows = pd.read_csv(path)

        botnet_ips = flows.loc[flows['target'] == 1, 'src_ip'].unique()
        normal_ips = flows.loc[flows['target'] == 0, 'src_ip'].unique()

        if len(botnet_ips) > num_ips_sample:
            botnet_ips = np.random.choice(botnet_ips, num_ips_sample, replace=False)
        if len(normal_ips) > num_ips_sample:
            normal_ips = np.random.choice(normal_ips, num_ips_sample, replace=False)

        sampled_ips = list(set(botnet_ips).union(set(normal_ips)))
        sampled_flows = []
        for ip in sampled_ips:
            sample = flows[flows['src_ip'] == ip]
            sample = sample.iloc[:max_sample_size]
            sampled_flows.append(sample)

        # Take some random flows
        sampled_flows.append(flows.iloc[:max_sample_size])
        sampled_flows = pd.concat(sampled_flows).sort_values(by='timestamp')

        sampled_flows.to_csv(mock_path/path.name, index=False)


def create_report_scenario_iot23(data, static_path, timestamp_col='timestamp'):

    report = ""

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    start_tstmp = data[timestamp_col].iloc[0]
    data.loc[:, 'sec'] = (data[timestamp_col] - start_tstmp).dt.total_seconds()

    num_flows = data.shape[0]
    report += '<h3>' + f'Num flows: {num_flows}' + ' </h3>'
    report += '</br>'

    attack_cnt = np.sum(data['target'])
    report += '<h3>' + f'Botnet flows: {attack_cnt}' + ' </h3>'
    report += '</br>'

    contamination_perc = attack_cnt / num_flows * 100
    report += '<h2> Contamination perc: {:.2f} </h2>'.format(contamination_perc)
    report += '</br>'

    plt.close('all')
    labels = []
    groups = data.groupby('detailed_label')
    for name, group in groups:
        sns.distplot(group['sec'], rug=True, kde=False)
        labels.append(name)
    plt.legend(labels=labels)
    plot_name = start_tstmp.strftime("{}.png".format(str(uuid4())[:5]))
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'

    plt.close('all')
    for name, group in groups:
        sns.distplot(group['sec'], rug=True)
    plt.legend(labels=labels)
    plot_name = "{}.png".format(str(uuid4())[:5])
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'
    plt.close('all')

    return report


def create_data_report_ctu13(dataset_path, report_path, timestamp_col='timestamp'):

    dataset_path = Path(dataset_path).resolve()
    static_path = report_path/'static'
    static_path.mkdir(parents=True, exist_ok=True)

    report = ''

    for path in sorted(dataset_path.iterdir()):

        name = path.name[:-len(path.suffix)]
        print(name)

        data = pd.read_csv(path)
        path_report = create_report_scenario_iot23(data, static_path, timestamp_col=timestamp_col)

        report += f'<h1> {name} </h1></br>'
        report += path_report
        report += '</br></br>'

    report = BASE.replace('{{STUFF}}', report)
    with open(report_path / 'report.html', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    dataset_path = '/home/emikmis/data/nids/IOT23-2'
    download_iot23(dataset_path)
