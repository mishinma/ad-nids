
import os
import time
import logging
import shutil
import multiprocessing as mp

from collections import Counter
from subprocess import CalledProcessError, check_output
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.aggregate import aggregate_features_pool
from ad_nids.utils.exception import DownloadError
from ad_nids.utils.misc import set_seed
from ad_nids.process.columns import IOT_23_ORIG_SCENARIO_NAME_MAPPING, IOT_23_ORIG_COLUMN_MAPPING, \
    IOT_23_HISTORY_LETTERS, IOT_23_REPLACE_EMPTY_ZERO_FEATURES, IOT_23_COLUMNS, IOT_23_META_COLUMNS,  \
    IOT_23_FEATURES, IOT_23_AGGR_COLUMNS, IOT_23_AGGR_FUNCTIONS, IOT_23_AGGR_META_COLUMNS, \
    IOT_23_CATEGORICAL_FEATURE_MAP, IOT_23_BINARY_FEATURES, IOT_23_NUMERICAL_FEATURES
from ad_nids.report.general import BASE

DATASET_NAME = 'IOT-23'
MAX_FLOWS_TO_PROCESS = 5000000  # 5M


def _parse_scenario_idx(name):
    return int(name.split('_')[0])


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


def cleanup_iot23(data_path):

    logging.info('Cleaning up the data')
    data_path = Path(data_path).resolve()

    for path in data_path.iterdir():
        name = path.with_suffix('').name
        logging.info(f'Processing scenario {name}')
        flows = pd.read_csv(path, skiprows=8, sep='\s+', nrows=MAX_FLOWS_TO_PROCESS,
                            header=None, names=list(IOT_23_ORIG_COLUMN_MAPPING.keys()))

        # drop the last row if less than MAX_FLOWS_TO_PROCESS
        last_row = flows.iloc[-1]
        if last_row['ts'] == '#close':
            flows = flows.drop(last_row.name, axis=0)

        flows = cleanup_iot23_flows(flows)
        flows.to_csv(path, index=False)


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

    attack_value_counts = data['detailed_label'].value_counts()
    report += '<h3> Labels distribution </h3>'
    report += pd.DataFrame(attack_value_counts).to_html()
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


def create_data_report_iot23(dataset_path, report_path, timestamp_col='timestamp'):

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


def create_dataset_iot23(dataset_path,
                         train_scenarios=None, test_scenarios=None, frequency=None,
                         test_size=None, random_seed=None,
                         create_hash=False):

    dataset_path = Path(dataset_path).resolve()
    logging.info("Creating dataset")

    if random_seed is not None:
        set_seed(random_seed)
        name = '{}_TEST_SIZE_{}_RANDOM_SEED_{}'.format(
            DATASET_NAME, test_size, random_seed
        )
        data = pd.concat([pd.read_csv(p) for p in dataset_path.iterdir()])
        data_outlier_idx = data['target'] == 1
        data_outlier = data.loc[data_outlier_idx]
        data_normal = data.loc[~data_outlier_idx]
        # take:  test_size normal, (1 - test_size) outlier
        train_normal, test_normal = train_test_split(data_normal, test_size=test_size, random_state=random_seed)
        train_outlier, test_outlier = train_test_split(data_outlier, test_size=(1-test_size), random_state=random_seed)
        train = pd.concat([train_normal, train_outlier], axis=0).sample(frac=1)
        test = pd.concat([test_normal, test_outlier], axis=0).sample(frac=1)

    else:
        name = '{}_TRAIN_{}_TEST_{}'.format(
            DATASET_NAME,
            '-'.join(map(str, train_scenarios)),
            '-'.join(map(str, test_scenarios)),
        )
        train_paths = [p for p in dataset_path.iterdir()
                       if _parse_scenario_idx(p.name) in train_scenarios]
        test_paths = [p for p in dataset_path.iterdir()
                      if _parse_scenario_idx(p.name) in test_scenarios]
        train = pd.concat([pd.read_csv(p) for p in train_paths])
        test = pd.concat([pd.read_csv(p) for p in test_paths])

    if frequency is not None:
        name += '_AGGR_{}'.format(frequency)
        feature_columns = list(IOT_23_AGGR_FUNCTIONS.keys())
        meta_columns = IOT_23_AGGR_META_COLUMNS
        # all numerical
        features_info = {
            'categorical_feature_map': {},
            'categorical_features': [],
            'binary_features': [],
            'numerical_features': [k for k in IOT_23_AGGR_FUNCTIONS.keys() if k != 'target']
        }
    else:
        feature_columns = list(IOT_23_FEATURES.keys())
        meta_columns = IOT_23_META_COLUMNS
        features_info = {
            'categorical_feature_map': IOT_23_CATEGORICAL_FEATURE_MAP,
            'categorical_features': list(IOT_23_CATEGORICAL_FEATURE_MAP.keys()),
            'binary_features': IOT_23_BINARY_FEATURES,
            'numerical_features': IOT_23_NUMERICAL_FEATURES
        }

    train_meta = train.loc[:, meta_columns]
    train = train.loc[:, feature_columns]
    test_meta = test.loc[:, meta_columns]
    test = test.loc[:, feature_columns]

    meta = {
        'data_hash': None,
        'dataset_name': DATASET_NAME,
        'test_size': test_size,
        'random_seed': random_seed,
        'frequency': frequency,
        'train_scenarios': train_scenarios,
        'test_scenarios': test_scenarios,
        'name': name
    }
    meta.update(features_info)

    dataset = Dataset(train, test, train_meta, test_meta, meta,
                      create_hash=create_hash)

    logging.info('Done!')

    return dataset


def _aggregate_flows_wkr(args):

    grp_name, grp = args

    flow_stats = {col_name: aggr_fn(grp)
                  for col_name, aggr_fn in IOT_23_AGGR_FUNCTIONS.items()}
    record = {
        'src_ip': grp_name[0],
        'time_window_start': grp_name[1],
        **flow_stats
    }

    return record


def aggregate_flows_iot23(data_path, aggr_path, processes=-1, frequency='T'):

    if processes == -1:
        processes = mp.cpu_count() - 1

    logging.info('Aggregating the data')
    data_path = Path(data_path).resolve()
    aggr_path = Path(aggr_path).resolve()
    aggr_path.mkdir(exist_ok=True)

    for path in data_path.iterdir():

        name = path.with_suffix('').name
        logging.info(f'Processing scenario {name}')
        start_time = time.time()

        out_path = aggr_path / path.name

        flows = pd.read_csv(path)
        flows['timestamp'] = pd.to_datetime(flows['timestamp'])
        grouped = flows.groupby(['src_ip', pd.Grouper(key='timestamp', freq=frequency)])
        aggr_flows = aggregate_features_pool(grouped, _aggregate_flows_wkr, processes)
        aggr_flows = pd.DataFrame.from_records(aggr_flows)
        aggr_flows['scenario'] = _parse_scenario_idx(name)
        aggr_flows = aggr_flows[IOT_23_AGGR_COLUMNS]
        aggr_flows = aggr_flows.sort_values(by='time_window_start').reset_index(drop=True)
        aggr_flows = aggr_flows.fillna(0)
        aggr_flows.to_csv(out_path, index=False)

        logging.info("Done {0:.2f}".format(time.time() - start_time))


if __name__ == '__main__':
    dataset_path = '/home/emikmis/data/nids/IOT23-2'
    download_iot23(dataset_path)
