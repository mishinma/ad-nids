import shutil
import os
import logging
import multiprocessing as mp

from timeit import default_timer as timer
from datetime import datetime, time
from dateutil.parser import parse, parserinfo
from subprocess import check_output
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from ad_nids.utils.aggregate import aggregate_features_pool
from ad_nids.utils.exception import DownloadError
from ad_nids.dataset import Dataset
from ad_nids.utils.misc import is_valid_ip, sample_df, fair_attack_sample
from ad_nids.process.columns import CIC_IDS2017_COLUMN_MAPPING, CIC_IDS2017_ATTACK_LABELS, \
    CIC_IDS2017_FEATURES, CIC_IDS2017_META_COLUMNS, \
    CIC_IDS2017_BINARY_FEATURES, CIC_IDS2017_CATEGORICAL_FEATURE_MAP, CIC_IDS2017_NUMERICAL_FEATURES, \
    CIC_IDS2017_AGGR_FUNCTIONS, CIC_IDS2017_AGGR_COLUMNS, CIC_IDS2017_AGGR_META_COLUMNS, \
    CIC_IDS2017_PROTOCOL_MAPPING
from ad_nids.report.general import BASE
from ad_nids.utils.misc import set_seed


DATASET_NAME = 'CIC-IDS2017'
OFFICE_HOURS = (time(7, 0, 0), time(21, 0, 0))
TOTAL_SCENARIOS = 8
ORIG_TRAIN_SCENARIOS = [3, 0, 1, 5]
ORIG_TEST_SCENARIOS = [2, 4, 6, 7]
ALL_SCENARIOS = list(range(TOTAL_SCENARIOS))

THRESHOLD_BATCH_N_SAMPLES = 10000
THRESHOLD_BATCH_PERC_OUTLIER = 5
TEST_BATCH_PERC_OUTLIER = 10
PREPARE_DATA_RANDOM_SEED = 42


def _parse_scenario(name):
    return int(name.split('_')[0])


def download_cicids17(data_path):

    logging.info('Downloading the dataset')

    data_path = Path(data_path).resolve()
    data_path.mkdir(parents=True)

    mycwd = os.getcwd()
    os.chdir(data_path)

    try:
        check_output(
            'wget --header="Host: 205.174.165.80" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" --header="Accept:'
            ' text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/'
            'signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" --header="Referer:'
            ' http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/" "http://205.174.165.80/CICDataset/CIC-IDS-2017/'
            'Dataset/GeneratedLabelledFlows.zip" -O "GeneratedLabelledFlows.zip" -c', shell=True
        )
    except Exception as e:
        raise DownloadError('Could not download the dataset')

    check_output(['unzip', "GeneratedLabelledFlows.zip"])
    os.remove(data_path / "GeneratedLabelledFlows.zip")

    middle_path = list(data_path.iterdir())[0]  # 'TrafficLabelling '
    for path in middle_path.iterdir():
        shutil.move(path, data_path / path.name)
    shutil.rmtree(middle_path)

    os.chdir(mycwd)  # go back where you came from

    # fix any possible enconding errors:
    for path in data_path.iterdir():
        with open(path, 'rb') as f:
            data = f.read()
        data = data.decode('utf-8', 'ignore')
        with open(path, 'w') as f:
            f.write(data)

    return


def cleanup_cidids17(data_path):

    logging.info('Cleaning up the data')

    data_path = Path(data_path).resolve()

    def parse_time(s):

        dt = parse(s, dayfirst=True)

        # they did not include am/pm info
        # if outside of office hours -> add pm
        if not OFFICE_HOURS[0] <= dt.time() <= OFFICE_HOURS[1]:
            dt = parse(s + ' pm', dayfirst=True)

        assert OFFICE_HOURS[0] <= dt.time() <= OFFICE_HOURS[1]

        return dt

    for idx, path in enumerate(sorted(data_path.iterdir())):

        # check if already processed
        try:
            datetime.strptime(path.name.split('_')[0], '%d-%m-%Y')
        except ValueError:
            pass
        else:
            return

        weekday = path.name.split('-')[0]
        assert parserinfo().weekday(weekday) is not None

        flows = pd.read_csv(path)

        flows = flows.dropna(how='all')
        flows = flows.rename(columns=lambda c: CIC_IDS2017_COLUMN_MAPPING.get(c.strip(), c))
        flows = flows[list(CIC_IDS2017_COLUMN_MAPPING.values())]
        assert len(flows.columns) == len(CIC_IDS2017_COLUMN_MAPPING)
        valid_ip_idx = flows['src_ip'].apply(is_valid_ip) & flows['dst_ip'].apply(is_valid_ip)
        flows = flows[valid_ip_idx]

        flows['timestamp'] = flows['timestamp'].apply(parse_time)
        day = flows['timestamp'].dt.strftime('%A')
        wrong_day_idx = day != weekday
        num_wrong_dates = np.sum(wrong_day_idx)
        if num_wrong_dates > 0:
            logging.info(f'Found {num_wrong_dates} flows with a wrong date')
            flows = flows[~wrong_day_idx]

        flows = flows.sort_values(by=['timestamp'])

        # Some rows in the files are just column names
        # Ignore them
        bad_rows = flows.index[flows['protocol'].astype(str).str.isalpha()]
        flows = flows.drop(bad_rows).reset_index(drop=True)
        flows['protocol'] = flows['protocol'].astype(int).apply(lambda s: CIC_IDS2017_PROTOCOL_MAPPING[s])

        # Fill na
        flows['flow_byts/s'] = flows['flow_byts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        flows['flow_pkts/s'] = flows['flow_pkts/s'].astype(np.float64)\
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # drop all other nan rows
        flows = flows.dropna(how='any')

        # Create a new file name
        flows['label_orig'] = flows['label']
        flows['label'] = flows['label'].apply(lambda x: CIC_IDS2017_ATTACK_LABELS.get(x, 'benign'))
        attack_labels = list(flows['label'].unique())
        attack_labels.remove('benign')
        if attack_labels:
            exp_labels = '-'.join(sorted(attack_labels))
        else:
            exp_labels = 'benign'

        flows['scenario'] = idx
        flows['target'] = (flows['label'] != 'benign').astype(np.int)

        exp_dt = flows.iloc[0]['timestamp']
        exp_date = exp_dt.strftime('%d-%m-%Y_%a')
        new_name = f'{idx}_{exp_date}_{exp_labels}.csv'

        flows.to_csv(path, index=False)
        shutil.move(path, path.parent/new_name)

    return


def create_mock_cicids17(data_path, mock_path, num_ips_sample=3, max_sample_size=1000,
                         mock_scenarios=None):

    if mock_scenarios is None:
        mock_scenarios = [2, 5]

    mock_path.mkdir(parents=True, exist_ok=True)

    for sc_i in mock_scenarios:
        logging.info(f'Processing scenario {sc_i}')
        path = [p for p in data_path.iterdir() if p.name.startswith(str(sc_i))][0]
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

        out_path = mock_path/path.name
        sampled_flows.to_csv(out_path, index=False)


def create_report_day_cicids(meta, static_path, timestamp_col):

    report = ""

    meta[timestamp_col] = pd.to_datetime(meta[timestamp_col], format='%Y/%m/%d %H:%M:%S')

    # assert only this day
    meta['day'] = meta[timestamp_col].dt.strftime('%m%d')
    assert len(meta['day'].unique()) == 1

    start_tstmp = meta[timestamp_col].iloc[0]
    meta.loc[:, 'sec'] = (meta[timestamp_col] - start_tstmp).dt.total_seconds()
    day = start_tstmp.strftime("%a %m/%d")
    #         print(day)
    report += '<h2> {} </h2>'.format(day)
    report += '</br>'

    if 'label' not in meta:
        meta.loc[:, 'label'] = 'benign'
        meta.loc[meta['target'] == 1, 'label'] = 'attack'

    label_counts = meta['label'].value_counts()
    report += '<div>' + pd.DataFrame(label_counts).to_html() + '</div>'
    report += '</br>'

    total_cnt = np.sum(label_counts)
    attack_cnt = np.sum([cnt for lbl, cnt in label_counts.iteritems()
                         if lbl != 'benign'])
    contamination_perc = attack_cnt / total_cnt * 100
    report += '<h2> Contamination perc: {:.2f} </h2>'.format(contamination_perc)
    report += '</br>'

    plt.close('all')
    for lbl in label_counts.index:
        sns.distplot(meta.loc[meta['label'] == lbl, 'sec'], rug=True, kde=False)
    plt.legend(labels=label_counts.index)
    plot_name = start_tstmp.strftime("{}.png".format(str(uuid4())[:5]))
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'

    plt.close('all')
    for lbl in label_counts.index:
        sns.distplot(meta.loc[meta['label'] == lbl, 'sec'], rug=True)
    plt.legend(labels=label_counts.index)
    plot_name = start_tstmp.strftime("{}.png".format(str(uuid4())[:5]))
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'
    plt.close('all')

    return report


def create_dataset_report_cicids17(dataset_path, report_path, timestamp_col='timestamp'):

    dataset_path = Path(dataset_path).resolve()
    static_path = report_path/'static'
    static_path.mkdir(parents=True, exist_ok=True)

    report = ''

    for path in dataset_path.iterdir():

        name = path.name[:-len(path.suffix)]
        print(name)

        df = pd.read_csv(path)
        path_report = create_report_day_cicids(df, static_path, timestamp_col)

        report += f'<h1> {name} </h1></br>'
        report += path_report
        report += '</br></br>'

    report = BASE.replace('{{STUFF}}', report)
    with open(report_path / 'report.html', 'w') as f:
        f.write(report)


def create_dataset_cicids17(dataset_path,
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
        train_normal, test_normal = train_test_split(data_normal,
                                                     test_size=test_size,
                                                     random_state=random_seed)
        train_outlier, test_outlier = train_test_split(data_outlier,
                                                       test_size=(1 - test_size),
                                                       random_state=random_seed)

        # Construct the threshold batch
        threshold_n_samples = THRESHOLD_BATCH_N_SAMPLES
        threshold_perc_outlier = THRESHOLD_BATCH_PERC_OUTLIER
        threshold_n_outlier_samples = int(threshold_n_samples * .01 * threshold_perc_outlier)
        threshold_n_normal_samples = threshold_n_samples - threshold_n_outlier_samples
        threshold_normal = sample_df(train_normal, threshold_n_normal_samples)
        threshold_outlier = fair_attack_sample(train_outlier, threshold_n_outlier_samples)

        # Resample the test batch
        test_perc_outlier = TEST_BATCH_PERC_OUTLIER
        test_n_normal_samples = test_normal.shape[0]
        test_n_outlier_samples = int(test_n_normal_samples * test_perc_outlier / (100 - test_perc_outlier))
        test_outlier = fair_attack_sample(test_outlier, test_n_outlier_samples)

        train = train_normal.sample(frac=1)
        threshold = pd.concat([threshold_normal, threshold_outlier], axis=0).sample(frac=1)
        test = pd.concat([test_normal, test_outlier], axis=0).sample(frac=1)

    else:
        name = '{}_TRAIN_{}_TEST_{}'.format(
            DATASET_NAME,
            '-'.join(map(str, train_scenarios)),
            '-'.join(map(str, test_scenarios)),
        )

        train_paths = [p for p in dataset_path.iterdir()
                       if _parse_scenario(p.name) in train_scenarios]
        test_paths = [p for p in dataset_path.iterdir()
                       if _parse_scenario(p.name) in test_scenarios]
        train = pd.concat([pd.read_csv(p) for p in train_paths])

        # Create threshold and clean test
        train_outlier_idx = train['target'] == 1
        train_outlier = train.loc[train_outlier_idx]
        train_normal = train.loc[~train_outlier_idx]

        # Construct the threshold batch
        threshold_n_samples = THRESHOLD_BATCH_N_SAMPLES
        threshold_perc_outlier = THRESHOLD_BATCH_PERC_OUTLIER
        threshold_n_outlier_samples = int(threshold_n_samples * .01 * threshold_perc_outlier)
        threshold_n_normal_samples = threshold_n_samples - threshold_n_outlier_samples
        threshold_normal = sample_df(train_normal, threshold_n_normal_samples)
        threshold_outlier = fair_attack_sample(train_outlier, threshold_n_outlier_samples)

        train = train_normal
        threshold = pd.concat([threshold_normal, threshold_outlier], axis=0).sample(frac=1)
        test = pd.concat([pd.read_csv(p) for p in test_paths])



    if frequency is not None:
        name += '_AGGR_{}'.format(frequency)
        feature_columns = list(CIC_IDS2017_AGGR_FUNCTIONS.keys())
        meta_columns = CIC_IDS2017_AGGR_META_COLUMNS
        # all numerical
        features_info = {
            'categorical_feature_map': {},
            'categorical_features': [],
            'binary_features': [],
            'numerical_features': [k for k in CIC_IDS2017_AGGR_FUNCTIONS.keys() if k != 'target']
        }
    else:
        feature_columns = list(CIC_IDS2017_FEATURES.keys())
        meta_columns = CIC_IDS2017_META_COLUMNS
        features_info = {
            'categorical_feature_map': CIC_IDS2017_CATEGORICAL_FEATURE_MAP,
            'categorical_features': list(CIC_IDS2017_CATEGORICAL_FEATURE_MAP.keys()),
            'binary_features': CIC_IDS2017_BINARY_FEATURES,
            'numerical_features': CIC_IDS2017_NUMERICAL_FEATURES
        }
    train_meta = train.loc[:, meta_columns]
    train = train.loc[:, feature_columns]
    test_meta = test.loc[:, meta_columns]
    test = test.loc[:, feature_columns]
    threshold_meta = threshold.loc[:, meta_columns]
    threshold = threshold.loc[:, feature_columns]

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

    dataset = Dataset(train, threshold, test, train_meta, threshold_meta, test_meta, meta)

    logging.info('Done!')

    return dataset


def _aggregate_flows_wkr(args):

    grp_name, grp = args

    flow_stats = {col_name: aggr_fn(grp)
                  for col_name, aggr_fn in CIC_IDS2017_AGGR_FUNCTIONS.items()}
    record = {
        'src_ip': grp_name[0],
        'time_window_start': grp_name[1],
        **flow_stats
    }

    return record


def aggregate_flows_cicids17(data_path, aggr_path, processes=-1, frequency='T'):

    if processes == -1:
        processes = mp.cpu_count() - 1

    logging.info('Aggregating the data')
    data_path = Path(data_path).resolve()
    aggr_path = Path(aggr_path).resolve()
    aggr_path.mkdir(exist_ok=True)

    for path in data_path.iterdir():
        sc_i = _parse_scenario(path.name)
        logging.info(f'Processing scenario {sc_i}')
        start_time = timer()

        out_path = aggr_path / path.name

        flows = pd.read_csv(path)
        flows['timestamp'] = pd.to_datetime(flows['timestamp'])
        grouped = flows.groupby(['src_ip', pd.Grouper(key='timestamp', freq=frequency)])
        aggr_flows = aggregate_features_pool(grouped, _aggregate_flows_wkr, processes)
        aggr_flows = pd.DataFrame.from_records(aggr_flows)
        aggr_flows['scenario'] = sc_i
        aggr_flows = aggr_flows[CIC_IDS2017_AGGR_COLUMNS]
        aggr_flows = aggr_flows.sort_values(by='time_window_start').reset_index(drop=True)
        aggr_flows = aggr_flows.fillna(0)
        aggr_flows.to_csv(out_path, index=False)

        logging.info("Done {0:.2f}".format(timer() - start_time))


if __name__ == '__main__':

    root_path = Path('/home/emikmis/data/cic-ids2017/')
    dataset_path = root_path/'data'/'CIC-IDS2017'
    report_path = root_path/'data_report'/'CIC-IDS2017'
    # cleanup_cidids17(dataset_path)
    create_dataset_report_cicids17(dataset_path, report_path)
