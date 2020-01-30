import shutil
import os
import logging

from datetime import datetime, time
from dateutil.parser import parse, parserinfo
from subprocess import CalledProcessError, check_output
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from ad_nids.utils.exception import DownloadError
from ad_nids.dataset import Dataset, create_meta
from ad_nids.utils.misc import sample_df, dd_mm_yyyy2mmdd
from ad_nids.process.columns import CIC_IDS2017_COLUMN_MAPPING, CIC_IDS2017_ATTACK_LABELS


DATASET_NAME = 'CIC-IDS2017'
OFFICE_HOURS = (time(7, 0, 0), time(21, 0, 0))


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

        exp_dt = flows.iloc[0]['timestamp']
        exp_date = exp_dt.strftime('%d-%m-%Y_%a')
        new_name = f'{idx}_{exp_date}_{exp_labels}.csv'

        flows.to_csv(path, index=False)
        shutil.move(path, path.parent/new_name)

    return


def create_report_day_cicids(meta, static_path):

    report = ""

    meta['timestamp'] = pd.to_datetime(meta['timestamp'], format='%Y/%m/%d %H:%M:%S')

    # assert only this day
    meta['day'] = meta['timestamp'].dt.strftime('%m%d')
    assert len(meta['day'].unique()) == 1

    meta['target'] = (meta['label'] != 'benign').astype(int)

    start_tstmp = meta['timestamp'].iloc[0]
    meta.loc[:, 'sec'] = (meta['timestamp'] - start_tstmp).dt.total_seconds()
    day = start_tstmp.strftime("%a %m/%d")
    #         print(day)
    report += '<h2> {} </h2>'.format(day)
    report += '</br>'
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


BASE = """
<!DOCTYPE html>
<html>
    <head>
    <style>
        table, th, td {
          border: 1px solid black;
          border-collapse: collapse;
        }
    </style>
    </head>
    <body>
    {{STUFF}}
    </body>
</html>
"""


def create_dataset_report_cicids17(dataset_path, report_path):

    dataset_path = Path(dataset_path).resolve()
    static_path = report_path/'static'
    static_path.mkdir(parents=True, exist_ok=True)

    report = ''

    for path in dataset_path.iterdir():

        name = path.name[:-len(path.suffix)]
        print(name)

        df = pd.read_csv(path)
        path_report = create_report_day_cicids(df, static_path)

        report += f'<h1> {name} </h1></br>'
        report += path_report
        report += '</br></br>'

    report = BASE.replace('{{STUFF}}', report)
    with open(report_path / 'report.html', 'w') as f:
        f.write(report)


def create_dataset_cicids17(dataset_path, train_scenarios, test_scenarios, create_hash=False):

    features = 'ORIG'

    train_scenario_idx = [s.split('_')[0] for s in train_scenarios]
    test_scenario_idx = [s.split('_')[0] for s in test_scenarios]

    dataset_name = '{}_TRAIN_{}_TEST_{}_{}'.format(
        DATASET_NAME,
        '-'.join(train_scenario_idx),
        '-'.join(test_scenario_idx),
        features
    )

    meta = create_meta(DATASET_NAME, train_scenarios, test_scenarios,
                       features=features, name=dataset_name)
    logging.info("Creating dataset {}...".format(meta['name']))

    train_paths = [dataset_path/f'{sc}.csv' for sc in train_scenarios]
    test_paths = [dataset_path/f'{sc}.csv' for sc in test_scenarios]

    # Naive concat
    # ToDo: protocol to meta
    meta_columns = ['flow_id', 'src_ip', 'dest_ip', 'src_port', 'dst_port', 'protocol',
                    'timestamp', 'label', 'label_orig']
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    train_meta = train.loc[:, meta_columns]
    train = train.drop(meta_columns, axis=1)
    train['target'] = (train_meta['label'] != 'benign').astype(np.int)

    test = pd.concat([pd.read_csv(p) for p in test_paths])
    test_meta = test.loc[:, meta_columns]
    test = test.drop(meta_columns, axis=1)
    train['target'] = (train_meta['label'] != 'benign').astype(np.int)

    logging.info("Done")

    return Dataset(train, test, train_meta, test_meta, meta, create_hash=create_hash)


if __name__ == '__main__':

    root_path = Path('/home/emikmis/data/cic-ids2017/')
    dataset_path = root_path/'data'/'CIC-IDS2017'
    report_path = root_path/'data_report'/'CIC-IDS2017'
    # cleanup_cidids17(dataset_path)
    create_dataset_report_cicids17(dataset_path, report_path)
