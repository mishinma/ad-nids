
import os
import time
import logging
import shutil
import multiprocessing as mp

from pathlib import Path
from subprocess import check_output
from uuid import uuid4

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


from ad_nids.utils.exception import DownloadError
from ad_nids.dataset import Dataset, create_meta
from ad_nids.process.aggregate import aggregate_extract_features
from ad_nids.process.columns import CTU_13_ORIG_COLUMN_MAPPING, TCP_FLAGS, CTU_13_PROTOS, \
    CTU_13_COLUMNS, CTU_13_FEATURES, CTU_13_META
from ad_nids.report import BASE


DATASET_NAME = 'CTU-13'

TOTAL_SCENARIOS = 13
ORIG_TRAIN_SCENARIOS = [3, 4, 5, 7, 10, 11, 12, 13]
ORIG_TEST_SCENARIOS = [1, 2, 6, 8, 9]
ALL_SCENARIOS = list(range(1, TOTAL_SCENARIOS + 1))


def download_ctu13(data_path):

    logging.info('Downloading the dataset')

    data_path = Path(data_path).resolve()
    data_path.mkdir(parents=True)
    data_root = data_path.parent

    mycwd = os.getcwd()
    os.chdir(data_root)

    try:
        check_output(
            'wget --header="Host: mcfp.felk.cvut.cz" '
            '--header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36" '
            '--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,'
            '*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" '
            '--header="Referer: https://www.stratosphereips.org/datasets-ctu13" '
            '"https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2" '
            '--no-check-certificate '
            '-O "CTU-13-Dataset.tar.bz2" -c', shell=True
        )
    except Exception as e:
        raise DownloadError('Could not download the dataset')

    check_output(['tar', '-xvf', "CTU-13-Dataset.tar.bz2"])
    os.remove(data_root / "CTU-13-Dataset.tar.bz2")

    for path in (data_root / "CTU-13-Dataset").glob('**/*.binetflow'):
        scenario_i = int(path.parent.name)
        new_name = '{:02d}.csv'.format(scenario_i)
        shutil.move(str(path), str(data_path/new_name))
    shutil.rmtree(str(data_root / "CTU-13-Dataset"))

    os.chdir(mycwd)  # go back where you came from

    return


def format_ctu_flows(flows):

    flows = flows.rename(columns=CTU_13_ORIG_COLUMN_MAPPING)

    flows['timestamp'] = pd.to_datetime(flows['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    flows = flows.sort_values(by='timestamp').reset_index(drop=True)

    flows['target'] = flows['label'].str.startswith('flow=From-Botnet').astype(np.int)

    # Type of Service
    flows['src_tos'] = (flows['src_tos'] == 0).astype(np.int)
    flows['dst_tos'] = (flows['dst_tos'] == 0).astype(np.int)

    # create flags
    trans_dir = {'fwd', 'bwd'}
    flag_cols = [f'{d}_{f}_flag' for d in trans_dir for f in TCP_FLAGS.values()]

    flows = flows.assign(**{fc: 0 for fc in flag_cols})

    tcp_state = flows.loc[flows['proto'] == 'tcp', ['state']]
    tcp_state[['fwd_state', 'bwd_state']] = tcp_state['state'].str.split('_', expand=True)

    for d in trans_dir:
        for flag, name in TCP_FLAGS.items():
            flag_col = f'{d}_{name}_flag'
            tcp_state[flag_col] = tcp_state[f'{d}_state'].str.contains(flag).astype(np.int)

    flows['fwd_dir'] = flows['dir'].str.contains('>').astype(np.int)
    flows['bwd_dir'] = flows['dir'].str.contains('<').astype(np.int)

    flows.loc[~flows['proto'].isin(CTU_13_PROTOS), 'proto'] = 'other'

    flows = flows[CTU_13_COLUMNS]
    return flows


def cleanup_ctu13(data_path):

    logging.info('Cleaning up the data')
    data_path = Path(data_path).resolve()

    for sc_i in ALL_SCENARIOS:
        logging.info(f'Processing scenario {sc_i}')
        sc_path = data_path/'{:02d}.csv'.format(sc_i)
        sc_flows = pd.read_csv(sc_path)
        sc_flows = format_ctu_flows(sc_flows)
        sc_flows.to_csv(sc_path, index=False)


def create_report_scenario_ctu13(data, static_path):

    report = ""

    data['timestamp'] = pd.to_datetime(data['timestamp'])

    start_tstmp = data['timestamp'].iloc[0]
    data.loc[:, 'sec'] = (data['timestamp'] - start_tstmp).dt.total_seconds()

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
    sns.distplot(data.loc[~data['target'].astype(bool), 'sec'], rug=True, kde=False)
    sns.distplot(data.loc[data['target'].astype(bool), 'sec'], rug=True, kde=False)
    plt.legend(labels=['normal', 'botnet'])
    plot_name = start_tstmp.strftime("{}.png".format(str(uuid4())[:5]))
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'

    plt.close('all')
    sns.distplot(data.loc[~data['target'].astype(bool), 'sec'], rug=True)
    sns.distplot(data.loc[data['target'].astype(bool), 'sec'], rug=True)
    plt.legend(labels=['normal', 'botnet'])
    plot_name = start_tstmp.strftime("{}.png".format(str(uuid4())[:5]))
    plot_path = os.path.join(static_path, plot_name)
    plt.savefig(plot_path)
    report += f'<img src="static/{plot_name}" alt="dataset visualization">'
    report += '</br>'
    plt.close('all')

    return report


def create_data_report_ctu13(dataset_path, report_path):

    dataset_path = Path(dataset_path).resolve()
    static_path = report_path/'static'
    static_path.mkdir(parents=True, exist_ok=True)

    report = ''

    for path in dataset_path.iterdir():

        name = path.name[:-len(path.suffix)]
        print(name)

        data = pd.read_csv(path)
        path_report = create_report_scenario_ctu13(data, static_path)

        report += f'<h1> {name} </h1></br>'
        report += path_report
        report += '</br></br>'

    report = BASE.replace('{{STUFF}}', report)
    with open(report_path / 'report.html', 'w') as f:
        f.write(report)


def create_dataset_ctu13(dataset_path, train_scenarios, test_scenarios, create_hash=False):

    features = 'ORIG'

    dataset_name = '{}_TRAIN_{}_TEST_{}_{}'.format(
        DATASET_NAME,
        '-'.join(train_scenarios),
        '-'.join(test_scenarios),
        features
    )

    meta = create_meta(DATASET_NAME, train_scenarios, test_scenarios,
                       features=features, name=dataset_name)
    logging.info("Creating dataset {}...".format(meta['name']))

    train_paths = [dataset_path/'{:02d}.csv'.format(i) for i in train_scenarios]
    test_paths = [dataset_path/'{:02d}.csv'.format(i) for i in test_scenarios]

    # Naive concat
    train = pd.concat([pd.read_csv(p) for p in train_paths])
    train_meta = train.loc[:, CTU_13_META]
    train = train.loc[:, CTU_13_FEATURES]

    test = pd.concat([pd.read_csv(p) for p in test_paths])
    test_meta = test.loc[:, CTU_13_META]
    test = test.loc[:, CTU_13_FEATURES]

    logging.info("Done")

    return Dataset(train, test, train_meta, test_meta, meta, create_hash=create_hash)


# def create_dataset_ctu13(aggr_path, train_scenarios, test_scenarios, frequency='T'):
#
#     aggr_path = Path(aggr_path).resolve()
#
#     meta = create_meta(DATASET_NAME, train_scenarios,
#                        test_scenarios, frequency, FEATURES)  # no hash created
#     logging.info("Creating dataset {}...".format(meta['name']))
#
#     train_paths = []
#     for sc in train_scenarios:
#         train_paths.extend((aggr_path / str(sc)).glob(f'*aggr_{frequency}.csv'))
#
#     test_paths = []
#     for sc in test_scenarios:
#         test_paths.extend((aggr_path / str(sc)).glob(f'*aggr_{frequency}.csv'))
#
#     # Naive concat
#     train = pd.concat([pd.read_csv(p) for p in train_paths])
#     train_meta = train.loc[:, ['sa', 'tws']]
#     train = train.loc[:, FLOW_STATS.keys()]
#
#     test = pd.concat([pd.read_csv(p) for p in test_paths])
#     test_meta = test.loc[:, ['sa', 'tws']]
#     test = test.loc[:, FLOW_STATS.keys()]
#
#     logging.info("Done")
#
#     return Dataset(train, test, train_meta, test_meta, meta)


# def aggregate_ctu_data(root_path, aggr_path, processes=-1,
#                        frequency='T', exist_ok=True):
#
#     if processes == -1:
#         processes = mp.cpu_count() - 1
#
#     root_path = Path(root_path).resolve()
#     aggr_path = Path(aggr_path).resolve()
#
#     scenarios = [s.name for s in root_path.iterdir()
#                  if s.name in map(str, ALL_SCENARIOS)]
#
#     for scenario in scenarios:
#         scenario_path = root_path/scenario
#         scenario_out_path = aggr_path/scenario
#         scenario_out_path.mkdir(parents=True, exist_ok=True)
#
#         logging.info("Processing scenario {}...".format(scenario))
#         start_time = time.time()
#
#         flow_file = [f for f in scenario_path.iterdir()
#                      if f.suffix == '.binetflow'][0]
#
#         out_name = "{}_aggr_{}.csv".format(flow_file, frequency)
#         out_path = scenario_out_path/out_name
#
#         # Don't overwrite if exist_ok is set
#         if out_path.exists() and exist_ok:
#             logging.info("Found existing; no overwrite")
#             continue
#
#         flows = pd.read_csv(scenario_path/flow_file)
#         flows = format_ctu_flows(flows)
#
#         aggr_flows = aggregate_extract_features(flows, frequency, processes)
#         aggr_flows.to_csv(out_path, index=False)
#
#         logging.info("Done {0:.2f}".format(time.time() - start_time))
#
#



if __name__ == '__main__':

    root_path = Path('/home/emikmis/data/ctu-13/')

    dataset_name = 'CTU-13'
    data_path = root_path / 'data'
    dataset_path = data_path / dataset_name

    download_ctu13(dataset_path)

#     # Example command
#     # python ctu13.py ../tests/data/ctu_mock/ ../tests/data/processed/ -p -1 -f T --overwrite --plot
#
#     parser = get_argparser()
#     args = parser.parse_args()
#
#     loglevel = getattr(logging, args.logging.upper(), None)
#     logging.basicConfig(level=loglevel)
#
#     train_scenarios = ['2', '9']
#     test_scenarios = ['3']
#
#     aggr_dir = args.aggr_dir if args.aggr_dir else args.root_dir
#     aggregate_ctu_data(args.root_dir, aggr_dir,
#                        args.processes, args.frequency)
#     dataset = create_aggr_ctu_dataset(
#         args.root_dir, train_scenarios=train_scenarios,
#         test_scenarios=test_scenarios, frequency=args.frequency
#     )
#     dataset.write_to(args.out_dir, visualize=args.plot,
#                      overwrite=args.overwrite, archive=args.archive)
