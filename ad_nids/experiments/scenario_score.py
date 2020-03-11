
import json
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ad_nids.utils.visualize import plot_instance_score
from ad_nids.utils.aggregate import aggregate_features_pool


def _aggregate_scores_wkr(args):

    grp_name, grp = args

    record = {
        'src_ip': grp_name[0],
        'time_window_start': grp_name[1],
        'instance_score': grp['instance_score'].max(),
        'target': np.int(grp['target'].sum() > 0),
        'is_outlier': np.int(grp['is_outlier'].sum() > 0),
    }

    return record


def log_plot_contiguous_scores(data, log_score_path, aggr_frequency=None,
                               threshold=None, processes=8):

    data_sc_grp = data.groupby('scenario')

    for sc, data_sc in data_sc_grp:

        if aggr_frequency is not None:

            logging.info(f'Aggregating scores for scenario {sc}')
            timestamp_col = data_sc_grp.select_dtypes('datetime').columns[0]
            data_sc = data_sc.sort_values(timestamp_col)
            aggr_sc = data_sc.groupby(['src_ip', pd.Grouper(key=timestamp_col, freq=aggr_frequency)])
            data_sc = aggregate_features_pool(aggr_sc, _aggregate_scores_wkr, processes=processes)
            data_sc = pd.DataFrame.from_records(data_sc)

        # time column may be different
        timestamp_col = data_sc_grp.select_dtypes('datetime').columns[0]
        data_sc = data_sc.sort_values(timestamp_col)
        time_start = data_sc.iloc[0]['time_window_start']
        data_sc['min_start'] = int((data_sc[timestamp_col] - time_start) / pd.Timedelta(minutes=1))

        logging.info('Logging scores')
        fname = '{:02d}_scores.csv'.format(sc)
        data_sc.to_csv(log_score_path/fname)

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        plot_instance_score(ax, scores=data_sc['instance_score'], target=data_sc['target'],
                            idx=data_sc['min_start'], threshold=threshold, xlabel='Min from Start')
        plot_fname = '{:02d}_scores.png'.format(sc)
        plt.savefig(str(log_score_path/plot_fname))
        plt.close()

        logging.info('Done')


def log_plot_random_scores(data, log_score_path, threshold=None):

    data_sc_grp = data.groupby('scenario')

    for sc, data_sc in data_sc_grp:

        logging.info('Logging scores')
        fname = '{:02d}_scores.csv'.format(sc)
        data_sc.to_csv(log_score_path/fname)

        data_sc = data_sc.sample(frac=1)  # shuffle data points
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        plot_instance_score(ax, scores=data_sc['instance_score'], target=data_sc['target'],
                            threshold=threshold, xlabel='Number of Instances')
        plot_fname = '{:02d}_scores.png'.format(sc)
        plt.savefig(str(log_score_path/plot_fname))
        plt.close()

        logging.info('Done')


def create_scenario_score_log_path(log_path, dataset, train=True, test=True):

    if not (train or test):
        raise ValueError('Either test or train must be True')

    sets = []
    if train:
        sets.append('train')
    if test:
        sets.append('test')

    # random or scenario split
    is_random_split = dataset.meta.get('train_scenarios') is not None
    frequency = dataset.meta.get('frequency')

    for _set in sets:

        try:
            preds = np.load(log_path / _set / 'eval.npz')
        except FileNotFoundError:
            logging.error(f'No predicted scores found for {_set} in {log_path}')
            continue

        meta = dataset.train_meta if _set == 'train' else dataset.test_meta
        data = pd.DataFrame.copy(meta)

        data['is_outlier'] = preds['is_outlier']
        data['target'] = preds['ground_truth']
        data['instance_score'] = np.nan_to_num(preds['instance_score'])

        # Fetch the threshold
        with open(log_path / 'eval_results.json', 'r') as f:
            eval_results = json.load(f)
        threshold = eval_results['threshold']
        data['threshold'] = threshold

        timestamp_col = [c for c in data.columns if 'time' in c][0]
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])

        if is_random_split:
            # No need to aggregate as data points were sampled randomly
            # Just split the scores per scenario and plot
            name = 'scores_per_scenario'
            if frequency is not None:
                name += f'_{frequency}'
            log_score_path = log_path/_set/name
            log_score_path.mkdir(exist_ok=True)
            log_plot_random_scores(data, log_score_path, threshold=threshold)
        else:

            if frequency == '3T':
                # 3Min is the largest aggr window, no need to aggr
                log_score_path = log_path / _set / f'scores_per_scenario_{frequency}'
                log_score_path.mkdir(exist_ok=True)
                log_plot_contiguous_scores(data, log_score_path, threshold=threshold)
            else:
                # log plot two times: using the orig freq and aggr 3T
                name = 'scores_per_scenario'
                if frequency is not None:
                    name += f'_{frequency}'
                log_score_path = log_path / _set / name
                log_score_path.mkdir(exist_ok=True)
                log_plot_contiguous_scores(data, log_score_path, threshold=threshold)

                aggr_frequency = '3T'
                log_score_path = log_path / _set / f'scores_per_scenario_{aggr_frequency}'
                log_score_path.mkdir(exist_ok=True)
                log_plot_contiguous_scores(data, log_score_path, aggr_frequency=aggr_frequency,
                                           threshold=threshold)
