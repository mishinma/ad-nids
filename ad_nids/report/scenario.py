
import json
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from json2html import json2html
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from alibi_detect.base import outlier_prediction_dict
from alibi_detect.utils.saving import load_detector, save_detector
from alibi_detect.utils.data import create_outlier_batch

from ad_nids.dataset import Dataset
from ad_nids.utils.misc import performance_asdict
from ad_nids.utils.visualize import plot_instance_score
from ad_nids.utils.report import merge_reports


def create_report_scenario(data, static_path, timestamp_col='timestamp'):

    report = ""

    num_flows = data.shape[0]
    attack_cnt = np.sum(data['target'])
    contamination_perc = attack_cnt / num_flows * 100

    report += '<h3>' + f'Num flows: {num_flows}' + ' </h3>'
    report += '<h3>' + f'Botnet flows: {attack_cnt}' + ' </h3>'
    report += '<h3> Contamination perc: {:.2f} </h3>'.format(contamination_perc)
    report += '</br>'

    y_gt = data['target']
    y_pred = data['is_outlier']
    cm = confusion_matrix(y_gt, y_pred)
    prf1s = precision_recall_fscore_support(y_gt, y_pred, average='binary')

    perf = performance_asdict(cm, prf1s)
    report += '<div>' + json2html.convert(perf) + '</div><br>'

    # plot score for a batch
    threshold = data.iloc[0]['threshold']

    normal_batch = data.loc[data['target'] == 0]
    num_normal_to_plot = 6000
    if normal_batch.shape[0] > num_normal_to_plot:
        normal_batch = normal_batch.sample(num_normal_to_plot)
    outlier_batch = data.loc[data['target'] == 1]
    batch = pd.concat([normal_batch, outlier_batch], axis=0).sample(frac=1)
    #     batch = batch.sort_values(timestamp_col)  # SORTING
    ylim = (batch['instance_score'].min(),
            min(10 *threshold, batch['instance_score'].quantile(0.99)))


    fig, ax = plt.subplots()
    plot_instance_score(ax, batch['instance_score'],
                        batch['target'], LABELS,
                        threshold=threshold, ylim=ylim)
    plot_name = "{}.png".format(str(uuid4())[:5])
    plt.savefig(static_path /plot_name)
    report += f'<img src="static/{plot_name}" alt="instance_score">'
    report += '</br>'
    plt.close('all')

    return report


def create_report_log_path(path, train=True):
    _set = 'train' if train else 'test'

    with open(path / 'config.json', 'r') as f:
        config = json.load(f)

    dataset_path = processed_path / (config['dataset_name'] + '_EVAL')
    #     print(dataset_path.name)
    #     dataset = Dataset.from_path(dataset_path)
    #     test = pd.read_csv(dataset_path/'test-meta.csv')
    if train:
        dataset_meta = DATASET_TO_TRAIN_META[dataset_path.name]
    else:
        dataset_meta = DATASET_TO_TEST_META[dataset_path.name]

    test = pd.DataFrame.copy(dataset_meta)
    test_preds = np.load(path / _set / 'eval.npz')

    test['is_outlier'] = test_preds['is_outlier']
    test['target'] = test_preds['ground_truth']
    test['instance_score'] = np.nan_to_num(test_preds['instance_score'])

    with open(path / 'eval_results.json', 'r') as f:
        eval_results = json.load(f)

    threshold = eval_results['threshold']
    od = load_detector(str(path / 'detector'))
    od.threshold = threshold
    test['threshold'] = threshold
    save_detector(od, str(path / 'detector'))

    test_grp = test.groupby('scenario')
    report_path = path / _set
    static_path = report_path / 'static'
    if static_path.exists():
        shutil.rmtree(static_path)
    static_path.mkdir(parents=True)

    if 'AGGR' in dataset_path.name:
        timestamp_col = 'time_window_start'
    else:
        timestamp_col = 'timestamp'
    reports = [(sc, create_report_scenario(flows, static_path, timestamp_col=timestamp_col))
               for sc, flows in test_grp]
    report = merge_reports(reports)

