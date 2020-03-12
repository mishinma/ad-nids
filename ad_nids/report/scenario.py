
import logging
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from json2html import json2html
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import ad_nids
from ad_nids.report.general import create_experiment_report
from ad_nids.utils.misc import performance_asdict
from ad_nids.utils.report import merge_reports, copy_to_static


templates_path = Path(ad_nids.__path__[0])/'templates'

with open(templates_path/'base.html', 'r') as f:
    BASE = f.read()

with open(templates_path/'experiment.html', 'r') as f:
    EXPERIMENT = f.read()


def create_single_scenario_report(data, score_plot_path, static_dir):

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

    score_plot_static_path = copy_to_static(score_plot_path, static_dir)
    report += f'<img src="{score_plot_static_path}" alt="instance_score">'
    report += '</br>'
    plt.close('all')

    return report


def create_subset_scenarios_report(log_path, static_dir, subset='train', sc_name_mapping=None):

    assert subset in ['train', 'test']

    subset_path = log_path / subset
    scenario_scores_paths = sorted(subset_path.glob('scores_per_scenario*'))

    if not scenario_scores_paths:
        return ''

    scenarios = [int(p.name.split('_')[0]) for p in scenario_scores_paths[0].glob('*.csv')]
    scenario_scores_reports = []

    for sc in scenarios:

        if sc_name_mapping is not None:
            sc_name = 'SCENARIO {:02d} {}'.format(sc, sc_name_mapping[sc])
        else:
            sc_name = 'SCENARIO {:02d}'.format(sc)

        sc_reports = []

        for scenario_scores_path in scenario_scores_paths:

            try:
                freq = scenario_scores_path.name.split('_')[3]
            except IndexError:
                freq = None

            sc_data_path = scenario_scores_path/'{:02d}_scores.csv'.format(sc)
            sc_data = pd.read_csv(sc_data_path)
            sc_score_plot_path = scenario_scores_path/'{:02d}_scores.png'.format(sc)

            sc_report = create_single_scenario_report(sc_data, sc_score_plot_path, static_dir)
            sc_reports.append((f'AGGR: {freq}', sc_report))

        scenario_scores_report = merge_reports(sc_reports, base=False, heading=3)
        scenario_scores_reports.append((sc_name, scenario_scores_report))

    return merge_reports(scenario_scores_reports, sort=False, base=False, heading=2)


def create_experiment_scenario_report(log_path, static_dir, exp_idx, sc_name_mapping=None):

    report = create_experiment_report(log_path, static_dir, exp_idx)
    report += '<br>'
    report += create_subset_scenarios_report(log_path, static_dir,
                                             subset='test', sc_name_mapping=sc_name_mapping)
    return report


def create_experiments_scenario_report(log_paths, static_dir, sc_name_mapping=None):

    reports = []

    for i, log_path in enumerate(log_paths):
        try:
            report = create_experiment_scenario_report(log_path, static_dir,
                                                       exp_idx=i + 1, sc_name_mapping=sc_name_mapping)
        except Exception as e:
            logging.exception(e)
            report = EXPERIMENT
        reports.append(report)

    reports = '\n<br><br>\n'.join(reports)
    final_report = BASE.replace('{{STUFF}}', reports)
    return final_report


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_root_path", type=str,
                        help="log directory")
    parser.add_argument("report_path", type=str,
                        help="report directory")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    log_root_path = Path(args.log_root_path).resolve()

    log_paths = list([p for p in log_root_path.iterdir() if p.is_dir()])
    log_paths = sorted(log_paths)

    report_path = args.report_path
    if report_path is None:
        report_path = log_root_path / 'reports'
    else:
        report_path = Path(report_path).resolve()

    static_path = report_path / 'static'
    static_path.mkdir(parents=True)

    experiments_report_path = report_path / 'experiments_report.html'
    logging.info(f"Creating all experiments report {experiments_report_path}")
    experiments_report = create_experiments_scenario_report(log_paths, static_path)
    with open(experiments_report_path, 'w') as f:
        f.write(experiments_report)

    server_path = Path(ad_nids.__path__[0])/'server'
    shutil.copy(str(server_path/'report_server.py'), str(report_path))
    shutil.copy(str(server_path/'run_server.sh'), str(report_path))

