import argparse
import json
import os
import shutil
import uuid
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from json2html import json2html

from generative_nids.utils import int_to_roman


templates_path = Path(__file__).parent/'templates'

with open(templates_path/'base.html', 'r') as f:
    BASE = f.read()

with open(templates_path/'experiment.html', 'r') as f:
    EXPERIMENT = f.read()

with open(templates_path/'dataset.html', 'r') as f:
    DATASET = f.read()


CONFIG_REPORT_FIELDS = {
    'model_parameters',
    'lr',
    'num_epochs',
    'optimizer'
}


def performance_asdict(y_true, cm, prf1s):

    p = np.sum(y_true)
    n = len(y_true) - p

    perf = dict(
        p=p,
        n=n,
        tp=cm[1][1],
        fp=cm[0][1],
        fn=cm[1][0],
        precision=round(prf1s[0], 2),
        recall=round(prf1s[1], 2),
        f1score=round(prf1s[2], 2),
    )

    return perf


def copy_to_static(loc_path, static_dir):
    new_name = str(uuid.uuid4()) + loc_path.suffix
    shutil.copy(loc_path, os.path.join(static_dir, new_name))
    rel_new_path = os.path.join(static_dir.name, new_name)
    return rel_new_path


def create_report(log_path, static_path, exp_idx=1):
    """ Create a simple HTML doc with summary. """

    with open(log_path / 'eval_results.json', 'r') as f:
        results = json.load(f)
    with open(log_path / 'config.json', 'r') as f:
        config = json.load(f)

    report = EXPERIMENT
    report = report.replace('{{EXPERIMENT_I}}}', int_to_roman(exp_idx))
    report = report.replace('{{ALGORITHM}}', config['algorithm'].upper())
    report = report.replace('{{DATASET_NAME}}', config['dataset_name'])
    report = report.replace('{{CONFIG_NAME}}', config['config_name'])
    config_report = {k: v for k, v in config.items() if k in CONFIG_REPORT_FIELDS}
    report = report.replace('{{CONFIG_PARAMS}}', json2html.convert(config_report))

    # train performance
    report = report.replace('{{THRESHOLD}}', '{:0.2f}'.format(results['threshold']))
    train_perf = performance_asdict(results['y_train'], results['train_cm'], results['train_prf1s'])
    report = report.replace('{{TRAIN_PERFORMANCE}}', json2html.convert(train_perf))

    # plot
    if (log_path / 'train_frontier.png').exists():
        static_img_path = copy_to_static(log_path / 'train_frontier.png', static_path)
        report = report.replace('{{TRAIN_FRONTIER}}', str(static_img_path))

    if (log_path / 'train_pr_curve.png').exists():
        static_img_path = copy_to_static(log_path / 'train_pr_curve.png', static_path)
        report = report.replace('{{TRAIN_PR_CURVE}}',  str(static_img_path))

    if (log_path / 'train_f1_curve.png').exists():
        static_img_path = copy_to_static(log_path / 'train_f1_curve.png', static_path)
        report = report.replace('{{TRAIN_F1_CURVE}}', str(static_img_path))

    # test performance
    test_perf = performance_asdict(results['y_test'], results['test_cm'], results['test_prf1s'])
    report = report.replace('{{TEST_PERFORMANCE}}', json2html.convert(test_perf))
    # plot test frontier
    if (log_path / 'test_frontier.png').exists():
        static_img_path = copy_to_static(log_path / 'test_frontier.png', static_path)
        report = report.replace('{{TEST_FRONTIER}}', str(static_img_path))

    return report


def create_datasets_report(log_paths, static_path):

    dataset2logs = {}
    for log_path in log_paths:
        with open(log_path / 'eval_results.json', 'r') as f:
            results = json.load(f)
        with open(log_path / 'config.json', 'r') as f:
            config = json.load(f)
        dataset2logs.setdefault(config['dataset_name'], []).append((log_path, results, config))

    dataset_names_sorted = sorted(dataset2logs.keys())
    dataset_reports = []

    for dataset_name in dataset_names_sorted:
        dataset_report = DATASET
        dataset_report = dataset_report.replace('{{DATASET_NAME}}', dataset_name)

        dataset_info = None
        dataset_img = None

        experiments = []
        for logs in dataset2logs[dataset_name]:
            log_path, results, config = logs

            if dataset_info is None:
                with open(os.path.join(log_path, 'dataset_meta.json'), 'r') as f:
                    dataset_info = json.load(f)
                del dataset_info['name']  # is already the header

                # Path to dataset visualization
                dataset_img = Path(config['dataset_path']) / 'data.png'

            config_name = config['config_name']
            model_name = results['model_name']
            train_perf = performance_asdict(results['y_train'],
                                            results['train_cm'],
                                            results['train_prf1s'])
            test_perf = performance_asdict(results['y_test'],
                                           results['test_cm'],
                                           results['test_prf1s'])
            exp = [
                config_name, model_name,
                train_perf['tp'], train_perf['fp'], train_perf['fn'],
                train_perf['precision'], train_perf['recall'], train_perf['f1score'],
                test_perf['tp'], test_perf['fp'], test_perf['fn'],
                test_perf['precision'], test_perf['recall'], test_perf['f1score']
            ]

            experiments.append(exp)

        experiments = pd.DataFrame.from_records(
            experiments, columns=[
                'config_name', 'model_name', 'train_tp', 'train_fp', 'train_fn',
                'train_pre', 'train_rec', 'train_f1',
                'test_tp', 'test_fp', 'test_fn',
                'test_pre', 'test_rec', 'test_f1',
            ])

        best_performance = experiments[['train_pre', 'train_rec', 'train_f1',
                                        'test_pre', 'test_rec', 'test_f1']].max()
        worst_performance = experiments[['train_pre', 'train_rec', 'train_f1',
                                        'test_pre', 'test_rec', 'test_f1']].min()

        rows = []
        for idx, exp in experiments.iterrows():
            row = f'<th>{idx + 1}</th>\n'
            row += f'<th>{exp["config_name"]}</th>\n'
            row += f'<th>{exp["model_name"]}</th>\n'
            for met, val in exp.iloc[2:].iteritems():
                is_best = met in best_performance and val == best_performance[met]
                is_worst = met in worst_performance and val == worst_performance[met]
                if is_best:
                    row += f'<th style="color:MediumSeaGreen;">{val}</th>\n'
                elif is_worst:
                    row += f'<th style="color:Tomato;">{val}</th>\n'
                else:
                    row += f'<th>{val}</th>\n'
            rows.append('<tr>' + row + '</tr>')
        rows = '\n'.join(rows)
        dataset_report = dataset_report.replace('{{EXPERIMENT_ROWS}}', rows)
        dataset_report = dataset_report.replace('{{DATASET_INFO}}', json2html.convert(dataset_info))
        if dataset_img.exists():
            static_img_path = copy_to_static(dataset_img, static_path)
            dataset_report = dataset_report.replace('{{DATASET_IMG}}', str(static_img_path))
        dataset_reports.append(dataset_report)

    dataset_reports = '<br><br>'.join(dataset_reports)
    report = BASE.replace('{{STUFF}}', dataset_reports)
    return report


def create_experiments_report(log_paths, static_path):

    reports = []

    for i, log_path in enumerate(log_paths):
        report = create_report(log_path, static_path, exp_idx=i+1)
        reports.append(report)

    reports = '\n<br><br>\n'.join(reports)
    final_report = BASE.replace('{{STUFF}}', reports)
    return final_report


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_root_path", type=str,
                        help="log directory")
    parser.add_argument("--report_path", type=str, default=None,
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

    static_path = report_path / 'static'
    static_path.mkdir(parents=True)

    datasets_report_path = report_path / 'datasets_report.html'
    logging.info(f"Creating all datasets report {datasets_report_path}")
    datasets_report = create_datasets_report(log_paths, static_path)
    with open(datasets_report_path, 'w') as f:
        f.write(datasets_report)

    experiments_report_path = report_path / 'experiments_report.html'
    logging.info(f"Creating all experiments report {experiments_report_path}")
    experiments_report = create_experiments_report(log_paths, static_path)
    with open(experiments_report_path, 'w') as f:
        f.write(experiments_report)

