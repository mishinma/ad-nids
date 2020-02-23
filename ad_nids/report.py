import argparse
import ad_nids
import os
import shutil
import uuid
import logging
import json
from pathlib import Path

import pandas as pd
from json2html import json2html

from ad_nids.utils import int_to_roman
from ad_nids.utils.misc import performance_asdict

templates_path = Path(__file__).parent/'templates'

with open(templates_path/'base.html', 'r') as f:
    BASE = f.read()

with open(templates_path/'experiment.html', 'r') as f:
    EXPERIMENT = f.read()

with open(templates_path/'dataset.html', 'r') as f:
    DATASET = f.read()


CONFIG_NOREPORT_FIELDS = [
    'dataset_name',
    'dataset_path',
]


def copy_to_static(loc_path, static_dir):
    new_name = str(uuid.uuid4()) + loc_path.suffix
    shutil.copy(loc_path, os.path.join(static_dir, new_name))
    rel_new_path = os.path.join(static_dir.name, new_name)
    return rel_new_path


def collect_plots(plot_paths, static_path):
    plots = ''
    for plot_path in plot_paths:
        static_plot_path = copy_to_static(plot_path, static_path)
        alt_text = plot_path.name[:-len(plot_path.suffix)]
        plots += f'<img src="{static_plot_path}" alt="{alt_text}"><br>\n'
    return plots


def create_experiment_report(log_path, static_path, exp_idx=1):
    """ Create a simple HTML doc with summary. """

    report = EXPERIMENT
    report = report.replace('{{EXPERIMENT_I}}}', int_to_roman(exp_idx))
    try:
        with open(log_path / 'config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        return report

    report = report.replace('{{ALGORITHM}}', config['experiment_name'].upper())
    report = report.replace('{{DATASET_NAME}}', config['dataset_name'])
    report = report.replace('{{CONFIG_NAME}}', config['config_name'])
    report = report.replace('{{LOG_DIR}}', str(log_path))

    config_report = {k: v for k, v in config.items()
                     if k not in CONFIG_NOREPORT_FIELDS}
    report = report.replace('{{CONFIG_PARAMS}}', json2html.convert(config_report))

    try:
        with open(log_path / 'eval_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        return report

    # train performance
    if results.get('threshold') is not None:
        report = report.replace('{{THRESHOLD}}', '{:0.2f}'.format(results['threshold']))

    if results.get('time_fit') is not None:
        report = report.replace('{{TIME_FIT}}',
                                '{:0.2f}'.format(results['time_fit']))

    if results.get('time_score_train') is not None:
        report = report.replace('{{TIME_SCORE_TRAIN}}',
                                '{:0.2f}'.format(results['time_score_train']))

    train_cm, train_prf1s = results.get('train_cm'), results.get('train_prf1s')
    if train_cm is not None:
        train_perf = performance_asdict(train_cm, train_prf1s)
        report = report.replace('{{TRAIN_PERFORMANCE}}', json2html.convert(train_perf))
    else:
        report = report.replace('{{TRAIN_PERFORMANCE}}', None)

    # train plots
    train_plots = collect_plots(log_path.glob('train_*.png'), static_path)
    report = report.replace('{{TRAIN_PLOTS}}', train_plots)

    # test performance
    report = report.replace('{{TIME_SCORE_TEST}}',
                            '{:0.2f}'.format(results.get('time_score_test')))
    test_cm, test_prf1s = results.get('test_cm'), results.get('test_prf1s')
    if test_cm is not None:
        test_perf = performance_asdict(test_cm, test_prf1s)
        report = report.replace('{{TEST_PERFORMANCE}}', json2html.convert(test_perf))
    else:
        report = report.replace('{{TEST_PERFORMANCE}}', None)

    # test plots
    test_plots = collect_plots(log_path.glob('test_*.png'), static_path)
    report = report.replace('{{TEST_PLOTS}}', test_plots)

    return report


def create_experiments_per_dataset_report(log_paths, static_path):

    dataset2logs = {}
    for log_path in log_paths:
        try:
            with open(log_path / 'config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.warning(f'Skipping. No config found {log_path}')
        else:
            dataset2logs.setdefault(config['dataset_name'], []).append((log_path, config))

    dataset_names_sorted = sorted(dataset2logs.keys())
    for logs in dataset2logs.values():
        logs.sort(key=lambda l: l[1]['experiment_name'])
    dataset_reports = []

    for dataset_name in dataset_names_sorted:
        dataset_report = DATASET
        dataset_report = dataset_report.replace('{{DATASET_NAME}}', dataset_name)

        dataset_info = None
        dataset_img = None

        experiments = []
        for logs in dataset2logs[dataset_name]:
            log_path, config = logs

            try:
                with open(log_path / 'eval_results.json', 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = {}

            if dataset_info is None:
                with open(os.path.join(log_path, 'dataset_meta.json'), 'r') as f:
                    dataset_info = json.load(f)
                del dataset_info['name']  # is already the header

                # Path to dataset visualization
                dataset_img = Path(config['dataset_path']) / 'data.png'

            exp_name = config['experiment_name']
            config_name = config['config_name']
            if results:
                time_fit = round(results['time_fit'], 2)
                time_score_train = round(results['time_score_train'], 2)
                time_score_test = round(results['time_score_test'], 2)
                train_perf = performance_asdict(results['train_cm'],
                                                results['train_prf1s'])
                test_perf = performance_asdict(results['test_cm'],
                                               results['test_prf1s'])
                exp = [
                    config_name, exp_name, time_fit,
                    time_score_train, train_perf['tp'], train_perf['fp'], train_perf['fn'],
                    train_perf['precision'], train_perf['recall'], train_perf['f1score'],
                    time_score_test, test_perf['tp'], test_perf['fp'], test_perf['fn'],
                    test_perf['precision'], test_perf['recall'], test_perf['f1score']
                ]
            else:
                exp = [config_name, exp_name] + [None]*15

            experiments.append(exp)

        experiments = pd.DataFrame.from_records(
            experiments, columns=[
                'config_name', 'model_name', 'time_fit',
                'time_score_train', 'train_tp', 'train_fp', 'train_fn',
                'train_pre', 'train_rec', 'train_f1',
                'time_score_test', 'test_tp', 'test_fp', 'test_fn',
                'test_pre', 'test_rec', 'test_f1'
            ])

        best_performance = pd.concat([
            experiments[['train_pre', 'train_rec', 'train_f1',
                         'test_pre', 'test_rec', 'test_f1']].max(),
            experiments[['time_fit', 'time_score_train', 'time_score_test']].min()
        ])
        worst_performance = pd.concat([
            experiments[['train_pre', 'train_rec', 'train_f1',
                         'test_pre', 'test_rec', 'test_f1']].min(),
            experiments[['time_fit', 'time_score_train', 'time_score_test']].max()
        ])
        rows = []
        for idx, exp in experiments.iterrows():
            row = f'<th>{idx + 1}</th>\n'
            for met, val in exp.iteritems():
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

        dataset_configs = [logs[1] for logs in dataset2logs[dataset_name]]
        configs_html = ''
        dataset_configs_frame = pd.DataFrame.from_records(dataset_configs).set_index('config_name')
        for _, configs_grp in dataset_configs_frame.groupby('experiment_name'):
            configs_grp = configs_grp.drop(CONFIG_NOREPORT_FIELDS, axis=1)
            configs_grp = configs_grp.dropna(axis=1)
            configs_html += configs_grp.to_html() + '\n<br>\n'
        dataset_report = dataset_report.replace('{{CONFIGS_TABLE}}', configs_html)

        dataset_reports.append(dataset_report)

    dataset_reports = '<br><br>'.join(dataset_reports)
    report = BASE.replace('{{STUFF}}', dataset_reports)

    return report


def create_experiments_report(log_paths, static_path):

    reports = []

    for i, log_path in enumerate(log_paths):
        report = create_experiment_report(log_path, static_path, exp_idx=i + 1)
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

    datasets_report_path = report_path / 'datasets_report.html'
    logging.info(f"Creating all datasets report {datasets_report_path}")
    datasets_report = create_experiments_per_dataset_report(log_paths, static_path)
    with open(datasets_report_path, 'w') as f:
        f.write(datasets_report)

    experiments_report_path = report_path / 'experiments_report.html'
    logging.info(f"Creating all experiments report {experiments_report_path}")
    experiments_report = create_experiments_report(log_paths, static_path)
    with open(experiments_report_path, 'w') as f:
        f.write(experiments_report)

    server_path = Path(ad_nids.__path__[0])/'server'
    shutil.copy(str(server_path/'report_server.py'), str(report_path))
    shutil.copy(str(server_path/'run_server.sh'), str(report_path))

