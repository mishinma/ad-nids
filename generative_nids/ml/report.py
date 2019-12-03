import argparse
import json
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from json2html import json2html

import matplotlib.pyplot as plt

from generative_nids.ml.utils import plot_precision_recall

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

ALGORITHM_SHORT_NAMES = {
    'autoencoder': 'ae',
    'isolationforest': 'if',
    'nearestneighbors': 'nn',
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


def create_report(results, config, static_path):
    """ Create a simple HTML doc with summary. """

    report = EXPERIMENT
    report = report.replace('{{ALGORITHM}}', config['algorithm'])
    import pdb; pdb.set_trace()
    report = report.replace('{{DATASET}}', config['dataset_name'])
    config_report = {k: v for k, v in config.items() if k in CONFIG_REPORT_FIELDS}
    report = report.replace('{{CONFIG_PARAMS}}', json2html.convert(config_report))

    # train performance
    train_perf = performance_asdict(results['y_train'], results['train_cm'], results['train_prf1s'])
    report = report.replace('{{TRAIN_PERFORMANCE}}', json2html.convert(train_perf))

    # plot train precision recall curve
    train_precisions = results['train_prf1_curve']['precisions']
    train_recalls = results['train_prf1_curve']['recalls']
    train_f1scores = results['train_prf1_curve']['f1scores']
    fig, ax = plt.subplots(1, 1)
    plot_precision_recall(ax, train_precisions, train_recalls)
    fig.savefig(static_path/'train_pr_curve.png')
    plt.close()
    report = report.replace('{{TRAIN_PR_CURVE}}', str(static_path/'train_pr_curve.png'))

    # test performance
    test_perf = performance_asdict(results['y_test'], results['test_cm'], results['test_prf1s'])
    report = report.replace('{{TEST_PERFORMANCE}}', json2html.convert(test_perf))

    report = BASE.replace('{{STUFF}}', report)

    return report


def create_report_datasets(dataset2logs):

    dataset_names_sorted = sorted(dataset2logs.keys())
    dataset_reports = []

    for dataset_name in dataset_names_sorted:
        dataset_report = DATASET
        dataset_report = dataset_report.replace('{{DATASET_NAME}}', dataset_name)

        dataset_info = None

        experiments = []
        for logs in dataset2logs[dataset_name]:
            log_path, results, config = logs

            if dataset_info is None:
                with open(os.path.join(log_path, 'dataset_meta.json'), 'r') as f:
                    dataset_info = json.load(f)
                del dataset_info['name']  # is already the header

            config_name = config['config_name']
            algorithm = config['algorithm']
            train_perf = performance_asdict(results['y_train'],
                                            results['train_cm'],
                                            results['train_prf1s'])
            test_perf = performance_asdict(results['y_test'],
                                           results['test_cm'],
                                           results['test_prf1s'])
            exp = [
                config_name, algorithm,
                train_perf['tp'], train_perf['fp'], train_perf['fn'],
                train_perf['precision'], train_perf['recall'], train_perf['f1score'],
                test_perf['tp'], test_perf['fp'], test_perf['fn'],
                test_perf['precision'], test_perf['recall'], test_perf['f1score']
            ]

            experiments.append(exp)

        experiments = pd.DataFrame.from_records(
            experiments, columns=[
                'config_name', 'algorithm', 'train_tp', 'train_fp', 'train_fn',
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
            alg_name = ALGORITHM_SHORT_NAMES[exp["algorithm"].lower()]
            row += f'<th>{alg_name}</th>\n'
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
        dataset_reports.append(dataset_report)


    dataset_reports = '<br><br>'.join(dataset_reports)
    report = BASE.replace('{{STUFF}}', dataset_reports)
    return report


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_root_path", type=str,
                        help="log directory")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    log_root_path = Path(args.log_root_path).resolve()

    log_paths = list([p for p in log_root_path.iterdir() if p.is_dir()])
    log_paths = sorted(log_paths)

    dataset2logs = {}

    for log_path in log_paths:

        report_path = (log_path/'report.html').resolve()
        logging.info(f"Creating report {report_path}")

        with open(log_path/'eval_results.json', 'r') as f:
            results = json.load(f)

        with open(log_path/'config.json', 'r') as f:
            config = json.load(f)

        report = create_report(results, config, log_path)
        with open(report_path, 'w') as f:
            f.write(report)

        dataset2logs.setdefault(config['dataset_name'], []).append((log_path, results, config))

    datasets_report_path = (log_root_path / 'report.html').resolve()
    logging.info(f"Creating datasets report {datasets_report_path}")
    report = create_report_datasets(dataset2logs)
    with open(datasets_report_path, 'w') as f:
        f.write(report)
