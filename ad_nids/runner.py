
import argparse
import logging
import json

from pathlib import Path

import ad_nids.experiments as experiments
from ad_nids.report import create_experiments_report, create_datasets_report
from ad_nids.utils.logging import get_log_dir, log_config


def run_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_exp_path", type=str,
                        help="log directory")
    parser.add_argument("--config_path", type=str, default=None,
                        help="directory with config files")
    parser.add_argument("--report_path", type=str, default=None,
                        help="report directory")
    parser.add_argument("--load", action="store_true",
                        help="load_outlier detector")
    parser.add_argument("--contam_percs", default=None)
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser


def runner():
    parser = run_parser()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    log_exp_path = Path(args.log_exp_path).resolve()

    if args.contam_percs is not None:
        contam_percs = json.loads(args.contam_percs)
    else:
        contam_percs = None

    if not args.load:

        config_exp_path = Path(args.config_path).resolve()
        config_paths = [p for p in config_exp_path.iterdir()
                        if p.suffix == '.json']

        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            try:
                run_fn = getattr(experiments, config['experiment_run_fn'])
            except AttributeError as e:
                logging.error(f"No such function "
                              f"{config['experiment_run_fn']}")
                continue
            log_path = get_log_dir(log_exp_path, config["config_name"])
            log_path.mkdir(parents=True)
            log_config(log_path, config)

            try:
                run_fn(config, log_path, do_plot_frontier=True, contam_percs=contam_percs)
            except Exception as e:
                logging.exception(e)

    else:
        # Re-evaluate existing log directories

        for log_path in log_exp_path.iterdir():
            try:
                with open(log_path/'config.json', 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logging.exception('Could not load config')
                continue

            try:
                run_fn = getattr(experiments, config['experiment_run_fn'])
            except AttributeError as e:
                logging.error(f"No such function "
                              f"{config['experiment_run_fn']}")
                continue

            try:
                run_fn(config, log_path, do_plot_frontier=True,
                       contam_percs=contam_percs, load_outlier_detector=args.load)
            except Exception as e:
                logging.exception(e)


    if args.report_path is not None:

        report_path = Path(args.report_path).resolve()
        report_path.mkdir(parents=True)
        static_path = report_path / 'static'
        static_path.mkdir()

        log_paths = list([p for p in log_exp_path.iterdir() if p.is_dir()])
        log_paths = sorted(log_paths)

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


if __name__ == '__main__':
    runner()
