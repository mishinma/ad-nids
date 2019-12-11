import argparse
import logging
import json

from pathlib import Path

from ad_nids.ml.report import create_datasets_report, create_experiments_report
from ad_nids.ml.run import run


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str,
                    help="directory with config files")
parser.add_argument("log_root_path", type=str,
                    help="log directory")
parser.add_argument("--report_path", type=str, default=None,
                    help="report directory")
parser.add_argument("-l", "--logging", type=str, default='INFO',
                    help="logging level")

args = parser.parse_args()

loglevel = getattr(logging, args.logging.upper(), None)
logging.basicConfig(level=loglevel)

config_root_path = Path(args.config_path).resolve()
log_root_path = Path(args.log_root_path).resolve()

if config_root_path.is_dir():
    config_paths = list(config_root_path.iterdir())
else:
    config_paths = [config_root_path]

config_paths = sorted(config_paths)

for config_path in config_paths:
    with open(config_path, 'r') as f:
        config = json.load(f)
    run(config, log_root_path)

logging.info('Creating reports...')
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
