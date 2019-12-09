
import logging
import json

from pathlib import Path

from generative_nids.ml.run import run
from generative_nids.ml.report import create_datasets_report, create_experiments_report


loglevel = 'INFO'
loglevel = getattr(logging, loglevel.upper(), None)
logging.basicConfig(level=loglevel)

config_root_path = Path('data/config').resolve()
log_root_path = Path('data/logs').resolve()

if config_root_path.is_dir():
    config_paths = list(config_root_path.iterdir())
else:
    config_paths = [config_root_path]

config_paths = sorted(config_paths)

for config_path in config_paths:
    with open(config_path, 'r') as f:
        config = json.load(f)
    # run(config, log_root_path, frontier=True)

log_paths = list([p for p in log_root_path.iterdir() if p.is_dir()])
log_paths = sorted(log_paths)

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
