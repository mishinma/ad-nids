
import logging
import json

from pathlib import Path

from generative_nids.ml.run import run
from generative_nids.ml.report import create_report, create_report_datasets


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

# for config_path in config_paths:
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#     run(config, log_root_path, frontier=True)

log_paths = list([p for p in log_root_path.iterdir() if p.is_dir()])
log_paths = sorted(log_paths)

dataset2logs = {}

for log_path in log_paths:

    logging.info(f"Creating report {log_path/'report.html'}")

    with open(log_path/'eval_results.json', 'r') as f:
        results = json.load(f)

    with open(log_path/'config.json', 'r') as f:
        config = json.load(f)

    report = create_report(results, config, log_path)
    with open(log_path/'report.html', 'w') as f:
        f.write(report)

    dataset2logs.setdefault(config['dataset_name'], []).append((log_path, results, config))

logging.info(f"Creating datasets report {log_root_path / 'report.html'}")
report = create_report_datasets(dataset2logs)
with open(log_root_path / 'report.html', 'w') as f:
    f.write(report)