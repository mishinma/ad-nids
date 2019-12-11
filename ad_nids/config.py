import argparse
import logging
import json

from pathlib import Path
from uuid import uuid4

import pandas as pd


LOG_SHOW_PARAMS = ['dataset_name', 'algorithm', 'model_parameters',
                   'data_standardization', 'lr', 'num_epochs', 'optimizer']


def config_dumps(config):
    log_config = {k: v for k, v in config.items() if k in LOG_SHOW_PARAMS}
    return json.dumps(log_config, indent=2)


def read_exp_params_csv(params_path):

    with open(params_path, 'r') as f:
        exp_name = f.readline().strip()

    exp_params = pd.read_csv(params_path, index_col=0, skiprows=1)

    return exp_name, exp_params


def create_configs(exp_params_path, dataset_paths, config_root_path):

    exp_name, exp_params = read_exp_params_csv(exp_params_path)

    config_root_path = Path(config_root_path).resolve()
    config_exp_path = config_root_path/exp_name

    uniq_str = str(uuid4())[:5]

    configs = []
    for idx, params in exp_params.iterrows():
        for dataset_path in dataset_paths:
            conf = dict(
                config_name='conf_{}_{:03d}'.format(uniq_str, idx),
                dataset_name=str(dataset_path.name),
                dataset_path=str(dataset_path),
            )
            conf.update(params)
            configs.append(conf)

    configs = pd.DataFrame.from_records(configs)

    config_exp_path.mkdir(parents=True)
    configs.to_csv(config_exp_path/'configs.csv', index=None)
    for _, conf in configs.iterrows():
        conf = conf.to_dict()
        logging.debug(json.dumps(conf, indent=4) + '\n')
        conf_full_path = (config_exp_path / conf['config_name']).with_suffix('.json')
        with open(conf_full_path, 'w') as f:
            json.dump(conf, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_params_path", type=str,
                        help="experiment parameters path")
    parser.add_argument("data_root_path", type=str,
                        help="data root path")
    parser.add_argument("config_root_path", type=str,
                        help="output config directory")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    data_root_path = Path(args.data_root_path).resolve()
    dataset_paths = list(data_root_path.iterdir())

    create_configs(args.exp_params_path, dataset_paths, args.config_root_path)
