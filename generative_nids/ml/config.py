import os
import argparse
import logging
import json

from pathlib import Path
from uuid import uuid4

import pandas as pd

from generative_nids.ml.modelwrapper import filter_model_params


def read_model_params_from_csv(params_path):

    with open(params_path, 'r') as f:
        alg = f.readline().strip()

    params_frame = pd.read_csv(params_path, index_col=0, skiprows=1)
    exp_params = []

    for idx, params in params_frame.iterrows():
        params = params.to_dict()
        model_params, other_params = filter_model_params(params, alg)
        exp_params.append((alg, (model_params, other_params)))

    return exp_params


def create_configs(data_root_path, config_out_path, model_params_path):

    config_out_path = Path(config_out_path)
    config_out_path.mkdir(exist_ok=True)

    data_root_path = Path(data_root_path).resolve()
    if data_root_path.is_dir():
        data_paths = list(data_root_path.iterdir())
    else:
        data_paths = [data_root_path]

    exp_params = read_model_params_from_csv(model_params_path)

    uniq_str = str(uuid4())[:5]

    idx = 0
    for alg, params in exp_params:

        model_params, other_params = params

        for data_path in data_paths:
            conf = dict()
            conf['dataset_name'] = str(data_path.name)
            conf['dataset_path'] = str(data_path)
            conf['algorithm'] = alg
            conf['model_parameters'] = model_params
            conf.update(other_params)

            logging.debug(json.dumps(conf, indent=4) + '\n')

            conf_name = 'conf_{}_{:03d}'.format(uniq_str, idx)
            conf['config_name'] = conf_name
            conf_full_path = (config_out_path / conf_name).with_suffix('.json')
            with open(conf_full_path, 'w') as f:
                json.dump(conf, f)

            idx += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str,
                        help="data root path")
    parser.add_argument("config_out_path", type=str,
                        help="output config directory")
    parser.add_argument("model_params_path", type=str,
                        help="model_parameters_path")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")

    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)

    create_configs(args.data_path, args.config_out_path, args.model_params_path)
