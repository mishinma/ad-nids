import argparse
import logging
import copy
import json

from pathlib import Path
from uuid import uuid4

import pandas as pd


def read_model_params_from_csv(params_path):

    params_frame = pd.read_csv(params_path, index_col=0)
    model_params = []

    for idx, exp in params_frame.iterrows():
        exp = exp.to_dict()
        alg = exp.pop('algorithm')
        model_params.append((alg, exp))

    return model_params


def create_configs(data_path, config_out_path, model_params_path):

    config_out_path = Path(config_out_path)
    config_out_path.mkdir(exist_ok=True)

    data_path = Path(data_path).resolve()
    if data_path.is_dir():
        data_path = list(data_path.iterdir())
    else:
        data_path = [data_path]

    data_names_paths = [(dp.name[:-len(dp.suffix)], dp) for dp in data_path]
    model_params = read_model_params_from_csv(model_params_path)

    uniq_str = str(uuid4())[:5]

    idx = 0
    for alg, mp in model_params:
        for name, path in data_names_paths:

            conf = dict()
            conf['dataset_name'] = name
            conf['dataset_path'] = str(path)
            conf['algorithm'] = alg
            conf['model_parameters'] = mp

            logging.debug(
                'DATASET {}; ALGORITHM {}; MODEL_PARAMETERS {}'.format(
                    conf['dataset_name'], conf['algorithm'], conf['model_parameters'])
            )

            conf_name = 'conf_{:03d}_{}.json'.format(idx, uniq_str)
            conf_full_path = config_out_path / conf_name
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
