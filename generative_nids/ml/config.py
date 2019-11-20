
import json

from pathlib import Path
from uuid import uuid4

import pandas as pd

ALGORITHM_PARAM_DTYPE = {
   'IsolationForest': {
        'n_estimators': int,
        'behaviour': str,
        'contamination': str
    },
   'NearestNeighbors': {
        'n_neighbors': int,
        'algorithm': str
    },
}


def parse_params(params_str, alg):
    params = params_str.split('\n')
    param_dtype = ALGORITHM_PARAM_DTYPE[alg]
    params = dict([_.strip() for _ in p.split(':')] for p in params)
    params = {k: param_dtype[k](v) for k, v in params.items()}
    return params


def configs_from_csv(experiments_path, config_out_dir):

    config_out_dir = Path(config_out_dir).resolve()
    config_out_dir.mkdir(exist_ok=True)

    experiments = pd.read_csv(experiments_path, index_col=0)

    for idx, exp in experiments.iterrows():

        # conf_name = f'conf_{str(uuid4())[:5]}.json'
        conf_name = f'conf_{idx}.json'

        exp['model_parameters'] = parse_params(
            exp['model_parameters'], exp['algorithm']
        )
        exp = exp.to_dict()
        exp['index'] = idx

        conf_full_path = config_out_dir/conf_name
        with open(conf_full_path, 'w') as f:
            json.dump(exp, f)


if __name__ == '__main__':
    experiments_path = '../tests/data/test_experiments.csv'
    config_out_dir = '../tests/data/configs/'
    configs_from_csv(experiments_path, config_out_dir)
