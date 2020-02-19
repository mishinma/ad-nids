import argparse
import logging
import json
import copy
import shutil

import pandas as pd
from pathlib import Path

import ad_nids.experiments.hyperparam_select as experiments
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_config
from ad_nids.utils.misc import set_seed, average_results, jsonify


DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 70]
DEFAULT_RANDOM_SEEDS = [11, 33, 55]
DEFAULT_SAMPLE_PARAMS = {
    'train': {'n_samples': 400000},
    'threshold': {'n_samples': 10000, 'perc_outlier': 5},
    'test': {'n_samples': 10000, 'perc_outlier': 5}
}


def parser_fit_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str,
                        help="log directory")
    parser.add_argument("--data_path", nargs='*',
                        help="data root path")
    parser.add_argument("--config_path", nargs='*',
                        help="directory with config files")
    parser.add_argument("--seeds", nargs='*',
                        help="random seeds")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser


def runner_fit_predict():
    parser = parser_fit_predict()
    args = parser.parse_args()

    loglevel = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(level=loglevel)
    log_root = Path(args.log_path).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    config_paths = []
    for config_path in args.config_path:
        config_path = Path(config_path).resolve()
        if config_path.is_dir():
            config_paths += list(config_path.iterdir())
        else:
            config_paths.append(config_path)

    dataset_paths = []
    for data_path in args.data_path:
        data_path = Path(data_path).resolve()
        if Dataset.is_dataset(data_path):
            dataset_paths.append(data_path)
        else:
            dataset_paths += [p for p in data_path.iterdir()
                              if Dataset.is_dataset(p)]

    random_seeds = [int(s) for s in args.seeds]
    if not random_seeds:
        random_seeds = DEFAULT_RANDOM_SEEDS

    for dataset_path in dataset_paths:

        logging.info(f'Loading dataset {dataset_path.name}')
        dataset = Dataset.from_path(dataset_path)

        for config_path in config_paths:

            configs = pd.read_csv(config_path)
            configs['dataset_path'] = str(dataset_path)
            configs['dataset_name'] = dataset_path.name

            for idx, config in configs.iterrows():

                run_fn = config.get('run_fn', 'no_fn')
                try:
                    run_fn = getattr(experiments, run_fn)
                except AttributeError as e:
                    logging.error(f"No such function {run_fn}")
                    continue

                config = config.to_dict()
                logging.info(f'Starting {config["config_name"]}')
                logging.info(json.dumps(config, indent=2))

                log_dir = get_log_dir(log_root, config)

                for rs in random_seeds:
                    set_seed(rs)
                    # Create a directory to store experiment logs
                    log_rs_dir = log_root/(log_dir.name + '_{}'.format(rs))
                    logging.info('Created a new log directory')
                    logging.info(f'{log_rs_dir}\n')

                    config_rs = copy.deepcopy(config)
                    config_rs['config_name'] += '_{}'.format(rs)

                    num_tries = config_rs.get('num_tries', 1)
                    i_run = 0

                    while True:
                        i_run += 1
                        if log_rs_dir.exists():
                            shutil.rmtree(str(log_rs_dir))
                        log_rs_dir.mkdir()
                        log_config(log_rs_dir, config_rs)
                        try:
                            # Pass data
                            run_fn(config_rs, log_rs_dir, dataset,
                                   DEFAULT_SAMPLE_PARAMS, DEFAULT_CONTAM_PERCS)
                        except Exception as e:
                            logging.exception(e)
                        else:
                            break
                        if i_run >= num_tries:
                            logging.warning('Model did NOT converge!')
                            break
                    with open(log_rs_dir/f'{i_run}.try', 'w') as f:
                        pass

                logging.info('Averaging the results for {}'.format(log_dir.name))
                #  We average results and save in another log dir
                #  So that we can reuse report module for generating reports
                log_ave_dir = log_root/(log_dir.name + '_AVE')
                log_ave_dir.mkdir()
                all_results = []
                for log_dir in log_root.glob(log_dir.name + '*'):
                    try:
                        with open(log_dir/'eval_results.json', 'r') as f:
                            results = json.load(f)
                    except FileNotFoundError:
                        continue
                    all_results.append(results)
                if all_results:
                    ave_results = average_results(all_results)
                    config_ave = copy.deepcopy(config)
                    config_ave['config_name'] += '_AVE'
                    log_config(log_ave_dir, config_ave)
                    with open(log_ave_dir / 'eval_results.json', 'w') as f:
                        json.dump(jsonify(ave_results), f)


def runner_predict():
    pass


if __name__ == '__main__':
    runner_fit_predict()
