
import argparse
import logging

import numpy as np
import pandas as pd

from alibi_detect.datasets import fetch_kdd
from sklearn.model_selection import train_test_split
from ad_nids.dataset import Dataset



parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str, nargs='?',
                    default='data/processed',
                    help="dataset directory")
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument("--percent10", action="store_true",
                        help="overwrite the data")
args = parser.parse_args()
dataset_path = args.dataset_path

rs = args.random_state
np.random.seed(rs)

logging.info("Fetching data")
X, y = fetch_kdd(percent10=args.percent10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rs)

train_df = pd.DataFrame(X_train)
train_df['lbl'] = y_train
test_df = pd.DataFrame(X_test)
test_df['lbl'] = y_test

meta = dict(
    name='KDDCUP99_{}'.format(rs),
    test_size=0.33,
    random_state=rs,
    percent10=args.percent10
)

name = 'KDDCUP99'
if args.percent10:
    name +='_0p10'
name += '_{}'.format(rs)

dataset = Dataset(train_df, test_df, meta=meta)
dataset.write_to(dataset_path, overwrite=True, visualize=True)


