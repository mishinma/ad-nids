
from pathlib import Path

import numpy as np
import pandas as pd

from alibi_detect.datasets import fetch_kdd
from sklearn.model_selection import train_test_split
from ad_nids.dataset import Dataset

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

X, y = fetch_kdd(percent10=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)

train_df = pd.DataFrame(X_train)
train_df['lbl'] = y_train
test_df = pd.DataFrame(X_test)
test_df['lbl'] = y_test

meta = dict(
    name='KDDCUP99_033',
    test_size=0.33
)

dataset = Dataset(train_df, test_df, meta=meta)
dataset_path = Path('data/processed')
dataset.write_to(dataset_path, overwrite=True, visualize=True)



