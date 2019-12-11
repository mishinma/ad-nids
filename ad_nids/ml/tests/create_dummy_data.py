
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import make_moons, make_blobs

from ad_nids.dataset import Dataset

RANDOM_STATE = 42
RANDOM_STATE_TEST = 24


def create_dummy_datasets(n_samples=300, outliers_fraction=0.15, random_state=RANDOM_STATE):
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    rng = np.random.RandomState(random_state)

    # Define datasets
    blobs_params = dict(random_state=random_state, n_samples=n_inliers, n_features=2)
    datasets = {
        "blobs_1":  make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0],
        "blobs_2": make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params)[0],
        "blobs_3": make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3], **blobs_params)[0],
        "blobs_4_3D":  make_blobs(
            centers=[[0, 0, 0], [0, 0, 0]], cluster_std=0.5,
            random_state=random_state, n_samples=n_inliers, n_features=3)[0],
        "blobs_5_15D": make_blobs(
            centers=[[0]*15, [0]*15], cluster_std=0.5,
            random_state=random_state, n_samples=n_inliers, n_features=15)[0],
        "moons_1": 4. * (make_moons(n_samples=n_inliers, noise=.05,
                                    random_state=random_state)[0] - np.array([0.5, 0.25])),
        "random_1": 14. * (rng.rand(n_inliers, 2) - 0.5)
    }

    y = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)]).astype(np.int)
    y = pd.DataFrame(np.expand_dims(y, axis=-1), columns=['lbl'])

    for name, x in datasets.items():
        x = np.concatenate([x, rng.uniform(low=-6, high=6,
                                           size=(n_outliers, x.shape[1]))], axis=0)
        x_columns = [f'x{i}' for i in range(x.shape[1])]
        x = pd.DataFrame(x, columns=x_columns)
        datasets[name] = pd.concat([x, y], axis=1)

    return datasets


n_samples_train = 300
outliers_fraction_train = .10
train_datasets = create_dummy_datasets(n_samples_train, outliers_fraction_train)

n_samples_test = 100
outliers_fraction_test = .20
test_datasets = create_dummy_datasets(n_samples_test, outliers_fraction_test, random_state=RANDOM_STATE_TEST)

data_path = Path('data/processed')

for name in train_datasets:
    train = train_datasets[name]
    test = test_datasets[name]
    meta = {'name': name}
    dataset = Dataset(train, test, meta=meta, create_hash=True)
    dataset.write_to(data_path, overwrite=True, plot=True)
