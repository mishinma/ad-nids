import json
import shutil
import zipfile
import hashlib
import logging

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from alibi_detect.utils.data import create_outlier_batch

from ad_nids.utils import plot_data_2d
from ad_nids.utils.misc import fair_attack_sample


def sample_df(df: pd.DataFrame,
              n: int):
    """ Sample n instances from the dataframe df. """
    if n < df.shape[0] + 1:
        replace = False
    else:
        replace = True
    return df.sample(n=n, replace=replace)


def create_meta(dataset_name, train_split, test_split, frequency=None,
                features=None, name=None, notes=None):

    if notes is None:
        notes = ''

    if name is None:
        name = '{}_TRAIN_{}_TEST_{}_{}_{}'.format(
            dataset_name, '-'.join(train_split), '-'.join(test_split), frequency, features
        )

    meta = {
        'data_hash': None,
        'dataset_name': dataset_name,
        'train_split': train_split,
        'test_split': test_split,
        'frequency': frequency,
        'features': features,
        'notes': notes,
        'name': name
    }

    return meta


def extract_dataset(arc_path, remove_archive=True):

    arc_path = Path(arc_path)

    with zipfile.ZipFile(arc_path) as ziph:
        ziph.extractall(arc_path.parent)

    if remove_archive:
        arc_path.remove()


def hash_from_frames(frames):
    """ Naive concat """

    # create hash
    data_hash = hashlib.md5()
    for frame in frames:
        data_hash.update(frame.to_csv().encode('utf-8'))

    return data_hash.hexdigest()


def hash_from_paths(paths):
    """ Naive concat """

    # create hash
    data_hash = hashlib.md5()
    for path in paths:
        with open(path, 'rb') as f:
            data_hash.update(f.read())

    return data_hash.hexdigest()


class Dataset:

    def __init__(self, train, test, train_meta=None, test_meta=None,
                 meta=None, create_hash=True):

        self.train = train
        self.test = test
        if train_meta is not None:
            self.train_meta = train_meta
        else:
            self.train_meta = pd.DataFrame()

        if test_meta is not None:
            self.test_meta = test_meta
        else:
            self.test_meta = pd.DataFrame()

        if meta is not None:
            self.meta = meta
        else:
            self.meta = {}

        if 'name' not in self.meta:
            meta['name'] = 'noname_{}'.format(str(uuid4())[:5])
        self.name = meta['name']

        if create_hash:
            self._create_hash()

    @staticmethod
    def is_dataset(some_path):
        some_path = Path(some_path)
        return (some_path/'train.csv').exists() and (some_path/'test.csv').exists()

    @classmethod
    def from_path(cls, dataset_path, only_meta=False):

        dataset_path = Path(dataset_path)

        try:
            assert dataset_path.exists()
        except AssertionError as e:
            raise ValueError(f'Dataset {dataset_path} does not exist')

        if not only_meta:
            train = pd.read_csv(dataset_path / 'train.csv')
            test = pd.read_csv(dataset_path / 'test.csv')
        else:
            train = pd.DataFrame()
            test = pd.DataFrame()

        train_meta_path = dataset_path / 'train-meta.csv'
        train_meta = None
        if train_meta_path.exists():
            train_meta = pd.read_csv(train_meta_path)

        test_meta_path = dataset_path / 'test-meta.csv'
        test_meta = None
        if test_meta_path.exists():
            test_meta = pd.read_csv(test_meta_path)

        meta_path = dataset_path / 'meta.json'
        meta = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)

        return Dataset(train, test, train_meta, test_meta, meta, create_hash=False)

    def _create_hash(self):
        data_hash = hash_from_frames([self.train, self.test])
        self.meta['data_hash'] = data_hash

    @staticmethod
    def _contamination(data):
        labels = data.iloc[:, -1]
        return labels.mean()

    @property
    def train_contamination_perc(self):
        val = self._contamination(self.train)
        return val*100

    @property
    def test_contamination_perc(self):
        val = self._contamination(self.test)
        return val*100

    @property
    def meta_columns(self):
        return list(self.train_meta.columns)

    @property
    def data_columns(self):
        return list(self.train.columns)

    def create_outlier_batch(self, n_samples, perc_outlier, train=True,
                             fair_sample=True, include_meta=True):
        """ Create a batch with a defined percentage of outliers. """

        data = self.train if train else self.test

        if include_meta:
            meta = self.train_meta if train else self.test_meta
            data = pd.concat([data, meta], axis=1)
            data = data.loc[:, ~data.columns.duplicated()]  # if there are duplicate columns, like target

        # separate inlier and outlier data
        normal = data[data['target'] == 0]
        outlier = data[data['target'] == 1]

        if n_samples == 1:
            n_outlier = np.random.binomial(1, .01 * perc_outlier)
            n_normal = 1 - n_outlier
        else:
            n_outlier = int(perc_outlier * .01 * n_samples)
            n_normal = int((100 - perc_outlier) * .01 * n_samples)

        # draw samples
        batch_normal = sample_df(normal, n_normal)

        if fair_sample:
            scenario_col = self.meta.get('scenario_col')
            assert scenario_col is not None
            batch_outlier = fair_attack_sample(outlier, num_sample=n_outlier, scenario_col=scenario_col)
        else:
            batch_outlier = sample_df(outlier, n_outlier)

        batch = pd.concat([batch_normal, batch_outlier], axis=0)
        batch = batch.sample(frac=1)

        is_outlier = batch['target'].values

        if include_meta and self.meta_columns:
            batch_meta = batch.loc[:, self.meta_columns]
        else:
            batch_meta = None

        batch = batch.loc[:, self.data_columns]
        batch.drop(columns=['target'], inplace=True)
        return batch, is_outlier, batch_meta

    def visualize(self, ax, train=True):

        data = self.train if train else self.test

        targets = data.loc[:, "target"]

        # ToDo: only numerical features
        data = data.select_dtypes(include=np.number)
        data.drop(columns='target', inplace=True)

        batch = create_outlier_batch(data, targets, n_samples=1000, perc_outlier=10)

        X, y = batch.data.astype('float32'), batch.target.astype('bool')
        num_dims = X.shape[1]

        if num_dims > 2:
            X = TSNE(n_components=2).fit_transform(X)

        X_norm, X_anom = X[y], X[~y]
        plot_data_2d(ax, X_norm, X_anom)

        title = 'Train data' if train else 'Test data'
        ax.set_title(title)

    def write_to(self, root_path, archive=False, overwrite=False, visualize=False):

        logging.info("Writing {} to {}".format(self.name, root_path))

        root_path = Path(root_path).resolve()

        dataset_path = root_path / self.name
        if dataset_path.exists() and not overwrite:
            logging.info("Found existing; no overwrite")
            return

        dataset_path.mkdir(parents=True, exist_ok=True)

        train_path = dataset_path / 'train.csv'
        test_path = dataset_path / 'test.csv'
        train_meta_path = dataset_path / 'train-meta.csv'
        test_meta_path = dataset_path / 'test-meta.csv'
        meta_path = dataset_path / 'meta.json'

        self.train.to_csv(train_path, index=False)
        self.test.to_csv(test_path, index=False)

        if not self.train_meta.empty:
            self.train_meta.to_csv(train_meta_path, index=False)

        if not self.test_meta.empty:
            self.test_meta.to_csv(test_meta_path, index=False)

        with open(meta_path, 'w') as f:
            json.dump(self.meta, f)

        if visualize:
            logging.info('Visualizing the data')
            fig, ax = plt.subplots(1, 2)
            try:
                self.visualize(ax[0], train=True)
                self.visualize(ax[1], train=False)
            except Exception as e:
                logging.exception("Failed to visualize_data")
            fig.savefig(dataset_path / 'data.png')
            plt.close()

        if archive:
            logging.info('Compressing the data')
            shutil.make_archive(dataset_path, 'zip', root_path, self.name)
