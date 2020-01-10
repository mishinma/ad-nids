import json
import shutil
import zipfile
import hashlib
import logging

from pathlib import Path
from uuid import uuid4

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from alibi_detect.utils.data import create_outlier_batch

from ad_nids.utils import plot_data_2d


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
        self.train_meta = train_meta
        self.test_meta = test_meta
        self.meta = meta

        if self.meta is not None and self.meta.get('name'):
            self.name = self.meta['name']
        else:
            self.name = 'noname_{}'.format(str(uuid4())[:5])

        if create_hash:
            self._create_hash()

    @staticmethod
    def is_dataset(some_path):
        some_path = Path(some_path)
        return (some_path/'train.csv').exists() and (some_path/'test.csv').exists()

    @classmethod
    def from_path(cls, dataset_path):

        dataset_path = Path(dataset_path)

        try:
            assert dataset_path.exists()
        except AssertionError as e:
            raise ValueError(f'Dataset {dataset_path} does not exist')

        train = pd.read_csv(dataset_path / 'train.csv')
        test = pd.read_csv(dataset_path / 'test.csv')

        train_meta_path = dataset_path / 'train-meta.json'
        train_meta = None
        if train_meta_path.exists():
            with open(train_meta_path, 'r') as f:
                train_meta = json.load(f)

        test_meta_path = dataset_path / 'test-meta.json'
        test_meta = None
        if test_meta_path.exists():
            with open(test_meta_path, 'r') as f:
                test_meta = json.load(f)

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

    def create_outlier_batch(self, train=True, n_samples=-1, perc_outlier=-1, scaler=None):
        """

        :param train:
        :param n_samples: -1 infer number of samples
        :param perc_outlier: -1 true contamination
        :param scaler:
        :return:
        """
        # ToDo: test this
        if train:
            data = self.train
            true_contamination = self.train_contamination_perc
        else:
            data = self.test
            true_contamination = self.test_contamination_perc

        targets = data.iloc[:, -1]
        data = data.iloc[:, :-1]

        if perc_outlier == -1:
            perc_outlier = true_contamination

        if n_samples == -1:
            # under/over sample outliers
            # take all normal data
            n_outlier = int(perc_outlier * .01 * data.shape[0])
            n_normal = int((100 - true_contamination) * .01 * data.shape[0])
            n_samples = n_outlier + n_normal
        if scaler is not None:
            data = scaler.transform(data)

        return create_outlier_batch(data, targets, n_samples, perc_outlier)

    def visualize(self, ax, train=True):

        data = self.train if train else self.test
        targets = data.iloc[:, -1]
        data = data.iloc[:, :-1]

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

        self.train.to_csv(train_path, index=None)
        self.test.to_csv(test_path, index=None)

        if self.train_meta is not None:
            self.train_meta.to_csv(train_meta_path, index=None)

        if self.test_meta is not None:
            self.test_meta.to_csv(test_meta_path, index=None)

        if self.meta is not None:
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

