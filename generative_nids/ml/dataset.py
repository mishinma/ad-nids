import json
import shutil
import zipfile
import hashlib
import logging

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def create_meta(dataset_name, train_split, test_split, frequency,
                features, name=None, notes=None):

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
                 meta=None, create_hash=True, scaler=None):

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

        self.scaler = scaler

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

    def loader(self, train=True, contamination=True, batch_size=None, shuffle=True):

        if train:
            x = self.train.iloc[:, :-1].to_numpy()
            y = self.train.iloc[:, -1].to_numpy()
        else:
            x = self.test.iloc[:, :-1].to_numpy()
            y = self.test.iloc[:, -1].to_numpy()

        if not contamination:
            x = x[y == 0]
            y = np.zeros_like(y)

        # ToDo: is it the right way to do it? keep for now
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return Dataloader(x, y, batch_size, shuffle)

    def _create_hash(self):
        data_hash = hash_from_frames([self.train, self.test])
        self.meta['data_hash'] = data_hash

    def write_to(self, root_path, archive=False, overwrite=False):

        root_path = Path(root_path).resolve()

        dataset_path = root_path / self.name
        dataset_path.mkdir(parents=True, exist_ok=overwrite)

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

        if archive:
            shutil.make_archive(dataset_path, 'zip', root_path, self.name)
            shutil.rmtree(dataset_path)


# ToDo: Dataloader class?
# Inspiration:
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)


class Dataloader:

    def __init__(self, x, y, batch_size=None, shuffle=True):
        self.x = x
        self.y = y  # if None no batches
        self.batch_size = None
        self.shuffle = shuffle
