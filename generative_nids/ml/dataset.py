import json
import shutil
import zipfile
import hashlib

from pathlib import Path

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

    def __init__(self, train, test, meta, update_hash=True):
        self.train = train
        self.test = test
        self.meta = meta

        if update_hash:
            self._update_hash()

    @classmethod
    def from_path(cls, dataset_path):

        dataset_path = Path(dataset_path)

        try:
            assert dataset_path.exists()
        except AssertionError as e:
            raise ValueError(f'Dataset {dataset_path} does not exist')

        train = pd.read_csv(dataset_path / 'train.csv')
        test = pd.read_csv(dataset_path / 'test.csv')

        with open(dataset_path / 'meta.json', 'r') as f:
           meta = json.load(f)

        return Dataset(train, test, meta, update_hash=False)

    def _update_hash(self):
        data_hash = hash_from_frames([self.train, self.test])
        self.meta['data_hast'] = data_hash

    def write_to(self, root_path, archive=False):

        root_path = Path(root_path).resolve()

        dataset_path = root_path / self.meta['name']
        dataset_path.mkdir(parents=True)

        train_path = dataset_path / 'train.csv'
        test_path = dataset_path / 'test.csv'
        meta_path = dataset_path / 'meta.json'

        self.train.to_csv(train_path, index=None)
        self.test.to_csv(test_path, index=None)

        with open(meta_path, 'w') as f:
            json.dump(self.meta, f)

        if archive:
            shutil.make_archive(dataset_path, 'zip', root_path, self.meta['name'])
            shutil.rmtree(dataset_path)
