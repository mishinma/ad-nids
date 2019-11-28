import os
import logging

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from joblib import dump

from generative_nids.ml.model import AE
from generative_nids.ml.utils import get_threshold

FIT_PARAMS = {"lr", "num_epochs", "optimizer", "device"}


class ModelWrapper(ABC):

    def fit(self, x, *args, **kwargs):
        pass

    def anomaly_score(self, x):
        pass

    def predict(self, x):
        pass

    def save(self, save_dir):
        pass

#
# class SklearnModelWrapper(ModelWrapper):
#
#     ALGORITHM = BaseEstimator
#
#     def __init__(self, hyperparams):
#         self.model = self.__class__.ALGORITHM(**hyperparams)
#
#     def fit(self, x):
#         self.model.fit(x)
#
#     def predict(self, x):
#         return self.model.predict(x)
#
#     def save(self, save_dir):
#         dump(self.model, os.path.join(save_dir, 'model.joblib'))


class IsolationForestModelWrapper(ModelWrapper):

    model_params = {"n_estimators", "behaviour", "contamination"}

    def __init__(self, model_params):
        self.model = IsolationForest(**model_params)

    def fit(self, loader, *args, **kwargs):
        self.model.fit(loader.x)

    def anomaly_score(self, loader, *args, **kwargs):
        return self.model.score_samples(loader.x)

    def predict(self, loader):
        return self.model.predict(loader.x)

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class NearestNeighborsModelWrapper(ModelWrapper):

    model_params = {"n_neighbors", "algorithm"}

    def __init__(self, model_params):
        self.model = NearestNeighbors(**model_params)
        self._threshold = None

    @property
    def threshold(self):
        if self._threshold is None:
            raise ValueError('Need to set threshold first')
        return self._threshold

    def fit(self, loader, *args, **kwargs):
        # ToDo: set threshold
        self.model.fit(loader.x)

    def anomaly_score(self, loader):
        distances, _ = self.model.kneighbors(loader.x)
        return np.mean(distances, axis=1)

    def predict(self, loader):
        # compute anomaly scores
        # threshold
        pass

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class AutoEncoderModelWrapper(ModelWrapper):

    criterion = nn.MSELoss()
    model_params = {"input_dim", "hidden_dim", "latent_dim", "num_hidden"}

    def __init__(self, model_params):
        self.model = AE(**model_params)
        self._threshold = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, loader, *args, **kwargs):

        lr = kwargs['lr']
        num_epochs = kwargs['num_epochs']

        optim_name = kwargs.get('optimizer', 'Adam')
        optimizer = getattr(optim, optim_name)(self.model.parameters(), lr=lr)

        x = torch.tensor(loader.x.astype(np.float32))
        x = x.to(self.device)

        for i in range(num_epochs):
            rec_x = self.model(x)
            loss = self.criterion(rec_x, x)

            # backprop
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def anomaly_score(self, loader):
        x = torch.tensor(loader.x.astype(np.float32))
        x = x.to(self.device)
        with torch.no_grad():
            rec_x = self.model(x)

        x, rec_x = x.numpy(), rec_x.numpy()
        scores = (np.square(x - rec_x)).mean(axis=1)
        return scores

    def save(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))

    # Load:
    #
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()


ALGORITHM2WRAPPER = {
    'IsolationForest': IsolationForestModelWrapper,
    'NearestNeighbors': NearestNeighborsModelWrapper,
    'Autoencoder': AutoEncoderModelWrapper
}


def create_model(algorithm, model_parameters):
    return ALGORITHM2WRAPPER[algorithm](model_parameters)


def is_param_required(param, algorithm):
    model_params = ALGORITHM2WRAPPER[algorithm].model_params
    return param in model_params


def filter_model_params(params, algorithm):

    if isinstance(params, dict):
        params_keys = set(params.keys())
    else:
        params_keys = set(params)

    algorithm_model_params = ALGORITHM2WRAPPER[algorithm].model_params
    model_keys = params_keys.intersection(algorithm_model_params)
    other_keys = params_keys - model_keys

    if isinstance(params, dict):
        model_params = {k: v for k, v in params.items() if k in model_keys}
        other_params = {k: v for k, v in params.items() if k in other_keys}
    else:
        model_params = model_keys
        other_params = other_keys

    return model_params, other_params
