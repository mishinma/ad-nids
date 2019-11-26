import os

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from joblib import dump

from generative_nids.ml.model import AutoEncoder
from generative_nids.utils.ml_utils import get_threshold


class ModelWrapper(ABC):

    @abstractmethod
    def fit(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def anomaly_score(self, x, *args, **kwargs):
        pass

    # @abstractmethod
    # def predict(self, x):
    #     pass

    @abstractmethod
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

    def __init__(self, model_params):
        self.model = IsolationForest(**model_params)

    def fit(self, x, *args, **kwargs):
        self.model.fit(x)

    def anomaly_score(self, x, *args, **kwargs):
        return self.model.score_samples(x)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class NearestNeighborsModelWrapper(ModelWrapper):

    def __init__(self, model_params):
        self.model = NearestNeighbors(**model_params)
        self._threshold = None

    @property
    def threshold(self):
        if self._threshold is None:
            raise ValueError('Need to set threshold first')
        return self._threshold

    def fit(self, x, *args, **kwargs):
        # ToDo: set threshold
        self.model.fit(x)

    def anomaly_score(self, x, *args, **kwargs):
        distances, _ = self.model.kneighbors(x)
        return np.mean(distances, axis=1)

    def predict(self, x):
        # compute anomaly scores
        # threshold
        pass

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class AutoEncoderModelWrapper(ModelWrapper):

    criterion = nn.MSELoss()

    def __init__(self, model_params):
        self.model = AutoEncoder(**model_params)

    def fit(self, x, *args, **kwargs):

        # ToDo: move to init
        optimizer = kwargs.get('optimizer')
        device = kwargs.get('device', 'cpu')
        epochs = kwargs.get('epochs')

        x = x.to(device)

        for i in range(epochs):

            rec_x = self.model(x)
            loss = self.criterion(rec_x, x)

            # backprop
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def anomaly_score(self, x, *args, **kwargs):
        pass

    def save(self, save_dir):
        torch.save(self.model, os.path.join(save_dir, 'model.pth'))


ALGORITHM2WRAPPER = {
    'IsolationForest': IsolationForestModelWrapper,
    'NearestNeighbors': NearestNeighborsModelWrapper,
    'AutoEncoder': AutoEncoderModelWrapper
}
