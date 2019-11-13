import os

from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from joblib import dump


class ModelWrapper(ABC):

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def anomaly_score(self, x):
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

    def fit(self, x):
        self.model.fit(x)

    def anomaly_score(self, x):
        return self.model.score_samples(x)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class NearestNeighborsModelWrapper(ModelWrapper):

    def __init__(self, model_params):
        self.model = NearestNeighbors(**model_params)

    def fit(self, x):
        self.model.fit(x)

    def anomaly_score(self, x):
        distances, _ = self.model.kneighbors(x)
        return np.mean(distances, axis=1)

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


ALGORITHM2WRAPPER = {
    'IsolationForest': IsolationForestModelWrapper,
    'NearestNeighbors': NearestNeighborsModelWrapper
}
