import os

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from joblib import dump

class ModelWrapper(ABC):

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class SklearnModelWrapper(ModelWrapper):

    ALGORITHM = BaseEstimator

    def __init__(self, hyperparams):
        self.model = self.__class__.ALGORITHM(**hyperparams)

    def fit(self, x):
        self.model.fit(x)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, save_dir):
        dump(self.model, os.path.join(save_dir, 'model.joblib'))


class IsolationForestModelWrapper(SklearnModelWrapper):

    ALGORITHM = IsolationForest


ALGORITHM2WRAPPER = {
    'IsolationForest': IsolationForestModelWrapper
}
