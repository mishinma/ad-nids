import sklearn

MODELS = {
    'IsolationForest':
        [sklearn.ensemble.IsolationForest,
         {'n_estimators': 100}]
}


class ModelWrapper:

    def __init__(self, name, model, conf):

        self.name = name
        self.model = model(**conf)
