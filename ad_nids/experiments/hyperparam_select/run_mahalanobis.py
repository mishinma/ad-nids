
import json
import logging

from timeit import default_timer as timer

import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from alibi_detect.od import Mahalanobis
from alibi_detect.utils.saving import save_detector

from ad_nids.utils.misc import jsonify
from ad_nids.utils.logging import log_plot_prf1_curve, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold

EXPERIMENT_NAME = 'mahalanobis'


def run_mahalanobis(config, log_dir, dataset, sample_params, contam_percs):

    n_train_samples = sample_params['train']['n_samples']
    n_threshold_samples = sample_params['threshold']['n_samples']
    perc_threshold_outlier = sample_params['threshold']['perc_outlier']
    n_test_samples = sample_params['test']['n_samples']
    perc_test_outlier = sample_params['test']['perc_outlier']

    X_train, y_train = dataset.create_outlier_batch(train=True, n_samples=n_train_samples,
                                                    perc_outlier=0)
    X_threshold, y_threshold = dataset.create_outlier_batch(train=True, n_samples=n_threshold_samples,
                                                            perc_outlier=perc_threshold_outlier)
    X_test, y_test = dataset.create_outlier_batch(train=True, n_samples=n_test_samples,
                                                  perc_outlier=perc_test_outlier)

    # ToDO only use numerical features
    numeric_features = dataset.meta['numerical_features']

    # normalize
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
    ])

    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train).astype(np.float32)
    X_threshold = preprocessor.transform(X_threshold).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    # Train the model on normal data
    logging.info('Fitting the model...')
    se = timer()
    od = Mahalanobis(
        threshold=None,
        n_components=config['n_components'],
        std_clip=config['std_clip'],
        start_clip=config['start_clip']
    )
    # od.score(X_train)
    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')

    # Compute the anomaly scores for train with anomalies
    # Select a threshold that maximises F1 Score
    logging.info(f'Selecting the optimal threshold...')
    se = timer()
    score_threshold = od.score(X_threshold)  # feature and instance lvl
    contam_percs = np.array(contam_percs)
    train_prf1_curve = precision_recall_curve_scores(
        y_threshold, score_threshold, 100 - contam_percs)
    best_threshold = select_threshold(
        train_prf1_curve['thresholds'],
        train_prf1_curve['f1scores'])
    od.threshold = best_threshold
    y_threshold_pred = (score_threshold > od.threshold).astype(int)
    time_score_train = timer() - se

    train_cm = confusion_matrix(y_threshold, y_threshold_pred)
    train_prf1s = precision_recall_fscore_support(
        y_threshold, y_threshold_pred, average='binary')
    logging.info(f'Done (train): {timer() - se}')

    # Compute anomaly scores for test
    logging.info('Computing test anomaly scores...')
    se = timer()
    X_test_pred = od.predict(X_test)
    y_test_pred = X_test_pred['data']['is_outlier']
    time_score_test = timer() - se
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_prf1s = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    logging.info(f'Done (test): {timer() - se}')

    eval_results = {
        'threshold': od.threshold,
        'train_prf1_curve': train_prf1_curve,
        'train_prf1s': train_prf1s,
        'train_cm': train_cm,
        'test_prf1s': test_prf1s,
        'test_cm': test_cm,
        'time_score_train': time_score_train,
        'time_score_test': time_score_test,
        'time_fit': time_fit,
        'model_name': od.meta['name']
    }

    # Log everything
    logging.info(f'Logging the results\n')
    save_detector(od, str(log_dir / 'detector'))
    with open(log_dir / 'eval_results.json', 'w') as f:
        json.dump(jsonify(eval_results), f)
    # log_preds(log_dir, 'test', X_test_pred, y_test)
    # log_preds(log_dir, 'train', X_threshold_pred, y_threshold)
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    # ToDo: subsample
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=['normal', 'outlier'])
