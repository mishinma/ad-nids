
import json
import logging

from timeit import default_timer as timer

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from alibi_detect.od import Mahalanobis
from alibi_detect.utils.saving import load_detector, save_detector

from ad_nids.utils.misc import jsonify, predict_batch
from ad_nids.utils.logging import log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score, log_preds
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold

EXPERIMENT_NAME = 'mahalanobis'


def run_mahalanobis(config, log_dir, experiment_data,
                    contam_percs=None, load_outlier_detector=False, i_run=0):

    # data
    train_normal_batch, threshold_batch, test_batch = experiment_data
    X_train, y_train = train_normal_batch.data, train_normal_batch.target
    X_threshold, y_threshold = threshold_batch.data, threshold_batch.target
    X_test, y_test = test_batch.data, test_batch.target


    if load_outlier_detector:
        # Load the model
        logging.info('Loading the model...')
        try:
            od = load_detector(str(log_dir/'detector'))
            # fetch the time it took to fit the model
            time_fit = 0.
        except Exception as e:
            logging.exception("Could not load the detector")
            raise e
    else:
        # Train the model on normal data
        logging.info('Fitting the model...')
        se = timer()
        od = Mahalanobis(
            threshold=None,
            n_components=config['n_components'],
            std_clip=config['std_clip'],
            start_clip=config['start_clip'],
        )
        time_fit = timer() - se
        logging.info(f'Done: {time_fit}')

    # Compute the anomaly scores for train  anomalies
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
    if X_test.shape[0] > int(1e5):
        X_test_pred = predict_batch(od, X_test, batch_size=int(1e5))
    else:
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
    log_preds(log_dir, 'test', X_test_pred, y_test)
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    ylim = (np.min(X_test_pred['data']['instance_score']),
            np.quantile(X_test_pred['data']['instance_score'], 0.99))
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=test_batch.target_names, ylim=ylim)

