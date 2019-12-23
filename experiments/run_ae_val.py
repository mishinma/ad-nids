
import shutil
import logging

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from alibi_detect.od import OutlierAE
# from alibi_detect.models.losses import recon_loss

from ad_nids.ml import build_ae, run_experiments, trainer
from ad_nids.config import config_dumps
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import get_log_dir, log_experiment, log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold

EXPERIMENT_NAME = 'ae'
DEFAULT_CONTAM_PERCS = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10])

np.random.seed(42)
tf.random.set_seed(42)


def run_ae_val(config, log_exp_dir, do_plot_frontier=False):
    logging.info(f'Starting {config["config_name"]}')
    logging.info(config_dumps(config))

    if config["experiment_name"] != EXPERIMENT_NAME:
        logging.warning(
            'Experiment name mismatch. Expected {}, got {}.'.format(
                EXPERIMENT_NAME, config['experiment_name'])
        )

    # Create dataset and loaders
    logging.info('Loading the dataset...')
    dataset = Dataset.from_path(config['dataset_path'])

    normal_batch = dataset.create_outlier_batch(train=True, perc_outlier=0)
    X_train, y_train = normal_batch.data.astype(np.float32), normal_batch.target
    scaler = None
    if config['data_standardization']:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

    # Create a directory to store experiment logs
    log_dir = get_log_dir(log_exp_dir, config["config_name"])
    log_dir.mkdir(parents=True)
    logging.info('Created a new log directory')
    logging.info(f'\ntensorboard --logdir {log_dir}\n')

    # Train the model on normal data
    logging.info('Fitting the model...')
    se = timer()
    input_dim = X_train.shape[1]
    ae = build_ae(config['hidden_dim'], config['encoding_dim'],
                  config['num_hidden'], input_dim)
    od = OutlierAE(threshold=0.0, ae=ae)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    # od.fit(X_train, optimizer=optimizer,
    #        epochs=config['num_epochs'], batch_size=config['batch_size'])
    mse = tf.losses.MeanSquaredError()
    trainer(od.ae, mse, X_train, epochs=config['num_epochs'], batch_size=config['batch_size'],
            log_dir=log_dir)
    time_fit = timer() - se
    logging.info(f'Done: {time_fit}')


    # Compute the anomaly scores for train with anomalies
    # Select a threshold that maximises F1 Score
    threshold_batch = dataset.create_outlier_batch(train=True, scaler=scaler)
    X_threshold, y_threshold = threshold_batch.data.astype(np.float32), threshold_batch.target
    logging.info(f'Selecting the optimal threshold...')
    se = timer()
    X_threshold_pred = od.predict(X_threshold)  # feature and instance lvl
    iscore_threshold = X_threshold_pred['data']['instance_score']
    train_prf1_curve = precision_recall_curve_scores(
        y_threshold, iscore_threshold, 100 - DEFAULT_CONTAM_PERCS)
    best_threshold = select_threshold(
        train_prf1_curve['thresholds'],
        train_prf1_curve['f1scores'])
    od.threshold = best_threshold
    y_threshold_pred = (iscore_threshold > od.threshold).astype(int)
    X_threshold_pred['data']['is_outlier'] = y_threshold_pred
    time_score_train = timer() - se

    train_cm = confusion_matrix(y_threshold, y_threshold_pred)
    train_prf1s = precision_recall_fscore_support(
        y_threshold, y_threshold_pred, average='binary')
    logging.info(f'Done (train): {timer() - se}')

    # Compute anomaly scores for test
    logging.info('Computing test anomaly scores...')
    test_batch = dataset.create_outlier_batch(train=False, scaler=scaler)
    X_test, y_test = test_batch.data.astype(np.float32), test_batch.target
    se = timer()
    X_test_pred = od.predict(X_test)
    y_test_pred = X_test_pred['data']['is_outlier']
    time_score_test = timer() - se
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_prf1s = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    logging.info(f'Done (test): {timer() - se}')

    eval_results = {
        'test_pred': X_test_pred,
        'train_pred': X_threshold_pred,
        'y_test': y_test,
        'y_train': y_threshold,
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
    try:
        log_experiment(log_dir, config, dataset.meta, od, eval_results)
        log_plot_prf1_curve(log_dir, train_prf1_curve)
        # ToDo: subsample
        log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                                labels=test_batch.target_names)
        if do_plot_frontier:
            input_dim = X_threshold.shape[1]
            if input_dim == 2:
                log_plot_frontier(log_dir, od, X_threshold, y_threshold, X_test, y_test)
            else:
                logging.warning(f"Cannot plot frontier for {input_dim} dims")

    except Exception as e:
        shutil.rmtree(log_dir)
        raise e


if __name__ == '__main__':
    run_experiments(run_ae)
