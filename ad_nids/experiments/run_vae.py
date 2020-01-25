
import json
import logging

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from alibi_detect.od import OutlierVAE
from alibi_detect.models.losses import elbo
from alibi_detect.models.autoencoder import VAE
from alibi_detect.utils.saving import load_detector, save_detector

from ad_nids.ml import build_net, trainer
from ad_nids.config import config_dumps
from ad_nids.utils.misc import jsonify, concatenate_preds
from ad_nids.utils.logging import log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold, \
    cov_elbo_type

EXPERIMENT_NAME = 'vae'
# Todo: should be a property of a dataset


def run_vae(config, log_dir, experiment_data,
            do_plot_frontier=False, contam_percs=None, load_outlier_detector=False):
    logging.info(f'Starting {config["config_name"]}')
    logging.info(config_dumps(config))

    if config["experiment_name"] != EXPERIMENT_NAME:
        logging.warning(
            'Experiment name mismatch. Expected {}, got {}.'.format(
                EXPERIMENT_NAME, config['experiment_name'])
        )

    # data
    train_normal_batch, train_outlier_batch, val_batch, test_batch = experiment_data
    X_train, y_train = train_normal_batch.data, train_normal_batch.target
    X_train_outlier, y_train_outlier = train_outlier_batch.data, train_outlier_batch.target
    X_val, y_val = val_batch.data, val_batch.target
    X_test, y_test = test_batch.data, test_batch.target

    if config['data_standardization']:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train_outlier = scaler.transform(X_train_outlier)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Create a directory to store experiment logs
    logging.info('Created a new log directory\n')
    logging.info(f'{log_dir}\n')
    logging.info(f'\n >>> tensorboard --host 0.0.0.0 --port 8888 --logdir {log_dir}\n')

    if load_outlier_detector:
        # Load the model
        logging.info('Loading the model...')
        try:
            od = load_detector(str(log_dir/'detector'))
            # fetch the time it took to fit the model
            with open(log_dir / 'eval_results.json', 'r') as f:
                time_fit = json.load(f)['time_fit']
        except Exception as e:
            logging.exception("Could not load the detector")
            raise e
    else:
        # Train the model on normal data
        logging.info('Fitting the model...')
        se = timer()
        input_dim = X_train.shape[1]
        latent_dim = config['latent_dim']
        encoder_dims = [input_dim] + json.loads(config['encoder_net'])
        encoder_net = build_net(encoder_dims)
        decoder_dims = [latent_dim] + json.loads(config['decoder_net']) + [input_dim]
        decoder_net = build_net(decoder_dims)
        vae = VAE(encoder_net, decoder_net, latent_dim)
        od = OutlierVAE(threshold=0.0, vae=vae, score_type='mse',
                        latent_dim=config['latent_dim'], samples=config['samples'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

        loss_fn_kwargs = {}
        loss_fn_kwargs.update(cov_elbo_type(cov_elbo=dict(sim=.1), X=X_train))
        trainer(od.vae, elbo, X_train, X_val=X_val, loss_fn_kwargs=loss_fn_kwargs,
                epochs=config['num_epochs'], batch_size=config['batch_size'],
                optimizer=optimizer, log_dir=log_dir)
        time_fit = timer() - se
        logging.info(f'Done: {time_fit}')

    # Compute the anomaly scores for train with anomalies
    # Select a threshold that maximises F1 Score
    logging.info(f'Selecting the optimal threshold...')
    se = timer()
    X_threshold_pred = od.predict(X_train)  # feature and instance lvl
    X_threshold_outlier_pred = od.predict(X_train_outlier)
    X_threshold_pred = concatenate_preds(X_threshold_pred, X_threshold_outlier_pred)
    y_threshold = np.concatenate([y_train, y_train_outlier])
    iscore_threshold = X_threshold_pred['data']['instance_score']
    contam_percs = np.array(contam_percs)
    train_prf1_curve = precision_recall_curve_scores(
        y_threshold, iscore_threshold, 100 - contam_percs)
    # todo save the corresponding contam percent
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
    if not load_outlier_detector:
        save_detector(od, str(log_dir / 'detector'))
    with open(log_dir / 'eval_results.json', 'w') as f:
        json.dump(jsonify(eval_results), f)
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    # log_preds(log_dir, 'test', X_test_pred, y_test)
    # log_preds(log_dir, 'train', X_threshold_pred, y_threshold)

    # ToDo: subsample
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=test_batch.target_names)
    if do_plot_frontier:
        input_dim = X_train.shape[1]
        X_threshold = np.concatenate([X_train, X_train_outlier])
        if input_dim == 2:
            log_plot_frontier(log_dir, od, X_threshold, y_threshold, X_test, y_test)
        else:
            logging.warning(f"Cannot plot frontier for {input_dim} dims")
