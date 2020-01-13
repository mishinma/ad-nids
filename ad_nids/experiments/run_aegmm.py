
import json
import logging

from timeit import default_timer as timer


import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from alibi_detect.od import OutlierAEGMM
from alibi_detect.models.autoencoder import eucl_cosim_features
from alibi_detect.models.gmm import gmm_params
from alibi_detect.models.losses import loss_aegmm
from alibi_detect.utils.saving import load_detector, save_detector

from tensorflow.keras.layers import Dense, InputLayer

from ad_nids.ml import trainer
from ad_nids.config import config_dumps
from ad_nids.utils.misc import jsonify
from ad_nids.utils.logging import log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold, concatenate_preds

EXPERIMENT_NAME = 'aegmm'
DEFAULT_CONTAM_PERCS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10]


def run_aegmm(config, log_dir, experiment_data,
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
        n_gmm = config['n_gmm']

        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(input_dim,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(6, activation=tf.nn.tanh),
                Dense(3, activation=tf.nn.tanh),
                Dense(latent_dim, activation=None)
            ])

        decoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(3, activation=tf.nn.tanh),
                Dense(6, activation=tf.nn.tanh),
                Dense(10, activation=tf.nn.tanh),
                Dense(input_dim, activation=None)
            ])

        gmm_density_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim + 2,)),
                Dense(10, activation=tf.nn.tanh),
                Dense(n_gmm, activation=tf.nn.softmax)
            ])

        # initialize outlier detector
        od = OutlierAEGMM(threshold=0.0,  # threshold for outlier score
                          encoder_net=encoder_net,  # can also pass AEGMM model instead
                          decoder_net=decoder_net,  # of separate encoder, decoder
                          gmm_density_net=gmm_density_net,  # and gmm density net
                          n_gmm=n_gmm,
                          recon_features=eucl_cosim_features)  # fn used to derive features
        # from the reconstructed
        # instances based on cosine
        # similarity and Eucl distance

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    loss_fn_kwargs = dict(
        w_energy=.1,
        w_cov_diag=.005
    )
    trainer(od.aegmm, loss_aegmm, X_train, X_val=X_val, loss_fn_kwargs=loss_fn_kwargs,
            epochs=config['num_epochs'], batch_size=config['batch_size'],
            optimizer=optimizer, log_dir=log_dir)
    # set GMM parameters
    x_recon, z, gamma = od.aegmm(X_train)
    od.phi, od.mu, od.cov, od.L, od.log_det_cov = gmm_params(z, gamma)
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
    if contam_percs is None:
        contam_percs = DEFAULT_CONTAM_PERCS
    contam_percs = np.array(contam_percs)
    train_prf1_curve = precision_recall_curve_scores(
        y_threshold, iscore_threshold, 100 - contam_percs)
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
