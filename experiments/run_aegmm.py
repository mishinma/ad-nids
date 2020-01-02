
import shutil
import logging

from timeit import default_timer as timer


import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from alibi_detect.od import OutlierAEGMM
from alibi_detect.models.autoencoder import eucl_cosim_features
from alibi_detect.models.gmm import gmm_params
from alibi_detect.models.losses import loss_aegmm
from tensorflow.keras.layers import Dense, InputLayer

from ad_nids.ml import run_experiments, trainer
from ad_nids.config import config_dumps
from ad_nids.dataset import Dataset
from ad_nids.utils.logging import log_experiment, log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score, log_preds
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold, cov_elbo_type

EXPERIMENT_NAME = 'aegmm'
DEFAULT_CONTAM_PERCS = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 3, 5, 10])


def run_aegmm(config, log_dir, do_plot_frontier=False):
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
    X_train, X_val = train_test_split(X_train, test_size=0.1)

    scaler = None
    if config['data_standardization']:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    # Create a directory to store experiment logs
    logging.info('Created a new log directory\n')
    logging.info(f'{log_dir}\n')
    logging.info(f'\n >>> tensorboard --host 0.0.0.0 --port 8888 --logdir {log_dir}\n')

    # Train the model on normal data
    logging.info('Fitting the model...')
    se = timer()
    input_dim = X_train.shape[1]
    latent_dim = config['latent_dim']
    n_gmm = config['n_gmm']

    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(input_dim,)),
            Dense(9, activation=tf.nn.tanh),
            Dense(6, activation=tf.nn.tanh),
            Dense(3, activation=tf.nn.tanh),
            Dense(latent_dim, activation=None)
        ])

    decoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim,)),
            Dense(3, activation=tf.nn.tanh),
            Dense(6, activation=tf.nn.tanh),
            Dense(9, activation=tf.nn.tanh),
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
    log_experiment(log_dir, od, eval_results)
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    log_preds(log_dir, 'test', X_test_pred, y_test)
    log_preds(log_dir, 'train', X_threshold_pred, y_threshold)

    # ToDo: subsample
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=test_batch.target_names)
    if do_plot_frontier:
        input_dim = X_threshold.shape[1]
        if input_dim == 2:
            log_plot_frontier(log_dir, od, X_threshold, y_threshold, X_test, y_test)
        else:
            logging.warning(f"Cannot plot frontier for {input_dim} dims")


if __name__ == '__main__':
    run_experiments(run_aegmm)
