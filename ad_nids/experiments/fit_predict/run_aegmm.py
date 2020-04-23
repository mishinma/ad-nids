
import json
import logging

from timeit import default_timer as timer


import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.layers import Dense, InputLayer, Dropout

from alibi_detect.od import OutlierAEGMM
from alibi_detect.models.autoencoder import eucl_cosim_features
from alibi_detect.models.gmm import gmm_params
from alibi_detect.models.losses import loss_aegmm
from alibi_detect.utils.saving import load_detector, save_detector

from ad_nids.ml import trainer, build_net, DataGenerator
from ad_nids.utils.misc import jsonify
from ad_nids.utils.logging import log_plot_prf1_curve,\
    log_plot_frontier, log_plot_instance_score, log_preds
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold

EXPERIMENT_NAME = 'aegmm'
EPOCH_SIZE = 500


def run_aegmm(config, log_dir, experiment_data, contam_percs=None,
              load_outlier_detector=False, i_run=0):

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
        input_dim = X_train.shape[1]
        latent_dim = config['latent_dim']
        n_gmm = config['n_gmm']

        encoder_hidden_dims = json.loads(config['encoder_net']) + [latent_dim]
        encoder_activations = [tf.nn.tanh] * (len(encoder_hidden_dims) - 1) + [None]
        encoder_net = build_net(input_dim, encoder_hidden_dims, encoder_activations)

        decoder_hidden_dims = json.loads(config['decoder_net']) + [input_dim]
        decoder_activations = [tf.nn.tanh] * (len(decoder_hidden_dims) - 1) + [None]
        decoder_net = build_net(latent_dim, decoder_hidden_dims, decoder_activations)

        gmm_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim + 2,)),
                Dense(3, activation=tf.nn.tanh),
                Dropout(0.5),
                Dense(10, activation=tf.nn.tanh),
                Dense(n_gmm, activation=tf.nn.softmax)
            ]
        )

        # initialize outlier detector
        od = OutlierAEGMM(threshold=0.0,  # threshold for outlier score
                          encoder_net=encoder_net,  # can also pass AEGMM model instead
                          decoder_net=decoder_net,  # of separate encoder, decoder
                          gmm_density_net=gmm_net,  # and gmm density net
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
        i_run_log_dir = log_dir / str(i_run)
        train_gen = DataGenerator(X_train, batch_size=config['batch_size'])

        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        if X_train.shape[0] > num_epochs * batch_size * EPOCH_SIZE:
            epoch_size = None
        else:
            epoch_size = EPOCH_SIZE

        trainer(od.aegmm, loss_aegmm, train_gen, X_val=X_threshold[y_threshold == 0], loss_fn_kwargs=loss_fn_kwargs,
                epochs=config['num_epochs'], epoch_size=epoch_size,
                optimizer=optimizer, log_dir=i_run_log_dir,
                checkpoint=True, checkpoint_freq=5)
        # set GMM parameters
        x_recon, z, gamma = od.aegmm(X_train)
        od.phi, od.mu, od.cov, od.L, od.log_det_cov = gmm_params(z, gamma)
        time_fit = timer() - se
        logging.info(f'Done: {time_fit}')

    # Compute the anomaly scores for train with anomalies
    # Select a threshold that maximises F1 Score
    logging.info(f'Selecting the optimal threshold...')
    se = timer()
    X_threshold_pred = od.predict(X_threshold)  # feature and instance lvl
    iscore_threshold = X_threshold_pred['data']['instance_score']
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
    X_test_pred = od.predict(X_test, batch_size=int(1e5))
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
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    log_preds(log_dir, 'test', X_test_pred, y_test)
    #log_preds(log_dir, 'train', X_threshold_pred, y_threshold)
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold)

