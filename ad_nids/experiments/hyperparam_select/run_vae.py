
import json
import logging

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from alibi_detect.od import OutlierVAE
from alibi_detect.models.losses import elbo
from alibi_detect.models.autoencoder import VAE
from alibi_detect.utils.saving import save_detector

from ad_nids.ml import build_net, trainer, DataGenerator
from ad_nids.utils.misc import jsonify
from ad_nids.utils.logging import log_plot_prf1_curve, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold, \
    cov_elbo_type

EXPERIMENT_NAME = 'vae'


def run_vae(config, log_dir, experiment_data, contam_percs, i_run=0):

    # data
    train_normal_batch, threshold_batch, test_batch = experiment_data
    X_train, y_train = train_normal_batch.data, train_normal_batch.target
    X_threshold, y_threshold = threshold_batch.data, threshold_batch.target
    X_test, y_test = test_batch.data, test_batch.target

    # Train the model on normal data
    logging.info('Fitting the model...')
    se = timer()
    input_dim = X_train.shape[1]
    latent_dim = config['latent_dim']

    encoder_hidden_dims = json.loads(config['encoder_net'])
    encoder_activations = [tf.nn.relu] * len(encoder_hidden_dims)
    encoder_net = build_net(input_dim, encoder_hidden_dims, encoder_activations)

    decoder_hidden_dims = json.loads(config['decoder_net']) + [input_dim]
    decoder_activations = [tf.nn.relu] * (len(decoder_hidden_dims) - 1) + [None]
    decoder_net = build_net(latent_dim, decoder_hidden_dims, decoder_activations)

    vae = VAE(encoder_net, decoder_net, latent_dim)
    od = OutlierVAE(threshold=0.0, vae=vae, score_type='mse',
                    latent_dim=config['latent_dim'], samples=config['samples'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    loss_fn_kwargs = {}
    loss_fn_kwargs.update(cov_elbo_type(cov_elbo=dict(sim=.1), X=X_train))
    i_run_log_dir = log_dir / str(i_run)
    train_gen = DataGenerator(X_train, batch_size=config['batch_size'])
    trainer(od.vae, elbo, train_gen, loss_fn_kwargs=loss_fn_kwargs,
            epochs=config['num_epochs'], epoch_size=config.get('epoch_size'),
            optimizer=optimizer, log_dir=i_run_log_dir,
            checkpoint=True, checkpoint_freq=5)
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
    log_plot_prf1_curve(log_dir, train_prf1_curve)
    ylim = (X_test_pred['data']['instance_score'].min(),
            X_test_pred['data']['instance_score'].quantile(0.99))
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=test_batch.target_names, y_lim=ylim)

