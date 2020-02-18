
import json
import logging

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from alibi_detect.od import OutlierAEGMM
from alibi_detect.models.autoencoder import eucl_cosim_features
from alibi_detect.models.gmm import gmm_params
from alibi_detect.models.losses import loss_aegmm
from alibi_detect.utils.saving import save_detector

from ad_nids.ml import build_net, trainer
from ad_nids.utils.misc import jsonify
from ad_nids.utils.logging import log_plot_prf1_curve, log_plot_instance_score
from ad_nids.utils.metrics import precision_recall_curve_scores, select_threshold

EXPERIMENT_NAME = 'aegmm'


def run_ae(config, log_dir, dataset, sample_params, contam_percs):

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

    numeric_features = dataset.meta['numerical_features']
    binary_features = dataset.meta['binary_features']
    categorical_feature_map = dataset.meta['categorical_feature_map']

    # normalize
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(categories=list(categorical_feature_map.values())),
         list(categorical_feature_map.keys())),
        ('bin', FunctionTransformer(), binary_features),
        ('num', StandardScaler(), numeric_features),
    ])

    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_threshold = preprocessor.transform(X_threshold)
    X_test = preprocessor.transform(X_test)

    # Create a directory to store experiment logs
    logging.info(f'\n >>> tensorboard --host 0.0.0.0 --port 8888 --logdir {log_dir}\n')

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

    gmm_hidden_dims = json.loads(config['gmm_density_net']) + [n_gmm]
    gmm_activations = [tf.nn.tanh] * (len(gmm_hidden_dims) - 1) + [tf.nn.softmax]
    gmm_net = build_net(latent_dim + 2, gmm_hidden_dims, gmm_activations)

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
    trainer(od.aegmm, loss_aegmm, X_train, X_val=X_val, loss_fn_kwargs=loss_fn_kwargs,
            epochs=config['num_epochs'], batch_size=config['batch_size'],
            optimizer=optimizer, log_dir=log_dir,
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
    log_plot_instance_score(log_dir, X_test_pred, y_test, od.threshold,
                            labels=['normal', 'outlier'])
