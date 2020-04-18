
import json
import logging

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from alibi_detect.utils.saving import load_detector

from ad_nids.utils.misc import jsonify, predict_batch
from ad_nids.utils.logging import log_preds


def run_eval(log_dir, eval_batch, results_name='test', batch_size=1000000):

    X_test, y_test = eval_batch.data, eval_batch.target

    od = load_detector(str(log_dir / 'detector'))

    # Compute anomaly scores for test
    logging.info('Computing test anomaly scores...')
    se = timer()
    if X_test.shape[0] <= 1000000:
        X_test_pred = od.predict(X_test)
    else:
        X_test_pred = predict_batch(od, X_test, batch_size=batch_size)
    y_test_pred = X_test_pred['data']['is_outlier']
    time_score_test = timer() - se
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_prf1s = precision_recall_fscore_support(y_test, y_test_pred, average='binary')

    # Save results
    log_preds(log_dir, results_name, X_test_pred, y_test)
    eval_results = jsonify({
        'test_prf1s': test_prf1s,
        'test_cm': test_cm,
        'time_score_test': time_score_test,
    })

    with open(log_dir / results_name / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f)
