import os

import numpy as np
import tensorflow as tf
from typing import Tuple


def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            y_train: np.ndarray = None,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            batch_size: int = 64,
            buffer_size: int = 1024,
            verbose: bool = True,
            log_metric:  Tuple[str, "tf.keras.metrics"] = None,
            log_dir: str = None,
            callbacks: tf.keras.callbacks = None) -> None:  # TODO: incorporate callbacks + LR schedulers
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    X_train
        Training batch.
    y_train
        Training labels.
    X_val
        Validation batch.
    y_val
        Validation labels.
    optimizer
        Optimizer used for training.
    loss_fn_kwargs
        Kwargs for loss function.
    epochs
        Number of training epochs.
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    log_metric
        Additional metrics whose progress will be displayed if verbose equals True.
    log_dir
        Directory to store logs
    callbacks
        Callbacks used during training.
    """
    # create dataset
    if y_train is None:  # unsupervised model
        train_data = X_train
    else:
        train_data = (X_train, y_train)

    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size)
    n_minibatch = int(np.ceil(X_train.shape[0] / batch_size))

    # validation dataset
    do_validation = 
    if X_val is not None:
        if y_val is None:
            val_data = X_val
        else:
            val_data = (X_val, y_val)
    else:
        val_data = None

    if log_dir is not None:
        train_log_dir = os.path.join(log_dir, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        if val_data is not None:
            val_log_dir = os.path.join(log_dir, 'val')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        else:
            val_summary_writer = None

    # iterate over epochs
    for epoch in range(epochs):
        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)

        # iterate over the batches of the dataset
        for step, train_batch in enumerate(train_data):

            if y_train is None:
                X_train_batch = train_batch
            else:
                X_train_batch, y_train_batch = train_batch

            with tf.GradientTape() as tape:
                preds = model(X_train_batch)

                if y_train is None:
                    ground_truth = X_train_batch
                else:
                    ground_truth = y_train_batch

                # compute loss
                if tf.is_tensor(preds):
                    args = [ground_truth, preds]
                else:
                    args = [ground_truth] + list(preds)

                if loss_fn_kwargs:
                    loss = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss = loss_fn(*args)

                if model.losses:  # additional model losses
                    loss += sum(model.losses)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if verbose:
                loss_val = loss.numpy()
                if loss_val.shape != (batch_size,) and loss_val.shape:
                    add_mean = np.ones((batch_size - loss_val.shape[0],)) * loss_val.mean()
                    loss_val = np.r_[loss_val, add_mean]

                pbar_values = [('loss', loss_val)]
                if log_metric is not None:
                    log_metric[1](ground_truth, preds)
                    pbar_values.append((log_metric[0], log_metric[1].result().numpy()))
                pbar.add(1, values=pbar_values)

            if log_dir:
                loss_val = loss.numpy()
                assert not loss_val.shape
                abs_step = epoch*n_minibatch + step
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_val, step=abs_step)
