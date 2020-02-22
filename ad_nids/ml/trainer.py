import os

import numpy as np
import tensorflow as tf
from typing import Tuple


class NANLossError(ValueError):
    pass


def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            X_val: np.ndarray = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            epoch_size: int = None,
            batch_size: int = 64,
            buffer_size: int = 1024,
            verbose: bool = True,
            log_metric:  Tuple[str, "tf.keras.metrics"] = None,
            log_dir: str = None,
            checkpoint: bool = True,
            checkpoint_freq: int = 10,
            lr_schedule: list = None,
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
    checkpoint
    lr_schedule
    callbacks
        Callbacks used during training.
    """

    train_data = tf.data.Dataset.from_tensor_slices(X_train)
    train_data = train_data.batch(batch_size)

    if epoch_size is None:
        n_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
    else:
        n_minibatch = epoch_size

    do_validation = X_val is not None

    if log_dir is not None:
        train_log_dir = os.path.join(log_dir, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if do_validation:
            val_log_dir = os.path.join(log_dir, 'val')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # iterate over epochs
    for epoch in range(epochs):

        train_data = train_data.shuffle(buffer_size=buffer_size)

        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)

        if lr_schedule is not None:
            scheduled_lr = lr_schedule[epoch]
            tf.keras.backend.set_value(optimizer.lr, scheduled_lr)

        # iterate over the batches of the dataset
        for step, X_train_batch in enumerate(train_data):

            if step > n_minibatch:
                return

            with tf.GradientTape() as tape:
                preds = model(X_train_batch)

                # compute loss
                if tf.is_tensor(preds):
                    args = [X_train_batch, preds]
                else:
                    args = [X_train_batch] + list(preds)

                if loss_fn_kwargs:
                    loss = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss = loss_fn(*args)

                if model.losses:  # additional model losses
                    loss += sum(model.losses)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss = loss.numpy()
            if np.isnan(loss):
                raise NANLossError

            if verbose:
                if loss.shape != (batch_size,) and loss.shape:
                    add_mean = np.ones((batch_size - loss.shape[0],)) * loss.mean()
                    loss = np.r_[loss, add_mean]

                pbar_values = [('loss', loss)]
                if log_metric is not None:
                    log_metric[1](X_train_batch, preds)
                    pbar_values.append((log_metric[0], log_metric[1].result().numpy()))
                pbar.add(1, values=pbar_values)

            if log_dir:
                abs_step = epoch * n_minibatch + step
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=abs_step)

        if do_validation:

            val_preds = model(X_val)

            if tf.is_tensor(val_preds):
                args = [X_val, val_preds]
            else:
                args = [X_val] + list(val_preds)

            if loss_fn_kwargs:
                val_loss = loss_fn(*args, **loss_fn_kwargs)
            else:
                val_loss = loss_fn(*args)

            if model.losses:  # additional model losses
                val_loss += sum(model.losses)

            abs_step = (epoch + 1) * n_minibatch
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step=abs_step)

        if checkpoint and epoch % checkpoint_freq == 0:
            checkpoint_dir = os.path.join(log_dir, 'weights')
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            abs_step = (epoch + 1) * n_minibatch
            weights_fname = f'weights_epoch_{epoch}_step_{abs_step}.ckpt'
            model.save_weights(os.path.join(checkpoint_dir, weights_fname))
