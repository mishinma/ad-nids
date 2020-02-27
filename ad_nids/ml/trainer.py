import os
import logging

import numpy as np
import tensorflow as tf
from typing import Tuple


class NANLossError(ValueError):
    pass


def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            train_gen,
            X_val: np.ndarray = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            epoch_size: int = None,
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

    do_validation = X_val is not None

    if log_dir is not None:
        logging.info(f'\n >>> tensorboard --host 0.0.0.0 --port 4000 --logdir {log_dir}\n')
        train_log_dir = os.path.join(log_dir, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if do_validation:
            val_log_dir = os.path.join(log_dir, 'val')
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    if epoch_size is not None:
        n_minibatch = int(epoch_size)
    else:
        n_minibatch = train_gen.n_minibatch

    # iterate over epochs
    for epoch in range(epochs):

        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)

        if lr_schedule is not None:
            scheduled_lr = lr_schedule[epoch]
            tf.keras.backend.set_value(optimizer.lr, scheduled_lr)

        # iterate over the batches of the dataset
        for step in range(n_minibatch):

            X_train_batch = next(train_gen)

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


class DataGenerator:

    def __init__(self, data, batch_size, buffer_size=1024):
        self.data = tf.data.Dataset.from_tensor_slices(data)
        self.data = self.data.batch(batch_size).shuffle(buffer_size=buffer_size)
        self.n_minibatches = int(np.ceil(data.shape[0] / batch_size))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._iter_data = iter(self.data)
        self._cnt = 0

    def __next__(self):
        if self._cnt >= self.n_minibatches:
            logging.info('Shuffling the data')
            self.data = self.data.shuffle(buffer_size=self.buffer_size)
            self._iter_data = iter(self.data)
            self._cnt = 0
        self._cnt += 1
        return next(self._iter_data)

    def __iter__(self):
        return self
