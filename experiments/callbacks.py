import os
from pathlib import Path
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from scipy import stats

import matplotlib.pyplot as plt


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

class VisCallback(tf.keras.callbacks.Callback):

    # https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/trainers/deterministic.py

    def __init__(self, path, xy_train, xy_test, model_name='', dataset_name='depth', tag_name=''):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.save_dir = os.path.join(path, 'save','{}_{}_{}_{}'.format(current_time, dataset_name, model_name, tag_name))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join(path, 'logs', '{}_{}_{}_{}_train-epoch'.format(current_time, dataset_name, model_name, tag_name))
        val_log_dir = os.path.join(path, 'logs', '{}_{}_{}_{}_val-epoch'.format(current_time, dataset_name, model_name, tag_name))
        train_bs_log_dir = os.path.join(path, 'logs', '{}_{}_{}_{}_train-batch'.format(current_time, dataset_name, model_name, tag_name))

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        self.train_bs_summary_writer = tf.summary.create_file_writer(train_bs_log_dir)

        # todo-low: sample here as well? Such that if ds or array is not shuffled we account for this
        if isinstance(xy_train, tf.data.Dataset):
            (x_train, y_train), (x_test, y_test) = iter(xy_train).get_next(), iter(xy_test).get_next()
        elif isinstance(xy_train, tuple):
            # tuple of np.ndarray's
            (x_train, y_train), (x_test, y_test) = xy_train, xy_test

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.model_name = model_name
        self.iter = 0
        self.n_sample = 9
        self.min_vloss = float('inf')
        # self.loss_fn = keras.losses.MeanSquaredError()

    def save_summary(self, loss, x, y, y_hat):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)
        tf.summary.scalar('loss', loss, step=self.iter)

        idx = np.random.choice(int(tf.shape(x)[0]), 9)
        tf.summary.image('x', [gallery(tf.gather(x, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y', [gallery(tf.gather(y, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y_hat', [gallery(tf.gather(y_hat, idx).numpy())], max_outputs=1, step=self.iter)

    def save(self, name):
        save_path = os.path.join(self.save_dir, name)
        if self.model_name == 'base':
            # Functional model or Sequential model
            self.model.save(save_path)
        else:
            self.model.save_weights(save_path, save_format='tf')

    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False).astype(np.int32)
        if isinstance(x, tf.Tensor):
            x_ = tf.gather(x, idx)
            y_ = tf.gather(y, idx)
        elif isinstance(x, np.ndarray):
            idx = np.sort(idx)
            x_ = x[idx,...]
            y_ = y[idx,...]
        return x_, y_

    def on_train_batch_end(self, batch, logs=None):
        """ Note, this setup doesn't track initial loss (of the untrained model) """

        val_loss_names = [i for i in logs if i.startswith("val_")]
        loss_names = [i for i in logs if i not in val_loss_names]
        # note: asumes only one keras metric is present
        loss = logs[loss_names[0]]

        with self.train_bs_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.iter)

        # # uncomment for more granular val_loss, downside is slower training due to additionally running the model
        # if self.iter % 10 == 0:
        #     x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(self.n_sample, self.x_test.shape[0]))
        #     y_hat = self.model.predict(x_test_batch, verbose=0)
        #     vloss = self.loss_fn(y_test_batch, y_hat)
        #     with self.val_summary_writer.as_default():
        #         self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)

        self.iter += 1

    def on_epoch_end(self, epoch, logs=None):
        val_loss_names = [i for i in logs if i.startswith("val_")]
        loss_names = [i for i in logs if i not in val_loss_names]
        # note: asumes only one keras metric is present
        loss, vloss = logs[loss_names[0]], logs[val_loss_names[0]]

        x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, self.n_sample)
        # todo-med: Note A has "y_hat = self.model(x, training=True)"
        if self.model_name == 'base':
            y_hat = self.model.predict(x_input_batch, verbose=0)
        else:
            y_hat, _ = self.model.predict(x_input_batch, verbose=0)
        with self.train_summary_writer.as_default():
            self.save_summary(loss, x_input_batch, y_input_batch, y_hat)

        x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(self.n_sample, self.x_test.shape[0]))
        # todo-med: Note A has "y_hat = self.model(x, training=True)"
        if self.model_name == 'base':
            y_hat = self.model.predict(x_test_batch, verbose=0)
        elif self.model_name == 'mve':
            y_hat, sigma = self.model.predict(x_test_batch, verbose=0)
        else:
            y_hat, _ = self.model.predict(x_test_batch, verbose=0)
        with self.val_summary_writer.as_default():
            self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)
            if self.model_name == 'mve':
                tf.summary.scalar('sigma', tf.reduce_mean(sigma), step=self.iter)
        if vloss < self.min_vloss:
            self.min_vloss = vloss
            self.save(f'model_vloss-{round(vloss, 3)}_itter-{self.iter}')

def get_checkpoint_callback(checkpoints_path):
    itters_per_ep = config.N_TRAIN / config.BS
    total_itters = itters_per_ep * config.EP
    save_itters = int(total_itters // 10) # save 10 times during training
    # save_ep = int(save_itters / itters_per_ep)
    # last_saved_ep = round(save_itters * 10 // itters_per_ep)
    print('total_itters:', total_itters)
    print('save_itters:', save_itters)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        # todo-low: tf tutorial saves all checkpoints to same folder https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options
        #   - what is the "checkpoint" file which is shared for all checkpoints? It contains prefixes for both an index file as well as for one or more data files
        #   - alternatively can just save checkpoints to different folders to create a separate "checkpoint" file for every saved weights -- filepath=os.path.join(checkpoints_path, 'ep_{epoch:02d}', 'weights.tf')
        filepath=os.path.join(checkpoints_path, 'ep_{epoch:02d}_weights.tf'),
        save_weights_only=True,
        # monitor='loss', # val_loss
        save_best_only=False,
        # mode='auto',
        save_freq=save_itters,
    )

    return checkpoint_callback

class CalibrationCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds_test):
        self.ds_test = ds_test
    
    def gen_calibration_plot(self, path):
        percentiles = np.arange(41)/40
        vals = []
        mu = []
        std = []
        y_test = []
        for step, (x_test_batch, y_test_batch) in enumerate(self.ds_test):
            #outs = np.array([self.model(x_test_batch, training=True) for _ in range(20)])
            #mu_batch = np.array([i[0] for i in outs])
            #std_batch = np.array([i[1] for i in outs])
            #total_mu = tf.math.reduce_mean(mu_batch, axis=0)
            #mu.append(total_mu)
            #std.append(tf.math.reduce_mean(std_batch + mu_batch, axis=0) - total_mu**2)
            mu_batch, std_batch = self.model(x_test_batch)
            mu.append(mu_batch)
            std.append(std_batch)
            y_test.append(y_test_batch)
        mu = np.array(mu)
        std = np.array(std)
        y_test = np.array(y_test)
        y_test = y_test.reshape(-1, *y_test.shape[-3:])
        mu = mu.reshape(-1, *mu.shape[-3:])
        std = tf.sqrt(std.reshape(-1, *std.shape[-3:]))
        for percentile in tqdm(percentiles):
            ppf_for_this_percentile = stats.norm.ppf(percentile, mu, std)
            vals.append((y_test <= ppf_for_this_percentile).mean())

        plt.clf()
        plt.scatter(percentiles, vals)
        plt.scatter(percentiles, percentiles)
        plt.title(str(np.mean(abs(percentiles - vals))))
        plt.show()
        plt.savefig(path)

    def on_epoch_end(self, epoch, logs=None):
        path = f"calibration_curve_{epoch}"
        self.gen_calibration_plot(path)