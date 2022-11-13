import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from scipy import stats

import matplotlib.pyplot as plt


import config

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

class BaseCallback(tf.keras.callbacks.Callback):

    # https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/trainers/deterministic.py

    def __init__(self, checkpoints_path, logs_path, model_name, xy_train, xy_test):

        self.save_dir = checkpoints_path

        train_log_dir = os.path.join(logs_path, 'train-epoch')
        val_log_dir = os.path.join(logs_path, 'val-epoch')
        train_bs_log_dir = os.path.join(logs_path, 'train-batch')

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
        self.min_vloss = float('inf')

    def save_summary(self, loss, x, y, y_hat):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)

        for k, v in loss.items():
            # both plot val and train losses on the same card in tfboard -- val_mse_loss -> mse_loss
            k = k if 'val_' not in k else k.replace('val_', '')
            tf.summary.scalar(k, v, step=self.iter)

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

class VisCallback(BaseCallback):

    def __init__(self, checkpoints_path, logs_path, model_name, xy_train, xy_test):
        super().__init__(checkpoints_path, logs_path, model_name, xy_train, xy_test)

    def on_train_batch_end(self, batch, logs=None):
        """ Note, this setup doesn't track initial loss (of the untrained model) """

        val_loss_names = [i for i in logs if i.startswith("val_")]
        loss_names = [i for i in logs if i not in val_loss_names]
        loss = {k: v for k, v in logs.items() if k in loss_names}
        with self.train_bs_summary_writer.as_default():
            for k, v in loss.items():
                tf.summary.scalar(k, v, step=self.iter)

        self.iter += 1

    def on_epoch_end(self, epoch, logs=None):
        val_loss_names = [i for i in logs if i.startswith("val_")]
        loss_names = [i for i in logs if i not in val_loss_names]
        # dicts
        loss = {k: v for k, v in logs.items() if k in loss_names}
        vloss = {k: v for k, v in logs.items() if k in val_loss_names}

        x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, config.BS)
        if self.model_name == 'base':
            y_hat = self.model(x_input_batch, training=True)
        elif self.model_name == 'vae_model':
            y_hat, _, _ = self.model(x_input_batch, training=True)
        else:
            y_hat, _ = self.model(x_input_batch, training=True)
        with self.train_summary_writer.as_default():
            self.save_summary(loss, x_input_batch, y_input_batch, y_hat)

        x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(100, self.x_test.shape[0]))
        if self.model_name == 'base':
            y_hat = self.model(x_test_batch, training=True)
        elif self.model_name == 'vae_model':
            y_hat, _, _ = self.model(x_test_batch, training=True)
        else:
            y_hat, _ = self.model(x_test_batch, training=True)
        with self.val_summary_writer.as_default():
            self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)

        total_val_loss = vloss['val_loss']
        if total_val_loss < self.min_vloss:
            self.min_vloss = vloss.numpy() if isinstance(total_val_loss, np.ndarray) else total_val_loss
            self.save("{:0.3f}vloss_{}iter.tf".format(self.min_vloss, self.iter))

class MVEVisCallback(BaseCallback):

    # more granular val_loss downside is slower training due to additionally running the model
    # (every n steps on_train_batch_end, instead of on on_epoch_end)

    def __init__(self, checkpoints_path, logs_path, model_name, xy_train, xy_test):
        super().__init__(checkpoints_path, logs_path, model_name, xy_train, xy_test)

    def save_summary(self, loss, x, y, y_hat, var):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)
        tf.summary.scalar('loss', loss, step=self.iter)
        tf.summary.scalar('variance', tf.reduce_mean(var), step=self.iter)

        idx = np.random.choice(int(tf.shape(x)[0]), 9)
        tf.summary.image('x', [gallery(tf.gather(x, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y', [gallery(tf.gather(y, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y_hat', [gallery(tf.gather(y_hat, idx).numpy())], max_outputs=1, step=self.iter)

    def on_train_batch_end(self, batch, logs=None):
        loss_names = [i for i in logs]
        # note: asumes only one keras metric is present
        loss = logs[loss_names[0]]

        with self.train_bs_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.iter)

        if self.iter % 100 == 0:
            x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, config.BS)
            loss, y_hat, var = self.model.loss_fn(x_input_batch, y_input_batch, return_var=True)
            with self.train_summary_writer.as_default():
                self.save_summary(loss, x_input_batch, y_input_batch, y_hat, var)

            x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(100, self.x_test.shape[0]))
            vloss, y_hat, var = self.model.loss_fn(x_test_batch, y_test_batch, return_var=True)
            with self.val_summary_writer.as_default():
                self.save_summary(vloss, x_test_batch, y_test_batch, y_hat, var)

            total_val_loss = vloss['val_loss']
            if total_val_loss < self.min_vloss:
                self.min_vloss = total_val_loss.numpy()
                self.save("{:0.3f}vloss_{}iter.tf".format(self.min_vloss, self.iter))

        self.iter += 1

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