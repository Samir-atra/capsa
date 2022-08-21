import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

# rm
from debug_minimal import DebugWrappar

import config
from models import unet, AutoEncoder, get_encoder, get_decoder
from run_utils import setup
from capsa import Wrapper, MVEWrapper, EnsembleWrapper, VAEWrapper
from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss, get_checkpoint_callback

(x_train, y_train), (x_test, y_test) = load_depth_data() # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)
ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN])
# ds_val = get_normalized_ds(x_train[-config.N_VAL:], y_train[-config.N_VAL:])
ds_test = get_normalized_ds(x_test, y_test)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

# todo-high: move this from here to preprocessing
# elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
#     x_divisor = 255. if x_.dtype == np.uint8 else 1.0
#     y_divisor = 255. if y_.dtype == np.uint8 else 1.0

#     x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
#     y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
# else:
#     print("unknown dataset type {} {}".format(type(x), type(y)))

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

    def __init__(self, x_train, y_train, x_test, y_test, n_sample=16, dataset_name="", trainer_name="", tag_name=""):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.iter = 0
        self.n_sample = n_sample
        self.min_vloss = float('inf')
        self.loss_fn = keras.losses.MeanSquaredError()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save','{}_{}_{}_{}'.format(current_time, dataset_name, trainer_name, tag_name))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join('logs', '{}_{}_{}_{}_train'.format(current_time, dataset_name, trainer_name, tag_name))
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs', '{}_{}_{}_{}_val'.format(current_time, dataset_name, trainer_name, tag_name))
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def save_summary(self, loss, x, y, y_hat):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)
        tf.summary.scalar('loss', loss, step=self.iter)

        idx = np.random.choice(int(tf.shape(x)[0]), 9)
        tf.summary.image('x', [gallery(tf.gather(x, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y', [gallery(tf.gather(y, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image('y_hat', [gallery(tf.gather(y_hat, idx).numpy())], max_outputs=1, step=self.iter)

    def save(self, name):
        self.model.save(os.path.join(self.save_dir, "{}.h5".format(name)))

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

    # def on_train_batch_begin(self, batch, logs=None):

    #     if self.iter % 10 == 0:

    #         x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, self.n_sample)
    #         # todo-high: Note A has "y_hat = self.model(x, training=True)"
    #         y_hat = self.model.predict(x_input_batch, verbose=0)
    #         loss = self.loss_fn(y_input_batch, y_hat)
    #         with self.train_summary_writer.as_default():
    #             self.save_summary(loss, x_input_batch, y_input_batch, y_hat)

    #         x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(self.n_sample, self.x_test.shape[0]))
    #         # todo-high: Note A has "y_hat = self.model(x, training=True)"
    #         y_hat = self.model.predict(x_test_batch, verbose=0)
    #         vloss = self.loss_fn(y_test_batch, y_hat)
    #         with self.val_summary_writer.as_default():
    #             self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)

    #         if vloss < self.min_vloss: # vloss.numpy()
    #             self.min_vloss = vloss # vloss.numpy()
    #             self.save(f'model_vloss_{self.iter}')

    #     self.iter += 1

    def on_train_batch_end(self, batch, logs=None):
        """ Note, this setup doesn't track initial loss (of the untrained model)
        """
        loss = logs['loss']
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', loss, step=self.iter)

        self.iter += 1

    def on_epoch_end(self, epoch, logs=None):
        """ use on_epoch_end because both losses are available only here
        Althoguh another solution is to put all the code below
        into on_train_batch_end and just calculate loss every n-th batch
        (this will give a more detailed curves) -- but a very big drawback
        is that to calcuate the loss you will need to run the model.
        So that setup will slow down the training considerably
        """
        loss, vloss = logs['loss'], logs['val_loss']

        x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, self.n_sample)
        # todo-high: Note A has "y_hat = self.model(x, training=True)"
        y_hat = self.model.predict(x_input_batch, verbose=0)
        with self.train_summary_writer.as_default():
            self.save_summary(loss, x_input_batch, y_input_batch, y_hat)

        x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(self.n_sample, self.x_test.shape[0]))
        # todo-high: Note A has "y_hat = self.model(x, training=True)"
        y_hat = self.model.predict(x_test_batch, verbose=0)
        with self.val_summary_writer.as_default():
            self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)

        if vloss < self.min_vloss: # vloss.numpy()
            self.min_vloss = vloss # vloss.numpy()
            self.save(f'model_vloss_{self.iter}')

import sys
from pathlib import Path
import time
import datetime

def train_base_model():
    vis_path, checkpoints_path, plots_path, logs_path = setup('base')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    # todo-high: support ds inside the callbacck
    x_train_, y_train_ = iter(ds_train).get_next()
    x_test_, y_test_ = iter(ds_test).get_next()

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = their_model.fit(ds_train, epochs=config.EP,
        validation_data=ds_test,
        verbose=0,
        # VisCallback(x_train[:64], y_train[:64], x_test[:64], y_test[:64])
        callbacks=[VisCallback(x_train_, y_train_, x_test_, y_test_)],
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train', False)
    # visualize_depth_map(model, ds_val, vis_path, 'val', False)
    visualize_depth_map(model, ds_test, vis_path, 'test', False)
    visualize_depth_map(model, ds_ood, vis_path, 'ood', False)

def train_ensemble_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('ensemble')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    model = EnsembleWrapper(their_model, num_members=1)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[keras.losses.MeanSquaredError()],
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_mve_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('mve')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    model = MVEWrapper(their_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_test,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    # visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_vae():
    vis_path, checkpoints_path, plots_path, logs_path = setup('vae')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    model = AutoEncoder()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR), #1e-4
        run_eagerly=True
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_vae_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('vae')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    model = VAEWrapper(
        get_encoder(),
        decoder = get_decoder((8, 10, 256)),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR), #1e-4
        loss=keras.losses.MeanSquaredError(),
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')



def train_debug():
    vis_path, checkpoints_path, plots_path, logs_path = setup('debug')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    # their_model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=config.LR),
    #     loss=keras.losses.MeanSquaredError(),
    # )

    model = DebugWrappar(their_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError()
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')


train_base_model()
# train_ensemble_wrapper()
# train_mve_wrapper()
# train_vae()
# train_vae_wrapper()
# train_debug()