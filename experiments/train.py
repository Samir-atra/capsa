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

    def __init__(self, x_train, y_train, x_test, y_test, dataset_name="", trainer_name="", tag_name=""):
        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        # self.images = []
        self.iter = 0
        self.BS = 16
        self.min_vloss = float('inf')

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save','{}_{}_{}_{}'.format(current_time, dataset_name, trainer_name, tag_name))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join('logs', '{}_{}_{}_{}_train'.format(current_time, dataset_name, trainer_name, tag_name))
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs', '{}_{}_{}_{}_val'.format(current_time, dataset_name, trainer_name, tag_name))
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def save_summary(self, loss, x, y, y_hat):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)
        tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)

        idx = np.random.choice(int(tf.shape(x)[0]), 9)
        tf.summary.image("x", [gallery(tf.gather(x, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image("y", [gallery(tf.gather(y, idx).numpy())], max_outputs=1, step=self.iter)
        tf.summary.image("y_hat", [gallery(tf.gather(y_hat, idx).numpy())], max_outputs=1, step=self.iter)

    def save(self, name):
        self.model.save(os.path.join(self.save_dir, "{}.h5".format(name)))

    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False).astype(np.int32)
        x_ = tf.gather(x, idx) # x[idx,...]
        y_ = tf.gather(y, idx) # y[idx,...]
        # todo-high: move this from here to preprocessing
        # elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
        #     idx = np.sort(idx)
        #     x_ = x[idx,...]
        #     y_ = y[idx,...]

        #     x_divisor = 255. if x_.dtype == np.uint8 else 1.0
        #     y_divisor = 255. if y_.dtype == np.uint8 else 1.0

        #     x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
        #     y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
        # else:
        #     print("unknown dataset type {} {}".format(type(x), type(y)))
        return x_, y_

    def on_train_batch_begin(self, batch_num, logs=None):
        # keys = list(logs.keys())
        # print("Got log keys: {}".format(keys))

        if self.iter % 100 == 0:
            print(self.iter)

            x_input_batch, y_input_batch = self.get_batch(self.x_train, self.y_train, self.BS) # todo-high: bs
            # loss, y_hat = self.run_train_step(x_input_batch, y_input_batch)
            # todo-high: Note A has below "y_hat = self.model(x, training=True)"
            y_hat = self.model.predict(x_input_batch, verbose=0)
            # todo-high: get loss from the self.model atribute
            loss = float('inf')
            with self.train_summary_writer.as_default():
                self.save_summary(loss, x_input_batch, y_input_batch, y_hat)

            x_test_batch, y_test_batch = self.get_batch(self.x_test, self.y_test, min(self.BS, self.x_test.shape[0]))
            # todo-high: Note A has below "y_hat = self.model(x, training=True)"
            y_hat = self.model.predict(x_input_batch, verbose=0)
            # todo-high: get valloss from the self.model atribute
            vloss = float('inf')
            with self.val_summary_writer.as_default():
                self.save_summary(vloss, x_test_batch, y_test_batch, y_hat)

            if vloss < self.min_vloss: # vloss.numpy()
                self.min_vloss = vloss # vloss.numpy()
                self.save(f'model_vloss_{self.iter}')

        self.iter += 1

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