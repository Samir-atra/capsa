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
from callbacks import VisCallback, MVEVisCallback, get_checkpoint_callback
from capsa import Wrapper, MVEWrapper, EnsembleWrapper, VAEWrapper
from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss

(x_train, y_train), (x_test, y_test) = load_depth_data() # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)

# idx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False).astype(np.int32)
# # convert to np array here because cannot index h5 with unsorted idxs 
# # and sorting random indexes here will result in the original (not sorted) data
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_train = x_train[idx,...]
# y_train = y_train[idx,...]

ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN])
# ds_val = get_normalized_ds(x_train[config.N_TRAIN:], y_train[config.N_TRAIN:])
ds_test = get_normalized_ds(x_test, y_test)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

# todo-high: move this from here to preprocessing
# elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
#     x_divisor = 255. if x_.dtype == np.uint8 else 1.0
#     y_divisor = 255. if y_.dtype == np.uint8 else 1.0

#     x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
#     y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)

# todo-high: vis uncertenty

def train_base_model():
    model_name = 'base'

    path, vis_path, checkpoints_path, plots_path, logs_path = setup(model_name)
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    # checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    vis_callback = VisCallback(f'{path}/tensorboard', ds_train, ds_test, model_name)
    history = their_model.fit(ds_train, epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger], #checkpoint_callback
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(their_model, ds_train, vis_path, 'train', False)
    # visualize_depth_map(their_model, ds_val, vis_path, 'val', False)
    visualize_depth_map(their_model, ds_test, vis_path, 'test', False)
    visualize_depth_map(their_model, ds_ood, vis_path, 'ood', False)

def train_ensemble_wrapper():
    model_name = 'ensemble'

    path, vis_path, checkpoints_path, plots_path, logs_path = setup(model_name)
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    model = EnsembleWrapper(their_model, num_members=1)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[keras.losses.MeanSquaredError()],
    )

    # checkpoint_callback = get_checkpoint_callback(logs_path)
    vis_callback = VisCallback(f'{path}/tensorboard', ds_train, ds_test, model_name)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger], #checkpoint_callback
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    # visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_mve_wrapper():
    model_name = 'mve'

    path, vis_path, checkpoints_path, plots_path, logs_path = setup(model_name)
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    model = MVEWrapper(their_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    # checkpoint_callback = get_checkpoint_callback(logs_path)
    vis_callback = MVEVisCallback(f'{path}/tensorboard', ds_train, ds_test, model_name) # VisCallback
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger], #checkpoint_callback
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    # visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

# def train_vae():
#     vis_path, checkpoints_path, plots_path, logs_path = setup('vae')
#     logger = CSVLogger(f'{logs_path}/log.csv', append=True)

#     model = AutoEncoder()
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=config.LR), #1e-4
#         run_eagerly=True
#     )

#     checkpoint_callback = get_checkpoint_callback(checkpoints_path)
#     history = model.fit(ds_train, epochs=config.EP,
#         validation_data=ds_test,
#         callbacks=[logger, checkpoint_callback],
#         verbose=0,
#     )
#     plot_loss(history, plots_path)
#     visualize_depth_map(model, ds_train, vis_path, 'train')
#     visualize_depth_map(model, ds_val, vis_path, 'val')
#     visualize_depth_map(model, ds_test, vis_path, 'test')
#     visualize_depth_map(model, ds_ood, vis_path, 'ood')

# def train_vae_wrapper():
#     vis_path, checkpoints_path, plots_path, logs_path = setup('vae')
#     logger = CSVLogger(f'{logs_path}/log.csv', append=True)

#     model = VAEWrapper(
#         get_encoder(),
#         decoder = get_decoder((8, 10, 256)),
#     )
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=config.LR), #1e-4
#         loss=keras.losses.MeanSquaredError(),
#     )

#     checkpoint_callback = get_checkpoint_callback(checkpoints_path)
#     history = model.fit(ds_train, epochs=config.EP,
#         validation_data=ds_test,
#         callbacks=[logger, checkpoint_callback],
#         verbose=0,
#     )
#     plot_loss(history, plots_path)
#     visualize_depth_map(model, ds_train, vis_path, 'train')
#     visualize_depth_map(model, ds_val, vis_path, 'val')
#     visualize_depth_map(model, ds_test, vis_path, 'test')
#     visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_debug():
    model_name = 'debug'

    path, vis_path, checkpoints_path, plots_path, logs_path = setup(model_name)
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

    # checkpoint_callback = get_checkpoint_callback(logs_path)
    vis_callback = VisCallback(f'{path}/tensorboard', ds_train, ds_test, model_name)
    history = model.fit(ds_train, epochs=config.EP,
        # todo-high: note
        validation_data=ds_val,
        callbacks=[vis_callback, logger], #checkpoint_callback
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')


# train_base_model()
# train_ensemble_wrapper()
train_mve_wrapper()
# train_vae()
# train_vae_wrapper()
# train_debug()