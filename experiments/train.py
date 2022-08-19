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
ds_val = get_normalized_ds(x_train[-config.N_VAL:], y_train[-config.N_VAL:])
# ds_test = get_normalized_ds(x_test, y_test)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

def train_base_model():
    vis_path, checkpoints_path, plots_path, logs_path = setup('base')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = their_model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(their_model, ds_train, vis_path, 'iid', False)
    visualize_depth_map(their_model, ds_ood, vis_path, 'ood', False)

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
    visualize_depth_map(model, ds_train, vis_path, 'iid')
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
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'iid')
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
    visualize_depth_map(model, ds_train, vis_path, 'iid')
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
    visualize_depth_map(model, ds_train, vis_path, 'iid')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')



def train_debug():
    vis_path, checkpoints_path, plots_path, logs_path = setup('debug')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = unet()
    model = DebugWrappar(their_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(ds_train, epochs=config.EP,
        validation_data=ds_val,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'iid')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')


# train_base_model()
# train_ensemble_wrapper()
# train_mve_wrapper()
# train_vae()
# train_vae_wrapper()
train_debug()