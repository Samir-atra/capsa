import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

import config
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from models import create
from run_utils import setup
from utils import load_depth_data, load_apollo_data, totensor_and_normalize, \
    visualize_depth_map, visualize_depth_map_uncertainty, plot_multiple, \
    plot_loss, get_checkpoint_callback

(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train, y_train = totensor_and_normalize(x_train, y_train)

_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood, y_test_ood = totensor_and_normalize(x_test_ood, y_test_ood)

def train_base_model():
    vis_path, checkpoints_path, plots_path, logs_path = setup('base')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = their_model.fit(x_train, y_train, epochs=config.EP, batch_size=config.BS,
        validation_split=0.2,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )
    plot_loss(history, plots_path)

    pred = their_model(x_train)
    visualize_depth_map(x_train, y_train, pred, vis_path)

def train_ensemble_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('ensemble')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    model = EnsembleWrapper(their_model, num_members=2)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[keras.losses.MeanSquaredError()],
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(x_train, y_train, epochs=config.EP, batch_size=config.BS,
        validation_split=0.2,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, vis_path, 'iid')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')

def train_mve_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('mve')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    model = MVEWrapper(their_model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR))

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(x_train, y_train, epochs=config.EP, batch_size=config.BS,
        validation_split=0.2,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
    )

    plot_loss(history, plots_path)
    plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path)


# train_base_model()
# train_ensemble_wrapper()
train_mve_wrapper()