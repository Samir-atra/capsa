import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

import config
from capsa import Wrapper, MVEWrapper, EnsembleWrapper, DropoutWrapper
from models import create
from run_utils import setup
from utils import load_depth_data, load_apollo_data, totensor_and_normalize, \
    visualize_depth_map, visualize_depth_map_uncertainty, plot_multiple, \
    plot_loss, get_checkpoint_callback, load_depth_as_dataset, Augment

train_ds, test_ds, inp_shape = load_depth_as_dataset()
train_batches = (
    train_ds
    .cache()
    .batch(config.BS)
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_ds.batch(config.BS)
_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood, y_test_ood = totensor_and_normalize(x_test_ood, y_test_ood)

def display(display_list, name):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig("sample" + str(name))

def plot_images():
    i = 0
    for images, masks in train_batches.take(10):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask], "train" + str(i))
        i += 1
    i = 0
    for images, masks in test_batches.take(10):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask], "test" + str(i))
        i += 1

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

    their_model = create(inp_shape)
    model = EnsembleWrapper(their_model, num_members=2)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[keras.losses.MeanSquaredError()],
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(train_batches, epochs=config.EP, batch_size=config.BS,
        validation_data=test_batches,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)

    def get_ensemble_uncertainty(x_train, model):
        outs = model(x_train)
        preds = np.concatenate((outs[0][np.newaxis, ...], outs[1][np.newaxis, ...]), 0)
        pred, epistemic = np.mean(preds, 0), np.std(preds, 0)
        return pred, epistemic

    pred, epistemic = get_ensemble_uncertainty(x_train, model)
    visualize_depth_map_uncertainty(x_train, y_train, pred, epistemic, vis_path, 'iid.png')

    pred_ood, epistemic_ood = get_ensemble_uncertainty(x_test_ood, model)
    visualize_depth_map_uncertainty(x_test_ood, y_test_ood, pred_ood, epistemic_ood, vis_path, 'ood.png')

def train_mve_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('mve')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(inp_shape, drop_prob=0.1)
    model = MVEWrapper(their_model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR), loss=tf.keras.losses.MeanSquaredError())

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(train_batches, epochs=config.EP, batch_size=config.BS,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
        validation_data=test_batches
    )

    plot_loss(history, plots_path)
    # plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path)

def train_dropout_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('dropout')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    model = create(inp_shape, drop_prob=0.1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR), loss=tf.keras.losses.MeanSquaredError())

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(train_batches, epochs=config.EP, batch_size=config.BS,
        callbacks=[logger, checkpoint_callback],
        verbose=0,
        validation_data=test_batches
    )

    plot_loss(history, plots_path)

def train_ensemble_mve_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup('ensemble_mve')
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(inp_shape)
    model = EnsembleWrapper(their_model, num_members=5, metric_wrapper=MVEWrapper)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[keras.losses.MeanSquaredError()],
    )

    checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    history = model.fit(train_batches, epochs=config.EP, batch_size=config.BS,
        validation_data=test_batches,
        callbacks=[logger, checkpoint_callback], 
        verbose=0,
    )
    plot_loss(history, plots_path)
#train_base_model()
# train_ensemble_wrapper()
# train_mve_wrapper()
#train_dropout_wrapper()
train_ensemble_mve_wrapper()