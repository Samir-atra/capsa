import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from debug_minimal import DebugWrappar

import config
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from models import unet

# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
def load_depth_data():
    train = h5py.File('/data/capsa/data/depth_train.h5', 'r')
    test = h5py.File('/data/capsa/data/depth_test.h5', 'r')
    return (train['image'], train['depth']), (test['image'], test['depth'])

def load_apollo_data():
    test = h5py.File('/data/capsa/data/apolloscape_test.h5', 'r')
    return (None, None), (test['image'], test['depth'])

def totensor_and_normalize(x, y):
    x = tf.convert_to_tensor(x, tf.float32)
    y = tf.convert_to_tensor(y, tf.float32)
    return x / 255., y / 255.

def _get_ds(x, y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(x.shape[0])
    ds = ds.batch(config.BS, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_normalized_ds(x, y, shuffle=True):
    x, y = totensor_and_normalize(x, y)
    return _get_ds(x, y, shuffle)

def plot_loss(history, plots_path, name='loss'):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='upper right')

    plt.savefig(f'{plots_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    plt.close()

def visualize_depth_map(model, ds, vis_path, name='map', plot_uncertainty=True):
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    col = 4 if plot_uncertainty else 3
    fig, ax = plt.subplots(6, col, figsize=(50, 50))

    x, y = iter(ds).get_next()
    if plot_uncertainty:
        pred, uncertainty = model(x)
    else:
        pred = model(x)

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)
        if plot_uncertainty:
            ax[i, 3].imshow(uncertainty[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{vis_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    plt.close()

# # todo-med: plot_multiple
# # todo-med: normalize tougher
# def plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path):

#     for i in range(0, num_save_times * config.BS, config.BS):

#         pred, variance = model(x) # (6, 128, 160, 1), (6, 128, 160, 1)
#         pred_ood, variance_ood = model(x_ood)

#         # normalize separately
#         # variance_normalized = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
#         # variance_ood_normalized = (variance_ood - np.min(variance_ood)) / (np.max(variance_ood) - np.min(variance_ood))

#         # normalize tougher
#         cat = tf.stack([variance, variance_ood]) #(6, 128, 160, 1), (6, 128, 160, 1) = (2, 6, 128, 160, 1)
#         cat_normalized = (cat - np.min(cat)) / (np.max(cat) - np.min(cat))
#         variance_normalized = cat_normalized[0]
#         variance_ood_normalized = cat_normalized[1]

#         visualize_depth_map_uncertainty(model, ds_train, vis_path, f'{i}_iid.png')
#         visualize_depth_map_uncertainty(model, ds_ood, vis_path, f'{i}_ood.png')

def load_model(path, ds):
    # path = tf.train.latest_checkpoint(checkpoints_path)



    their_model = unet()
    # todo-med: make work with other wrappers
    model = MVEWrapper(their_model) #EnsembleWrapper(their_model, num_members=1) 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )


    '''
    model = unet()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=keras.losses.MeanSquaredError(),
    )
    '''


    # https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-574517363
    _ = model.fit(ds, epochs=1)
    x, y = iter(ds).get_next()
    # _, _ = model(x[:config.BS])
    _ = model(x[:config.BS])

    load_status = model.load_weights(path)
    # used as validation that all variable values have been restored from the checkpoint
    load_status.assert_consumed()
    print(f'Successfully loaded weights from {path}.')
    return model