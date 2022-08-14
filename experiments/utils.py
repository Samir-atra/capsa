import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import config
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from models import create

# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
def load_depth_data():
    train = h5py.File("/home/iaroslavelistratov/data/depth_train.h5", "r")
    test = h5py.File("/home/iaroslavelistratov/data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def load_apollo_data():
    test = h5py.File("/home/iaroslavelistratov/data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])

def totensor_and_normalize(x, y):
    x = tf.convert_to_tensor(x[:config.N_SAMPLES], tf.float32)
    y = tf.convert_to_tensor(y[:config.N_SAMPLES], tf.float32)
    return x / 255. , y / 255.

def get_checkpoint_callback(checkpoints_path):
    itters_per_ep = config.N_SAMPLES / config.BS
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
        save_freq=save_itters, # batches, not epochs
    )

    return checkpoint_callback

def plot_loss(history, plots_path, name='loss'):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='upper right')

    plt.savefig(f'{plots_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    # plt.show()
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

def plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path, prefix=''):
    # because if just to range(0, N_SAMPLES, BS) it's num_save_times=N_SAMPLES/BS which can be a huge number if dataset is big
    num_save_times = min(config.N_SAMPLES//config.BS, config.NUM_PLOTS)

    for i in range(0, num_save_times * config.BS, config.BS):
        x, y = x_train[i:i+6], y_train[i:i+6]
        x_ood, y_ood = x_test_ood[i:i+6], y_test_ood[i:i+6]

        pred, variance = model(x) # (6, 128, 160, 1), (6, 128, 160, 1)
        pred_ood, variance_ood = model(x_ood)

        # normalize separately
        # variance_normalized = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
        # variance_ood_normalized = (variance_ood - np.min(variance_ood)) / (np.max(variance_ood) - np.min(variance_ood))

        # normalize tougher
        cat = tf.stack([variance, variance_ood]) #(6, 128, 160, 1), (6, 128, 160, 1) = (2, 6, 128, 160, 1)
        cat_normalized = (cat - np.min(cat)) / (np.max(cat) - np.min(cat))
        variance_normalized = cat_normalized[0]
        variance_ood_normalized = cat_normalized[1]

        visualize_depth_map_uncertainty(x, y, pred, variance_normalized, vis_path, f'{prefix}{i}_iid.png')
        visualize_depth_map_uncertainty(x_ood, y_ood, pred_ood, variance_ood_normalized, vis_path, f'{prefix}{i}_ood.png')

def load_model(path, x, y):
    # path = tf.train.latest_checkpoint(checkpoints_path)

    their_model = create(x.shape[1:])
    # todo-med: make work with other wrappers
    model = MVEWrapper(their_model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR))

    # https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-574517363
    _ = model.fit(x, y, epochs=1, batch_size=config.BS, verbose=0)
    _, _ = model(x[:config.BS])

    load_status = model.load_weights(path)
    # used as validation that all variable values have been restored from the checkpoint
    load_status.assert_consumed()
    print(f'Successfully loaded weights from {path}.')
    return model