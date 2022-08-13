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

def plot_loss(history, plots_path, name='loss'):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='upper right')

    plt.savefig(f'{plots_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    # plt.show()
    plt.close()

def visualize_depth_map(x, y, pred, visualizations_path, name='map'):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(6, 3, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{visualizations_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    # plt.show()
    plt.close()

# hacky ugly way - should reuse visualize_depth_map
def visualize_depth_map_uncertainty(x, y, pred, uncertain, visualizations_path, name='map'):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(6, 4, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)
        ax[i, 3].imshow(uncertain[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{visualizations_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    # plt.show()
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