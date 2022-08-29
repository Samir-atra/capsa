import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from debug_minimal import DebugWrappar

import config
from losses import MSE
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
    ds = ds.batch(config.BS)
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

def visualize_depth_map(model, ds, vis_path, name='map', plot_uncertainty=True, is_show=False):
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    col = 4 if plot_uncertainty else 3
    fig, ax = plt.subplots(6, col, figsize=(20, 20))

    x, y = iter(ds).get_next()
    if plot_uncertainty:
        pred, uncertainty = model(x, training=True)
    else:
        pred = model(x, training=True)

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)
        if plot_uncertainty:
            ax[i, 3].imshow(uncertainty[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{vis_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    if is_show:
        plt.show()
    else:
        plt.close()

# todo-low: reduce code duplication, combine with 'visualize_depth_map'
def visualize_vae_depth_map(model, ds, vis_path, name='map', plot_uncertainty=True, is_show=False):
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    col = 3 if plot_uncertainty else 2
    fig, ax = plt.subplots(6, col, figsize=(16, 20))

    x, y = iter(ds).get_next()
    if plot_uncertainty:
        pred, uncertainty = model(x, training=True)
    else:
        pred = model(x, training=True)

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(tf.clip_by_value(pred[i], clip_value_min=0, clip_value_max=1))
        if plot_uncertainty:
            ax[i, 2].imshow(uncertainty[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{vis_path}/{name}.pdf', bbox_inches='tight', format='pdf')
    if is_show:
        plt.show()
    else:
        plt.close()

# plt.savefig(f'{vis_path}/{name}.pdf', bbox_inches='tight', format='pdf')
# plt.close()
plt.show()

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

def select_best_checkpoint(model_path):
    checkpoints_path = os.path.join(model_path, 'checkpoints')
    model_name = model_path.split('/')[-2]

    l = sorted(glob.glob(os.path.join(checkpoints_path, '*.tf*')))
    # l = [i.split('/')[-1].split('.')[0] for i in l]
    # >>> ['ep_128_weights', 'ep_77_weights', 'ep_103_weights']

    # -1.702vloss_100iter.tf.data-00000-of-00001
    # l = [i.split('/')[-1].split('.')[1] for i in l]
    # >>> ['190vloss_900iter', '317vloss_6700iter', '386vloss_1900iter']
    l_split = [float(i.split('/')[-1].split('vloss')[0]) for i in l]
    # >>> ['-0.155', '-1.183', '0.951']

    # weights_names = list(set(l_split))
    # for i, name in enumerate(sorted(weights_names)):
    #     path = f'{checkpoints_path}/{name}.tf'
    #     model = load_model(path, ds_train)

    # select lowest loss
    min_loss = min(l_split)
    # represent same model
    model_paths = [i for i in l if str(min_loss) in i]
    path = model_paths[0].split('.tf')[0]
    return f'{path}.tf', model_name

def load_model(path, model_name, ds):
    # path = tf.train.latest_checkpoint(checkpoints_path)

    # d = {
    #     'base' : unet,
    #     'mve' : MVEWrapper,
    #     'vae' : AutoEncoder,
    #     # 'ensemble' : EnsembleWrapper,
    # }

    if model_name in ['base', 'notebook_base']:
        model = unet()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    elif model_name in ['mve', 'notebook_mve']:
        their_model = unet()
        model = MVEWrapper(their_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    elif model_name in ['ae_model', 'notebook_ae_model']:
        model = AutoEncoder()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    elif model_name == 'ensemble':
        their_model = unet()
        model = EnsembleWrapper(their_model, num_members=1)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    # https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-574517363
    _ = model.fit(ds, epochs=1, verbose=0)
    load_status = model.load_weights(path)

    # base mode tires to load optimizer as well, so load_status gives error
    if model_name not in ['base', 'notebook_base']:
        # used as validation that all variable values have been restored from the checkpoint
        load_status.assert_consumed()
    print(f'Successfully loaded weights from {path}.')
    return model

def notebook_select_gpu(idx):
    # # https://www.tensorflow.org/guide/gpu#using_a_single_gpu_on_a_multi-gpu_system
    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[idx], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)