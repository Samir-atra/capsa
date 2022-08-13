N_SAMPLES = 64 # 256
BS = 32 # 8
EP = 32 # 256
LR = 5e-5 # 5e-5


import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
# import pandas as pd

from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from models import create
from run_utils import setup
from utils import load_depth_data, load_apollo_data, visualize_depth_map, visualize_depth_map_uncertainty, plot_loss

# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

### https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train = tf.convert_to_tensor(x_train[:N_SAMPLES], tf.float32) #16, 256, 1024
y_train = tf.convert_to_tensor(y_train[:N_SAMPLES], tf.float32) #16, 256, 1024
x_train /= 255.
y_train /= 255.

_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood = tf.convert_to_tensor(x_test_ood[:N_SAMPLES], tf.float32) #16, 256, 1024
y_test_ood = tf.convert_to_tensor(y_test_ood[:N_SAMPLES], tf.float32) #16, 256, 1024
x_test_ood /= 255.
y_test_ood /= 255.


def train_base_model():
    visualizations_path, checkpoints_path, plots_path, logs_path = setup()
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    # trainer = trainer_obj(model, opts, args.learning_rate)

    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=keras.losses.MeanSquaredError(),
    )

    history = their_model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0) # 10000 epochs
    plot_loss(history, plots_path)

    # todo-high: inference on val set
    pred = their_model(x_train)
    visualize_depth_map(x_train, y_train, pred, visualizations_path)

def train_ensemble_wrapper():
    visualizations_path, checkpoints_path, plots_path, logs_path = setup()
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    # trainer = trainer_obj(model, opts, args.learning_rate)

    model = EnsembleWrapper(their_model, num_members=2)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=5e-5)],
        loss=[keras.losses.MeanSquaredError()],
        # metrics=[[keras.metrics.CosineSimilarity(name='cos')]],
    )

    history = model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0) # 10000 epochs
    plot_loss(history, plots_path)

    def get_ensemble_uncertainty(x_train, model):
        outs = model(x_train)
        preds = np.concatenate((outs[0][np.newaxis, ...], outs[1][np.newaxis, ...]), 0)
        pred, epistemic = np.mean(preds, 0), np.std(preds, 0)
        return pred, epistemic

    pred, epistemic = get_ensemble_uncertainty(x_train, model)
    visualize_depth_map_uncertainty(x_train, y_train, pred, epistemic, visualizations_path, 'iid.png')

    pred_ood, epistemic_ood = get_ensemble_uncertainty(x_test_ood, model)
    visualize_depth_map_uncertainty(x_test_ood, y_test_ood, pred_ood, epistemic_ood, visualizations_path, 'ood.png')

# train_base_model()
# train_ensemble_wrapper()



visualizations_path, checkpoints_path, plots_path, logs_path = setup()
logger = CSVLogger(f'{logs_path}/log.csv', append=True)

their_model = create(x_train.shape[1:])
# trainer = trainer_obj(model, opts, args.learning_rate)

model = MVEWrapper(their_model)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    # loss=keras.losses.MeanSquaredError(),
)

itters_per_ep = N_SAMPLES / BS
total_itters = itters_per_ep * EP
save_itters = int(total_itters // 10) # save 10 times during training
# save_ep = int(save_itters / itters_per_ep)
# last_saved_ep = round(save_itters * 10 // itters_per_ep)
print('total_itters:', total_itters)
print('save_itters:', save_itters)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
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

history = model.fit(x_train, y_train, epochs=EP, batch_size=BS, callbacks=[logger, model_checkpoint_callback], verbose=0) # 10000 epochs
plot_loss(history, plots_path)

### need this to load weights
del model
their_model = create(x_train.shape[1:])
model = MVEWrapper(their_model)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR))
# https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-574517363
_ = model.fit(x_train, y_train, epochs=1, batch_size=BS, verbose=0)
_, _ = model(x_train)
latest = tf.train.latest_checkpoint(checkpoints_path)
load_status = model.load_weights(latest)
# used as validation that all variable values have been restored from the checkpoint
load_status.assert_consumed()
print(f'successfully loaded weights from {latest}')


### plot
# because if just to range(0, N_SAMPLES, BS) it's num_save_times=N_SAMPLES/BS which can be a huge number if dataset is big
num_save_times = min(N_SAMPLES//BS, 10)

for i in range(0, num_save_times*BS, BS):
    # print(i, i+6)
    x, y = x_train[i:i+6], y_train[i:i+6]
    x_ood, y_ood = x_test_ood[i:i+6], y_test_ood[i:i+6]

    pred, variance = model(x) # (6, 128, 160, 1) and (6, 128, 160, 1)
    pred_ood, variance_ood = model(x_ood)

    # normalize separately
    # variance_normalized = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
    # variance_ood_normalized = (variance_ood - np.min(variance_ood)) / (np.max(variance_ood) - np.min(variance_ood))

    # normalize tougher
    cat = tf.stack([variance, variance_ood]) #(6, 128, 160, 1), (6, 128, 160, 1) = (2, 6, 128, 160, 1)
    cat_normalized = (cat - np.min(cat)) / (np.max(cat) - np.min(cat))
    variance_normalized = cat_normalized[0]
    variance_ood_normalized = cat_normalized[1]

    visualize_depth_map_uncertainty(x, y, pred, variance_normalized, visualizations_path, f'{i}_iid.png')
    visualize_depth_map_uncertainty(x_ood, y_ood, pred_ood, variance_ood_normalized, visualizations_path, f'{i}_ood.png')