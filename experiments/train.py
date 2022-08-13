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
    plot_loss

(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train, y_train = totensor_and_normalize(x_train, y_train)

_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood, y_test_ood = totensor_and_normalize(x_test_ood, y_test_ood)

def train_base_model():
    vis_path, checkpoints_path, plots_path, logs_path = setup()
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])
    their_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=keras.losses.MeanSquaredError(),
    )

    history = their_model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0)
    plot_loss(history, plots_path)

    pred = their_model(x_train)
    visualize_depth_map(x_train, y_train, pred, vis_path)

def train_ensemble_wrapper():
    vis_path, checkpoints_path, plots_path, logs_path = setup()
    logger = CSVLogger(f'{logs_path}/log.csv', append=True)

    their_model = create(x_train.shape[1:])

    model = EnsembleWrapper(their_model, num_members=2)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=5e-5)],
        loss=[keras.losses.MeanSquaredError()],
        # metrics=[[keras.metrics.CosineSimilarity(name='cos')]],
    )

    history = model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0)
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

# train_base_model()
# train_ensemble_wrapper()



vis_path, checkpoints_path, plots_path, logs_path = setup()
logger = CSVLogger(f'{logs_path}/log.csv', append=True)

their_model = create(x_train.shape[1:])
model = MVEWrapper(their_model)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR))

itters_per_ep = config.N_SAMPLES / config.BS
total_itters = itters_per_ep * config.EP
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

history = model.fit(x_train, y_train, epochs=config.EP, batch_size=config.BS, callbacks=[logger, model_checkpoint_callback], verbose=0) # 10000 epochs
plot_loss(history, plots_path)
plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path)