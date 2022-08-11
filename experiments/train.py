import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
# import pandas as pd

from models import create
from run_utils import setup
from utils import load_depth_data, load_apollo_data, visualize_depth_map, visualize_depth_map_uncertainty, plot_loss

# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LR = 5e-5



visualizations_path, checkpoints_path, plots_path, logs_path = setup()
logger = CSVLogger(f'{logs_path}/log.csv', append=True)

### https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train = tf.convert_to_tensor(x_train[:256], tf.float32) #16, 256, 1024
y_train = tf.convert_to_tensor(y_train[:256], tf.float32) #16, 256, 1024
x_train /= 255.
y_train /= 255.

_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood = tf.convert_to_tensor(x_test_ood[:256], tf.float32) #16, 256, 1024
y_test_ood = tf.convert_to_tensor(y_test_ood[:256], tf.float32) #16, 256, 1024
x_test_ood /= 255.
y_test_ood /= 255.

their_model = create(input_shape=x_train.shape[1:], drop_prob=0.0, activation=tf.nn.relu, num_class=1)
# trainer = trainer_obj(model, opts, args.learning_rate)





### train base model only (debugging)
# their_model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=LR),
#     loss=keras.losses.MeanSquaredError(),
# )

# history = their_model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0) # 10000 epochs
# plot_loss(history, plots_path)

# todo-high: inference on val set
# pred = model(x_train, verbose=0)
# visualize_depth_map(x_train, y_train, pred, visualizations_path)



from capsa import (
    Wrapper,
    MVEWrapper,
    EnsembleWrapper,
)

model = EnsembleWrapper(their_model, num_members=2)
model.compile(
    optimizer=[keras.optimizers.Adam(learning_rate=LR)],
    loss=[keras.losses.MeanSquaredError()],
    # metrics=[[
    #     # keras.metrics.MeanSquaredError(name='mse'),
    #     keras.metrics.CosineSimilarity(name='cos'),
    # ]],
)

history = model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[logger], verbose=0) # 10000 epochs
plot_loss(history, plots_path)

def get_ensemble_uncertainty(x_train, model):
    outs = model(x_train)
    preds = np.concatenate((outs[0][np.newaxis, ...], outs[1][np.newaxis, ...]), 0)
    pred = np.mean(preds, 0)
    epistemic = np.std(preds, 0)

    return pred, epistemic

pred, epistemic = get_ensemble_uncertainty(x_train, model)
visualize_depth_map_uncertainty(x_train, y_train, pred, epistemic, visualizations_path, 'iid.png')

pred_ood, epistemic_ood = get_ensemble_uncertainty(x_test_ood, model)
visualize_depth_map_uncertainty(x_test_ood, y_test_ood, pred_ood, epistemic_ood, visualizations_path, 'ood.png')