import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
# import pandas as pd

from models import create
from utils import visualize_depth_map, plot_loss

def _load_depth():
    train = h5py.File("/home/iaroslavelistratov/data/depth_train.h5", "r")
    test = h5py.File("/home/iaroslavelistratov/data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def load_depth():
    return _load_depth()

def load_apollo():
    test = h5py.File("/home/iaroslavelistratov/data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])

### https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
(x_train, y_train), (x_test, y_test) = load_depth()
# (X_train, y_train), (X_test, y_test), y_train_scale = load_dataset('depth')
x_train = tf.convert_to_tensor(x_train[:16], tf.float32) #256 # 1024
y_train = tf.convert_to_tensor(y_train[:16], tf.float32) #256 # 1024

x_train /= 255.
y_train /= 255.

their_model = create(input_shape=x_train.shape[1:], drop_prob=0.0, activation=tf.nn.relu, num_class=1)
# trainer = trainer_obj(model, opts, args.learning_rate)

their_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=keras.losses.MeanSquaredError(),
)

history = their_model.fit(x_train, y_train, epochs=500, batch_size=16) # 10000 epochs
plot_loss(history)

visualize_depth_map(x_train, y_train, their_model)