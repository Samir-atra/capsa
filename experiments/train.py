import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import pandas as pd

from models import create
from run_utils import setup
from utils import load_depth_data, visualize_depth_map, plot_loss

visualizations_path, checkpoints_path, plots_path, logs_path = setup()

### https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/train_depth.py#L34
(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train = tf.convert_to_tensor(x_train[:256], tf.float32) #16, 256, 1024
y_train = tf.convert_to_tensor(y_train[:256], tf.float32) #16, 256, 1024

x_train /= 255.
y_train /= 255.

their_model = create(input_shape=x_train.shape[1:], drop_prob=0.0, activation=tf.nn.relu, num_class=1)
# trainer = trainer_obj(model, opts, args.learning_rate)

their_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=keras.losses.MeanSquaredError(),
)

history = their_model.fit(x_train, y_train, epochs=256, batch_size=8) # 10000 epochs

plot_loss(history, plots_path)
visualize_depth_map(x_train, y_train, their_model, visualizations_path)