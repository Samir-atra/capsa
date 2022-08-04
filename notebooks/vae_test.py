import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys
import glob
import functools
from capsa import HistogramWrapper, HistogramCallback, wrap, VAEWrapper
from train_dataloader import TrainingDatasetLoader
import mitdeeplearning as mdl
from tqdm import tqdm


n_filters = 12  # base number of convolutional filters
import functools

"""Function to define a standard CNN model"""


def make_standard_classifier(n_outputs=1):
    Conv2D = functools.partial(
        tf.keras.layers.Conv2D, padding="same", activation="relu"
    )
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation="relu")

    inp = tf.keras.Input((64, 64, 3))
    x = Conv2D(filters=1 * n_filters, kernel_size=5, strides=2)(inp)
    x = BatchNormalization()(x)
    x = Conv2D(filters=2 * n_filters, kernel_size=5, strides=2)(inp)
    x = BatchNormalization()(x)
    x = Conv2D(filters=4 * n_filters, kernel_size=5, strides=2)(inp)
    x = BatchNormalization()(x)
    x = Conv2D(filters=6 * n_filters, kernel_size=5, strides=2)(inp)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(n_outputs, activation=None, name="dense2")(x)

    return tf.keras.Model(inp, x)


latent_dim = 32


def make_face_decoder_network():
    # Functionally define the different layer types we will use
    Conv2DTranspose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="same", activation="relu"
    )
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation="relu")
    Reshape = tf.keras.layers.Reshape

    # Build the decoder network using the Sequential API
    decoder = tf.keras.Sequential(
        [
            # Transform to pre-convolutional generation
            Dense(units=4 * 4 * 6 * n_filters),  # 4x4 feature maps (with 6N occurances)
            Reshape(target_shape=(4, 4, 6 * n_filters)),
            # Upscaling convolutions (inverse of encoder)
            Conv2DTranspose(filters=4 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=2 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=1 * n_filters, kernel_size=5, strides=2),
            Conv2DTranspose(filters=3, kernel_size=5, strides=2),
        ]
    )
    decoder.build((None, latent_dim,))
    return decoder


standard_classifier = make_standard_classifier()

wrapped_classifier = VAEWrapper(
    standard_classifier,
    decoder=make_face_decoder_network(),
    latent_dim=latent_dim,
    epistemic=False,
)
# Training hyperparameters
batch_size = 32
num_epochs = 2  # keep small to run faster
learning_rate = 1e-5

data_path = tf.keras.utils.get_file(
    "train_face.h5", "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
)
dataloader = TrainingDatasetLoader(data_path, batch_size=batch_size)


wrapped_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

history = wrapped_classifier.fit(
    dataloader,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[HistogramCallback()],
)


import mitdeeplearning as mdl

test_faces = mdl.lab2.get_test_faces()
all_outs = [wrapped_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
# predictions = all_outs
predictions = [out[0] for out in all_outs]
biases = [out[1] for out in all_outs]

keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
for group, key in zip(test_faces, keys):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.hstack(group))
    plt.title(key, fontsize=15)

predictions = tf.squeeze(tf.sigmoid(predictions))
predictions = predictions.numpy().mean(1)
biases = np.asarray(biases).mean(1)
print(predictions)
print(biases)
