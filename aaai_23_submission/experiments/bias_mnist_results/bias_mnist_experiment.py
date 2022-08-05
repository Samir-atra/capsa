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
import mitdeeplearning as mdl
from tqdm import tqdm
import functools


n_filters = 3  # base number of convolutional filters
latent_dim = 100

"""Function to define a standard CNN model"""

def make_standard_classifier(n_outputs=1):
  Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
  BatchNormalization = tf.keras.layers.BatchNormalization
  Flatten = tf.keras.layers.Flatten
  Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

  model = tf.keras.Sequential([
    Conv2D(filters=1*n_filters, kernel_size=5,  strides=2),
    BatchNormalization(),
    
    Conv2D(filters=2*n_filters, kernel_size=5,  strides=2),
    BatchNormalization(),

    Flatten(),
    Dense(128),
    Dense(n_outputs, activation=None),
  ])
  model.build((None, 28, 28, 1))
  return model


def make_mnist_decoder_network():
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
            Dense(units=7 * 7 * 2 * n_filters),  # 4x4 feature maps (with 6N occurances)
            Reshape(target_shape=(7, 7, 2 * n_filters)),
            # Upscaling convolutions (inverse of encoder)
            Conv2DTranspose(filters=2* n_filters, kernel_size=5, strides=2),
            Conv2DTranspose(filters=1* n_filters, kernel_size=5, strides=2)
        ]
    )
    decoder.build((None, latent_dim,))
    return decoder


standard_classifier = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
wrapped_classifier = HistogramWrapper(
    standard_classifier,
)
# Training hyperparameters
batch_size = 32
num_epochs =  2 # keep small to run faster
learning_rate = 1e-5

def get_mnist(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # # Preprocess the data (these are NumPy arrays)
    x_train = x_train[..., np.newaxis].astype(np.float32) / 255.0
    x_test = x_test[..., np.newaxis].astype(np.float32) / 255.0

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_mnist(flatten=True)

wrapped_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
history = wrapped_classifier.fit(
    x_train,
    y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[HistogramCallback()]
)



all_outs = [wrapped_classifier(x_test, softmax=True)]
bias = np.array([i[1] for i in all_outs])
bias = tf.squeeze(bias)

num_images = 10
lowest_indices = np.argpartition(bias, num_images)[:num_images]
fig = plt.figure()
fig.subplots_adjust(hspace=0.6)
for img in range(num_images):
    ax = fig.add_subplot(num_images/5, 5, img + 1)
    img_to_show = tf.reshape(tf.squeeze(x_test[lowest_indices[img]]), (28, 28))
    ax.imshow(img_to_show, interpolation="nearest")
    
    ax.set_title(
        str("{:.5e}".format(bias[lowest_indices[img]]) + "\n"),
        y=1.08,
    )
    
plt.tight_layout()
plt.savefig("lowest_indices.PNG")

highest_indices = np.argpartition(bias, -1 * num_images)[-1 * num_images : ]
fig = plt.figure()
fig.subplots_adjust(hspace=0.6)
for img in range(num_images):
    ax = fig.add_subplot(num_images/5, 5, img + 1)
    img_to_show = tf.reshape(tf.squeeze(x_test[highest_indices[img]]), (28, 28))
    ax.imshow(img_to_show, interpolation="nearest")
    
    ax.set_title(
        str("{:.5e}".format(bias[highest_indices[img]]) + "\n"),
        y=1.08,
    )
    
plt.tight_layout()
plt.savefig("highest_indices.PNG")




