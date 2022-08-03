# Import Tensorflow 2.0
import tensorflow as tf

import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from capsa import HistogramCallback, HistogramWrapper, Wrapper, wrap

# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl
import h5py
import sys

import requests

requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def call(self, inputs):
        return inputs / 255.0


# Get the training data: both images from CelebA and ImageNet
from rsa import verify


data_path = tf.keras.utils.get_file(
    "train_face.h5", "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
)

loader = mdl.lab2.TrainingDatasetLoader(data_path)
images = loader.images
labels = loader.images
face_images = images[np.where(labels == 1)[0]]
not_face_images = images[np.where(labels == 0)[0]]

idx_face = 30  # @param {type:"slider", min:0, max:50, step:1}
idx_not_face = 25  # @param {type:"slider", min:0, max:50, step:1}

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face] / 255.0)
plt.title("Face")
plt.grid(False)

plt.subplot(1, 2, 2)
plt.imshow(not_face_images[idx_not_face] / 255.0)
plt.title("Not Face")
plt.grid(False)

plt.show()

number_of_training_examples = loader.get_train_size()
(images, labels) = loader.get_batch(100)
face_images = images[np.where(labels==1)[0]]
not_face_images = images[np.where(labels==0)[0]]

idx_face = 24 #@param {type:"slider", min:0, max:50, step:1}
idx_not_face = 26 #@param {type:"slider", min:0, max:50, step:1}

plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face])
plt.title("Face"); plt.grid(False)

plt.subplot(1, 2, 2)
plt.imshow(not_face_images[idx_not_face])
plt.title("Not Face"); plt.grid(False)
'''
n_train_samples = images.shape[0]
train_inds = np.random.permutation(np.arange(n_train_samples))

print("Transforming data")
labels = labels[train_inds]

### Define the CNN model ###

n_filters = 12  # base number of convolutional filters

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
    x = Conv2D(filters=2 * n_filters, kernel_size=5, strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=4 * n_filters, kernel_size=5, strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=6 * n_filters, kernel_size=5, strides=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation=None, name="dense1")(x)
    x = Dense(n_outputs, activation=None, name="dense2")(x)

    return tf.keras.Model(inp, x)


# Training hyperparameters
batch_size = 32
num_epochs = 2  # keep small to run faster
learning_rate = 1e-3


wrapped_classifier = HistogramWrapper(make_standard_classifier())
wrapped_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

loss_history = mdl.util.LossHistory(smoothing_factor=0.99)  # to record loss evolution
plotter = mdl.util.PeriodicPlotter(sec=2, scale="semilogy")
if hasattr(tqdm, "_instances"):
    tqdm._instances.clear()  # clear if it exists

history = wrapped_classifier.fit(
    images, labels, epochs=2, batch_size=batch_size, callbacks=[HistogramCallback()]
)

test_faces = mdl.lab2.get_test_faces()
all_outs = [wrapped_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
predictions = [out[0] for out in all_outs]
biases = [out[1] for out in all_outs]

standard_classifier_probs = tf.squeeze(tf.sigmoid(predictions))
biases = tf.squeeze(biases)

print(standard_classifier_probs)
print(biases)

'''
