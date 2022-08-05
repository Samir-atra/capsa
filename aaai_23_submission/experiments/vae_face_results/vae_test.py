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
import functools


n_filters = 12  # base number of convolutional filters
latent_dim = 100
data_path = tf.keras.utils.get_file(
    "train_face.h5", "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
)
test_dataloader = TrainingDatasetLoader(data_path, batch_size=32, training=False)
selected_inds = test_dataloader.train_inds
sorted_inds = np.sort(selected_inds)
train_img = (test_dataloader.images[sorted_inds, :, :, ::-1] / 255.0).astype(np.float32)
train_label = test_dataloader.labels[sorted_inds, ...]
plt.imshow(train_img[0])
plt.savefig("testimg.png")
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

    Conv2D(filters=4*n_filters, kernel_size=3,  strides=2),
    BatchNormalization(),

    Conv2D(filters=6*n_filters, kernel_size=3,  strides=2),
    BatchNormalization(),

    Flatten(),
    Dense(512),
    Dense(n_outputs, activation=None),
  ])
  model.build((None, 64, 64, 3))
  return model


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
num_epochs = 6  # keep small to run faster
learning_rate = 1e-5

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
    callbacks=[HistogramCallback()]
)



test_faces = mdl.lab2.get_test_faces()
all_outs = [wrapped_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
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


test_outs = wrapped_classifier(train_img, softmax=True)
print(test_outs)
test_biases = np.squeeze(test_outs[1])
sorted_bias_inds = np.argsort(test_biases)
sorted_biases = np.array(test_biases[sorted_bias_inds])
sorted_images = np.array(train_img[sorted_bias_inds])
num_images = 20

num_samples = len(train_img) // num_images
all_imgs = []
all_bias = []
for percentile in range(num_images):
    cur_imgs = sorted_images[percentile * num_samples : (percentile + 1) * num_samples]
    cur_bias = sorted_biases[percentile * num_samples : (percentile + 1) * num_samples]
    avged_imgs = tf.reduce_mean(cur_imgs, axis=0)
    all_imgs.append(avged_imgs)
    all_bias.append(tf.reduce_mean(cur_bias))

fig = plt.figure()
fig.subplots_adjust(hspace=0.6)
for img in range(num_images):
    ax = fig.add_subplot(num_images/5, 5, img + 1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_to_show = sorted_images[img]
    ax.imshow(img_to_show, interpolation="nearest")
plt.subplots_adjust(wspace=0.20,hspace=0.20)

plt.savefig("least_biased_faces.PNG")
plt.clf()

fig = plt.figure()
fig.subplots_adjust(hspace=0.6)
for img in range(-1 * num_images, 0):
    ax = fig.add_subplot(num_images/5, 5, -1 * img)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_to_show = sorted_images[img]
    ax.imshow(img_to_show, interpolation="nearest")
plt.subplots_adjust(wspace=0.20,hspace=0.20)
plt.savefig("highest_biased_faces.PNG")

fig = plt.figure()
fig.subplots_adjust(hspace=0.6)
for img in range(num_images):
    ax = fig.add_subplot(num_images/5, 5, img + 1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img_to_show = all_imgs[img]
    ax.imshow(img_to_show, interpolation="nearest")
plt.subplots_adjust(wspace=0.20,hspace=0.20)
plt.savefig("percentile.PNG")