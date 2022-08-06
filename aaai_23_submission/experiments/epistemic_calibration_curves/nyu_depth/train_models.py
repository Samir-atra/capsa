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
from capsa import HistogramWrapper, HistogramCallback, wrap, VAEWrapper, DropoutWrapper, EnsembleWrapper
from tqdm import tqdm
import functools
from create_models import create

epistemic_named_wrappers = {
        #"DropoutWrapper": DropoutWrapper,
        "EnsembleWrapper": EnsembleWrapper,
        #"VAEWrapper": VAEWrapper,
}

train_data = np.load("../../data/training_data_labeled.npy")
train_labels = np.load("../../data/depths_labeled.npy")
num_epochs = 1

for name, wrapper in epistemic_named_wrappers.items():
    original_model = create(train_data[0].shape)
    wrapped_model = wrapper(original_model, num_members=1)
    wrapped_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = wrapped_model.fit(
        train_data,
        train_labels,
        epochs=num_epochs,
        batch_size=32,
    )