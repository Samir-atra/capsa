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


class TrainingDatasetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, training=True):

        print("Opening {}".format(data_path))
        sys.stdout.flush()

        self.cache = h5py.File(data_path, "r")

        print("Loading data into memory...")
        sys.stdout.flush()
        self.images = self.cache["images"][:]
        self.labels = self.cache["labels"][:].astype(np.float32)
        self.image_dims = self.images.shape
        total_train_samples = self.image_dims[0]

        if training:
            n_train_samples = np.arange(int(0.7 * total_train_samples))
        else:
            n_train_samples = int(0.7 * total_train_samples) + np.arange(int(0.3 * total_train_samples))

        self.train_inds = np.random.permutation(n_train_samples)

        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]
        self.batch_size = batch_size

    def get_train_size(self):
        return self.train_inds.shape[0]

    def __len__(self):
        return int(np.floor(self.get_train_size() / self.batch_size))

    def __getitem__(self, index):
        selected_pos_inds = np.random.choice(
            self.pos_train_inds, size=self.batch_size // 2, replace=False
        )
        selected_neg_inds = np.random.choice(
            self.neg_train_inds, size=self.batch_size // 2, replace=False
        )
        selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :, :, ::-1] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]

        return np.array(train_img), np.array(train_label)

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[: 10 * n : 10]]
        return (self.images[most_prob_inds, ...] / 255.0).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]
    
    def return_sample_batch(self):
        return self.__getitem__(0)