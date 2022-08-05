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

test_faces = mdl.lab2.get_test_faces()
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
probs = np.array([1.9163518e-29, 2.5964466e-30, 1.5205098e-30, 7.6509994e-32])


logits = np.array([-67.99536,  -68.94391,  -70.070045, -71.775635])
log_probs = -1 * (logits - np.mean(logits)) + np.mean(logits)
probs = tf.math.softmax(logits)

probs = np.round(probs/sum(probs), 4)

for group, key, prob in zip(test_faces,keys,probs): 
  plt.figure(figsize=(5,5))
  plt.imshow(np.hstack(group))
  plt.title(key + f": {prob}", fontsize=15)
  plt.savefig(key)

