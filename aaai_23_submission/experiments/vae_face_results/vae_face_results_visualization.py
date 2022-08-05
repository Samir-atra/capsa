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
probs = np.array([1.8098767e-29, 4.0885006e-30, 1.3895825e-30, 1.1570064e-31])
probs = np.round(probs/sum(probs), 4)

for group, key, prob in zip(test_faces,keys,probs): 
  plt.figure(figsize=(5,5))
  plt.imshow(np.hstack(group))
  plt.title(key + f": {prob}", fontsize=15)
  plt.savefig(key)

