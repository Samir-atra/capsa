import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from capsa import Wrapper, HistogramWrapper, VAEWrapper, EnsembleWrapper, DropoutWrapper, MVEWrapper
from capsa.utils import get_user_model, plt_vspan, plot_results, plot_loss
from scipy import stats
from tqdm import tqdm

from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss, load_model, totensor_and_normalize
from models import unet
import config
from callbacks import CalibrationCallback


def gen_calibration_plot(model, path, ds_test):
        percentiles = np.arange(41)/40
        vals = []
        mu = []
        std = []
        y_test = []
        for step, (x_test_batch, y_test_batch) in enumerate(ds_test):
            #outs = np.array([self.model(x_test_batch, training=True) for _ in range(20)])
            #mu_batch = np.array([i[0] for i in outs])
            #std_batch = np.array([i[1] for i in outs])
            #total_mu = tf.math.reduce_mean(mu_batch, axis=0)
            #mu.append(total_mu)
            #std.append(tf.math.reduce_mean(std_batch + mu_batch, axis=0) - total_mu**2)
            mu_batch, std_batch = model(x_test_batch)
            mu.append(mu_batch)
            std.append(std_batch)
            y_test.append(y_test_batch)
        mu = np.array(mu)
        std = np.array(std)
        print(tf.math.reduce_mean(std))
        y_test = np.array(y_test)
        y_test = y_test.reshape(-1, *y_test.shape[-3:])
        mu = mu.reshape(-1, *mu.shape[-3:])
        std = std.reshape(-1, *std.shape[-3:])
        
        for percentile in tqdm(percentiles):
            ppf_for_this_percentile = stats.norm.ppf(percentile, mu, std)
            vals.append((y_test <= ppf_for_this_percentile).mean())

        plt.clf()
        plt.scatter(percentiles, vals)
        plt.scatter(percentiles, percentiles)
        plt.title(str(np.mean(abs(percentiles - vals))))
        plt.show()
        plt.savefig(path)
        