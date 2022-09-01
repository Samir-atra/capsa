import numpy as np
import tensorflow as tf
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

from capsa import MVEWrapper
from models import AutoEncoder

def gen_calibration_plot(model, ds, path=None):
    mu_ = []
    std_ = []
    y_test_ = []

    # x_test_batch, y_test_batch = iter(ds).get_next()
    for (x_test_batch, y_test_batch) in ds:
        # todo-med: better use model.predict, it's more optimized
        mu_batch, std_batch = model(x_test_batch)
        mu_.append(mu_batch)
        std_.append(std_batch)
        y_test_.append(y_test_batch)

    mu = np.concatenate(mu_) #(3029, 128, 160, 1)
    std = np.concatenate(std_) #(3029, 128, 160, 1)
    y_test = np.concatenate(y_test_) #(3029, 128, 160, 1)

    if isinstance(model, MVEWrapper):
        std = np.sqrt(std)

    vals = []
    percentiles = np.arange(41)/40
    for percentile in percentiles:
        # returns the value at the n% percentile e.g., stats.norm.ppf(0.5, 0, 1) == 0.0
        # in other words, if have a normal distrib. with mean 0 and std 1, 50% of data falls below and 50% falls above 0.0.
        ppf_for_this_percentile = stats.norm.ppf(percentile, mu, std) # (3029, 128, 160, 1)
        vals.append((y_test <= ppf_for_this_percentile).mean()) # (3029, 128, 160, 1) -> scalar

    plt.plot(percentiles, vals)
    plt.plot(percentiles, percentiles)
    plt.title(str(np.mean(abs(percentiles - vals))))
    plt.show()
    if path != None:
        plt.savefig(path)