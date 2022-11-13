import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from capsa import ControllerWrapper, EnsembleWrapper, MVEWrapper, VAEWrapper

from capsa.utils import (
    get_user_model,
    plot_loss,
    get_preds_names,
    plot_risk_2d,
    plot_epistemic_2d,
)

from bb_data import load_split_data

def get_bias_bounty_data(name="split"):
    if name == "split":
        x_train, y_train = load_split_data()
        
    return x_train, y_train

def main():
    user_model = get_user_model()


if __name__ == "__main__":
    main()