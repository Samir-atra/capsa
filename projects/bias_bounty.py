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

def get_bias_bounty_data():
    data_path="/data/bias_bounty/split/"
    

def main():
    user_model = get_user_model()


if __name__ == "__main__":
    main()