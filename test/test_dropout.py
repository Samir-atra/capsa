import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from capsa import Wrapper, DropoutWrapper
from capsa.utils import get_user_model, plot_loss, get_preds_names, \
    plot_risk_2d, plot_epistemic_2d
from data import get_data_v2

def test_regression():

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    # user can interact with a MetricWrapper directly
    model = DropoutWrapper(user_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = model.fit(ds_train, epochs=30)
    plot_loss(history)

    y_hat, risk = model(x_val)
    preds_names = get_preds_names(history)

    plot_risk_2d(x_val, y_val, y_hat, risk, preds_names[0])
    # plot_epistemic_2d(x, y, x_val, y_val, y_hat, risk)

test_regression()
