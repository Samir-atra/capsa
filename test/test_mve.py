import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

<<<<<<< HEAD
from capsa import (
    Wrapper,
    MVEWrapper,
    HistogramWrapper,
    HistogramCallback,
    EnsembleWrapper,
)
from capsa.utils import (
    get_user_model,
    plt_vspan,
    plot_results,
    plot_loss,
    get_preds_names,
)
from data import get_data_v1, get_data_v2


def plot_aleatoric(x_val, y_val, y_pred, variance, label):
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, variance, s=0.5, label=label)
    plt_vspan()
    plt.legend()
    plt.savefig("aleatoric.PNG")
    plt.show()


def test_regression(use_case=None):

    their_model = get_user_model()
    ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)
=======
from capsa import ControllerWrapper, MVEWrapper, HistogramWrapper, HistogramCallback
from capsa.utils import get_user_model, plot_loss, get_preds_names, plot_risk_2d
from data import get_data_v2


def test_regression(use_case):

    user_model = get_user_model()
    ds_train, ds_val, _, _, x_val, y_val = get_data_v2(batch_size=256, is_show=False)
>>>>>>> 9c21c08ed2bca7491bcf18a9f3e4bf140deb00ef

    # user can interact with a MetricWrapper directly
    if use_case == 1:
        model = MVEWrapper(user_model)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-3),
            loss=keras.losses.MeanSquaredError(),
        )
        history = model.fit(ds_train, epochs=10)
        plot_loss(history)

        risk_tensor = model(x_val)

    # user can interact with a MetricWrapper through Wrapper (what we call a 'controller wrapper')
    elif use_case == 2:
        model = ControllerWrapper(user_model, metrics=[MVEWrapper])

        model.compile(
            # user needs to specify optim and loss for each metric
            optimizer=keras.optimizers.Adam(learning_rate=2e-3),
            loss=keras.losses.MeanSquaredError(),
        )

        history = model.fit(ds_train, epochs=10)
        plot_loss(history)

        metrics_out = model(x_val)
        risk_tensor = metrics_out["mve"]

    preds_names = get_preds_names(history)
    plot_risk_2d(x_val, y_val, risk_tensor, preds_names[0])


def test_regression_predict():

    user_model = get_user_model()
    ds_train, ds_val, _, _, _, _ = get_data_v2(batch_size=256, is_show=False)

    model = ControllerWrapper(user_model, metrics=[MVEWrapper])

    model.compile(
        # user needs to specify optim and loss for each metric
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss=keras.losses.MeanSquaredError(),
    )

    history = model.fit(ds_train, epochs=10)
    plot_loss(history)

    # predict cats batch output to a single tensor under the hood
    # metrics_out is a list (of len 1) of tuples (x_val_batch, y_val_batch)
    metrics_out = model.predict(ds_val)
    risk_tensor = metrics_out["mve"]

    # need this for plotting -- cat all batches
    # list(ds_val) is a list (of len num of batches) of tuples (x_val_batch, y_val_batch)
    cat = np.concatenate(list(ds_val), 1)  # (2, 2304, 1)
    x_val, y_val = cat[0], cat[1]

    preds_names = get_preds_names(history)
    plot_risk_2d(x_val, y_val, risk_tensor, preds_names[0])


# def test_bias(use_case):

#     user_model = get_user_model()
#     ds_train, ds_val, _, _, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

#     ### use case 1 - user can interact with a MetricWrapper directly
#     if use_case == 1:
#         model = HistogramWrapper(user_model)

#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )

#         history = model.fit(ds_train, epochs=10, callbacks=[HistogramCallback()])
#         plot_loss(history)

#         risk_tensor = model(x_val)

#     ### use case 2 - user can interact with a MetricWrapper through Wrapper (what we call a 'controller wrapper')
#     elif use_case == 2:

#         model = ControllerWrapper(user_model, metrics=[HistogramWrapper])

#         model.compile(
#             # user needs to specify optim and loss for each metric
#             optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )

#         history = model.fit(ds_train, epochs=10, callbacks=[HistogramCallback()])
#         plot_loss(history)

#         metrics_out = model(x_val)
#         risk_tensor = metrics_out["histogram"]

#     plot_risk_2d(x_val, y_val, risk_tensor, "histogram")


test_regression(1)
test_regression(2)
test_regression_predict()
