import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def get_user_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def get_decoder():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def plot_loss(history):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc="upper right")
    plt.show()


def get_preds_names(history):
    l = list(history.history.keys())
    # cut of keras metric names -- e.g., 'MVEW_0_loss' -> 'MVEW_0'
    l_split = [i.rsplit("_", 1)[0] for i in l]
    # remove duplicates (if specified multiple keras metrics)
    return list(dict.fromkeys(l_split))


def plt_vspan():
    plt.axvspan(-6, -4, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.axvspan(4, 6, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.xlim([-6, 6])


def plot_epistemic_2d(x, y, x_val, y_val, y_pred, risk, k=3):
    # x, y = x[:, 0], y[:, 0] # x (20480, ), y (20480, )
    # x_val, y_val = x_val[:, 0], y_val[:, 0] # x_val (2304, ), y_val (2304, )
    # risk = risk[:, None]
    plt.scatter(x, y, label="train data")
    plt.plot(x_val, y_val, "g-", label="ground truth")
    plt.plot(x_val, y_pred, "r-", label="pred")
    plt.fill_between(
        x_val[:, 0],
        (y_pred - k * risk)[:, 0],
        (y_pred + k * risk)[:, 0],
        alpha=0.2,
        color="r",
        linestyle="-",
        linewidth=2,
        label="epistemic",
    )
    plt.legend()
    plt.show()


def plot_risk_2d(x_val, y_val, risk_tens, label):
    if risk_tens.aleatoric != None and risk_tens.epistemic != None:
        Exception(
            "RiskTensor has both aleatoric and epistemic uncertainties, please specify which one you would like to plot."
        )
    elif risk_tens.aleatoric != None:
        risk = risk_tens.aleatoric
    elif risk_tens.epistemic != None:
        risk = risk_tens.epistemic
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, risk_tens.y_hat, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, risk, s=0.5, label=label)
    plt_vspan()
    plt.legend()
    plt.show()


def _get_out_dim(model):
    return model.layers[-1].output_shape[1:]


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1:]
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_log_var))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def reverse_model(model, latent_dim):
    inputs = tf.keras.Input(shape=latent_dim)
    i = len(model.layers) - 1
    while type(model.layers[i]) != layers.InputLayer and i >= 0:
        if i == len(model.layers) - 1:
            x = reverse_layer(model.layers[i])(inputs)
        else:
            if type(model.layers[i - 1]) == layers.InputLayer:
                original_input = model.layers[i - 1].input_shape
                x = reverse_layer(model.layers[i], original_input)(x)
            else:
                x = reverse_layer(model.layers[i])(x)
        i = i - 1
    return tf.keras.Model(inputs, x)


def reverse_layer(layer, output_shape=None):
    config = layer.get_config()
    layer_type = type(layer)
    unchanged_layers = [layers.Activation, layers.BatchNormalization, layers.Dropout]
    # TODO: handle global pooling separately
    pooling_1D = [
        layers.MaxPooling1D,
        layers.AveragePooling1D,
        layers.GlobalMaxPooling1D,
    ]
    pooling_2D = [
        layers.MaxPooling2D,
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
    ]
    pooling_3D = [
        layers.MaxPooling3D,
        layers.AveragePooling3D,
        layers.GlobalMaxPooling3D,
    ]
    conv = [layers.Conv1D, layers.Conv2D, layers.Conv3D]

    if layer_type == layers.Dense:
        config["units"] = layer.input_shape[-1]
        return layers.Dense.from_config(config)
    elif layer_type in unchanged_layers:
        return type(layer).from_config(config)
    elif layer_type in pooling_1D:
        return layers.UpSampling1D(size=config["pool_size"])
    elif layer_type in pooling_2D:
        return layers.UpSampling2D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in pooling_3D:
        return layers.UpSampling3D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in conv:
        if output_shape is not None:
            config["filters"] = output_shape[0][-1]

        if layer_type == layers.Conv1D:
            return layers.Conv1DTranspose.from_config(config)
        elif layer_type == layers.Conv2D:
            return layers.Conv2DTranspose.from_config(config)
        elif layer_type == layers.Conv3D:
            return layers.Conv3DTranspose.from_config(config)
    else:
        raise NotImplementedError()



def copy_layer(layer, override_activation=None):
    # if no_activation is False, layer might or
    # might not have activation (depending on the config)
    layer_conf = layer.get_config()
    if override_activation:
        layer_conf["activation"] = override_activation
    # works for any serializable layer
    return type(layer).from_config(layer_conf)
