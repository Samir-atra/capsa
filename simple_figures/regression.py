import argparse
import tensorflow as tf
from capsa import Wrapper, EnsembleWrapper, VAEWrapper, MVEWrapper, DropoutWrapper, HistogramWrapper, HistogramCallback
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from colorline import colorline, rainbow_fill_between


parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="show plots using a gui")
parser.add_argument("--naive", action="store_true", help="no capsa")
parser.add_argument("--epistemic", action="store_true", help="epistemic uncertainty")
parser.add_argument("--aleatoric", action="store_true", help="aleatoric uncertainty")
parser.add_argument("--bias", action="store_true", help="bias")
args = parser.parse_args()


def gen_data():
    np.random.seed(0)
    x_train = np.random.normal(1.5, 2, (7000, 1))
    x_train = np.expand_dims(x_train[np.abs(x_train) < 4], 1)
    x_test = np.expand_dims(np.linspace(-6, 6, 1000), 1)

    # compute the labels
    y_train = x_train ** 3 / 10
    y_test = x_test ** 3 / 10

    # add baseline aleatoric noise to training
    y_train += np.random.normal(0, 0.2, x_train.shape)

    # add concentration of label noise centered at x=-0.75
    y_train += np.random.normal(0, 0.8*np.exp(-(x_train+0.75)**2))

    return (x_train, y_train), (x_test, y_test)


def get_model(input_shape=(1,)):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(1, None),
        ]
    )




model = get_model()
(x_train, y_train), (x_test, y_test) = gen_data()


# Naive model: no capsa, deterministic network. Plot the predicted values
if args.naive:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
    y_pred = model(x_test).numpy()

    plt.figure(figsize=(8,4))
    plt.gca().set_rasterization_zorder(0)
    plt.scatter(x_train, y_train, s=2, c='k', alpha=0.15, zorder=-1)
    plt.plot(x_test, y_pred, 'r')
    plt.xlim(-6, 6); plt.ylim(-12, 12)
    plt.savefig("export/reg_naive.pdf")
    if args.gui:
        plt.show()
    plt.close()


# Bias: Should support this in Capsa, but for now we hardcode a KDE
# implementation here for demo.
if args.bias:
    train_bias = np.zeros(x_train.shape)
    test_bias = np.zeros(x_test.shape)
    for sample in x_train[::10]:
        train_bias += ss.norm.pdf(x_train, sample, scale=0.2)
        test_bias += ss.norm.pdf(x_test, sample, scale=0.2)

    plt.figure(figsize=(8,4))
    # plt.plot(x_test, np.log(1/(1+bias**0.5)))
    # plt.scatter(x_train[::5], np.abs(np.random.normal(0, train_bias[::5]**0.5))/20., s=1, c=train_bias[::5]**0.5, alpha=0.3)
    plt.scatter(x_train[::25], 0*x_train[::25], s=25, c='k', alpha=0.1, zorder=-1)
    plt.gca().set_rasterization_zorder(0)
    # plt.plot(x_test, test_bias**0.5)
    colorline(x_test.ravel(), test_bias.ravel()**0.5, test_bias.ravel()**0.5, cmap='viridis')
    plt.xlim(x_test.min(), x_test.max())
    plt.ylim(-1, (test_bias**0.5).max()*1.1)
    plt.savefig("export/reg_bias.pdf")
    if args.gui:
        plt.show()
    plt.close()

# Epistemic: Ensemble variance between predicted outputs of model members
if args.epistemic:
    epistemic_model = EnsembleWrapper(model, num_members=10)
    epistemic_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    epistemic_model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test))
    # epistemic_model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))

    outputs = epistemic_model(x_test).numpy()
    y_pred = outputs.mean(0)
    y_std = outputs.std(0)

    plt.figure(figsize=(8,4))
    plt.scatter(x_train, y_train, s=2, c='k', alpha=0.1, zorder=-1)
    plt.gca().set_rasterization_zorder(0)
    # plt.plot(x_test, y_pred, zorder=1)
    sigma = np.maximum((10*y_std).ravel(), 0.2)
    rainbow_fill_between(plt.gca(),
        X=x_test.ravel(),
        Y1=y_pred.ravel()-sigma,
        Y2=y_pred.ravel()+sigma,
        colors=sigma,
        cmap='viridis_r',
        alpha=0.8, zorder=2)
    plt.xlim(-6, 6); plt.ylim(-12, 12)
    plt.savefig("export/reg_epistemic.pdf")
    if args.gui:
        plt.show()
    plt.close()


# Aleatoric: MVE wrapper
if args.aleatoric:
    aleatoric_model = MVEWrapper(model)
    aleatoric_model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=tf.keras.losses.MeanSquaredError())
    aleatoric_model.fit(x_train, y_train, batch_size=128, epochs=80, validation_data=(x_test, y_test))

    y_pred, y_var = aleatoric_model(x_test)
    y_pred, y_std = y_pred.numpy(), np.sqrt(y_var.numpy())

    plt.figure(figsize=(8,4))
    # plt.scatter(x_train, y_train, s=2, c='k', alpha=0.1, zorder=-1)
    # plt.plot(x_test, y_pred, zorder=1)
    sigma = np.maximum((3*y_std).ravel(), 0.1)
    rainbow_fill_between(plt.gca(),
        X=x_test.ravel(),
        Y1=y_pred.ravel()-sigma,
        Y2=y_pred.ravel()+sigma,
        colors=sigma,
        cmap='viridis_r',
        alpha=0.8, zorder=2)
    plt.gca().set_rasterization_zorder(0)


    # plt.fill_between(x_test.ravel(), (y_pred-3*y_std).ravel(), (y_pred+3*y_std).ravel(), alpha=0.3, zorder=2)
    plt.xlim(-6, 6); plt.ylim(-12, 12);
    plt.savefig("export/reg_aleatoric.pdf", transparent=True)
    if args.gui:
        plt.show()
    plt.close()


import pdb; pdb.set_trace()
