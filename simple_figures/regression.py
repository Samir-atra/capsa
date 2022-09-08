import argparse
import tensorflow as tf
from capsa import Wrapper, EnsembleWrapper, VAEWrapper, MVEWrapper, DropoutWrapper, HistogramWrapper, HistogramCallback
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import scipy.stats as ss


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
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
    y_pred = model(x_test).numpy()

    plt.scatter(x_train, y_train, s=1, c='k')
    plt.plot(x_test, y_pred)
    if args.gui:
        plt.show()


    plt.close()


# Bias: Should support this in Capsa, but for now we hardcode a KDE
# implementation here for demo.
if args.bias:
    bias = np.zeros(x_test.shape)
    for sample in x_train[::10]:
        bias += ss.norm.pdf(x_test, sample, scale=0.2)

    plt.scatter(x_train, y_train, s=1, c='k')
    plt.scatter(x_test, y_test, s=1, c=bias**0.5)
    if args.gui:
        plt.show()
    plt.close()

    plt.plot(x_test, bias**0.5)
    if args.gui:
        plt.show()
    plt.close()


# Epistemic: Ensemble variance between predicted outputs of model members
if args.epistemic:
    epistemic_model = EnsembleWrapper(model, num_members=5)
    epistemic_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    epistemic_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

    import pdb; pdb.set_trace()
    ### Getting error: TypeError: compile() got an unexpected keyword
    # argument 'weighted_metrics'


# Aleatoric: MVE wrapper
if args.aleatoric:
    aleatoric_model = MVEWrapper(model)
    aleatoric_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    aleatoric_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

    ### Getting error: ValueError: Error when checking model target: the list
    # of Numpy arrays that you are passing to your model is not the size the
    # model expected. Expected to see 2 array(s), but instead got the following
    # list of 1 arrays: [array([[0.],



plt.scatter(x_train[::5], y_train[::5], s=1)
plt.show()
import pdb; pdb.set_trace()

wrapped_model = EnsembleWrapper(model, num_members=5)
wrapped_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
wrapped_model.fit(x_train, y_train, batch_size=64, epochs=70, validation_data=(orig_x_test, y_test), callbacks=[HistogramCallback()])

'''
x_test_x = np.linspace(-1.5, 2.5, 50)
x_test_y = np.linspace(-1.0, 2.0, 50)
all_x_test = []
for i in x_test_x:
    for j in x_test_y:
        all_x_test.append([i, j])
x_test = np.asarray(all_x_test)

_, std_linspace = wrapped_model(x_test)
'''

preds = wrapped_model(orig_x_test)
y_pred = tf.reduce_mean(preds, axis=0)
std = tf.math.reduce_std(preds, axis=0)
#plt.scatter(x_test[:, 0], x_test[:, 1], c=std_linspace)
#plt.scatter(orig_x_test[:, 0], orig_x_test[:, 1], c=std)


plt.scatter(x_train, y_train, s=0.5)
plt.scatter(orig_x_test, y_test, s=0.5)
plt.savefig("data.pdf")
plt.savefig("data.png")

plt.clf()
#plt.scatter(orig_x_test, std.numpy())
plt.scatter(np.squeeze(orig_x_test), std.numpy())
plt.savefig("ensemble.pdf")
plt.savefig("ensemble.png")

plt.clf()
plt.hist(orig_x_test)
plt.savefig("gt.pdf")
plt.savefig("gt.png")
