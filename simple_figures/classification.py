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


def gen_data(noise=True):
    x, y = datasets.make_moons(n_samples=60000, noise=0.1)

    # mask = np.random.choice(2, y.shape, p=[0.5, 0.5])
    if noise:
        random_variable = ss.multivariate_normal([-0.7, 0.75], [[0.03, 0.0], [0.0, 0.05]])
        p_flip = random_variable.pdf(x)
        p_flip = p_flip / (2 * p_flip.max())
        flip = p_flip > np.random.rand(p_flip.shape[0])

        y[flip] = 1 - y[flip]

    x = x.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return (x_train, y_train), (x_test, y_test)


def get_model(input_shape=(2,), dropout_rate=0.1):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(16, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(1, "sigmoid"),
        ]
    )



# Prep the data/model for any/all of the metrics
model = get_model()
(x_train, y_train), (x_test, y_test) = gen_data()
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 100), np.linspace(-1, 1.5, 100))
grid = np.stack([xx, yy], axis=-1)

# Naive model: no capsa, deterministic network. Plot the decision boundary
if args.naive:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_train, y_train, batch_size=64, epochs=4, validation_data=(x_test, y_test))
    grid_pred = model(grid).numpy()

    plt.scatter(x_train[:,0], x_train[:,1], s=1, c=y_train)
    plt.contour(xx, yy, grid_pred[:,:,0], [0.5])
    if args.gui:
        plt.show()
    plt.close()

# Bias: Should support this in Capsa, but for now we hardcode a KDE
# implementation here for demo.
if args.bias:
    bias = np.zeros((grid.shape[0], grid.shape[1]))
    for sample in x_train[::15]:
        bias += ss.multivariate_normal.pdf(grid, sample, cov=0.025*np.eye(2))
    plt.pcolormesh(xx, yy, bias**0.75, cmap='Blues_r')
    if args.gui:
        plt.show()
    plt.close()


# Epistemic: Ensemble variance between predicted outputs of model members
if args.epistemic:
    epistemic_model = EnsembleWrapper(model, num_members=5)
    epistemic_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
    epistemic_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

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



import pdb; pdb.set_trace()
