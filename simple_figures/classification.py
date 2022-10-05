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
        random_variable = ss.multivariate_normal([-0.7, 0.8], [[0.03, 0.0], [0.0, 0.05]])
        p_flip = random_variable.pdf(x)
        p_flip = p_flip / (3 * p_flip.max())
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
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(2, None),
        ]
    )



# Prep the data/model for any/all of the metrics
model = get_model()
(x_train, y_train), (x_test, y_test) = gen_data()
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
grid = np.stack([xx, yy], axis=-1)

# Naive model: no capsa, deterministic network. Plot the decision boundary
if args.naive:
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
    # grid_pred = tf.nn.sigmoid(model(grid)).numpy()[:,:,[0]]

    i = y_train == 0
    plt.figure(figsize=(8,4))
    plt.scatter(x_train[i,0][::20], x_train[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_train[~i,0][::20], x_train[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)
    plt.gca().set_rasterization_zorder(0)
    # plt.contour(xx, yy, grid_pred[:,:,0], [0.5])
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    plt.savefig("export/cls_naive.pdf")
    if args.gui:
        plt.show()
    plt.close()



# Bias: Should support this in Capsa, but for now we hardcode a KDE
# implementation here for demo.
if args.bias:
    bias = np.zeros((grid.shape[0], grid.shape[1]))
    for sample in x_train[::15]:
        bias += ss.multivariate_normal.pdf(grid, sample, cov=0.02*np.eye(2))

    i = y_train == 0
    plt.figure(figsize=(8,4))
    plt.pcolormesh(xx, yy, bias**.5, cmap='viridis', alpha=0.9, zorder=-2)
    plt.clim(bias.min(), (bias**.5).max()*0.75)
    plt.scatter(x_train[i,0][::20], x_train[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_train[~i,0][::20], x_train[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)

    plt.gca().set_rasterization_zorder(0)
    plt.savefig("export/cls_bias.pdf")
    if args.gui:
        plt.show()
    plt.close()


# Epistemic: Ensemble variance between predicted outputs of model members
if args.epistemic:
    epistemic_model = EnsembleWrapper(model, num_members=5)
    epistemic_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    epistemic_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

    # grid_pred = tf.nn.sigmoid(epistemic_model(grid)).numpy()[:,:,:,[0]]
    unc = tf.nn.softmax(epistemic_model(grid.reshape(-1, 2))).numpy().std(0).sum(-1)
    unc = unc.reshape(200, 200)

    i = y_train == 0
    plt.figure(figsize=(8,4))
    plt.pcolormesh(xx, yy, np.log(1e-1+unc), cmap='viridis_r', alpha=0.9, zorder=-2)
    plt.scatter(x_train[i,0][::20], x_train[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_train[~i,0][::20], x_train[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)

    plt.gca().set_rasterization_zorder(0)
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    plt.savefig("export/cls_epistemic.pdf")
    if args.gui:
        plt.show()
    plt.close()





# Aleatoric: MVE wrapper
if args.aleatoric:
    y_train, y_test = np.expand_dims(y_train, -1), np.expand_dims(y_test, -1)
    aleatoric_model = MVEWrapper(model)
    aleatoric_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    aleatoric_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

    var = aleatoric_model(grid.reshape(-1, 2))[1].numpy().reshape(200, 200, 2)
    std = var.sum(-1) ** 0.5

    i = y_train[:,0] == 0
    plt.figure(figsize=(8,4))
    plt.pcolormesh(xx, yy, std, cmap='viridis_r', alpha=0.9, zorder=-2)
    plt.scatter(x_train[i,0][::20], x_train[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_train[~i,0][::20], x_train[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)

    plt.gca().set_rasterization_zorder(0)
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    plt.savefig("export/cls_aleatoric.pdf")
    if args.gui:
        plt.show()
    plt.close()



    ### Getting error: ValueError: Error when checking model target: the list
    # of Numpy arrays that you are passing to your model is not the size the
    # model expected. Expected to see 2 array(s), but instead got the following
    # list of 1 arrays: [array([[0.],



# import pdb; pdb.set_trace()
