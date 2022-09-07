import tensorflow as tf
from capsa import Wrapper, EnsembleWrapper, VAEWrapper, MVEWrapper, DropoutWrapper, HistogramWrapper, HistogramCallback
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def gen_cubic_data():
    np.random.seed(0)
    x_train = np.random.normal(1.5,4, (10000, 1))
    x_train = np.expand_dims(x_train[np.abs(x_train) < 4], 1)
    x_test = np.random.normal(1.5,2, (1000, 1))
    x_test = np.expand_dims(x_test[np.abs(x_test) < 6], 1)

    y_train = x_train **3 / 10
    y_test = x_test ** 3/10
    
    y_train += np.random.normal(0, 0.2, (len(x_train), 1))
    y_test += np.random.normal(0, 0.2, (len(x_test), 1))
    
    x_train = np.concatenate((x_train, np.random.normal(1.5, 0.3, 4096)[:, np.newaxis]), 0)
    y_train = np.concatenate((y_train, np.random.normal(1.5, 0.6, 4096)[:, np.newaxis]), 0)

    x_test = np.concatenate((x_test, np.random.normal(1.5, 0.3, 256)[:, np.newaxis]), 0)
    y_test = np.concatenate((y_test, np.random.normal(1.5, 0.6, 256)[:, np.newaxis]), 0)
    
    return (x_train, y_train), (x_test, y_test)

def gen_moons_data(noise=True):
    x, y = datasets.make_moons(n_samples=60000, noise=0.1)

    mask = np.random.choice(2, y.shape, p=[0.5, 0.5])
    if noise:
        for i in range(len(x)):
            if -0.5 < x[i][0] < 0.0 and x[i][1] > 0.8 and mask[i]:
                y[i] = 1

    x = x.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return (x_train, y_train), (x_test, y_test)
    

def get_toy_model_classification(input_shape=(2,), dropout_rate=0.1):
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

def get_toy_model(input_shape=(1,)):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(64, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(16, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(1, None),
        ]
    )

    


model = get_toy_model()
(x_train, y_train), (orig_x_test, y_test) = gen_cubic_data()

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