import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import Wrapper, HistogramWrapper, VAEWrapper, EnsembleWrapper, DropoutWrapper, MVEWrapper
from capsa.utils import get_user_model, plt_vspan, plot_results, plot_loss
from regression import get_data_v1, get_data_v2
from scipy import stats
from tqdm import tqdm

their_model = get_user_model()
x_train, y_train, x_val, y_val = get_data_v2(batch_size=256)
model = DropoutWrapper(their_model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    loss=tf.keras.losses.MeanSquaredError(),
)
history = model.fit(x_train, y_train, epochs=10)

mu, std = model(x_val, training=False)
print(mu.shape, std.shape)
percentiles = np.arange(100)/100
vals = []
for percentile in percentiles:
  ppf_for_this_percentile = stats.norm.ppf(percentile, mu, std)
  print("y_val", y_val[0], "ppf", ppf_for_this_percentile[0], "mu", mu[0], "std", std[0])
  vals.append((y_val <= ppf_for_this_percentile).mean())

print(np.mean(abs(percentiles - vals)))

plt.clf()
plt.scatter(percentiles, vals)
plt.scatter(percentiles, percentiles)
plt.show()
plt.savefig("calibration_curve_dropout.png")
