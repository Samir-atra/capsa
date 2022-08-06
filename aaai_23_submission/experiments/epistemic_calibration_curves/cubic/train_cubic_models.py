import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import Wrapper, HistogramWrapper, VAEWrapper, EnsembleWrapper, DropoutWrapper
from capsa.utils import get_user_model, plt_vspan, plot_results, plot_loss
from regression import get_data_v1, get_data_v2
from scipy import stats
from tqdm import tqdm

their_model = get_user_model()
x_train, y_train, x_val, y_val = get_data_v2(batch_size=256)

model = VAEWrapper(their_model, kl_weight=4, bias=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    loss=tf.keras.losses.MeanSquaredError(),
)
history = model.fit(x_train, y_train, epochs=30)

x_train = tf.squeeze(x_train)
y_pred, predictive_entropy= model(x_train)
epsilon = 1e-05
#predictive_entropy = -tf.reduce_sum(tf.math.multiply(y_pred, tf.math.log(y_pred + epsilon)), axis=-1)
preds_test = np.array(y_pred)
targets_test = y_train
targets_pred = preds_test.mean(axis=0).reshape(-1)
residuals = targets_pred - targets_test.reshape(-1)
stdevs = predictive_entropy.numpy().reshape(-1)


# Define a normalized bell curve we'll be using to calculate calibration
norm = stats.norm(loc=0, scale=1)


def calculate_density(percentile):
    '''
    Calculate the fraction of the residuals that fall within the lower
    `percentile` of their respective Gaussian distributions, which are
    defined by their respective uncertainty estimates.
    '''
    # Find the normalized bounds of this percentile
    upper_bound = norm.ppf(percentile)

    # Normalize the residuals so they all should fall on the normal bell curve
    normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the fraction of residuals that fall within the bounds
    density = num_within_quantile / len(residuals)
    return density


predicted_pi = np.linspace(0, 1, 100)
observed_pi = [calculate_density(quantile)
               for quantile in tqdm(predicted_pi, desc='Calibration')]

calibration_error = ((predicted_pi - observed_pi)**2).sum()
print('Calibration error = %.2f' % calibration_error)

import seaborn as sns


# Set figure defaults
width = 4  # Because it looks good
fontsize = 12
rc = {'figure.figsize': (width, width),
      'font.size': fontsize,
      'axes.labelsize': fontsize,
      'axes.titlesize': fontsize,
      'xtick.labelsize': fontsize,
      'ytick.labelsize': fontsize,
      'legend.fontsize': fontsize}
sns.set(rc=rc)
sns.set_style('ticks')

# Plot settings
figsize = (4, 4)

# Plot the calibration curve
fig_cal = plt.figure(figsize=figsize)
ax_ideal = sns.lineplot([0, 1], [0, 1], label='ideal')
_ = ax_ideal.lines[0].set_linestyle('--')
ax_gp = sns.lineplot(x=predicted_pi, y=observed_pi, label='Dropout')
ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                           alpha=0.2, label='miscalibration area')
_ = ax_ideal.set_xlabel('Expected cumulative distribution')
_ = ax_ideal.set_ylabel('Observed cumulative distribution')
_ = ax_ideal.set_xlim([0, 1])
_ = ax_ideal.set_ylim([0, 1])
plt.savefig("calibrationcurve")