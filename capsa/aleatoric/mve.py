import tensorflow as tf
from tensorflow import keras
import numpy as np

from ..utils import MLP, _get_out_dim, copy_layer

def Gaussian_NLL(y, mu, sigma, reduce=True):
    # https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/losses/continuous.py
    ax = list(range(1, len(y.shape)))

    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

class MVEWrapper(keras.Model):

    def __init__(self, base_model, is_standalone=True):
        super(MVEWrapper, self).__init__()

        self.metric_name = 'mve'
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = keras.Model(
                inputs=base_model.inputs,
                outputs=base_model.layers[-2].output,
            )

        output_layer = base_model.layers[-1]
        # self.out_y = copy_layer(output_layer)
        self.out_mu = copy_layer(output_layer, override_activation="linear")
        self.out_logvar = copy_layer(output_layer, override_activation="linear")

    @staticmethod
    def neg_log_likelihood(y, mu, logvariance):
        variance = tf.exp(logvariance)
        return logvariance + (y-mu)**2 / variance

    def loss_fn(self, x, y, return_var=False, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)

        # y_hat = self.out_y(features)
        # mu = self.out_mu(features)
        # logvariance = self.out_logvar(features)

        # loss = tf.reduce_mean(
        #     self.compiled_loss(y, y_hat, regularization_losses=self.losses),
        # )
        # # tf.print('mse', tf.convert_to_tensor(loss))

        # loss += tf.reduce_mean(
        #     self.neg_log_likelihood(y, mu, logvariance)
        # ) # (N_SAMPLES, 128, 160, 1) -> ( )
        # # # tf.print('gaussian_nll', tf.reduce_mean(self.neg_log_likelihood(y, mu, logvariance)))




        mu = self.out_mu(features)
        logsigma = self.out_logvar(features)
        # softplus is a smooth approximation of relu. Like relu, softplus always takes on positive values.
        sigma = tf.nn.softplus(logsigma) + 1e-6

        loss = Gaussian_NLL(y, mu, tf.math.sqrt(sigma))
        # tf.print('loss:', loss)

        if return_var:
            return loss, mu, sigma
        else:
            return loss, mu




        # mu = self.out_mu(features)
        # logvariance = self.out_logvar(features)

        # loss = tf.reduce_mean(
        #     self.neg_log_likelihood(y, mu, logvariance)
        # )

        # if return_var:
        #     return loss, mu, tf.exp(logvariance)
        # else:
        #     return loss, mu

    def train_step(self, data, prefix=None):
        x, y = data

        with tf.GradientTape() as t:
            loss, y_hat = self.loss_fn(x, y)
        # self.compiled_metrics.update_state(y, y_hat)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if prefix is None:
            prefix = self.metric_name
        # return {f'{prefix}_{m.name}': m.result() for m in self.metrics}
        return {f'{prefix}_loss': loss}

    def test_step(self, data):
        x, y = data
        loss, y_hat = self.loss_fn(x, y)
        prefix = self.metric_name
        return {f'{prefix}_loss': loss}

    def call(self, x, training=False, return_risk=True, features=None):

        if self.is_standalone:
            features = self.feature_extractor(x, training)
        # y_hat = self.out_y(features)
        y_hat = self.out_mu(features)

        if return_risk:
            logsigma = self.out_logvar(features)
            sigma = tf.nn.softplus(logsigma) + 1e-6
            return (y_hat, sigma)

            # logvariance = self.out_logvar(features)
            # variance = tf.exp(logvariance)
            # return (y_hat, variance)
        else:
            return y_hat