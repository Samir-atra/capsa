import tensorflow as tf
from tensorflow import keras

from ..utils import MLP, _get_out_dim, copy_layer
import numpy as np

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
        self.out_y = copy_layer(output_layer)
        self.out_mu = copy_layer(output_layer, override_activation="linear")
        self.out_logsigma = copy_layer(output_layer, override_activation="linear")

    @staticmethod
    def nll_loss(y, mu, sigma, reduce=True):
        ax = list(range(1, len(y.shape)))

        logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
        loss = tf.reduce_mean(-logprob, axis=ax)
        return tf.reduce_mean(loss) if reduce else loss 

    def loss_fn(self, x, y, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)

        y_hat = self.out_y(features)
        logsigma = self.out_logsigma(features)
        

        loss = tf.reduce_mean(
            self.nll_loss(y, y_hat, tf.nn.softplus(logsigma) + 1e-6)
        ) # (N_SAMPLES, 128, 160, 1) -> ( )

        return loss, y_hat

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
    
    @tf.function
    def wrapped_train_step(self, x, y, features, prefix):
        with tf.GradientTape() as t:
            loss, predictor_y = self.loss_fn(x, y, features)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return (
            {f"{prefix}_{m.name}": m.result() for m in self.metrics},
            tf.gradients(loss, features),
        )

    def test_step(self, data, prefix=None):
        x, y = data
        loss, y_hat = self.loss_fn(x, y)
        if prefix is None:
            prefix = self.metric_name
        return {f'{prefix}_loss': loss}

    def call(self, x, training=False, return_risk=True, features=None):

        if self.is_standalone:
            features = self.feature_extractor(x, training)
        y_hat = self.out_y(features)

        if return_risk:
            logsigma = self.out_logsigma(features)
            std = tf.nn.softplus(logsigma) + 1e-6
            return (y_hat, std**2)
        else:
            return y_hat