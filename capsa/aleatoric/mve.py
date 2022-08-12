import tensorflow as tf
from tensorflow import keras

from ..utils import MLP, _get_out_dim, copy_layer


# class MVEWrapper(keras.Model):

#     def __init__(self, base_model, is_standalone=True):
#         super(MVEWrapper, self).__init__()

#         self.metric_name = 'mve'
#         self.is_standalone = is_standalone

#         if is_standalone:
#             self.feature_extractor = keras.Model(
#                 inputs=base_model.inputs,
#                 outputs=base_model.layers[-2].output,
#             )

#         output_layer = base_model.layers[-1]
#         # self.out_y = copy_layer(output_layer)
#         self.out_mu = copy_layer(output_layer, override_activation="linear")
#         self.out_logvar = copy_layer(output_layer, override_activation="linear")

#     @staticmethod
#     def neg_log_likelihood(y, mu, logvariance):
#         variance = tf.exp(logvariance)
#         return logvariance + (y-mu)**2 / variance

#     def loss_fn(self, x, y, features=None):
#         if self.is_standalone:
#             features = self.feature_extractor(x, training=True)

#         # y_hat = self.out_y(features)
#         mu = self.out_mu(features)
#         logvariance = self.out_logvar(features)

#         # loss = tf.reduce_mean(
#         #     self.compiled_loss(y, y_hat, regularization_losses=self.losses),
#         # )

#         # tf.print('mse', tf.convert_to_tensor(loss))

#         loss = tf.reduce_mean(self.neg_log_likelihood(y, mu, logvariance))

#         # tf.print('gaussian_nll', tf.reduce_mean(self.neg_log_likelihood(y, mu, logvariance)))

#         return loss #, y_hat

#     def train_step(self, data, prefix=None):
#         x, y = data

#         with tf.GradientTape() as t:
#             loss = self.loss_fn(x, y)
#         # self.compiled_metrics.update_state(y, y_hat)

#         trainable_vars = self.trainable_variables
#         gradients = t.gradient(loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         if prefix is None:
#             prefix = self.metric_name
#         # return {f'{prefix}_{m.name}': m.result() for m in self.metrics}
#         return {f'{prefix}_loss': loss}

#     def call(self, x, training=False, return_risk=True, features=None):

#         if self.is_standalone:
#             features = self.feature_extractor(x, training)
#         # y_hat = self.out_y(features)

#         # todo-high: - not y_pred so not MSE supervised
#         mu = self.out_mu(features)
#         logvariance = self.out_logvar(features)
#         variance = tf.exp(logvariance)
#         return mu, variance


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

    def loss_fn(self, x, y, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)

        # y_hat = self.out_y(features)
        mu = self.out_mu(features)
        logvariance = self.out_logvar(features)

        # loss = tf.reduce_mean(
        #     self.compiled_loss(y, y_hat, regularization_losses=self.losses),
        # )

        # tf.print('mse', tf.convert_to_tensor(loss))
        

        loss = self.neg_log_likelihood(y, mu, logvariance) # (N_SAMPLES, 128, 160, 1)
        # tf.print('loss 1', tf.convert_to_tensor(loss.shape))

        loss = tf.reduce_mean(loss, axis=[1, 2])  # (N_SAMPLES, 1)
        # tf.print('loss 2', tf.convert_to_tensor(loss.shape))

        loss = tf.reduce_mean(loss) # ( )
        # tf.print('loss 3', tf.convert_to_tensor(loss.shape))


        # tf.print('gaussian_nll', tf.reduce_mean(self.neg_log_likelihood(y, mu, logvariance)))

        return loss #, y_hat

    def train_step(self, data, prefix=None):
        x, y = data

        with tf.GradientTape() as t:
            loss = self.loss_fn(x, y)
        # self.compiled_metrics.update_state(y, y_hat)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if prefix is None:
            prefix = self.metric_name
        # return {f'{prefix}_{m.name}': m.result() for m in self.metrics}
        return {f'{prefix}_loss': loss}

    def call(self, x, training=False, return_risk=True, features=None):

        if self.is_standalone:
            features = self.feature_extractor(x, training)
        # y_hat = self.out_y(features)

        mu = self.out_mu(features)
        logvariance = self.out_logvar(features)
        variance = tf.exp(logvariance)
        return mu, variance