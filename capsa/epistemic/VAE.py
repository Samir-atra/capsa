from random import sample
import tensorflow as tf
from tensorflow import keras

from keras import backend as K


class Sampling(keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAEWrapper(keras.Model):
    def __init__(self, base_model, decoder):
        super(VAEWrapper, self).__init__()

        self.metric_name = "VAEWrapper"

        self.feature_extractor = tf.keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )

        # after sampling (using both out_mu and out_logvar) z has ch 4
        self.mean_layer = tf.keras.layers.Dense(80, activation=None)
        self.log_std_layer = tf.keras.layers.Dense(80, activation=None)
        self.sampling_layer = Sampling()

        # last_layer = base_model.layers[-1]
        # self.output_layer = copy_layer(last_layer)

        self.decoder = decoder

    def reconstruction_loss(self, mu, log_std, x, training=True):
        # Calculates the VAE reconstruction loss by sampling and then feeding the latent vector through the decoder.
        if training:
            sampled_latent_vector = self.sampling_layer([mu, log_std]) # (B, 80)
            reconstruction = self.decoder(sampled_latent_vector, training=True) # (B, 128, 160, 3)
        else:
            reconstruction = self.decoder(mu, training=False) # (B, 128, 160, 3)

        assert(mu.shape[1:] == (80))
        assert(log_std.shape[1:] == (80))
        assert(x.shape[1:] == (128, 160, 3))
        assert(reconstruction.shape[1:] == (128, 160, 3))

        mse_loss = tf.math.square(x - reconstruction) # (B, 8, 10, 1)
        mse_loss = tf.reduce_sum(mse_loss, axis=[1, 2, 3]) # (B, ) 
        mse_loss = tf.reduce_mean(mse_loss) # (, )

        # https://keras.io/examples/generative/vae/
        kl_loss = -0.5 * (1 + log_std - tf.square(mu) - tf.exp(log_std)) # (B, 80)
        kl_loss = tf.reduce_sum(kl_loss, axis=1) # (B, ) 
        kl_loss = tf.reduce_mean(kl_loss) # (, )

        loss = mse_loss + kl_loss
        return loss, mse_loss, kl_loss, reconstruction

    def loss_fn(self, x):
        extractor_out = self.feature_extractor(x, training=True)

        # predictor_y = self.output_layer(extractor_out)

        # compiled_loss = (
        #     self.compiled_loss(y, predictor_y, regularization_losses=self.losses),
        # )

        mu = self.mean_layer(extractor_out) # (B, 80)
        log_std = self.log_std_layer(extractor_out) # (B, 80)

        loss, mse_loss, kl_loss, _ = self.reconstruction_loss(mu=mu, log_std=log_std, x=x)
        # return tf.reduce_mean(loss + compiled_loss), predictor_y
        return loss, mse_loss, kl_loss


    def train_step(self, data):
        x, _ = data

        with tf.GradientTape() as t:
            loss, mse_loss, kl_loss = self.loss_fn(x)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss':loss, 'mse_loss':mse_loss, 'kl_loss':kl_loss}

    def test_step(self, data):
        x, _ = data
        loss, mse_loss, kl_loss = self.loss_fn(x)
        return {'loss':loss, 'mse_loss':mse_loss, 'kl_loss':kl_loss}

    def call(self, x, training=False, return_risk=True):
        features = self.feature_extractor(x, training=training) # (B, 80)
        mu = self.mean_layer(features, training=training) # (B, 80)
        log_std = self.log_std_layer(features, training=training) # (B, 80)

        if training:
            sampled_latent_vector = self.sampling_layer([mu, log_std]) # (B, 80)
            reconstruction = self.decoder(sampled_latent_vector, training=True) # (B, 128, 160, 3)
        else:
            reconstruction = self.decoder(mu, training=False) # (B, 128, 160, 3)

        if return_risk:
            return reconstruction, tf.reduce_sum(tf.math.square(x - reconstruction), axis=-1)
        else:
            return reconstruction