import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class DebugWrappar(keras.Model):
    def __init__(self, base_model):
        super(DebugWrappar, self).__init__()

        self.metric_name = "debug"
        self.base_model = base_model
        self.loss_tracker = keras.metrics.Mean()
        self.val_loss_tracker = keras.metrics.Mean()

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

    def compile(self, optimizer, loss):
        super(DebugWrappar, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_hat = self.base_model(x)
            loss = self.loss_fn(y, y_hat)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # todo-high: almost exactly same as train_step -- reduce code duplication
    def test_step(self, data):
        x, y = data

        y_hat = self.base_model(x, training=False)
        loss = self.loss_fn(y, y_hat)
        self.val_loss_tracker.update_state(loss)
        return {"val_loss": self.val_loss_tracker.result()}

    def call(self, x, training=False, return_risk=True, features=None):
        outs = []

        out = self.base_model(x)
        outs.append(out)

        preds = tf.stack(outs)
        return tf.math.reduce_mean(preds, 0), tf.math.reduce_std(preds, 0)