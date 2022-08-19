import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class DebugWrappar(keras.Model):
    def __init__(self, base_model):
        super(DebugWrappar, self).__init__()

        self.metric_name = "debug"
        self.base_model = base_model

    def train_step(self, data):
        keras_metrics = {}

        _ = self.base_model.train_step(data)
        for m in self.base_model.metrics:
            keras_metrics[f"{self.metric_name}_{m.name}"] = m.result()

        return keras_metrics

    # todo-high: almost exactly same as train_step -- reduce code duplication
    def test_step(self, data):
        keras_metrics = {}

        _ = self.base_model.test_step(data)
        for m in self.base_model.metrics:
            keras_metrics[f"{self.metric_name}_{m.name}"] = m.result()

        return keras_metrics

    def call(self, x, training=False, return_risk=True, features=None):
        outs = []

        out = self.base_model(x)
        outs.append(out)

        preds = tf.stack(outs)
        return tf.math.reduce_mean(preds, 0), tf.math.reduce_std(preds, 0)