import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class DebugWrappar(keras.Model):
    def __init__(self, base_model, metric_wrapper=None, is_standalone=True):
        super(DebugWrappar, self).__init__()

        self.metric_name = "debug"
        self.is_standalone = is_standalone
        self.metric_wrapper = metric_wrapper

        self.model_name = (f"usermodel" if self.metric_wrapper is None else f"{m.metric_name}")
        self.base_model = base_model

    def train_step(self, data):
        keras_metrics = {}

        # user model
        if self.metric_wrapper is None:
            _ = self.base_model.train_step(data)
            for m in self.base_model.metrics:
                keras_metrics[f"{self.model_name}_{m.name}"] = m.result()

        # one of our metrics
        else:
            keras_metric = self.base_model.train_step(data, self.model_name)
            keras_metrics.update(keras_metric)

        return keras_metrics

    # todo-high: almost exactly same as train_step -- reduce code duplication
    def test_step(self, data):
        keras_metrics = {}

        # user model
        if self.metric_wrapper is None:
            _ = self.base_model.test_step(data)
            for m in self.base_model.metrics:
                keras_metrics[f"{self.model_name}_{m.name}"] = m.result()

        # one of our metrics
        else:
            keras_metric = self.base_model.test_step(data, self.model_name)
            keras_metrics.update(keras_metric)

        return keras_metrics

    def call(self, x, training=False, return_risk=True, features=None):
        outs = []

        # ensembling the user model
        if self.metric_wrapper is None:
            out = self.base_model(x)

        # ensembling one of our own metrics
        else:
            out = self.base_model(x, training, return_risk, features)
        outs.append(out)

        preds = tf.stack(outs)
        return tf.math.reduce_mean(preds, 0), tf.math.reduce_std(preds, 0)
