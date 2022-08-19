import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class DebugWrappar(keras.Model):
    def __init__(self, base_model, metric_wrapper=None, is_standalone=True):
        super(DebugWrappar, self).__init__()

        self.metric_name = "debug"
        self.is_standalone = is_standalone
        self.base_model = base_model

        self.metric_wrapper = metric_wrapper
        self.metrics_compiled = {}

    def compile(self, optimizer, loss):
        super(DebugWrappar, self).compile()

        m = self.base_model
        m = (m if self.metric_wrapper is None else self.metric_wrapper(m, self.is_standalone))
        m_name = (f"usermodel" if self.metric_wrapper is None else f"{m.metric_name}")
        m.compile(optimizer, loss)
        self.metrics_compiled[m_name] = m

    def train_step(self, data):
        keras_metrics = {}

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper is None:
                _ = wrapper.train_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f"{name}_{m.name}"] = m.result()

            # ensembling one of our metrics
            else:
                keras_metric = wrapper.train_step(data, name)
                keras_metrics.update(keras_metric)

        return keras_metrics

    # todo-high: almost exactly same as train_step -- reduce code duplication
    def test_step(self, data):
        keras_metrics = {}

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper is None:
                _ = wrapper.test_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f"{name}_{m.name}"] = m.result()

            # ensembling one of our metrics
            else:
                keras_metric = wrapper.test_step(data, name)
                keras_metrics.update(keras_metric)

        return keras_metrics

    def call(self, x, training=False, return_risk=True, features=None):
        outs = []
        for wrapper in self.metrics_compiled.values():

            # ensembling the user model
            if self.metric_wrapper is None:
                out = wrapper(x)

            # ensembling one of our own metrics
            else:
                out = wrapper(x, training, return_risk, features)
            outs.append(out)

        preds = tf.stack(outs)
        return tf.math.reduce_mean(preds, 0), tf.math.reduce_std(preds, 0)
