import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class EnsembleWrapper(keras.Model):
    def __init__(
        self, base_model, metric_wrapper=None, num_members=1, is_standalone=True
    ):
        super(EnsembleWrapper, self).__init__()

        self.metric_name = "ensemble"
        self.is_standalone = is_standalone
        self.base_model = base_model

        self.metric_wrapper = metric_wrapper
        self.num_members = num_members
        self.metrics_compiled = {}

    def compile(self, optimizer, loss, metrics=[None]):
        super(EnsembleWrapper, self).compile()

        # if user passes only 1 optimizer and loss_fn yet they specified e.g. num_members=3,
        # duplicate that one optimizer and loss_fn for all members in the ensemble
        if type(optimizer) != list:
            optim_conf = optim.serialize(optimizer)
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
        if type(loss) != list:
            loss = [loss for _ in range(self.num_members)]

        if len(optimizer) or len(loss) < self.num_members:
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
            # losses and *most* keras metrics are stateless, no need to serialize as above
            loss = [loss[0] for _ in range(self.num_members)]
            metrics = [metrics[0] for _ in range(self.num_members)]

        base_model_config = self.base_model.get_config()
        assert base_model_config != {}, "Please implement get_config()."

        for i in range(self.num_members):

            if isinstance(self.base_model, keras.Sequential):
                m = keras.Sequential.from_config(base_model_config)
            elif isinstance(self.base_model, keras.Model):
                m = keras.Model.from_config(base_model_config)
            else:
                raise Exception(
                    "Please provide a Sequential, Functional or subclassed model."
                )

            m = (
                m
                if self.metric_wrapper is None
                else self.metric_wrapper(m, self.is_standalone)
            )
            m_name = (
                f"usermodel_{i}"
                if self.metric_wrapper is None
                else f"{m.metric_name}_{i}"
            )
            m.compile(optimizer[i], loss[i], metrics[i])
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

    def wrapped_train_step(self, x, y, features, prefix):
        keras_metrics = {}

        accum_grads = tf.zeros_like(features)
        scalar = 1 / self.num_members

        for name, wrapper in self.metrics_compiled.items():
            keras_metric, grad = wrapper.wrapped_train_step(
                x, y, features, f"{prefix}_{name}"
            )
            keras_metrics.update(keras_metric)
            accum_grads += tf.scalar_mul(scalar, grad[0])
        return keras_metrics, [accum_grads]

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
        y_hat = tf.math.reduce_mean(preds, 0)

        if return_risk:
            # https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/gen_depth_results.py#L445-L446
            return y_hat, tf.sqrt(tf.math.reduce_variance(preds, 0))
        else:
            return y_hat