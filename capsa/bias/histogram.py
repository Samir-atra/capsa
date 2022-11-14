from numpy import histogram
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from ..controller_wrapper import ControllerWrapper
from ..base_wrapper import BaseWrapper
from ..utils import copy_layer
from ..risk_tensor import RiskTensor

class HistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            if type(self.model) == ControllerWrapper:
                for name, m in self.model.metric_compiled.items():
                    if name == "HistogramWrapper":
                        m.histogram_layer.update_state()
            else:
                self.model.histogram_layer.update_state()


class HistogramWrapper(BaseWrapper):
    """Tracks the feature histogram given a model.
    Calculates feature probabilities by discretizing features before the last layer.
    To calculate the bias of a sample, we calculate probability of those combinations of features
    occurring in the distribution learned so far.
    Example usage outside of the ``ControllerWrapper`` (standalone):
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = HistogramWrapper(user_model)
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(self, base_model, is_standalone=True, num_bins=5):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        is_standalone : bool
            Indicates whether or not the metric wrapper will be used inside the ``ControllerWrapper``.
        num_bins: int
            The number of bins to discretize the histogram distribution into.
        """
        super(HistogramWrapper, self).__init__(base_model, is_standalone)
        self.base_model = base_model
        self.metric_name = "histogram"
        self.is_standalone = is_standalone

        self.histogram_layer = HistogramLayer(num_bins=num_bins)

    def loss_fn(self, x, y, features=None):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor
            Ground truth label.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``loss_fn`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.
        Returns
        -------
        loss : tf.Tensor
            Float, reflects how well does the algorithm perform given the ground truth label,
            predicted label and the metric specific loss function. In this case it is
            0 because ``HistogramWrapper`` does not introduce an additional loss function,
            and the compiled loss is already added in the parent class ``BaseWrapper.train_step()``.
        y_hat : tf.Tensor
            Predicted label.
        """
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)
        self.histogram_layer(features, training=True)
        out = self.output_layer(features)
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

    def call(self, x, training=False, return_risk=True, features=None):
        """
        Forward pass of the model
        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``call`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.
        T : int, default 20
            Number of forward passes with different dropout masks.
        Returns
        -------
        out : capsa.RiskTensor
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the bias
            uncertainty estimate (tf.Tensor).
        """
        if self.is_standalone:
            features = self.feature_extractor(x, training=False)
        
        y_hat = self.output_layer(features)
        if not return_risk:
            return RiskTensor(y_hat)
        else:
            bias = self.histogram_layer(features, training=training)
            return RiskTensor(y_hat, bias=bias)


class HistogramLayer(tf.keras.layers.Layer):
    """Custom layer that calculates feature histograms at every epoch. 
    Discretizing input features into `num_bins` per dimension, and resets the histogram at every epoch.
    """

    def __init__(self, num_bins=5):
        """
        Parameters
        ----------
        num_bins: int
            The number of bins to discretize the histogram distribution into.
        """

        super(HistogramLayer, self).__init__()
        self.num_bins = num_bins

    def build(self, input_shape):
        # Constructs the layer the first time that it is called
        self.frequencies = tf.Variable(
            initial_value=tf.zeros((self.num_bins, input_shape[-1])), trainable=False
        )

        self.feature_dim = input_shape[1:]
        self.num_batches = tf.Variable(initial_value=0, trainable=False)
        self.edges = tf.Variable(
            initial_value=tf.zeros((self.num_bins + 1, input_shape[-1])),
            trainable=False,
        )
        self.minimums = tf.Variable(
            initial_value=tf.zeros(input_shape[1:]), trainable=False
        )
        self.maximums = tf.Variable(
            initial_value=tf.zeros(input_shape[1:]), trainable=False
        )

    def call(self, inputs, training=True):
        # Updates frequencies if we are training
        if training:
            self.minimums.assign(
                tf.math.minimum(tf.reduce_min(inputs, axis=0), self.minimums)
            )
            self.maximums.assign(
                tf.math.maximum(tf.reduce_max(inputs, axis=0), self.maximums)
            )
            histograms_this_batch = tfp.stats.histogram(
                inputs,
                self.edges,
                axis=0,
                extend_lower_interval=True,
                extend_upper_interval=True,
            )
            self.frequencies.assign(tf.add(self.frequencies, histograms_this_batch))
            self.num_batches.assign_add(1)
        else:
            # Returns the probability of a datapoint occurring if we are in inference mode
            # Normalize histograms
            hist_probs = tf.divide(
                self.frequencies, tf.reduce_sum(self.frequencies, axis=0)
            )
            # Get the corresponding bins of the features
            bin_indices = tf.cast(
                tfp.stats.find_bins(
                    inputs,
                    self.edges,
                    extend_lower_interval=True,
                    extend_upper_interval=True,
                ),
                tf.dtypes.int32,
            )

            # Multiply probabilities together to compute bias
            second_element = tf.repeat(
                [tf.range(tf.shape(inputs)[1])], repeats=[tf.shape(inputs)[0]], axis=0
            )
            indices = tf.stack([bin_indices, second_element], axis=2)

            probabilities = tf.gather_nd(hist_probs, indices)
            logits = tf.reduce_sum(tf.math.log(probabilities), axis=1)
            logits = logits - tf.math.reduce_mean(logits) #log probabilities are the wrong sign if we don't subtract the mean
            return tf.math.softmax(logits)

    def update_state(self):
        self.edges.assign(tf.linspace(self.minimums, self.maximums, self.num_bins + 1))
        self.minimums.assign(tf.zeros(self.feature_dim))
        self.maximums.assign(tf.zeros(self.feature_dim))

        self.frequencies.assign(tf.zeros((self.num_bins, self.feature_dim[-1])))