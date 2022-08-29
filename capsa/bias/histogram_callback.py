from numpy import histogram
import tensorflow as tf
from tensorflow import keras
from ..wrapper import Wrapper
from ..utils import copy_layer
import tensorflow_probability as tfp
import numpy as np
from ..epistemic import VAEWrapper
from .histogram import HistogramWrapper
class HistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            if type(self.model) == HistogramWrapper or type(self.model) == VAEWrapper:
                self.model.histogram_layer.update_state()
            elif type(self.model) == Wrapper:
                for name, m in self.model.metric_compiled.items():
                    if name == "HistogramWrapper":
                        m.histogram_layer.update_state()