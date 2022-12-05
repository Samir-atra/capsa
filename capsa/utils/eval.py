import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np



def expected_calibration_error(logits,labels_true,name=None,num_bins=10):
    """Expected Calibration Error (ECE) for classification.
    ECE is defined as the expected difference between the observed and predicted
    probabilities. See [1] for details.
    Args:
        num_bins: Number of bins to use when computing ECE.
        logits=logits predicted by the model for given input or inputs
        labels_true=the true labels of the input or inputs
        name: Optional name for the tf.operations created by this function.
    Returns:
        ECE metric.
    Raises:
        ValueError: If `num_bins` is not a positive integer.
    [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning (ICML 2017). arXiv:1706.04599.
    """
    if num_bins <= 0:
        raise ValueError("Number of bins must be positive.")

    ece = tfp.stats.expected_calibration_error(num_bins=num_bins,logits=logits,labels_true=labels_true,name=name)
    
    return ece

def maximum_calibration_error(logits,labels_true,name=None,num_bins=10):
    """Maximum Calibration Error (MCE) for classification.
    MCE is defined as the expected difference between the observed and predicted
    probabilities. See [1] for details.
    Args:
        num_bins: Number of bins to use when computing MCE.
        logits=logits predicted by the model for given input or inputs
        labels_true=the true labels of the input or inputs
        name: Optional name for the tf.operations created by this function.
    Returns:
        MCE metric.
    Raises:
        ValueError: If `num_bins` is not a positive integer.
    [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning (ICML 2017). arXiv:1706.04599.
    """
    if num_bins <= 0:
        raise ValueError("Number of bins must be positive.")

    with tf.name_scope(name or 'expected_calibration_error'):
        logits = tf.convert_to_tensor(logits)
        labels_true = tf.convert_to_tensor(labels_true)
        if labels_predicted is not None:
            labels_predicted = tf.convert_to_tensor(labels_predicted)

    # Compute empirical counts over the events defined by the sets
    # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
    # of predicted probabilities in each probability bin.
    event_bin_counts, pmean_observed = tfp._compute_calibration_bin_statistics(
        num_bins, logits=logits, labels_true=labels_true,
        labels_predicted=labels_predicted)

    # Compute the marginal probability of observing a probability bin.
    event_bin_counts = tf.cast(event_bin_counts, tf.float32)
    bin_n = tf.reduce_sum(event_bin_counts, axis=0)
    pbins = bin_n / tf.reduce_sum(bin_n)  # Compute the marginal bin probability

    # Compute the marginal probability of making a correct decision given an
    # observed probability bin.
    tiny = np.finfo(np.float32).tiny
    pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

    # Compute the MCE statistic as defined in reference [1].
    mce = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))

    return mce



