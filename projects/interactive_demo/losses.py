import tensorflow as tf

def MSE(y, y_, reduce=True):
    # https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/losses/continuous.py
    ax = list(range(1, len(y.shape)))
    mse = tf.reduce_mean((y-y_)**2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse

# def gaussian_NLL(y, mu, sigma, reduce=True):
#     # https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/losses/continuous.py
#     ax = list(range(1, len(y.shape)))

#     logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
#     loss = tf.reduce_mean(-logprob, axis=ax)
#     return tf.reduce_mean(loss) if reduce else loss

# def neg_log_likelihood(y, mu, logvariance):
#     variance = tf.exp(logvariance)
#     loss = logvariance + (y-mu)**2 / variance

#     ax = list(range(1, len(y.shape)))
#     reduced = tf.reduce_mean(loss, axis=ax)
#     return tf.reduce_mean(loss)