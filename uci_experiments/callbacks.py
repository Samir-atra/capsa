import tensorflow as tf
import numpy as np
class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, y_scale, model_type):
        super(LossCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type
        self.min_nll = float('inf')
        self.min_rmse = float('inf')
        self.scale = y_scale
        self.drop_prob = 0.2
        self.lam = 1e-3
        self.l = 1e-2
    
    def calculate_nll(self, mu, var, y):
        logvariance = tf.math.log(var)
        return tf.reduce_mean(logvariance + (y-mu)**2 / var)
    
    def calculate_rmse(self, mu, y):
        return tf.math.sqrt(tf.reduce_mean(
            tf.math.square(mu - y),
        ))
    
    def gen_mixture_model_preds(self, outputs):
        mu_batch = tf.stack([output[0] for output in outputs])
        var_batch = tf.stack([output[1] for output in outputs])

        mean_mu = tf.reduce_mean(mu_batch, axis=0)
        var = tf.reduce_mean(var_batch + mu_batch**2, axis=0) - mean_mu**2
        print(var)
        return mean_mu, var

    def gen_preds(self, test_batch_x):
        if self.model_type == "ensemble":
            preds = self.model(test_batch_x)
            return tf.math.reduce_mean(preds, 0), tf.math.reduce_std(preds, 0)
        elif self.model_type == "ensemble + mve":
            preds = self.model(test_batch_x)
            return self.gen_mixture_model_preds(preds)
        elif self.model_type == "dropout":
            preds = tf.stack([self.model(test_batch_x, training=True) for _ in range(20)])
            tau = self.l**2 * (1-self.drop_prob) / (2. * self.lam)
            var = tf.math.reduce_variance(preds, 0) + tau**-1
            return tf.math.reduce_mean(preds, 0), tf.math.sqrt(var)
        elif self.model_type == "vae":
            return self.model(test_batch_x, per_pixel=True)
        elif self.model_type == "vae + dropout":
            preds = tf.stack([self.model(test_batch_x, training=True, per_pixel=True) for _ in range(20)])
            return self.gen_mixture_model_preds(preds)
        else:
            preds = tf.stack([self.model.metric_compiled["VAEWrapper"](test_batch_x) for _ in range(20)])
            return self.gen_mixture_model_preds(preds)
        
    def on_epoch_end(self, epoch, logs=None):
        test_batch_x = self.x_test
        test_batch_y = self.y_test
        mu, var = self.gen_preds(test_batch_x)
        nll = self.calculate_nll(mu, var, test_batch_y)
        nll += np.log(self.scale[0,0])
        if nll < self.min_nll:
            self.min_nll = nll
        print("\nnll", nll.numpy())
        
        rmse = self.calculate_rmse(mu, test_batch_y)
        rmse *= self.scale[0,0]
        if rmse < self.min_rmse:
            self.min_rmse = rmse
        print("rmse", rmse.numpy())

    def on_train_end(self, logs=None):
        print("\nmin NLL", self.min_nll)
        print("min RMSE", self.min_rmse)
        return self.min_nll, self.min_rmse
