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
        self.drop_prob = 0.05
        self.lam = 1e-3
        self.l = 0.5
        self.tau = self.l**2 * (1-self.drop_prob) / (2 * self.lam)
        print(self.tau, 1/self.tau)
    
    def calculate_nll(self, y, mu, sigma, reduce=True):
        ax = list(range(1, len(y.shape)))
        logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
        loss = tf.reduce_mean(-logprob, axis=ax)
        return tf.reduce_mean(loss) if reduce else loss 
    
    def calculate_rmse(self, mu, y):
        return tf.math.sqrt(tf.reduce_mean(
            tf.math.square(mu - y),
        ))
    
    def gen_mixture_model_preds(self, outputs):
        mu_batch = tf.stack([output[0] for output in outputs])
        var_batch = tf.stack([output[1] for output in outputs])

        mean_mu = tf.reduce_mean(mu_batch, axis=0)
        var = tf.reduce_mean(var_batch + mu_batch**2, axis=0) - mean_mu**2
        return mean_mu, var

    def gen_preds(self, test_batch_x):
        if self.model_type == "ensemble":
            preds = self.model(test_batch_x)
            return tf.math.reduce_mean(preds, 0), tf.math.reduce_variance(preds, 0)
        elif self.model_type == "ensemble + mve":
            preds = self.model(test_batch_x)
            return self.gen_mixture_model_preds(preds)
        elif self.model_type == "dropout":
            preds = tf.stack([self.model(test_batch_x, training=True) for _ in range(5)])
            var = tf.math.reduce_variance(preds, 0)
            return tf.math.reduce_mean(preds, 0), var + self.tau**-1
        elif self.model_type == "vae":
            return self.model(test_batch_x, per_pixel=False)
        elif self.model_type == "vae + dropout":
            return self.model(test_batch_x, per_pixel=False, training=True, composed=3)
        else:
            
            preds = tf.stack([self.model(test_batch_x, training=True) for _ in range(20)])
            return self.gen_mixture_model_preds(preds)

        
    def on_epoch_end(self, epoch, logs=None):
        test_batch_x = self.x_test
        test_batch_y = self.y_test
        mu, var = self.gen_preds(test_batch_x)
        nll = self.calculate_nll(mu, test_batch_y, tf.sqrt(var))
        nll += np.log(self.scale[0,0])
        
        if nll < self.min_nll:
            self.min_nll = nll
        
        print("\nnll", nll.numpy())
        
        rmse = self.calculate_rmse(mu, test_batch_y)
        rmse *= self.scale[0,0]
        
        if rmse < self.min_rmse:
            self.min_rmse = rmse
        
        print("rmse", rmse.numpy())
