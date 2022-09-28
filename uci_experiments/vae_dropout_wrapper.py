import tensorflow as tf
from tensorflow import keras
from capsa import VAEWrapper

class VAEDropoutWrapper(keras.Model):
    """ This is a wrapper!

    Args:
        base_model (model): a model

    """
    def calculate_nll(self, y, mu, sigma, reduce=True):
        ax = list(range(1, len(y.shape)))
        logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
        loss = tf.reduce_mean(-logprob, axis=ax)
        return tf.reduce_mean(loss) if reduce else loss 
    
    def calculate_rmse(self, mu, y):
        return tf.math.sqrt(tf.reduce_mean(
            tf.math.square(mu - y),
        ))
    
    def get_toy_model(self, input_shape=(1,), dropout_rate=0.0):
        reg = 1e-3
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Dense(50, "relu", kernel_regularizer=tf.keras.regularizers.L2(reg)),
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Dense(10, "relu"),
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Dense(1, None, kernel_regularizer=tf.keras.regularizers.L2(reg)),
            ]
        )

    def __init__(self, kl_weight, dropout_probability, input_shape, decoder, latent_dim, y_scale):
        super(VAEDropoutWrapper, self).__init__()

        self.dropout_model = self.get_toy_model(input_shape, dropout_probability)
        self.vae_model = VAEWrapper(self.get_toy_model(input_shape), decoder=decoder, bias=False, latent_dim=latent_dim, kl_weight=kl_weight)

        self.drop_prob = dropout_probability
        self.lam = 1e-3
        self.l = 0.2
        self.tau = self.l**2 * (1-self.drop_prob) / (2 * self.lam)
        self.y_scale = y_scale

    def compile(self, *args, **kwargs):
        """ Compile the wrapper

        Args:
            optimizer (optimizer): the optimizer

        """
        super().compile(*args, **kwargs)
        self.dropout_model.compile(*args, **kwargs)
        self.vae_model.compile(*args, **kwargs)

    @tf.function
    def train_step(self, data):
        drop_metrics = self.dropout_model.train_step(data)
        vae_metrics = self.vae_model.train_step(data)
        return drop_metrics


    def call(self, x, training=False, return_risk=True):
        preds_dropout = tf.stack([self.dropout_model(X_test, training=True) for _ in range(5)])
        var_dropout = tf.math.reduce_variance(preds_dropout, 0) + self.tau**-1

        preds_vae, var_vae = self.vae_model(X_test, per_pixel=False)
        
        min_nll = float('inf')
        min_rmse = float('inf')
        min_nll_weight = -1
        min_rmse_weight = -1
        for i in range(100):
            weight = i/100
            total_var = (1 - weight) * var_dropout + weight * var_vae
            total_pred = (1 - weight) * preds_dropout + weight * preds_vae
            nll_run = self.calculate_nll(y_test, total_pred, tf.sqrt(total_var)) + np.log(self.y_scale[0,0])
            rmse_run = self.calculate_rmse(total_pred, y_test) * self.y_scale[0,0]
            if nll_run < min_nll:
                min_nll = nll_run
                min_nll_weight = weight
            if rmse_run < min_rmse:
                min_rmse = rmse_run
                min_rmse_weight = weight
        return min_rmse, min_nll

class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test):
        super(LossCallback, self).__init__()
        self.x_test = x_test
        self.nll = float('inf')
        self.rmse = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        rmse, nll = self.model(self.x_test)
        print("\n", nll.numpy(), rmse.numpy())  
        if nll < self.nll:
            self.nll = nll
        if rmse < self.rmse:
            self.rmse = rmse

    def on_train_end(self, logs=None):
        print("NLL", self.nll.numpy(), "RMSE", self.rmse.numpy()) 

def get_decoder(input_shape, latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(latent_dim, )),
            tf.keras.layers.Dense(10, "relu"),
            tf.keras.layers.Dense(50, "relu"),
            tf.keras.layers.Dense(input_shape[0], None),
        ]
    )


from hparams import h_params
from dataloader import *
datasets_and_input_shapes = {"boston" : (13, ), "power-plant" : (4, ), "yacht" : (6, ), "concrete" : (8, ), "naval" : (16, ), "energy-efficiency" : (8, ), "kin8nm" : (8, ), "protein" : (9, )}

for ds in ["yacht"]:
    nlls = []
    rmses = []
    for _ in range(10):
        inp_shape = datasets_and_input_shapes[ds]
        (X_train, y_train), (X_test, y_test), y_scale = load_dataset(ds)

        latent_dim = inp_shape[0] // 2
        decoder = get_decoder(inp_shape, latent_dim)
        model = VAEDropoutWrapper(input_shape=inp_shape, decoder=decoder, dropout_probability=0.1, latent_dim=latent_dim, kl_weight=h_params[ds]["kl-weight"], y_scale=y_scale)
        lr = h_params[ds]['learning_rate']
        batch_size = h_params[ds]['batch_size']

        loss_c = LossCallback(X_test)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)
        model.fit(X_train, y_train, epochs=40, batch_size=batch_size, callbacks=[loss_c], verbose=0)

        nlls.append(loss_c.nll)
        rmses.append(loss_c.rmse)
    
    print(ds, "NLL: ", tf.math.reduce_mean(nlls).numpy(), "+/-", tf.math.reduce_std(nlls).numpy())
    print(ds, "RMSEs: ", tf.math.reduce_mean(rmses).numpy(), "+/-", tf.math.reduce_std(rmses).numpy())