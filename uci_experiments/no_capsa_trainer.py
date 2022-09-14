import tensorflow as tf
from dataloader import *
from hparams import h_params
from capsa import MVEWrapper


datasets_and_input_shapes = {"boston" : (13, ), "power-plant" : (4, ), "yacht" : (6, ), "concrete" : (8, ), "naval" : (16, ), "energy-efficiency" : (8, ), "kin8nm" : (8, ), "protein" : (9, )}

class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
                tf.print("\n Min RMSE: ", self.model.min_RMSE, "Min NLL: ", self.model.min_NLL)

class Ensemble(tf.keras.Model):
        def __init__(self, input_shape, lr,  y_scale, num_ensembles=5,):
                super(Ensemble, self).__init__()
                self.models = [self.get_toy_model(input_shape) for _ in range(num_ensembles)]
                self.optimizers = [tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(num_ensembles)]
                self.RMSE = tf.keras.metrics.Mean(name='RMSE')
                self.NLL = tf.keras.metrics.Mean(name='NLL')
                self.min_RMSE = float('inf')
                self.min_NLL = float('inf')
                self.y_scale = y_scale
        
        def get_toy_model(self, input_shape=(1,)):
                inputs = tf.keras.Input(shape=input_shape)
                x = tf.keras.layers.Dense(50, "relu")(inputs)
                mu = tf.keras.layers.Dense(1, "linear")(x)
                logsigma = tf.keras.layers.Dense(1, "linear")(x)
                return tf.keras.Model(inputs, [mu, logsigma])

        def nll_loss(self, y, mu, sigma, reduce=True):
                ax = list(range(1, len(y.shape)))

                logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
                loss = tf.reduce_mean(-logprob, axis=ax)
                return tf.reduce_mean(loss) if reduce else loss 

        
        def train_step(self, data):
                losses = []
                rmses = []
                x, y = data
                for (model, optimizer) in zip(self.models, self.optimizers):
                        with tf.GradientTape() as tape:
                                y_pred, logsigma = model(x, training=True)  # Forward pass
                                loss = self.nll_loss(y, y_pred, tf.nn.softplus(logsigma) + 1e-6)
                                losses.append(loss)
                        trainable_vars = model.trainable_variables
                        gradients = tape.gradient(loss, trainable_vars)
                        optimizer.apply_gradients(zip(gradients, trainable_vars))
                        rmses.append(tf.math.sqrt(tf.math.reduce_mean((y_pred - y)**2)))
                self.NLL.update_state(losses)
                self.RMSE.update_state(rmses)
                # Return a dict mapping metric names to current value
                return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
                x, y = data
                preds = []
                all_sigmas = []
                for i, model in enumerate(self.models):
                        y_pred, logsigma = model(x, training=False)
                        preds.append(y_pred)
                        all_sigmas.append(tf.nn.softplus(logsigma) + 1e-6)
                preds = tf.stack(preds)
                all_sigmas = tf.stack(all_sigmas)
                final_mu = tf.math.reduce_mean(preds, axis=0)
                var = tf.reduce_mean(all_sigmas**2 + tf.square(preds), axis=0) - tf.square(final_mu)
                nll = self.nll_loss(y, final_mu, tf.math.sqrt(var))
                nll += np.log(self.y_scale[0,0])
                rmse = tf.sqrt(tf.reduce_mean((y - final_mu)**2))
                rmse *= self.y_scale[0,0]

                if nll < self.min_NLL:
                        self.min_NLL = nll
                
                if rmse < self.min_RMSE:
                        self.min_RMSE = rmse
                
                self.NLL.update_state([nll])
                self.RMSE.update_state([rmse])
                return {m.name: m.result() for m in self.metrics}
                

ds = "boston"
inp_shape = datasets_and_input_shapes[ds]
lr = h_params[ds]['learning_rate']
(X_train, y_train), (X_test, y_test), y_scale = load_dataset(ds)
batch_size = h_params[ds]['batch_size']
ensemble = Ensemble(inp_shape, lr, y_scale)
ensemble.compile('adam', tf.keras.losses.MeanSquaredError(), run_eagerly=True)

ensemble.fit(X_train, y_train, epochs=40, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[LoggingCallback()])
