import tensorflow as tf
from dataloader import *
from hparams import h_params
from callbacks import LossCallback
from capsa import Wrapper, EnsembleWrapper, VAEWrapper, MVEWrapper, DropoutWrapper
import time

reg = 1e-3
drop_prob = 0.1
lam = 1e-3
l = 0.2
tau = l**2 * (1-drop_prob) / (2 * lam)

def calculate_nll(y, mu, sigma, reduce=True):
        ax = list(range(1, len(y.shape)))
        logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
        loss = tf.reduce_mean(-logprob, axis=ax)
        return tf.reduce_mean(loss) if reduce else loss 
    
def calculate_rmse(mu, y):
    return tf.math.sqrt(tf.reduce_mean(
        tf.math.square(mu - y),
    ))

def get_toy_model(input_shape=(1,), dropout_rate=0.0):
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

def get_decoder(input_shape, latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(latent_dim, )),
            tf.keras.layers.Dense(10, "relu"),
            tf.keras.layers.Dense(50, "relu"),
            tf.keras.layers.Dense(input_shape[0], None),
        ]
    )

datasets_and_input_shapes = {"boston" : (13, ), "power-plant" : (4, ), "yacht" : (6, ), "concrete" : (8, ), "naval" : (16, ), "energy-efficiency" : (8, ), "kin8nm" : (8, ), "protein" : (9, )}

def train(dataset=None, trials=5):
    nll = {}
    rmse = {}
    time_taken = {}
    if dataset is None:
        dataset = datasets_and_input_shapes.keys()
    for ds in dataset:
        nll[ds] = {}
        rmse[ds] = {}
        nlls = []
        rmses = []
        times = []
        for t in range(trials):
            tic = time.time()
            inp_shape = datasets_and_input_shapes[ds]
            (X_train, y_train), (X_test, y_test), y_scale = load_dataset(ds)

            latent_dim = inp_shape[0] // 2
            decoder = get_decoder(inp_shape, latent_dim)
            vae = VAEWrapper(get_toy_model(inp_shape), decoder=decoder, bias=False, latent_dim=latent_dim, kl_weight=h_params[ds]["kl-weight"])
            dropout = get_toy_model(inp_shape, dropout_rate=0.1)
            lr = h_params[ds]['learning_rate']
            batch_size = h_params[ds]['batch_size']


            vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.MeanSquaredError())
            dropout.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.MeanSquaredError())

            #loss_c = LossCallback(X_test, y_test, y_scale, model_type, dropout)

            vae.fit(X_train, y_train, epochs=40, batch_size=batch_size, validation_data=(X_test, y_test))
            dropout.fit(X_train, y_train, epochs=40, batch_size=batch_size, validation_data=(X_test, y_test))
            
            preds_dropout = tf.stack([dropout(X_test, training=True) for _ in range(5)])
            var_dropout = tf.math.reduce_variance(preds_dropout, 0) + tau**-1

            preds_vae, var_vae = vae(X_test, per_pixel=False)
            
            min_nll = float('inf')
            min_rmse = float('inf')
            min_nll_weight = -1
            min_rmse_weight = -1
            for i in range(100):
                weight = i/100
                total_var = (1 - weight) * var_dropout + weight * var_vae
                total_pred = (1 - weight) * preds_dropout + weight * preds_vae
                nll_run = calculate_nll(y_test, total_pred, tf.sqrt(total_var)) + np.log(y_scale[0,0])
                rmse_run = calculate_rmse(total_pred, y_test) * y_scale[0,0]
                if nll_run < min_nll:
                    min_nll = nll_run
                    min_nll_weight = weight
                if rmse_run < min_rmse:
                    min_rmse = rmse_run
                    min_rmse_weight = weight
            
            print("ds", ds, "trial", t, "nll", min_nll.numpy(), "rmse", min_rmse.numpy(), "nll weight", min_nll_weight, "min_rmse_weight", min_rmse_weight)
            pure_vae_nll = calculate_nll(y_test, preds_vae, tf.sqrt(var_vae))+ np.log(y_scale[0,0])
            pure_vae_rmse = calculate_rmse(y_test, preds_vae)* y_scale[0,0]
            print("pure VAE NLL", pure_vae_nll, "pure VAE RMSE", pure_vae_rmse)
            pure_vae_nll = calculate_nll(y_test, preds_dropout, tf.sqrt(var_dropout)) + np.log(y_scale[0,0])
            pure_vae_rmse = calculate_rmse(y_test, preds_dropout)* y_scale[0,0]
            print("pure Dropout NLL", pure_vae_nll, "pure Dropout RMSE", pure_vae_rmse)

            toc = time.time()
            times.append(toc - tic)
            nlls.append(min_nll)
            rmses.append(min_rmse)
        nll[ds] = {"mean" : np.mean(nlls), "std": np.std(nlls)}
        rmse[ds] = {"mean" : np.mean(rmses), "std": np.std(rmses)}
        time_taken[ds] = {"mean" : np.mean(times), "std" : np.std(times)}
    return nll, rmse, time_taken

all_nll= {}
all_rmse = {}
all_times = {}
nll, rmse, times = train()
print("NLL", nll)
print("RMSE", rmse)
print("times", times)
'''
log = False
if log:
    print("***NLL****")
    with open("ensemble_mve_40.txt", "w") as f:
        for model_type, v in all_nll.items():
            for dataset, results in v.items():
                    f.write(model_type + " " + dataset + " " + str(results["mean"]) + " +/- " + str(results["std"]) + "\n") 

    with open("ensemble_mve_rmse_40.txt", "w") as f:
        for model_type, v in all_rmse.items():
            for dataset, results in v.items():
                    f.write(model_type + " " + dataset + " " + str(results["mean"]) + " +/- " + str(results["std"]) + "\n") 

    with open("ensemble_mve_times_40.txt", "w") as f:
        for model_type, v in all_times.items():
            for dataset, results in v.items():
                    f.write(model_type + " " + dataset + " " + str(results["mean"]) + " +/- " + str(results["std"]) + "\n") 

print("NLL", all_nll)
print("RMSE", all_rmse)
'''