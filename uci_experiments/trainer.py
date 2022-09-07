import tensorflow as tf
from dataloader import *
from hparams import h_params
from callbacks import LossCallback
from capsa import Wrapper, EnsembleWrapper, VAEWrapper, MVEWrapper 

def get_toy_model(input_shape=(1,), dropout_rate=0.1):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(50, "relu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(1, None),
        ]
    )

def get_decoder(input_shape, latent_dim, dropout_rate=0.1):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(latent_dim, )),
            #tf.keras.layers.Dense(50, "relu"),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(50, "linear"),
            tf.keras.layers.Dense(input_shape[0], None),
        ]
    )

datasets_and_input_shapes = {"boston" : (13, ), "power-plant" : (4, ), "yacht" : (6, ), "concrete" : (8, ), "naval" : (16, ), "energy-efficiency" : (8, ), "kin8nm" : (8, ), "protein" : (9, )}

def get_model(model_type, inp_shape, dataset):
    model = get_toy_model(inp_shape, dropout_rate=0.0)
    latent_dim = inp_shape[0] // 2
    if model_type == "ensemble":
        return EnsembleWrapper(model, num_members=2)
    elif model_type == "ensemble + mve":
        return EnsembleWrapper(model, metric_wrapper=MVEWrapper)
    elif model_type == "dropout":
        return get_toy_model(inp_shape, dropout_rate=0.2)
    elif model_type == "vae":
        decoder = get_decoder(inp_shape, latent_dim, dropout_rate=0.0)
        return VAEWrapper(model, decoder=decoder, bias=False, latent_dim=latent_dim, kl_weight=h_params[dataset]["kl-weight"])
    elif model_type == "vae + dropout":
        return VAEWrapper(get_toy_model(inp_shape, dropout_rate=0.1), bias=False, latent_dim=latent_dim)
    else:
        model = get_toy_model(inp_shape, dropout_rate=0.1)
        return Wrapper(model, metrics=[VAEWrapper(model, bias=False), MVEWrapper])

def train(model_type, dataset=None):
    if dataset is None:
        dataset = datasets_and_input_shapes.keys()
    for ds in dataset:
        inp_shape = datasets_and_input_shapes[ds]
        (X_train, y_train), (X_test, y_test), y_scale = load_dataset(ds)
        wrapped_model = get_model(model_type, inp_shape, ds)
        lr = h_params[ds]['learning_rate']
        batch_size = h_params[ds]['batch_size']
        wrapped_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.MeanSquaredError(),)
        wrapped_model.fit(X_train, y_train, epochs=80, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=LossCallback(X_test, y_test, y_scale, model_type))

train("vae", ["kin8nm"])