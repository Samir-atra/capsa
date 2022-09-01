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

datasets_and_input_shapes = {"boston" : (13, ), "power-plant" : (4, ), "yacht" : (6, ), "concrete" : (8, ), "naval" : (16, ), "energy-efficiency" : (8, ), "kin8nm" : (8, ), "protein" : (9, )}

def train_ensemble(dataset=None):
    if dataset is None:
        dataset = datasets_and_input_shapes.keys()
    for ds in dataset:
        inp_shape = datasets_and_input_shapes[ds]
        (X_train, y_train), (X_test, y_test), y_scale = load_dataset(ds)
        model = get_toy_model(inp_shape)
        wrapped_model = EnsembleWrapper(model, num_members=2)
        wrapped_model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=LossCallback(X_test, y_test, y_scale, "ensemble"))

train_ensemble(["power-plant"])