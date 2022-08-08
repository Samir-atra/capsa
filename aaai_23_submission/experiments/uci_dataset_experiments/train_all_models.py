from capsa import Wrapper, HistogramWrapper, VAEWrapper, EnsembleWrapper, DropoutWrapper, MVEWrapper
from custom_training_loop import Trainer
import tensorflow as tf
from dataloader import load_dataset
from hparams import h_params
import keras_tuner
import numpy as np
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
dataset = "protein"
num_trials = 5
num_epochs = None
input_shape = datasets_and_input_shapes[dataset]
batch_size = 32
lr = h_params[dataset]['learning_rate']

mve_rmse = []
mve_nll = []

dropout_rmse = []
dropout_nll = []

ensemble_rmse = []
ensemble_nll = []

dropout_ensemble_rmse = []
dropout_ensemble_nll = []

(X_train, y_train), (X_test, y_test), y_scale = load_dataset(dataset)



for trial in range(num_trials):
    
    # MVE and Dropout
    print(dataset)
    print(f"MVE and Dropout trial {trial}")
    model = MVEWrapper(get_toy_model(input_shape))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),)
    trainer = Trainer(model, dataset, batch_size=batch_size)
    model, min_rmse, min_nll = trainer.train()
    mve_nll.append(min_nll)
    mve_rmse.append(min_rmse)
    print("***")

    # Pure Dropout
    print(f"Dropout trial {trial}")
    model = DropoutWrapper(get_toy_model(input_shape))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),)
    trainer = Trainer(model, dataset, batch_size=batch_size)
    model, min_rmse, min_nll = trainer.train()
    dropout_nll.append(min_nll)
    dropout_rmse.append(min_rmse)
    print("***")
    
    
    # Pure Ensemble
    print(f"Ensemble trial {trial}")
    model = EnsembleWrapper(get_toy_model(input_shape, dropout_rate=0.0), metric_wrapper=MVEWrapper)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),)
    trainer = Trainer(model, dataset, batch_size=batch_size)
    model, min_rmse, min_nll = trainer.train()
    ensemble_nll.append(min_nll)
    ensemble_rmse.append(min_rmse)

    # Ensemble + Dropout
    print(f"Ensemble + Dropout trial {trial}")
    model = EnsembleWrapper(get_toy_model(input_shape), metric_wrapper=MVEWrapper)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),)
    trainer = Trainer(model, dataset, batch_size=batch_size, has_dropout=True)
    model, min_rmse, min_nll = trainer.train()
    dropout_ensemble_nll.append(min_nll)
    dropout_ensemble_rmse.append(min_rmse)
    
print("MVE + Dropout \n RMSE: ", sum(mve_rmse)/len(mve_rmse), "+/-", np.std(np.array(mve_rmse)), "\n", sum(mve_nll)/len(mve_nll), "+/-", np.std(np.array(mve_nll)),)
print("Pure Dropout \n RMSE: ", sum(dropout_rmse)/len(dropout_rmse), "+/-", np.std(np.array(dropout_rmse)), "\n", sum(dropout_nll)/len(dropout_nll), "+/-", np.std(np.array(dropout_nll)),)
print("Pure Ensemble \n RMSE: ", sum(ensemble_rmse)/len(ensemble_rmse), "+/-", np.std(np.array(ensemble_rmse)), "\n", sum(ensemble_nll)/len(ensemble_nll), "+/-", np.std(np.array(ensemble_nll)),)
print("Dropout + Ensemble \n RMSE: ", sum(dropout_ensemble_rmse)/len(dropout_ensemble_rmse), "+/-", np.std(np.array(dropout_ensemble_rmse)), "\n", sum(dropout_ensemble_nll)/len(dropout_ensemble_nll), "+/-", np.std(np.array(dropout_ensemble_nll)),)
