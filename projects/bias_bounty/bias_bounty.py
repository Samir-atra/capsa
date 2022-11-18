import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

from capsa import ControllerWrapper, EnsembleWrapper, MVEWrapper, VAEWrapper

from capsa.utils import (
    plot_loss,
    get_preds_names,
    plot_risk_2d,
    plot_epistemic_2d,
)

from bb_data import load_split_data


def get_bias_bounty_data(name="split"):
    if name == "split":
        x_train, y_train = load_split_data()
        
    return x_train, y_train


def get_model(model_name="3_layer"):
    if model_name == "4_layer":
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(256, 256, 3)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.Conv2D(16, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(3),
            ]
        )
    if model_name == "3_layer":
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(256, 256, 3)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.Conv2D(16, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(3, activation="relu"),
            ]
        )


def get_ordinal_loss_f():
    # Labels
    # y[:, 0] --> Skin tone, value range = [0, 9]
    # y[:, 1] --> Gender, value range = [0, 1]
    # y[:, 2] --> Age category, value range = [0, 3]
    return


def main():
    checkpoint_filepath = "/home/elahehahmadi/assets/ckpts/bias_bounty"
    user_model = get_model()

    train_x, train_y = load_split_data()

    # normalize y
    train_y[:, 0] = train_y[:, 0] / 9
    train_y[:, 2] = train_y[:, 0] / 3

    print("train_y shape: {}".format(train_y.shape))

    model = EnsembleWrapper(user_model, num_members=3)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss=keras.losses.MeanSquaredError()
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq="epoch",
        mode='max')

    random_pred = model.call(train_x[1:2, :, :, :])

    print("random_pred shape: {}".format(random_pred.shape))

    history = model.fit(train_x, train_y, batch_size=32, epochs=30, callbacks=[model_checkpoint_callback], verbose=1)

    plot_loss(history, show_plt=False, save=True, path_to_save=os.path.join(checkpoint_filepath, "loss_plot.png"))

    try:
        model.save(os.path.join(checkpoint_filepath, "cnn_model"))
    except:
        print("An exception occurred when saving the model")


if __name__ == "__main__":
    main()