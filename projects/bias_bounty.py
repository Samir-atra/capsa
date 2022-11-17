import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

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


def get_model():
    model = keras.models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Dense(3, activation='relu'))
    return model


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
    y[:, 0] = y[:, 0] / 9
    y[:, 2] = y[:, 0] / 3

    model = EnsembleWrapper(user_model, num_members=3)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss=keras.losses.MeanSquaredError()
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        mode='max',
        save_best_only=True)

    history = model.fit(train_x, train_y, epochs=30, callbacks=[model_checkpoint_callback], verbose=1)

    model.save(os.path.join(checkpoint_filepath, "cnn_model.h5"))

    plot_loss(history, show_plt=False, save=True, path_to_save=os.path.join(checkpoint_filepath, "loss_plot.png"))

    risk_tensor = model(train_x)

    plot_risk_2d(
        train_x,
        train_y,
        risk_tensor,
        model.metric_name,
        show_plt=False,
        save=True,
        path_to_save=os.path.join(checkpoint_filepath, "plot_risk_2d.png")
    )


if __name__ == "__main__":
    main()