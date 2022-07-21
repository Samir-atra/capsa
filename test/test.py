import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from capsa.utils import get_user_model, plt_vspan, plot_results, plot_loss, get_preds_names
from data import get_data_v1, get_data_v2

def plot_aleatoric(x_val, y_val, y_pred, variance, label):
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, variance, s=.5, label=label)
    plt_vspan()
    plt.legend()
    plt.show()

def test_regression(use_case=None):

    their_model = get_user_model()
    ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

    # user can interact with a MetricWrapper directly
    if use_case == 1:
        model = MVEWrapper(their_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-3),
            loss=keras.losses.MeanSquaredError(),
        )
        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        y_pred, variance = model(x_val)

    # user can interact with a MetricWrapper through Wrapper (what we call a "controller wrapper")
    elif use_case == 2:
        model = Wrapper(their_model, metrics=[MVEWrapper])

        model.compile(
            # user needs to specify optim and loss for each metric
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            # note reduction needs to be NONE, model reduces to mean under the hood
            loss=[keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        metrics_out = model(x_val)
        y_pred, variance = metrics_out[0]

    preds_names = get_preds_names(history)
    plot_aleatoric(x_val, y_val, y_pred, variance, preds_names[0])

def test_regression_predict():

    their_model = get_user_model()
    ds_train, ds_val, _, _ = get_data_v2(batch_size=256)

    model = Wrapper(their_model, metrics=[MVEWrapper])

    model.compile(
        # user needs to specify optim and loss for each metric
        optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
        # note reduction needs to be NONE, model reduces to mean under the hood
        loss=[keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
    )

    history = model.fit(ds_train, epochs=30)
    plot_loss(history)

    # predict cats batch output to a single tensor under the hood
    # metrics_out is a list (of len 1) of tuples (x_val_batch, y_val_batch)
    metrics_out = model.predict(ds_val)
    y_pred, variance = metrics_out[0]

    # need this for plotting -- cat all batches
    # list(ds_val) is a list (of len num of batches) of tuples (x_val_batch, y_val_batch)
    cat = np.concatenate(list(ds_val), 1) # (2, 2304, 1)
    x_val, y_val = cat[0], cat[1]

    preds_names = get_preds_names(history)
    plot_aleatoric(x_val, y_val, y_pred, variance, preds_names[0])


def test_ensemble(use_case):

    if use_case == 1:

        their_model = get_user_model()
        x, y, x_val, y_val = get_data_v1()

        model = EnsembleWrapper(their_model, num_members=5)
        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
            loss=[keras.losses.MeanSquaredError()],
            # metrics=[[
            #     # keras.metrics.MeanSquaredError(name='mse'),
            #     keras.metrics.CosineSimilarity(name='cos'),
            # ]],
        )

        history = model.fit(x, y, epochs=100)
        plot_loss(history)

        outs = model(x_val)
        preds_names = get_preds_names(history)

        plt.plot(x_val, y_val, 'r-', label="ground truth")
        plt.scatter(x, y, label="train data")
        for i, out in enumerate(outs):
            plt.plot(x_val, out, label=preds_names[i])
        plt.legend(loc='upper left')
        plt.show()

    elif use_case == 2:

        their_model = get_user_model()
        ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

        model = EnsembleWrapper(their_model, MVEWrapper, num_members=5)

        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
            # metrics=[[
            #     # keras.metrics.MeanSquaredError(name='mse'),
            #     keras.metrics.CosineSimilarity(name='cos'),
            # ]],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        outs = model(x_val)
        preds_names = get_preds_names(history)

        fig, axs = plt.subplots(2)
        axs[0].scatter(x_val, y_val, s=.5, label="gt")
        plt_vspan()
        for i, out in enumerate(outs):
            y_pred, variance = out
            axs[0].scatter(x_val, y_pred, s=.5)
            axs[1].scatter(x_val, variance, s=.5, label=preds_names[i])

        plt.ylim([0, 1])
        plt_vspan()
        plt.legend(loc='upper left')
        plt.show()

    elif use_case == 3:

        their_model = get_user_model()
        ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

        model = Wrapper(
            their_model, 
            metrics=[
                # VAEWrapper,
                EnsembleWrapper(their_model, MVEWrapper, is_standalone=False, num_members=5),
            ]
        )

        model.compile(
            optimizer=[
                # [keras.optimizers.Adam(learning_rate=2e-3)],
                [keras.optimizers.Adam(learning_rate=2e-3)],
            ],
            loss=[
                # [keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
                [keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
            ],
            # metrics=[
            #     # [keras.metrics.MeanSquaredError(name='mse')],
            #     [keras.metrics.CosineSimilarity(name='cos')],
            # ],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)


        metrics_out = model(x_val)
        preds_names = get_preds_names(history)
        mve_ensemble = metrics_out[0] # ['EnsembleWrapper']

        # _, epistemic = metrics_out['VAEWrapper']
        # epistemic_normalized = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic))

        fig, axs = plt.subplots(2)
        axs[0].scatter(x_val, y_val, s=.5, label="gt")
        plt_vspan()
        for i in range(len(mve_ensemble)):
            y_hat2, variance = mve_ensemble[i]
            axs[0].scatter(x_val, y_hat2, s=.5)
            axs[1].scatter(x_val, variance, s=.5, label=preds_names[i])
            # axs[1].scatter(x_val, epistemic_normalized, s=.5, label="epistemic_{i+1}")

        # plt.ylim([0, 1])
        plt_vspan()
        plt.legend(loc='upper left')
        plt.show()


def test_exceptions(use_case):

    x, y, x_val, y_val = get_data_v1()

    # get_config not implemented
    if use_case == 1:

        class CustomModel(keras.Model):
            def __init__(self, hidden_units):
                super(CustomModel, self).__init__()
                self.hidden_units = hidden_units
                self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

            def call(self, inputs):
                x = inputs
                for layer in self.dense_layers:
                    x = layer(x)
                return x

            # def get_config(self):
            #     return {"hidden_units": self.hidden_units}

            # @classmethod
            # def from_config(cls, config):
            #     return cls(**config)

        try:
            their_model = CustomModel([16, 32, 16])

            model = EnsembleWrapper(their_model, num_members=5)
            model.compile(
                optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
                loss=[keras.losses.MeanSquaredError()],
            )
            history = model.fit(x, y, epochs=100)

        except AssertionError:
            print(f'test_exceptions_{use_case} worked!')

    elif use_case == 2:

        try:
            class SquareNums():
                def __init__(self):
                    self.config = {1}

                def get_config(self):
                    return self.config

                def call(self, x):
                    return  x**2

            their_model = SquareNums()

            model = EnsembleWrapper(their_model, num_members=5)
            model.compile(
                optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
                loss=[keras.losses.MeanSquaredError()],
            )
            history = model.fit(x, y, epochs=100)

        except Exception:
            print(f'test_exceptions_{use_case} worked!')

test_regression(1)
test_regression(2)
test_regression_predict()

test_ensemble(1)
test_ensemble(2)
test_ensemble(3)

test_exceptions(1)
test_exceptions(2)