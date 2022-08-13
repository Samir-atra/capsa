import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_depth_data():
    train = h5py.File("/home/iaroslavelistratov/data/depth_train.h5", "r")
    test = h5py.File("/home/iaroslavelistratov/data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def load_apollo_data():
    test = h5py.File("/home/iaroslavelistratov/data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])

def visualize_depth_map(x, y, pred, visualizations_path, name='map.png'):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(6, 3, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{visualizations_path}/{name}')
    # plt.show()
    plt.close()

# hacky ugly way - should reuse visualize_depth_map
def visualize_depth_map_uncertainty(x, y, pred, uncertain, visualizations_path, name='map.png'):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    fig, ax = plt.subplots(6, 4, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)
        ax[i, 3].imshow(uncertain[i, :, :, 0], cmap=cmap)

    plt.savefig(f'{visualizations_path}/{name}')
    # plt.show()
    plt.close()

def plot_loss(history, plots_path, name='loss.png'):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='upper right')

    plt.savefig(f'{plots_path}/{name}')
    # plt.show()
    plt.close()