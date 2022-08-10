import numpy as np
import matplotlib.pyplot as plt

def visualize_depth_map(x, y, model=None):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    pred = model.predict(x)
    fig, ax = plt.subplots(6, 3, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=cmap)
        ax[i, 2].imshow(pred[i, :, :, 0], cmap=cmap)

    plt.savefig('/home/iaroslavelistratov/capsa/experiments/artifacts/heatmap.png')
    plt.show()

def plot_loss(history):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='upper right')

    plt.savefig('/home/iaroslavelistratov/capsa/experiments/artifacts/loss.png')
    plt.show()