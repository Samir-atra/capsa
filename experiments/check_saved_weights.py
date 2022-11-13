import os
import glob

from tensorflow import keras
import matplotlib.pyplot as plt

import config
from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss, select_best_checkpoint, load_model

(x_train, y_train), (x_test, y_test) = load_depth_data() # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)
ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN], shuffle=True)
ds_val = get_normalized_ds(x_train[config.N_TRAIN:], y_train[config.N_TRAIN:], shuffle=True)
ds_test = get_normalized_ds(x_test, y_test, shuffle=True)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

vis_path = os.path.join(config.MODEL_PATH, 'visualizations_loaded')
os.makedirs(vis_path, exist_ok=True)

path, model_name = select_best_checkpoint()
# https://github.com/tensorflow/models/issues/2676#issuecomment-444242182
model = load_model(path, model_name, ds_train)

# todo-med: fix
name = '-'
plot_uncertainty = True if model_name is not 'base' else False
visualize_depth_map(model, ds_train, vis_path, f'{name}_train', plot_uncertainty)
visualize_depth_map(model, ds_val, vis_path, f'{name}_val', plot_uncertainty)
visualize_depth_map(model, ds_test, vis_path, f'{name}_test', plot_uncertainty)
visualize_depth_map(model, ds_ood, vis_path, f'{name}_ood', plot_uncertainty)