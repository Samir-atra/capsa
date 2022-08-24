import os
import glob

from tensorflow import keras
import matplotlib.pyplot as plt

import config
from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss, load_model, totensor_and_normalize

from visualizations import gen_calibration_plot

(x_train, y_train), (x_test, y_test) = load_depth_data() # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)
ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN], shuffle=True)
ds_val = get_normalized_ds(x_train[-config.N_VAL:], y_train[-config.N_VAL:], shuffle=True)
ds_test = get_normalized_ds(x_test, y_test, shuffle=True)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

checkpoints_path = os.path.join(config.MODEL_PATH)
vis_path = os.path.join(config.MODEL_PATH, 'temp_visualizations')
#os.makedirs(vis_path, exist_ok=True)

l = sorted(glob.glob(os.path.join(checkpoints_path, '*.index*')))
l = [i.split('/')[-1].split('.index')[0] for i in l]
weights_names = list(set(l))
# >>> ['ep_128_weights', 'ep_77_weights', 'ep_103_weights', 'ep_52_weights', 'ep_26_weights']
#weights_names = ['model_vloss--2.091_itter-782']
for i, name in enumerate(sorted(weights_names)):
    print("currently loading", name)
    path = f'{checkpoints_path}/{name}'
    model = load_model(path, ds_train)
    #visualize_depth_map(model, ds_train, vis_path, f'{name}_train', plot_uncertainty=False)
    #visualize_depth_map(model, ds_val, vis_path, f'{name}_val', plot_uncertainty=False)
    #visualize_depth_map(model, ds_test, vis_path, f'{name}_test', plot_uncertainty=False)
    #visualize_depth_map(model, ds_ood, vis_path, f'{name}_ood', plot_uncertainty=False)
    gen_calibration_plot(model, name + ".png", ds_test)