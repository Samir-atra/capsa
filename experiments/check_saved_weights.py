import os
import glob

from tensorflow import keras
import matplotlib.pyplot as plt

import config
from utils import load_depth_data, load_apollo_data, get_normalized_ds, \
    visualize_depth_map, plot_loss, load_model

(x_train, y_train), (x_test, y_test) = load_depth_data() # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)
ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN])
ds_val = get_normalized_ds(x_train[-config.N_VAL:], y_train[-config.N_VAL:])
# ds_test = get_normalized_ds(x_test, y_test)

_, (x_ood, y_ood) = load_apollo_data() # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

checkpoints_path = os.path.join(config.MODEL_PATH, 'checkpoints')
vis_path = os.path.join(config.MODEL_PATH, 'temp_visualizations')
os.makedirs(vis_path, exist_ok=True)

l = sorted(glob.glob(os.path.join(checkpoints_path, '*.tf*')))
l = [i.split('/')[-1].split('.')[0] for i in l]
weights_names = list(set(l))
# >>> ['ep_128_weights', 'ep_77_weights', 'ep_103_weights', 'ep_52_weights', 'ep_26_weights']

for name in weights_names:
    path = f'{checkpoints_path}/{name}.tf'
    model = load_model(path, ds_train)
    visualize_depth_map(model, ds_train, vis_path, 'train')
    visualize_depth_map(model, ds_val, vis_path, 'val')
    visualize_depth_map(model, ds_test, vis_path, 'test')
    visualize_depth_map(model, ds_ood, vis_path, 'ood')