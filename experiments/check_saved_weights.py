import os
import glob

from tensorflow import keras
import matplotlib.pyplot as plt

import config
from utils import load_depth_data, load_apollo_data, totensor_and_normalize, \
    visualize_depth_map, visualize_depth_map_uncertainty, plot_multiple, \
    plot_loss, load_model

(x_train, y_train), (x_test, y_test) = load_depth_data()
x_train, y_train = totensor_and_normalize(x_train, y_train)

_, (x_test_ood, y_test_ood) = load_apollo_data()
x_test_ood, y_test_ood = totensor_and_normalize(x_test_ood, y_test_ood)

# todo-med: move these lines somewhere else
checkpoints_path = '/home/iaroslavelistratov/results/job_00/checkpoints'
vis_path = '/home/iaroslavelistratov/results/job_00/temp_visualizations'
os.makedirs(vis_path, exist_ok=True)

l = sorted(glob.glob(os.path.join(checkpoints_path, '*.tf*')))
l = [i.split('/')[-1].split('.')[0] for i in l]
weights_names = list(set(l))
# >>> ['ep_128_weights', 'ep_77_weights', 'ep_103_weights', 'ep_52_weights', 'ep_26_weights']

for name in weights_names:
    path = f'{checkpoints_path}/{name}.tf'
    model = load_model(path, x_train, y_train)
    plot_multiple(model, x_train, y_train, x_test_ood, y_test_ood, vis_path, prefix=f'{name}_')