''' 
used only for the demo
abstracts away individual metric wrapper names
'''

import tensorflow as tf
from tensorflow import keras

from losses import MSE
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from utils import load_depth_data, load_apollo_data, get_normalized_ds, visualize_depth_map

import notebooks.configs.demo as config

def AleatoricWrapper(user_model):
    model = MVEWrapper(user_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )
    return model

def EpistemicWrapper(user_model):
    model = EnsembleWrapper(user_model, num_members=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )
    return model

def vis_depth_map(model, vis_path, ds_train, ds_test=None, ds_ood=None, plot_uncertainty=True):
    tf.autograph.set_verbosity(2)
    visualize_depth_map(model, ds_train, vis_path, title='Train Dataset', plot_uncertainty=plot_uncertainty)
    if ds_test != None:
        visualize_depth_map(model, ds_test, vis_path, title='Test Dataset', plot_uncertainty=plot_uncertainty)
    if ds_ood != None:
        visualize_depth_map(model, ds_ood, vis_path, title='O.O.D Dataset', plot_uncertainty=plot_uncertainty)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'