''' 
used only for the demo
abstracts away individual metric wrapper names
'''

import tensorflow as tf
from tensorflow import keras

from losses import MSE
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from utils import notebook_select_gpu, load_depth_data, load_apollo_data, get_normalized_ds

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

notebook_select_gpu(0)