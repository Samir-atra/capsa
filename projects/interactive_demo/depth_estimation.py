import sys
import tensorflow as tf

from models import unet
from wrap import AleatoricWrapper, EpistemicWrapper, vis_depth_map
from utils import get_datasets, notebook_select_gpu, load_model, \
    select_best_checkpoint, gen_ood_comparison

from run_utils import setup

ALEATORIC_PATH = '/home/elahehahmadi/themis-ai/capsa/projects/interactive_demo/models/aleatoric'
EPISTEMIC_PATH = '/home/elahehahmadi/themis-ai/capsa/projects/interactive_demo/models/epistemic'

notebook_select_gpu(0)

ds_train = get_datasets(only_train=True)


path_aleatoric, _ = select_best_checkpoint(ALEATORIC_PATH)
model_name_aleatoric = 'mve'
print("Aleatoric: path: {}, model_name: {}".format(path_aleatoric, model_name_aleatoric))
trained_aleatoric = load_model(path_aleatoric, model_name_aleatoric, ds_train, quiet=False)

path_epistemic, _ = select_best_checkpoint(EPISTEMIC_PATH)
model_name_epistemic = 'ensemble'
print("Epistemic: path: {}, model_name: {}".format(path_epistemic, model_name_epistemic))
trained_epistemic = load_model(path_epistemic, model_name_epistemic, ds_train, quiet=False)

print("Done")