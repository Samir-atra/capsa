### used for demo


import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BS = 32
EP = 70 # 20
LR = 5e-5

N_TRAIN = 25000
N_VAL = 27250 - N_TRAIN
NUM_PLOTS = 10

# source code path for logging
SOURCE_PATH = '/home/iaroslavelistratov/capsa'

# logs, plots, visualizations, checkpoints, etc. will be saved there
LOGS_PATH = '/home/iaroslavelistratov/results'
# load model from
# MODEL_PATH = '/data/capsa/depth/mve/20220827-211807'

# logs, plots, visualizations, checkpoints, etc. will be saved there
# LOGS_PATH = '/data/capsa/depth/results'
# load model from
BASE_PATH = '/data/capsa/depth/results/base/20220829-135832'
ALEATORIC_PATH = '/data/capsa/depth/results/mve/20220827-211807'
EPISTEMIC_PATH = '/data/capsa/depth/results/ensemble/20220829-172414-3members'


timedelta = 3 # 3 for Iaro, 0 for ET