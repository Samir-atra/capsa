import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# base
# BS = 32
# EP = 100
# LR = 1e-4 #5e-5

# # ensemble
# BS = 32
# EP = 256
# LR = 5e-5

# # mve
BS = 32
EP = 40 # 70
LR = 5e-5

# # vae
# BS = 32
# EP = 10
# LR = 1e-4

N_TRAIN = 25000 #24576 #1024 #1024 #8192 #16384
N_VAL = 4096 #2684 #1024 #1024 #1024
NUM_PLOTS = 10

# logs, plots, visualizations, checkpoints, etc. will be saved there
LOGS_PATH = '/home/sadhanalolla/results' #'/data/capsa/depth'
# source code path for logging
SOURCE_PATH = '/home/sadhanalolla/capsa'
# optional, used only in check_saved_weights.py
MODEL_PATH = '/home/sadhanalolla/results/mve/job_07/tensorboard/save/20220825-105053_depth_mve_'