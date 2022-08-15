import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# base
# BS = 32
# EP = 100
# LR = 1e-4 #5e-5

# # ensebmle
# BS = 32
# EP = 256
# LR = 5e-5

# # mve
# BS = 32 # 8
# EP = 48 # 256
# LR = 5e-5

# # vae
BS = 32 # 8
EP = 10 # 256
LR = 1e-4

N_TRAIN = 1024 #8192 #16384
N_VAL = 1024 #1024
NUM_PLOTS = 10

# logs, plots, visualizations, checkpoints, etc. will be saved there
LOGS_PATH = '/home/sadhanalolla/results' #'/data/capsa/depth'
# source code path for logging
SOURCE_PATH = '/home/sadhanalolla/capsa'
# optional, used only in check_saved_weights.py
#MODEL_PATH = '/home/iaroslavelistratov/results/mve/job_00'