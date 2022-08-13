import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

N_SAMPLES = 64 #8192 # 256
BS = 32 # 8
EP = 48 # 256
LR = 5e-5 # 5e-5

NUM_PLOTS = 10