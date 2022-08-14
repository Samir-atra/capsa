import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# base
BS = 32
EP = 100
LR = 5e-5

# # ensebmle
# BS = 32
# EP = 256
# LR = 5e-5

# # mve
# BS = 32 # 8
# EP = 48 # 256
# LR = 5e-5

N_TRAIN = 64 #16384
N_VAL = 64
NUM_PLOTS = 10