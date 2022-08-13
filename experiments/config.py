import os
# tf logging - don't print INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# base
BS = 32
EP = 256
LR = 5e-5

# # ensebmle
# BS = 32
# EP = 256
# LR = 5e-5

# # mve
# BS = 32 # 8
# EP = 48 # 256
# LR = 5e-5

N_SAMPLES = 16384
NUM_PLOTS = 10