# https://github.com/IaroslavElistratov/gnn-motion-forecasting/blob/master/core/run_utils.py

import time
import datetime

import os

import logging
import shutil
# import yaml

# def _read_hyperparameters(dir, gpu_worker_id):
#     '''
#     Reads hyperparameters from the data-server. It's reasonable to do so,
#     as the image doesn't need to be rebuild each time we change hyperparameters.

#     For hyperparameter tuning, use environment variables `DRL_GPU_WORKER_ID` to execute the exact same
#     functions (for each job in the jobarray), except for the hyperparameters specified in the config.yaml
#     '''

#     with open(dir, 'r') as p: # todo-low: use os.join everythere
#         config = yaml.safe_load(p)[gpu_worker_id]

#     return config

def _create_folders(target_dir='/home/iaroslavelistratov/results', tag='test'):
    ''' Put logs of all jobs in the same jobarray into one directory '''
    
    # create that dir only once, all the following jobs of the same array will use it
    possible_jobarray_names = ['job_%.2d' % i for i in range(50)]
    for jobarray_name in possible_jobarray_names:
        path = os.path.join(target_dir, jobarray_name)
        if not os.path.exists(path):
            # add time and tag
            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # path = f'{path}_{current_time}_{tag}'
            os.makedirs(path)
            break

    # for each job create a unique folder for logging
    # path = os.path.join(target_dir, jobarray_name, tag)

    # within each job's folder create semantically meaningful subfolders
    visualizations_path = f'{path}/visualizations'
    checkpoints_path = f'{path}/checkpoints'
    source_path = f'{path}/source'
    plots_path = f'{path}/plots'
    logs_path = f'{path}/logs'

    os.makedirs(visualizations_path)
    os.makedirs(checkpoints_path)
    os.makedirs(source_path)
    os.makedirs(plots_path)
    os.makedirs(logs_path)

    return visualizations_path, checkpoints_path, source_path, plots_path, logs_path

# def _create_logger(target_dir):
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False

#     output_handler = logging.FileHandler(f'{target_dir}/output.log')
#     stdout_handler = logging.StreamHandler() # sys.stdout

#     logger.addHandler(output_handler)
#     logger.addHandler(stdout_handler)

#     # https://github.com/camptocamp/pytest-odoo/issues/15
#     logging.getLogger('PIL').setLevel(logging.INFO)
#     logging.getLogger('matplotlib').setLevel(logging.INFO)
    
#     return logger

def _log_model_source(target_dir, algorithm_name='unet'):
    ''' Log raw model object file source -- copy it from the image to the data-server '''

    name_to_path = {
        'unet': '/home/iaroslavelistratov/capsa/experiments',
        # 'prediction_cnn': f'{pwd}/core/models/prediction/cnn',
        # 'prediction_attn': f'{pwd}/core/models/prediction/cnn_attention',
        # 'prediction_gnn': f'{pwd}/core/models/prediction/cnn_gnn',
    }

    model_path = name_to_path[algorithm_name]

    for root, dirs, files in os.walk(model_path):
        for f in files: 
            if f.endswith('.py'):
                file_src = os.path.join(root, f)
                file_trg = os.path.join(target_dir, f)
                shutil.copy(file_src, file_trg)
                os.chmod(file_trg, int('664', base=8))


# def _log_hyperparameters(logger, config):
#     for k, v in config.items():
#         logger.info(f'{k}: {v}')
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     logger.info(f'device: {device}')
#     if device == 'cuda':
#         logger.info(f'device_name: {torch.cuda.get_device_name(0)}')
#     logger.info('\n')

def setup():
    # pwd = os.getcwd()
    # log_dir = f'{pwd}/examples'
    # data_dir = f'{pwd}/data/'

    # config = _read_hyperparameters(config_location, gpu_worker_id)
    # algorithm_name = config['algorithm_name']

    visualizations_path, checkpoints_path, source_path, plots_path, logs_path = _create_folders()
    # logger = _create_logger(logs_path)
    # _log_hyperparameters(logger, config)
    _log_model_source(source_path)

    # return config, logger, visualizations_path, plots_path, train_dir, val_dir
    return visualizations_path, checkpoints_path, plots_path, logs_path