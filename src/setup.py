import os
import sys
import torch
import logging
import numpy as np
from pprint import pprint
from dotmap import DotMap
from logging import Formatter
from logging.handlers import RotatingFileHandler
from time import strftime, localtime, time

from src.utils import load_json, save_json


def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


def process_config(config_path, custom_split=None, custom_gpu_device=None,
                   override_dotmap=None):
    config_json = load_json(config_path)
    if custom_split is not None:
        config_json['data_params']['split'] = str(custom_split)
    if custom_gpu_device is not None:
        config_json['gpu_device'] = int(custom_gpu_device)
    # so we dont need to manually change folders
    config_json['exp_name'] = '{}_{}'.format(
        config_json['exp_name'],
        config_json['data_params']['split'])
    return _process_config(config_json, override_dotmap=override_dotmap)


def _process_config(config_json, override_dotmap=None):
    """
    Processes config file:
        1) Converts it to a DotMap
        2) Creates experiments path and required subdirs
        3) Set up logging
    """
    config = DotMap(config_json)
    if override_dotmap is not None:
        config.update(override_dotmap)

    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    exp_base = config.exp_base
    timestamp = strftime('%Y-%m-%d--%H_%M_%S', localtime())
    exp_dir = os.path.join(exp_base, "experiments",
                           config.exp_name, timestamp)

    # create some important directories to be used for the experiment.
    config.summary_dir = os.path.join(exp_dir, "summaries/")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")

    makedirs([config.summary_dir, config.checkpoint_dir,
              config.out_dir, config.log_dir])

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info(
        "Configurations and directories successfully set up.")

    return config


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        '{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        '{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    try:
        call(["nvcc", "--version"])
    except:
        pass
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))
