import os, json
from copy import deepcopy
from src.agents.agents import *
from src.setup import process_config
from src.utils import load_json


def run(config_path, custom_split=None, custom_gpu_device=None):
    config = process_config(config_path, 
                            custom_split=custom_split,
                            custom_gpu_device=custom_gpu_device)
    
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)

    try:
        agent.run()
        agent.finalise()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--split', help='standard|sparse|disjoint')
    parser.add_argument('--gpu-device', help='0-9')
    args = parser.parse_args()

    run(args.config, custom_split=args.split, custom_gpu_device=args.gpu_device)
