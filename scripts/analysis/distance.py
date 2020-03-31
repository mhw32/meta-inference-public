"""
Given a single image. Find all edits of it. Compute average 
embedding distance across each set of rotations. Average over
the test set. What we hope is that the model is "invariant" 
to transformations that it was trained on.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from src.agents.agents import *
from src.datasets.mnist import \
    ALL_ROTATIONS, ALL_SCALES, ALL_SHEARS
from src.datasets.norb import \
    load_norb_numpy, \
    ALL_ELEVATION, ALL_AZIMUTH, ALL_LIGHTING
from src.datasets.utils import \
    BagOfDatasets, sample_minibatch


def _load_meta_agent(exp_dir, gpu_device, checkpoint_name='model_best.pth.tar'):
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    config = checkpoint['config']
    # overwrite GPU since we might want to use a different GPU
    config.gpu_device = gpu_device
    config.data_params.split = 'standard'
    config.exp_dir = exp_dir
    config.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

    print("Loading trained Agent from filesystem")
    AgentClass = globals()[config.agent]
    agent = AgentClass(config)
    agent.load_checkpoint(checkpoint_name)
    agent.model.eval()

    return agent


def save_or_load_datasets_to_cache(datasets, save_dir):
    caches = []
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(len(datasets)):
        cache_file_i = os.path.join(save_dir, 'cache_{}.npy'.format(i))
        if os.path.isfile(cache_file_i):
            print("found cached dataset {}/{}. loading...".format(i+1, len(datasets)))
            cache_i = np.load(cache_file_i)
        else:
            print("cache-ing each train dataset {}/{}".format(i+1, len(datasets)))
            loader_i = DataLoader(datasets[i], batch_size=100, shuffle=False)
            pbar = tqdm(total=len(loader_i))
            cache_i = []
            for image, label in loader_i:
                cache_i.append(image.numpy())
                pbar.update()
            pbar.close()
            cache_i = np.concatenate(cache_i, axis=0)
            np.save(cache_file_i, cache_i)
            print("saved cache to filesystem.")
        caches.append(cache_i)
    return caches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('out_dir', type=str, help='where to store embeddings')
    parser.add_argument('cache_dir', type=str, help='where to get/fetch cached items')
    parser.add_argument('--gpu-device', type=int, default=0, help='0-9')
    args = parser.parse_args()

    agent = _load_meta_agent(args.exp_dir, args.gpu_device)
    dataset_name = agent.config.data_params.dataset

    if dataset_name in ['rotate', 'scale', 'shear']:
        rotation_datasets = load_many_rotated_mnist(
            agent.config.data_params.data_dir,
            agent.config.data_params.image_size,
            train=False,
            rotations=ALL_ROTATIONS,
        )
        scale_datasets = load_many_scaled_mnist(
            agent.config.data_params.data_dir,
            agent.config.data_params.image_size,
            train=False,
            scales=ALL_SCALES,
        )
        shear_datasets = load_many_sheared_mnist(
            agent.config.data_params.data_dir,
            agent.config.data_params.image_size,
            train=False,
            shears=ALL_SHEARS,
        )
        rotation_caches = save_or_load_datasets_to_cache(
            rotation_datasets, 
            os.path.join(args.cache_dir, 'rotations'),
        )
        scale_caches = save_or_load_datasets_to_cache(
            scale_datasets, 
            os.path.join(args.cache_dir, 'scales'),
        )
        shear_caches = save_or_load_datasets_to_cache(
            shear_datasets, 
            os.path.join(args.cache_dir, 'shears'),
        )
        t1_datasets = rotation_datasets
        t2_datasets = scale_datasets
        t3_datasets = shear_datasets
        t1_caches = rotation_caches
        t2_caches = scale_caches
        t3_caches = shear_caches
    elif dataset_name == 'norb':
        elevation_datasets = [
            load_norb_numpy(
                i, 
                concept='elevation', 
                image_size=agent.config.data_params.image_size, 
                train=False,
            )
            for i in ALL_ELEVATION
        ]
        azimuth_datasets = [
            load_norb_numpy(
                i, 
                concept='azimuth', 
                image_size=agent.config.data_params.image_size, 
                train=False,
            )
            for i in ALL_AZIMUTH
        ]
        lighting_datasets = [
            load_norb_numpy(
                i, 
                concept='lighting', 
                image_size=agent.config.data_params.image_size, 
                train=False,
            )
            for i in ALL_LIGHTING
        ]
        elevation_caches = save_or_load_datasets_to_cache(
            elevation_datasets, 
            os.path.join(args.cache_dir, 'elevations'),
        )
        azimuth_caches = save_or_load_datasets_to_cache(
            azimuth_datasets, 
            os.path.join(args.cache_dir, 'azimuths'),
        )
        lighting_caches = save_or_load_datasets_to_cache(
            lighting_datasets, 
            os.path.join(args.cache_dir, 'lightings'),
        )
        t1_datasets = elevation_datasets
        t2_datasets = azimuth_datasets
        t3_datasets = lighting_datasets
        t1_caches = elevation_caches
        t2_caches = azimuth_caches
        t3_caches = lighting_caches
    else:
        raise Exception('dataset_name {} not recognized'.format(dataset_name))

    t1_loaders = [
        DataLoader(dset, batch_size=100, shuffle=False)
        for dset in t1_datasets
    ]
    t2_loaders = [
        DataLoader(dset, batch_size=100, shuffle=False)
        for dset in t2_datasets
    ]
    t3_loaders = [
        DataLoader(dset, batch_size=100, shuffle=False)
        for dset in t3_datasets
    ]

    print('***** Transformation 1: {} datasets *****'.format(len(t1_loaders)))

    t1_codes = []
    for i, (loader, cache) in enumerate(zip(t1_loaders, t1_caches)):
        print('--> Dataset {}/{}'.format(i+1, len(t1_loaders)))
        pbar = tqdm(total=len(loader))
        codes = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                batch_size = len(data)
                data = data.to(agent.device)
                if agent.config.model_params.name == 'ns':
                    data = data.unsqueeze(0)
                    code = agent.model.extract_codes(data)
                else:
                    sample_list = sample_minibatch_from_cache(
                        cache, batch_size,
                        agent.config.model_params.n_data_samples)
                    sample_list = sample_list.to(agent.device)
                    code = agent.model.extract_codes(data, sample_list)
                code = code.cpu().numpy()
                codes.append(code)
                pbar.update()
        pbar.close()
        codes = np.concatenate(codes)
        t1_codes.append(codes)
    t1_codes = np.stack(t1_codes)

    print('***** Transformation 2: {} datasets *****'.format(len(t2_loaders)))

    t2_codes = []
    for i, (loader, cache) in enumerate(zip(t2_loaders, t2_caches)):
        print('--> Dataset {}/{}'.format(i+1, len(t2_loaders)))
        pbar = tqdm(total=len(loader))
        codes = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                batch_size = len(data)
                data = data.to(agent.device)
                if agent.config.model_params.name == 'ns':
                    data = data.unsqueeze(0)
                    code = agent.model.extract_codes(data)
                else:
                    sample_list = sample_minibatch_from_cache(
                        cache, batch_size,
                        agent.config.model_params.n_data_samples)
                    sample_list = sample_list.to(agent.device)
                    code = agent.model.extract_codes(data, sample_list)
                code = code.cpu().numpy()
                codes.append(code)
                pbar.update()
            pbar.close()
        codes = np.concatenate(codes)
        t2_codes.append(codes)
    t2_codes = np.stack(t2_codes)

    print('***** Transformation 3: {} datasets *****'.format(len(t3_loaders)))

    t3_codes = []
    for i, (loader, cache) in enumerate(zip(t3_loaders, t3_caches)):
        print('--> Dataset {}/{}'.format(i+1, len(t3_loaders)))
        pbar = tqdm(total=len(loader))
        codes = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                batch_size = len(data)
                data = data.to(agent.device)
                if agent.config.model_params.name == 'ns':
                    data = data.unsqueeze(0)
                    code = agent.model.extract_codes(data)
                else:
                    sample_list = sample_minibatch_from_cache(
                        cache, batch_size,
                        agent.config.model_params.n_data_samples)
                    sample_list = sample_list.to(agent.device)
                    code = agent.model.extract_codes(data, sample_list)
                code = code.cpu().numpy()
                codes.append(code)
                pbar.update()
            pbar.close()
        codes = np.concatenate(codes)
        t3_codes.append(codes)
    t3_codes = np.stack(t3_codes)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    np.save(os.path.join(args.out_dir, 't1_embeddings.npy'), t1_codes)
    np.save(os.path.join(args.out_dir, 't2_embeddings.npy'), t2_codes)
    np.save(os.path.join(args.out_dir, 't3_embeddings.npy'), t3_codes)
