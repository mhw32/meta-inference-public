import os
import torch
import numpy as np
from tqdm import tqdm
from src.datasets.norb import load_norb, NORB_ROOT
from torchvision import transforms
from torch.utils.data import DataLoader

NUMPY_ROOT = '/mnt/fs5/wumike/datasets/norb/numpy'


if  __name__ == "__main__":
    if not os.path.isdir(NUMPY_ROOT):
        os.makedirs(NUMPY_ROOT)

    train_dset = load_norb(NORB_ROOT, train=True)
    test_dset = load_norb(NORB_ROOT, train=False)

    train_loader = loader = DataLoader(train_dset, batch_size=100, shuffle=False)
    test_loader = loader = DataLoader(test_dset, batch_size=100, shuffle=False)

    pbar = tqdm(total=len(train_loader))
    train_imgs, train_infos, train_targets = [], [], []
    for batch_idx, (img, info, target) in enumerate(train_loader):
        train_imgs.append(img)
        train_infos.append(info)
        train_targets.append(target)
        pbar.update()
    pbar.close()

    train_imgs = torch.cat(train_imgs, dim=0).numpy()
    train_infos = torch.cat(train_infos, dim=0).numpy()
    train_targets = torch.cat(train_targets).numpy()

    pbar = tqdm(total=len(test_loader))
    test_imgs, test_infos, test_targets = [], [], []
    for batch_idx, (img, info, target) in enumerate(test_loader):
        test_imgs.append(img)
        test_infos.append(info)
        test_targets.append(target)
        pbar.update()
    pbar.close()

    test_imgs = torch.cat(test_imgs, dim=0).numpy()
    test_infos = torch.cat(test_infos, dim=0).numpy()
    test_targets = torch.cat(test_targets).numpy()

    infos = np.concatenate((train_infos, test_infos), axis=0)
    elevations, azimuths, lightings = infos[:, 1], infos[:, 2], infos[:, 3]
    elevations_unique = np.unique(elevations)
    azimuths_unique = np.unique(azimuths)
    lightings_unique = np.unique(lightings)

    train_elevation_imgs = []
    train_elevation_infos = []
    train_elevation_targets = []
    for i in elevations_unique:
        choose = train_infos[:, 1] == i
        train_elevation_imgs.append(train_imgs[choose])
        train_elevation_infos.append(train_infos[choose])
        train_elevation_targets.append(train_targets[choose])

    train_azimuth_imgs = []
    train_azimuth_infos = []
    train_azimuth_targets = []
    for i in azimuths_unique:
        choose = train_infos[:, 2] == i
        train_azimuth_imgs.append(train_imgs[choose])
        train_azimuth_infos.append(train_infos[choose])
        train_azimuth_targets.append(train_targets[choose])

    train_lighting_imgs = []
    train_lighting_infos = []
    train_lighting_targets = []
    for i in lightings_unique:
        choose = train_infos[:, 3] == i
        train_lighting_imgs.append(train_imgs[choose])
        train_lighting_infos.append(train_infos[choose])
        train_lighting_targets.append(train_targets[choose])

    test_elevation_imgs = []
    test_elevation_infos = []
    test_elevation_targets = []
    for i in elevations_unique:
        choose = test_infos[:, 1] == i
        test_elevation_imgs.append(test_imgs[choose])
        test_elevation_infos.append(test_infos[choose])
        test_elevation_targets.append(test_targets[choose])

    test_azimuth_imgs = []
    test_azimuth_infos = []
    test_azimuth_targets = []
    for i in azimuths_unique:
        choose = test_infos[:, 2] == i
        test_azimuth_imgs.append(test_imgs[choose])
        test_azimuth_infos.append(test_infos[choose])
        test_azimuth_targets.append(test_targets[choose])

    test_lighting_imgs = []
    test_lighting_infos = []
    test_lighting_targets = []
    for i in lightings_unique:
        choose = test_infos[:, 3] == i
        test_lighting_imgs.append(test_imgs[choose])
        test_lighting_infos.append(test_infos[choose])
        test_lighting_targets.append(test_targets[choose])

    for i in range(len(elevations_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'train_elevation_dataset_{}.npz'.format(i)),
                 imgs=train_elevation_imgs[i],
                 infos=train_elevation_infos[i],
                 targets=train_elevation_targets[i])
    for i in range(len(azimuths_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'train_azimuth_dataset_{}.npz'.format(i)),
                 imgs=train_azimuth_imgs[i],
                 infos=train_azimuth_infos[i],
                 targets=train_azimuth_targets[i])
    for i in range(len(lightings_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'train_lighting_dataset_{}.npz'.format(i)),
                 imgs=train_lighting_imgs[i],
                 infos=train_lighting_infos[i],
                 targets=train_lighting_targets[i])

    for i in range(len(elevations_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'test_elevation_dataset_{}.npz'.format(i)),
                 imgs=test_elevation_imgs[i],
                 infos=test_elevation_infos[i],
                 targets=test_elevation_targets[i])

    for i in range(len(azimuths_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'test_azimuth_dataset_{}.npz'.format(i)),
                 imgs=test_azimuth_imgs[i],
                 infos=test_azimuth_infos[i],
                 targets=test_azimuth_targets[i])

    for i in range(len(lightings_unique)):
        np.savez(os.path.join(NUMPY_ROOT, 'test_lighting_dataset_{}.npz'.format(i)),
                 imgs=test_lighting_imgs[i],
                 infos=test_lighting_infos[i],
                 targets=test_lighting_targets[i])
