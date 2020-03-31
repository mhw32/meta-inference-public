import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from src.agents.agents import *
from src.datasets.norb import SmallNORB, NORB_ROOT
from torchvision.datasets import MNIST


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, default='path to experiment directory')
    parser.add_argument('--reduction', type=str, default='pca', help='pca|mds|tsne')
    parser.add_argument('--gpu-device', type=int, default=0, help='0-9')
    args = parser.parse_args()
    assert args.reduction in ['pca', 'mds', 'tsne']

    agent = _load_meta_agent(args.exp_dir, args.gpu_device)
    dataset_name = agent.config.data_params.dataset

    datasets = agent.train_datasets
    caches = agent.train_caches
    latents = []
    labels  = []

    for i, (dataset, cache) in enumerate(zip(datasets, caches)):
        print('Embedding Dataset ({}/{}).'.format(i, len(datasets)))
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=100, shuffle=False)
            pbar = tqdm(total=len(loader))
            for batch_idx, (data, label) in enumerate(loader):
                batch_size = len(data)
                data = data.to(agent.device)
                if agent.config.model_params.name == 'ns':
                    data = data.unsqueeze(0)
                    codes = agent.model.extract_codes(data)
                else:
                    sample_list = sample_minibatch_from_cache(
                        cache, batch_size,
                        agent.config.model_params.n_data_samples)
                    sample_list = sample_list.to(agent.device)
                    codes = agent.model.extract_codes(data, sample_list)
                codes = codes.cpu().numpy()
                label = label.cpu().numpy()
                latents.append(codes)
                labels.append(label)
                pbar.update()
            pbar.close()
    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)

    if args.reduction == 'pca':
        reducer = PCA(n_components=2)
        reduced_dataset = reducer.fit_transform(latents)  
        reduced_dir = os.path.join(args.exp_dir, 'pca')
    elif args.reduction == 'mds':
        reducer = MDS(n_components=2)
        latents = latents.astype(np.float64)
        reduced_dataset = reducer.fit_transform(latents)
        reduced_dir = os.path.join(args.exp_dir, 'mds')
    else:
        reducer = TSNE(n_components=2)
        reduced_dataset = reducer.fit_transform(latents)
        reduced_dir = os.path.join(args.exp_dir, 'tsne')
    
    if not os.path.isdir(reduced_dir):
        os.makedirs(reduced_dir)

    np.save(os.path.join(reduced_dir, 'embeddings.npy'), reduced_dataset)
    np.save(os.path.join(reduced_dir, 'embedding_labels.npy'), labels)

