r"""
Adapted from https://github.com/pyro-ppl/pyro/blob/dev/examples/inclined_plane.py.

An experiment consists of a box sliding down an include plane of some length and degrees 
(for example, 2 meters and 30 degrees incline). For each experiment, we measure the descent 
time using a device with a known measurement error of 20 milliseconds. BUT... each time 
we get a new box (made of a different material) and we know it has a different coefficient 
of friction... the classic inference question would be to infer the coefficient from the 
noisey measurement. 

Let's take this one step further. 

Suppose we have many such planes of varying lengths and degrees. Can we leverage a set of
experiments on each plane to infer the coefficient of friction mu for new planes of lengths
and incline that we have not seen before?

Assume that there is a simulator that returns the amount of time that it takes a box to slide 
down an inclined plane as a function of mu (up to measurement error). 
"""
import os
import sys
import math
import cPickle
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data

from src.datasets.utils import BagOfDatasets
from src.models.shared import GaussianEncoder, MeanMLPEncoder
from src.utils import gaussian_log_pdf, AverageMeter, save_checkpoint

little_g = 9.8  # m/s/s
fixed_variance = 0.1  # for gaussian decoder


class InferenceNetwork(nn.Module):
    def __init__(self, input_dim=1, z_dim=1, mlp_dim=2, hidden_dim=10):
        super(InferenceNetwork, self).__init__()
        self.mlp = MeanMLPEncoder(input_dim + z_dim, mlp_dim)
        self.encoder = GaussianEncoder( input_dim + mlp_dim, z_dim,
                                        hidden_dim=hidden_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, dset):
        ss = self.mlp(dset)  # sufficient statistics
        x_ss = torch.cat([x, ss.squeeze(0)], dim=1)
        z_mu, z_logvar = self.encoder(x_ss)

        return z_mu, z_logvar


def sample_minibatch(dataset, batch_size, sample_size):
    minibatch_x, minibatch_z = [], []
    for i in xrange(batch_size):
        items_x, items_z = [], []
        indexes = np.random.choice( np.arange(len(dataset)), 
                                    replace=False, size=sample_size)
        for index in indexes:
            item_x, item_z = dataset.__getitem__(index)
            items_x.append(item_x)
            items_z.append(item_z)
        
        items_x = torch.stack(items_x)
        items_z = torch.stack(items_z)
        minibatch_x.append(items_x)
        minibatch_z.append(items_z)

    minibatch_x = torch.stack(minibatch_x)
    minibatch_z = torch.stack(minibatch_z)
    
    return minibatch_x, minibatch_z


def compiled_inference_objective(z, z_mu, z_logvar):
    r"""NOTE: (x,z) are sampled from p(x,z), a known graphical model

    Compiled inference uses a different objective:
    https://arxiv.org/pdf/1610.09900.pdf

    Proof of objective:

    loss_func = E_{p(x)}[KL[p(z|x) || q_\phi(z|x)]]
            = \int_x p(x) \int_z p(z|x) log(p(z|x)/q_\phi(z|x)) dz dx
            = \int_x \int_z p(x,z) log(p(z|x)/q_\phi(z|x)) dz dx
            = E_{p(x,z)}[log(p(z|x)/q_\phi(z|x))]
        \propto E_{p(x,z)}[-log q_\phi(z|x)]
    """
    log_q_z_given_x = gaussian_log_pdf(z, z_mu, z_logvar)
    return -torch.mean(log_q_z_given_x)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save outputs')
    parser.add_argument('--n-mlp-samples', type=int, default=20,
                        help='number of samples to draw from dataset [default: 20]')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='minibatch size [default: 64]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [default: 10]')
    parser.add_argument('--evaluate-only', action='store_true', default=False,
                        help='assumes trained model is in out_dir/model_best.pth.tar')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # reproducibility
    # this is important as it will generate the same datasets
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if args.cuda else 'cpu')

    def simulate(N_obs, plane_length, plane_degrees, min_mu=0.0001, max_mu=1.0, time_measurement_sigma=0.02):
        r"""The forward simulator, which does numerical integration of the equations of motion
        in steps of size dt, and optionally includes measurement noise.
        """
        mu0s = torch.tensor(np.random.uniform(low=min_mu, high=max_mu, size=N_obs))
        phi = plane_degrees * np.pi / 180.
        noise_sigma = time_measurement_sigma
        
        def _simulate(mu, dt=0.005):
            T = torch.zeros(())
            velocity = torch.zeros(())
            displacement = torch.zeros(())
            acceleration = torch.tensor(little_g * np.sin(phi)) - \
                torch.tensor(little_g * np.cos(phi)) * mu

            if acceleration.numpy() <= 0.0:  # the box doesn't slide if the friction is too large
                return torch.tensor(1.0e5)   # return a very large time instead of infinity

            while displacement.numpy() < plane_length:  # otherwise slide to the end of the inclined plane
                displacement += velocity * dt
                velocity += acceleration * dt
                T += dt

            return T + noise_sigma * torch.randn(())

        t0s = torch.tensor([_simulate(mu) for mu in mu0s])
        
        return mu0s, t0s

    plane_lengths = np.array([2, 4, 6, 8, 10])
    plane_degrees = np.array([20, 30, 40, 50, 60])
    planes = list(product(plane_lengths, plane_degrees))
    n_planes = len(planes)

    # now we are amortizing over these planes; generate a dataset for each 
    # of these planes.
    train_datasets, val_datasets = [], []

    for plane in planes:
        train_mu, train_obs = simulate(1000, plane[0], plane[1])
        val_mu, val_obs = simulate(1000, plane[0], plane[1])
        train_dataset = data.TensorDataset(train_obs, train_mu)
        val_dataset = data.TensorDataset(val_obs, val_mu)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_dataset = BagOfDatasets(train_datasets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loaders = [torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
        for dataset in val_datasets]

    model = InferenceNetwork()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, data_list in enumerate(train_loader):
            x_list = [data[0] for data in data_list]
            batch_size = len(x_list[0])

            loss = 0
            for i in xrange(n_planes):
                x_i = x_list[i]
                x_i = x_i.to(device)

                context_x_i, context_z_i = sample_minibatch(
                    train_datasets[i], batch_size, args.n_mlp_samples)
                context_x_i = context_x_i.to(device)
                context_z_i = context_z_i.to(device)
                context_x_z_i = torch.cat([context_x_i, context_z_i], dim=2)

                z_mu_i, z_logvar_i = model(x_i, context_x_z_i)
                loss_i = compiled_inference_objective(z_i, z_mu_i, z_logvar_i)
                loss += loss_i

            loss_meter.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size,
                        len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        -loss_meter.avg))

        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, -loss_meter.avg))
        return loss_meter.avg

    def val(epoch):
        model.eval()
        loss_meters = [AverageMeter() for _ in xrange(n_planes)]

        with torch.no_grad():
            for i in xrange(args.n_train_models):
                val_loader = val_loaders[i]

                for x_i, _ in val_loader:
                    batch_size = len(x_i)
                    x_i = x_i.to(device)

                    context_x_i, context_z_i = sample_minibatch(
                        val_loader.dataset, batch_size, args.n_mlp_samples)
                    context_x_i = context_x_i.to(device)
                    context_z_i = context_z_i.to(device)
                    context_x_z_i = torch.cat([context_x_i, context_z_i], dim=2)

                    z_mu_i, z_logvar_i = model(x_i, context_x_z_i)
                    loss_i = compiled_inference_objective(z_i, z_mu_i, z_logvar_i)

                    loss_meters[i].update(loss_i.item(), batch_size)

        loss_meter_avgs = [meter.avg for meter in loss_meters]
        loss_meter_avgs = np.array(loss_meter_avgs)

        print('====> Test Epoch: {}\tAverage Loss: {:.4f}'.format(
            epoch, np.mean(loss_meter_avgs)))
        
        return loss_meter_avgs

    if not args.evaluate_only: 
        best_loss = sys.maxint

        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch)
            val_losses = val(epoch)
            val_loss_sum = sum(val_losses)

            is_best = val_loss_sum < best_loss
            best_loss = min(val_loss_sum, best_loss)

            save_checkpoint({
                'state_dict': infernet.state_dict(),
                'cmd_line_args': args,
            }, is_best, folder=args.out_dir)

    print('loading best performing model')
    checkpoint = torch.load(os.path.join(args.out_dir, 'model_best.pth.tar'))
    state_dict = checkpoint['state_dict']

    # NOTE: this includes out-of-sample
    plane_lengths = np.arange(1, 20, 1)
    plane_degrees = np.arange(5, 85, 5)
    planes = list(product(plane_lengths, plane_degrees))
    n_planes = len(planes)

    # out-of-sample models -- measure how well we can do 
    # inference out of the box (no learning)
    test_datasets = []
    for plane in planes:
        mu, obs = simulate(1000, plane[0], plane[1])
        test_dataset = data.TensorDataset(obs, mu)
        test_datasets.append(test_dataset)

    test_loaders = [torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
        for dataset in test_datasets] 

    model = InferenceNetwork()
    model = model.to(device)
    model = model.eval()
    model.load_state_dict(state_dict)

    def inference_error(loader):
        r"""We can compute error b/c we have the closed form
        numerical integration (see above)."""
        mu_gt, mu_pred = [], []

        with torch.no_grad():
            z_infer = []
            for obs, mu in loader:
                batch_size = len(obs)
                obs = obs.to(device)

                context_x, context_z = sample_minibatch(
                    loader.dataset, batch_size, args.n_mlp_samples)
                context_x = context_x.to(device)
                context_z = context_z.to(device)
                context_x_z = torch.cat([context_x, context_z], dim=2)

                z_mu, _ = model(obs, context_x_z)
                
                mu_gt.append(mu.numpy())
                mu_pred.append(z_mu.cpu().numpy())
        
        mu_gt = np.concatenate(mu_gt, axis=0)
        mu_pred = np.concatenate(mu_pred, axis=0)

        error = mean_squared_error(mu_gt, mu_pred)

        return error

    results = []
    for i in xrange(n_planes):
        test_loader = test_loaders[i]
        error_i = inference_error(test_loader)
        plane_length_i = plane_lengths[i]
        plane_degree_i = plane_degrees[i]
        results.append([plane_length_i, plane_degree_i, error_i])
    
    results = np.array(results)
    np.save(os.path.join(args.out_dir, 'inference_generalization.npy'), results)
