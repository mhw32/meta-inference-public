import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from src.datasets.utils import BagOfDatasets
from src.utils import AverageMeter, save_checkpoint
from src.models.expfam import (
    RNNEncoder,
    MeanMLPEncoder,
    GaussianEncoder, 
    Toy_FixedVarianceGaussianDecoder,
    Toy_FixedVarianceLogNormalDecoder,
    Toy_ExponentialDecoder,
)
from src.datasets.expfam import (
    build_gaussian_distributions,
    build_lognormal_distributions,
    build_exponential_distributions,
)
from src.objectives.expfam import (
    log_gaussian_marginal_estimate,
    log_lognormal_marginal_estimate,
    log_exponential_marginal_estimate,
    gaussian_elbo_loss,
    lognormal_elbo_loss,
    exponential_elbo_loss,
)

STDEV = 0.1


class AmortizedDVAE(nn.Module):
    def __init__(self, n_gaussian_datasets, n_lognormal_datasets, n_exponential_datasets,
                 input_dim=2, z_dim=2, summary_dim=64, hidden_dim=64):
        super(AmortizedDVAE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        assert self.z_dim == 2 

        self.sample_summary_net = MeanMLPEncoder(self.input_dim, self.summary_dim)
        self.meta_summary_net = MeanMLPEncoder(self.summary_dim, self.summary_dim)
        self.encoder = GaussianEncoder(self.summary_dim * 2, self.z_dim, 
                                       hidden_dim=self.hidden_dim)
        self.decoders = [] 
        self.decoder_dists =[]
        for i in range(n_gaussian_datasets):
            decoder_i = Toy_FixedVarianceGaussianDecoder(x_stdev=STDEV)
            self.decoders.append(decoder_i)
            self.decoder_dists.append('gaussian')
        for i in range(n_lognormal_datasets):
            decoder_i = Toy_FixedVarianceLogNormalDecoder(x_stdev=STDEV)
            self.decoders.append(decoder_i)
            self.decoder_dists.append('lognormal')
        for i in range(n_exponential_datasets):
            decoder_i = Toy_ExponentialDecoder()
            self.decoders.append(decoder_i)
            self.decoder_dists.append('exponential')
        self.decoders = nn.ModuleList(self.decoders)
        self.n_gaussian_datasets = n_gaussian_datasets
        self.n_lognormal_datasets = n_lognormal_datasets
        self.n_exponential_datasets = n_exponential_datasets

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def get_sample_summary(self, dset):
        return self.sample_summary_net(dset)

    def get_meta_summary(self, dset):
        return self.meta_summary_net(dset)

    def forward(self, meta_x, meta_x_list, i):
        # meta_x is a meta_dataset
        # meta_x_list is a list of meta_datasets
        meta_e = self.get_sample_summary(meta_x)
        meta_e_list = [self.get_sample_summary(meta_x_i) 
                       for meta_x_i in meta_x_list]
        meta_e_list = torch.stack(meta_e_list)
        meta_e_list = meta_e_list.permute(1, 0, 2).contiguous()
        meta_e2 = self.get_meta_summary(meta_e_list)

        x_input = torch.cat([meta_e, meta_e2], dim=1)
        z_mu, z_logvar = self.encoder(x_input)
        
        z = self.reparameterize(z_mu, z_logvar)
        x_s1, x_s2 = self.decoders[i](z)

        return x_s1, x_s2, z, z_mu, z_logvar

    def estimate_marginal(self, meta_x, meta_x_list, i, n_samples=100):
        with torch.no_grad():
            meta_e = self.get_sample_summary(meta_x)
            meta_e_list = [self.get_sample_summary(meta_x_i) 
                           for meta_x_i in meta_x_list]
            meta_e_list = torch.stack(meta_e_list)
            meta_e_list = meta_e_list.permute(1, 0, 2).contiguous()
            meta_e2 = self.get_meta_summary(meta_e_list)

            x_input = torch.cat([meta_e, meta_e2], dim=1)
            z_mu, z_logvar = self.encoder(x_input)

            x_s1_list, x_s2_list, z_list = [], [], []
            decoder_dist = self.decoder_dists[i]

            for _ in range(n_samples):
                z_i = self.reparameterize(z_mu, z_logvar)
                x_s1_i, x_s2_i = self.decoders[i](z_i)
                x_s1_list.append(x_s1_i)
                x_s2_list.append(x_s2_i)
                z_list.append(z_i)

            if decoder_dist == 'gaussian':
                log_marginal = log_gaussian_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, x_s2_list, 
                    z_list, z_mu, z_logvar)
            elif decoder_dist == 'lognormal':
                log_marginal = log_lognormal_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, x_s2_list, 
                    z_list, z_mu, z_logvar)
            elif decoder_dist == 'exponential':
                log_marginal = log_exponential_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, z_list, z_mu, z_logvar)
            else:
                raise Exception('decoder_dist %s not supported.' % decoder_dist)
                
        return log_marginal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save outputs')
    parser.add_argument('--n-gaussian-distributions', type=int, default=30,
                        help='number of gaussian distributions [default: 30]')
    parser.add_argument('--n-lognormal-distributions', type=int, default=30,
                        help='number of lognormal distributions [default: 30]')
    parser.add_argument('--n-exponential-distributions', type=int, default=30,
                        help='number of exponential distributions [default: 30]')
    parser.add_argument('--exponential-constant', type=float, default=1000.,
                        help='loss constant for exponential [default: 1000]')
    parser.add_argument('--test-frequency', type=int, default=100,
                        help='frequency of testing [default: 100]')
    parser.add_argument('--n-meta-datasets', type=int, default=10,
                        help='number of meta datasets from each dataset [default: 10]')
    parser.add_argument('--n-meta-samples', type=int, default=20,
                        help='number of samples to represent a meta distribution [default: 20]')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='minibatch size [default: 1]')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='number of hidden dimensions [default: 128]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default: 0.0002]')
    # no more real sense of an "epoch", we iter=number of gradient steps
    parser.add_argument('--n-iters', type=int, default=10000,
                        help='number of epochs [default: 10000]')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='interval for printing [default: 5]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.sample_dir = os.path.join(args.out_dir, 'samples')
    args.n_datasets = (args.n_gaussian_distributions + 
                       args.n_lognormal_distributions + 
                       args.n_exponential_distributions)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    train_gaussian_datasets, test_gaussian_datasets, _, gaussian_dataset_params = \
        build_gaussian_distributions(
            args.n_gaussian_distributions, 10000, prior=5, std=STDEV)

    train_lognormal_datasets, test_lognormal_datasets, _, lognormal_dataset_params = \
        build_lognormal_distributions(
            args.n_lognormal_distributions, 10000, prior=5, std=STDEV)

    train_exponential_datasets, test_exponential_datasets, _, exponential_dataset_params = \
        build_exponential_distributions(
            args.n_exponential_distributions, 10000, prior=3)

    train_datasets = train_gaussian_datasets + train_lognormal_datasets + train_exponential_datasets
    test_datasets = test_gaussian_datasets + test_lognormal_datasets + test_exponential_datasets
    dataset_params = gaussian_dataset_params + lognormal_dataset_params + exponential_dataset_params
    dataset_types = (['gaussian'] * args.n_gaussian_distributions + 
                     ['lognormal'] * args.n_lognormal_distributions + 
                     ['exponential'] * args.n_exponential_distributions)

    train_bag = BagOfDatasets(train_datasets)
    test_bag = BagOfDatasets(test_datasets)

    model = AmortizedDVAE(args.n_gaussian_distributions, 
                          args.n_lognormal_distributions,
                          args.n_exponential_distributions,
                          hidden_dim=args.hidden_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def step(iter):
        model.train()
        loss = 0
        gaussian_loss, lognormal_loss, exponential_loss = 0, 0, 0

        # list of (n_dataset, batch_size, n_meta_datasets + 1, n_meta_samples, 2)
        meta_datasets = train_bag.sample_meta_datasets(
            args.n_meta_datasets + 1, args.batch_size, args.n_meta_samples)

        for i in range(args.n_datasets):
            dataset_type = dataset_types[i]
            meta_dataset_i = meta_datasets[i]
            meta_x_mlp_i = meta_dataset_i[-1]    # shape: (batch_size, n_meta_samples, 2)
            meta_x_rnn_i = meta_dataset_i[:-1]  # shape: (n_meta_datasets, batch_size, n_meta_samples, 2)
            meta_x_rnn_i = [meta_x_rnn_i[j] for j in range(args.n_meta_datasets)]  # make to list

            meta_x_mlp_i = meta_x_mlp_i.to(device)
            meta_x_rnn_i = [meta_x.to(device) for meta_x in meta_x_rnn_i]

            x_s1_i, x_s2_i, z_i, z_mu_i, z_logvar_i = \
                model(meta_x_mlp_i, meta_x_rnn_i, i)

            if dataset_type == 'gaussian':
                loss_i = gaussian_elbo_loss(meta_x_mlp_i.permute(1, 0, 2), 
                                            x_s1_i, x_s2_i, z_i, z_mu_i, z_logvar_i)
                gaussian_loss += loss_i
            elif dataset_type == 'lognormal':
                loss_i = lognormal_elbo_loss(meta_x_mlp_i.permute(1, 0, 2), 
                                             x_s1_i, x_s2_i, z_i, z_mu_i, z_logvar_i)
                lognormal_loss += loss_i
            elif dataset_type == 'exponential':
                loss_i = exponential_elbo_loss( meta_x_mlp_i.permute(1, 0, 2), 
                                                x_s1_i, z_i, z_mu_i, z_logvar_i)
                loss_i = loss_i * args.exponential_constant
                exponential_loss += loss_i
            loss += loss_i
       
        loss /= float(args.n_datasets)
        gaussian_loss /= float(args.n_gaussian_distributions)
        lognormal_loss /= float(args.n_lognormal_distributions)
        exponential_loss /= float(args.n_exponential_distributions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, (gaussian_loss, lognormal_loss, exponential_loss)


    def get_marginal(repetitions=10):
        model.eval()
        log_marginals = np.zeros(args.n_datasets)
        
        with torch.no_grad():
            for _ in range(repetitions):  # do this for ten times!
                meta_datasets = test_bag.sample_meta_datasets(
                    args.n_meta_datasets + 1, args.batch_size, args.n_meta_samples)

                for i in range(args.n_datasets):
                    dataset_type = dataset_types[i]
                    meta_dataset_i = meta_datasets[i]
                    meta_x_mlp_i = meta_dataset_i[-1]    # shape: (batch_size, n_meta_samples, 2)
                    meta_x_rnn_i = meta_dataset_i[:-1]  # shape: (n_meta_datasets, batch_size, n_meta_samples, 2)
                    meta_x_rnn_i = [meta_x_rnn_i[j] for j in range(args.n_meta_datasets)]  # make to list

                    meta_x_mlp_i = meta_x_mlp_i.to(device)
                    meta_x_rnn_i = [meta_x.to(device) for meta_x in meta_x_rnn_i]

                    log_marginal_i = model.estimate_marginal(
                        meta_x_mlp_i, meta_x_rnn_i, i, n_samples=10)
                    log_marginals[i] += log_marginal_i
            
            log_marginals /= float(repetitions)
        
        return log_marginals


    best_loss = sys.maxsize
    train_elbo_db = np.zeros(args.n_iters)
    test_log_marginal_db = np.zeros((args.n_iters // args.test_frequency, args.n_datasets))

    for i in range(1, args.n_iters + 1):
        train_elbo, (gaussian_elbo, lognormal_elbo, exponential_elbo) = step(i)
        train_elbo_db[i - 1] = train_elbo

        print('Iter: {}\tTrain ELBO: {}'.format(i, train_elbo))
        print('\tGaussian ELBO: {}'.format(gaussian_elbo))
        print('\tLog Normal ELBO: {}'.format(lognormal_elbo))
        print('\tExponential ELBO: {}'.format(exponential_elbo))

        if i % args.test_frequency == 0:
            test_log_marginal = get_marginal(repetitions=10)
            test_log_marginal_db[i // args.test_frequency - 1, :] = test_log_marginal

            print('Test Average Log Marginal: {}'.format(np.mean(test_log_marginal)))

            is_best = sum(test_log_marginal) < best_loss
            best_loss = min(sum(test_log_marginal), best_loss)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'cmd_line_args': args,
                'train_datasets': train_datasets,
                'test_datasets': test_datasets,
                'dataset_params': dataset_params,
            }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_elbo.npy'), train_elbo_db)
        np.save(os.path.join(args.out_dir, 'test_log_marginal.npy'), test_log_marginal_db)
