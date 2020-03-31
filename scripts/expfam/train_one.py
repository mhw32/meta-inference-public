r"""
Train a graphical model where each data point is a dataset. As a
toy example, we use a bunch of different Gaussian distributions. 
Then, the latent variable should hopefully be some sufficient statstic.
"""
import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from src.datasets.utils import BagOfDatasetsExpFam as BagOfDatasets
from src.utils import (
    AverageMeter, 
    save_checkpoint,
    log_mean_exp,
)
from src.models.expfam import (
    RNNEncoder,
    MeanMLPEncoder,
    GaussianEncoder, 
    Toy_FixedVarianceGaussianDecoder,
    Toy_FixedVarianceLogNormalDecoder,
    Toy_GaussianDecoder,
    Toy_LogNormalDecoder,
    Toy_ExponentialDecoder,
)
from src.datasets.expfam import (
    build_gaussian_distributions,
    build_lognormal_distributions,
    build_exponential_distributions,
)
from src.objectives.expfam import (
    gaussian_elbo_loss,
    lognormal_elbo_loss,
    exponential_elbo_loss,
)

STDEV = 0.1

DIST_TO_Z_DIM = {
    'fixed_var_gaussian': 2,
    'fixed_var_lognormal': 2,
    'gaussian': 4,
    'lognormal': 4,
    'exponential': 2,
}

DIST_TO_GENERATOR_ARGS = {
    'fixed_var_gaussian': {'std': STDEV, 'prior': 5},
    'fixed_var_lognormal': {'std': STDEV, 'prior': 2},
    'gaussian': {'std': None, 'prior': 10},
    'lognormal': {'std': None, 'prior': 2},
    'exponential': {'prior': 3}
}

DIST_TO_DECODER_CLS = {
    'fixed_var_gaussian': Toy_FixedVarianceGaussianDecoder,
    'fixed_var_lognormal': Toy_FixedVarianceLogNormalDecoder,
    'gaussian': Toy_GaussianDecoder,
    'lognormal': Toy_LogNormalDecoder,
    'exponential': Toy_ExponentialDecoder,
}


class AmortizedDVAE(nn.Module):
    r"""This model takes as input a meta-dataset and a sequence of meta-datasets,
    all of which are drawn from the same data distribution. 

    We characterize a meta-dataset using 2 RNNs. One to derive a embedding over
    samples from each meta-dataset, and one to derive an embedding over meta-datasets.

    @param distribution: string
                         fixed_var_gaussian|fixed_var_lognormal|gaussian|lognormal|exponential
    """
    def __init__(self, n_datasets, input_dim=2, summary_dim=64, hidden_dim=10, distribution='gaussian'):
        super(AmortizedDVAE, self).__init__()
        assert distribution  in ['fixed_var_gaussian', 'fixed_var_lognormal', 'gaussian', 
                                 'lognormal', 'exponential']
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.distribution = distribution

        # 2 for just learning means and 4 for means and variance
        z_dim = DIST_TO_Z_DIM[distribution]
        decoder_cls = DIST_TO_DECODER_CLS[distribution]
        decoder_params = {'x_stdev': STDEV} if distribution == 'fixed_var_gaussian' else {}

        self.sample_summary_net = MeanMLPEncoder(self.input_dim, self.summary_dim)
        self.meta_summary_net = MeanMLPEncoder(self.summary_dim, self.summary_dim)
        self.encoder = GaussianEncoder(self.summary_dim * 2, z_dim, 
                                       hidden_dim=self.hidden_dim)
        self.decoders = nn.ModuleList([decoder_cls(**decoder_params) 
                                       for _ in range(n_datasets)])
        self.distribution = distribution
    
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

        if self.distribution == 'gaussian' or self.distribution == 'lognormal':
            # we need to bound the variance...
            z_mu[:, 2:] = torch.sigmoid(z_mu[:, 2:])
            z_mu[:, 2:] = (z_mu[:, 2:] - 1) * 5

        z = self.reparameterize(z_mu, z_logvar)
        x_s1, x_s2 = self.decoders[i](z)

        return x_s1, x_s2, z, z_mu, z_logvar

    def estimate_marginal(self, meta_x, meta_x_list, i, n_samples=1000):
        with torch.no_grad():
            meta_e = self.get_sample_summary(meta_x)
            meta_e_list = [self.get_sample_summary(meta_x_i)
                           for meta_x_i in meta_x_list]
            meta_e_list = torch.stack(meta_e_list)
            meta_e_list = meta_e_list.permute(1, 0, 2).contiguous()
            meta_e2 = self.get_meta_summary(meta_e_list)

            x_input = torch.cat([meta_e, meta_e2], dim=1)
            z_mu, z_logvar = self.encoder(x_input)

            if self.distribution == 'gaussian' or self.distribution == 'lognormal':
                # we need to bound the variance...
                z_mu[:, 2:] = torch.sigmoid(z_mu[:, 2:])
                z_mu[:, 2:] = (z_mu[:, 2:] - 1) * 5

            x_s1_list, x_s2_list, z_list = [], [], []

            for _ in range(n_samples):
                z_i = self.reparameterize(z_mu, z_logvar)
                x_s1_i, x_s2_i = self.decoders[i](z_i)
                x_s1_list.append(x_s1_i)
                x_s2_list.append(x_s2_i)
                z_list.append(z_i)

            if self.distribution in ['fixed_var_gaussian', 'gaussian']:
                log_marginal = log_gaussian_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, x_s2_list, 
                    z_list, z_mu, z_logvar)
            elif self.distribution == 'exponential':
                log_marginal = log_exponential_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, z_list, z_mu, z_logvar)
            elif self.distribution in ['fixed_var_lognormal', 'lognormal']:
                log_marginal = log_lognormal_marginal_estimate(
                    meta_x.permute(1, 0, 2), x_s1_list, x_s2_list, 
                    z_list, z_mu, z_logvar)

        return log_marginal


DIST_TO_ELBO_FN = {
    'fixed_var_gaussian': gaussian_elbo_loss,
    'fixed_var_lognormal': lognormal_elbo_loss,
    'gaussian': gaussian_elbo_loss,
    'lognormal': lognormal_elbo_loss,
    'exponential': exponential_elbo_loss, 
}


DIST_TO_DATA_FN =  {
    'fixed_var_gaussian': build_gaussian_distributions,
    'fixed_var_lognormal': build_lognormal_distributions,
    'gaussian': build_gaussian_distributions,
    'lognormal': build_lognormal_distributions,
    'exponential': build_exponential_distributions,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save outputs')
    parser.add_argument('--distribution', type=str, default='fixed_var_gaussian',
                        help='fixed_var_gaussian|fixed_var_lognormal|exponential')
    parser.add_argument('--test-frequency', type=int, default=100,
                        help='frequency of testing [default: 100]')
    parser.add_argument('--n-datasets', type=int, default=30,
                        help='number of datasets [default: 30]')
    parser.add_argument('--n-meta-datasets', type=int, default=10,
                        help='number of meta datasets from each dataset [default: 10]')
    parser.add_argument('--n-meta-samples', type=int, default=20,
                        help='number of samples to represent a meta distribution [default: 20]')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='minibatch size [default: 1]')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='number of hidden dimensions [default: 64]')
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

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    data_generator_fn = DIST_TO_DATA_FN[args.distribution]
    data_generator_args = DIST_TO_GENERATOR_ARGS[args.distribution]

    train_datasets, test_datasets, dataset_classes, dataset_params = \
        data_generator_fn(args.n_datasets, 10000, **data_generator_args)

    train_bag = BagOfDatasets(train_datasets)
    test_bag = BagOfDatasets(test_datasets)

    model = AmortizedDVAE(args.n_datasets, 
                          distribution=args.distribution,
                          hidden_dim=args.hidden_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elbo_loss_fn = DIST_TO_ELBO_FN[args.distribution]


    def step(iter):
        model.train()
        loss = 0

        # list of (n_dataset, batch_size, n_meta_datasets + 1, n_meta_samples, 2)
        meta_datasets = train_bag.sample_meta_datasets(
            args.n_meta_datasets + 1, args.batch_size, args.n_meta_samples)

        for i in range(args.n_datasets):
            meta_dataset_i = meta_datasets[i]
            meta_x_mlp_i = meta_dataset_i[-1]    # shape: (batch_size, n_meta_samples, 2)
            meta_x_rnn_i = meta_dataset_i[:-1]  # shape: (n_meta_datasets, batch_size, n_meta_samples, 2)
            meta_x_rnn_i = [meta_x_rnn_i[j] for j in range(args.n_meta_datasets)]  # make to list

            meta_x_mlp_i = meta_x_mlp_i.to(device)
            meta_x_rnn_i = [meta_x.to(device) for meta_x in meta_x_rnn_i]

            x_s1_i, x_s2_i, z_i, z_mu_i, z_logvar_i = \
                model(meta_x_mlp_i, meta_x_rnn_i, i)

            if args.distribution == 'exponential':
                loss_i = elbo_loss_fn(  meta_x_mlp_i.permute(1, 0, 2), x_s1_i, 
                                        z_i, z_mu_i, z_logvar_i)
            else:
                loss_i = elbo_loss_fn(  meta_x_mlp_i.permute(1, 0, 2), x_s1_i, x_s2_i, 
                                        z_i, z_mu_i, z_logvar_i)
            loss += loss_i
       
        loss /= float(args.n_datasets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    def get_marginal(repetitions=10):
        model.eval()
        log_marginals = np.zeros(args.n_datasets)
        
        with torch.no_grad():
            for _ in range(repetitions):  # do this for ten times!
                meta_datasets = test_bag.sample_meta_datasets(
                    args.n_meta_datasets + 1, args.batch_size, args.n_meta_samples)

                for i in range(args.n_datasets):
                    meta_dataset_i = meta_datasets[i]
                    meta_x_mlp_i = meta_dataset_i[-1]    # shape: (batch_size, n_meta_samples, 2)
                    meta_x_rnn_i = meta_dataset_i[:-1]  # shape: (n_meta_datasets, batch_size, n_meta_samples, 2)
                    meta_x_rnn_i = [meta_x_rnn_i[j] for j in range(args.n_meta_datasets)]  # make to list

                    meta_x_mlp_i = meta_x_mlp_i.to(device)
                    meta_x_rnn_i = [meta_x.to(device) for meta_x in meta_x_rnn_i]

                    log_marginal_i = model.estimate_marginal(meta_x_mlp_i, meta_x_rnn_i, i, n_samples=10)
                    log_marginals[i] += log_marginal_i
            
            log_marginals /= float(repetitions)
        
        return log_marginals


    best_loss = sys.maxsize
    train_elbo_db = np.zeros(args.n_iters)
    test_log_marginal_db = np.zeros((args.n_iters // args.test_frequency, args.n_datasets))

    for i in range(1, args.n_iters + 1):
        train_elbo = step(i)
        train_elbo_db[i - 1] = train_elbo

        print('Iter: {}\tTrain ELBO: {}'.format(i, train_elbo))

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
                'dataset_classes': dataset_classes,
                'dataset_params': dataset_params,
            }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_elbo.npy'), train_elbo_db)
        np.save(os.path.join(args.out_dir, 'test_log_marginal.npy'), test_log_marginal_db)
