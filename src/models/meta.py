import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.shared import \
    FCResBlock, Conv2d3x3, SharedConvolutionalEncoder, \
    MeanMLPEncoder
from src.objectives.elbo import \
    log_bernoulli_marginal_estimate


class MetaVAE(nn.Module):
    def __init__(self, n_datasets, z_dim, summary_dim=400, hidden_dim=400):
        super(MetaVAE, self).__init__()
        self.n_datasets = n_datasets
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.summary_dim = summary_dim

        self.input_dim = 784
        self.summary_net = MeanMLPEncoder(self.input_dim, self.summary_dim)
        self.encoder_net = LinearInferenceNetwork(
            self.input_dim + self.summary_dim, self.z_dim,
            hidden_dim=self.hidden_dim)
        self.decoder_nets = nn.ModuleList([
            LinearObservationDecoder(
                self.input_dim, self.z_dim, hidden_dim=self.hidden_dim) 
            for _ in range(self.n_datasets)
        ])

        self.input_dim = 784

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_dset, i):
        batch_size, n_samples = x.size(0), x_dset.size(1)
        x = x.view(batch_size, self.input_dim)
        x_dset = x_dset.view(batch_size, x_dset.size(1), self.input_dim)
        summary = self.summary_net(x_dset)
        x_summary = torch.cat([x, summary.squeeze(0)], dim=1)
        z_mu, z_logvar = self.encoder_net(x_summary)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decoder_nets[i](z)

        return x, x_mu, z, z_mu, z_logvar

    def extract_codes(self, x, x_dset):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)
        x_dset = x_dset.view(batch_size, x_dset.size(1), self.input_dim)
        enc_summary = self.summary_net(x_dset)
        enc_input = torch.cat([x, enc_summary], dim=1)
        z_mu, _ = self.encoder_net(enc_input)
        return z_mu

    def estimate_marginal(self, x, x_dset, i, n_samples=100):
        with torch.no_grad():
            batch_size, n_samples = x.size(0), x_dset.size(1)
            x = x.view(batch_size, self.input_dim)
            x_dset = x_dset.view(batch_size, n_samples, self.input_dim)
            summary = self.summary_net(x_dset)
            x_summary = torch.cat([x, summary.squeeze(0)], dim=1)
            z_mu, z_logvar = self.encoder_net(x_summary)

            x_mu_list, z_list = [], []
            
            for _ in range(n_samples):
                z_i = self.reparameterize(z_mu, z_logvar)
                x_mu_i = self.decoder_nets[i](z_i)
                z_list.append(z_i)
                x_mu_list.append(x_mu_i)

            log_p_x = log_bernoulli_marginal_estimate(
                x, x_mu_list, z_list, z_mu, z_logvar)
        
        return log_p_x


class LinearInferenceNetwork(nn.Module):
    r"""Parametrizes q(z|x).
    @param k: integer
              number of data samples
    @param input_dim: integer
                      number of input dimension.
    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 400]
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, z_dim, hidden_dim=400):
        super(LinearInferenceNetwork, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.z_dim * 2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        h2 = self.fc3(h1)
        mu, logvar = torch.chunk(h2, 2, dim=1)

        return mu, logvar


class LinearObservationDecoder(nn.Module):
    r"""Parametrizes p(x|z).
    Architecture design modeled after:
    https://github.com/jmtomczak/vae_vampprior
    @param input_dim: integer
                      number of input dimension.
    @param z_dim: integer
                  number of latent dimensions.
    @param hidden_dim: integer [default: 400]
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, z_dim, hidden_dim=400):
        super(LinearObservationDecoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        mu = torch.sigmoid(h)

        return mu  # parameters of bernoulli
