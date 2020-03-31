import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.shared import \
    FCResBlock, Conv2d3x3, SharedConvolutionalEncoder, \
    MeanMLPEncoder
from src.objectives.elbo import log_bernoulli_marginal_estimate


class ConvMetaVAE(nn.Module):
    def __init__(self, n_datasets, z_dim, summary_dim=400, n_hidden=3, hidden_dim=400, n_channels=1):
        super(ConvMetaVAE, self).__init__()

        self.n_datasets = n_datasets
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.summary_dim = summary_dim
        self.n_channels = n_channels

        # NOTE: flip this when we want deeper networks
        self.encoder_net = SharedConvolutionalEncoder(self.n_channels)
        self.feature_dim = 256*4*4
        self.summary_net = MeanMLPEncoder(self.feature_dim, self.summary_dim)
        self.inference_net = ConvInferenceNetwork(
            self.feature_dim, self.summary_dim, self.z_dim,
            self.n_hidden, self.hidden_dim)
        self.decoder_nets = nn.ModuleList([
            ConvObservationDecoder(self.z_dim, self.hidden_dim, n_channels=self.n_channels)
            for _ in range(self.n_datasets)
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_dset, i):
        batch_size, n_samples = x.size(0), x_dset.size(1)

        # NOTE: flip this when we want deeper networks
        h = self.encoder_net(x)
        h_dset = self.encoder_net(x_dset)
        h = h.view(batch_size, 256*4*4)
        h_dset = h_dset.view(batch_size, n_samples, 256*4*4)
        summary = self.summary_net(h_dset)
        z_mu, z_logvar = self.inference_net(h, summary)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decoder_nets[i](z)

        return x, x_mu, z, z_mu, z_logvar

    def extract_codes(self, x, x_dset):
        batch_size, n_samples = x.size(0), x_dset.size(1)
        h = self.encoder_net(x)
        h_dset = self.encoder_net(x_dset)
        h = h.view(batch_size, 256*4*4)
        h_dset = h_dset.view(batch_size, n_samples, 256*4*4)
        summary = self.summary_net(h_dset)
        z_mu, z_logvar = self.inference_net(h, summary)
        return z_mu

    def estimate_marginal(self, x, x_dset, i, n_samples=100):
        with torch.no_grad():
            batch_size, n_samples = x.size(0), x_dset.size(1)

            # NOTE: flip this when we want deeper networks
            h = self.encoder_net(x)
            h_dset = self.encoder_net(x_dset)
            h = h.view(batch_size, 256*4*4)
            h_dset = h_dset.view(batch_size, n_samples, 256*4*4)
            summary = self.summary_net(h_dset)
            z_mu, z_logvar = self.inference_net(h, summary)

            x_mu_list, z_list = [], []

            for _ in range(n_samples):
                z_i = self.reparameterize(z_mu, z_logvar)
                x_mu_i = self.decoder_nets[i](z_i)
                z_list.append(z_i)
                x_mu_list.append(x_mu_i)

            log_p_x = log_bernoulli_marginal_estimate(
                x, x_mu_list, z_list, z_mu, z_logvar)

        return log_p_x


class ConvInferenceNetwork(nn.Module):
    def __init__(self, input_dim, summary_dim, z_dim, n_hidden, hidden_dim):
        super(ConvInferenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.summary_dim = summary_dim
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.fc_h = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.summary_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=2 * self.hidden_dim, n=2 * self.n_hidden,
                                       batch_norm=True)
        self.fc_params = nn.Linear(2 * self.hidden_dim, 2 * self.z_dim)

    def forward(self, h, summary):
        # combine c and x
        batch_size = h.size(0)
        eh = self.fc_h(h)

        # embed c and expand for broadcast addition
        ec = self.fc_c(summary)

        # concatenate
        e = F.elu(torch.cat([eh, ec], dim=1))

        # apply fc layer(s)
        e = self.fc_res_block(e)

        # affine transformation to parameters
        z_params = self.fc_params(e)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar


class ConvObservationDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_channels=1):
        super(ConvObservationDecoder, self).__init__()
        # self.summary_dim = summary_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_channels =  n_channels

        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_initial = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_linear = nn.Linear(self.hidden_dim, 256*4*4)

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(256, 256),
            Conv2d3x3(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Conv2d3x3(256, 128),
            Conv2d3x3(128, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            Conv2d3x3(128, 64),
            # TODO: this is a little different here but maybe it's ok?
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
        ])

        self.conv_final = nn.Conv2d(64, self.n_channels, kernel_size=1)

    # def forward(self, z, summary):
    def forward(self, z):
        batch_size = z.size(0)
        e = self.fc_z(z)

        e = F.elu(self.fc_initial(e))
        e = self.fc_linear(e)
        e = e.view(-1, 256, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = F.elu(e)

        x_mu = self.conv_final(e)
        x_mu = torch.sigmoid(x_mu)

        return x_mu
