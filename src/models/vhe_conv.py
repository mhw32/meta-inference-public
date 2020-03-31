import os
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from src.utils import bernoulli_log_pdf
from src.objectives.elbo import log_bernoulli_marginal_estimate_sets
from src.models.shared import \
    FCResBlock, Conv2d3x3, SharedConvolutionalEncoder


class ConvHomoEncoder(nn.Module):
    def __init__(self, c_dim, z_dim, n_hidden=3, hidden_dim=400, n_channels=1):
        super(ConvHomoEncoder, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.encoder_net = SharedConvolutionalEncoder(self.n_channels)
        self.feature_dim = 256*4*4
        self.statistic_net = ConvStatisticNetwork(
            self.feature_dim, c_dim, hidden_dim)
        self.inference_net = ConvInferenceNetwork(
            self.feature_dim , c_dim, z_dim, n_hidden, hidden_dim)
        self.latent_decoder = ConvLatentDecoder(c_dim, z_dim, n_hidden, hidden_dim)
        self.observation_decoder = ConvObservationDecoder(c_dim, z_dim, hidden_dim, n_channels=n_channels)

        # initialize weights
        self.apply(self.weights_init)

    def forward(self, x, x_dset):
        batch_size = x.size(0)
        n_samples = x_dset.size(1)

        h = self.encoder_net(x)
        h_dset = self.encoder_net(x_dset)
        h = h.view(batch_size, self.feature_dim)
        h_dset = h_dset.view(batch_size, n_samples, 256*4*4)
        c_mu, c_logvar = self.statistic_net(h_dset)
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        qz_mean, qz_logvar = self.inference_net(h, c)
        z = self.reparameterize_gaussian(qz_mean, qz_logvar)
        cz_mean, cz_logvar = self.latent_decoder(c)
        x_mu = self.observation_decoder(z, c)

        outputs = (
            (c_mu, c_logvar),
            (qz_mean, qz_logvar, cz_mean, cz_logvar),
            (x, x_mu)
        )

        return outputs

    def gaussian_elbo(self, outputs, reduce=True):
        (c_mu, c_logvar), (q_mu, q_logvar, p_mu, p_logvar), (x, x_mu, x_logvar) = outputs
        batch_size = x.size(0)
        recon_loss = gaussian_log_pdf(x.view(batch_size, -1),
                                       x_mu.view(batch_size, -1),
                                       x_logvar.view(batch_size, -1))
        kl_c = -0.5 * (1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
        kl_c = torch.sum(kl_c, dim=1)
        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=1)

        ELBO = -recon_loss + kl_z + kl_c

        if reduce:
            return torch.mean(ELBO)
        else:
            return ELBO # (n_datasets)

    def bernoulli_elbo(self, outputs, reduce=True):
        (c_mu, c_logvar), (q_mu, q_logvar, p_mu, p_logvar), (x, x_mu) = outputs
        batch_size = x.size(0)
        recon_loss = bernoulli_log_pdf(x.view(batch_size, -1),
                                       x_mu.view(batch_size, -1))
        kl_c = -0.5 * (1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
        kl_c = torch.sum(kl_c, dim=1)
        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=1)

        ELBO = -recon_loss + kl_z + kl_c

        if reduce:
            return torch.mean(ELBO)
        else:
            return ELBO # (n_datasets)

    def elbo(self, outputs, reduce=True):
        return self.bernoulli_elbo(outputs, reduce=reduce)

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def extract_codes(self, x, x_dset):
        batch_size = x.size(0)
        n_samples = x_dset.size(1)

        h = self.encoder_net(x)
        h_dset = self.encoder_net(x_dset)
        h = h.view(batch_size, 256*4*4)
        h_dset = h_dset.view(batch_size, n_samples, 256*4*4)

        c_mu, c_logvar = self.statistic_net(h_dset)
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        qz_mean, _ = self.inference_net(h, c)

        return qz_mean

    def estimate_marginal(self, x, x_dset, n_samples=100):
        with torch.no_grad():
            elbo_list = []
            for i in range(n_samples):
                outputs = self.forward(x, x_dset)
                elbo = self.elbo(outputs, reduce=False)
                elbo_list.append(elbo)

            # bernoulli decoder
            log_p_x = log_bernoulli_marginal_estimate_sets(elbo_list)

        return log_p_x


class ConvStatisticNetwork(nn.Module):
    def __init__(self, input_dim, c_dim, hidden_dim):
        super(ConvStatisticNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.prepool = PrePool(self.input_dim, self.hidden_dim)
        self.postpool = PostPool(self.hidden_dim, self.c_dim)

    def forward(self, h):
        batch_size, sample_size, _ = h.size()
        eh = self.prepool(h)
        eh = eh.view(batch_size, sample_size, -1)
        e_mean = torch.mean(eh, dim=1)
        c_mu, c_logvar = self.postpool(e_mean)

        return c_mu, c_logvar


class ConvInferenceNetwork(nn.Module):
    """
    Inference network q(z|h, c) gives approximate posterior over latent variables.
    """
    def __init__(self, input_dim, c_dim, z_dim, n_hidden, hidden_dim):
        super(ConvInferenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.fc_h = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=2 * self.hidden_dim, n=2 * self.n_hidden,
                                       batch_norm=True)
        self.fc_params = nn.Linear(2 * self.hidden_dim, 2 * self.z_dim)

    def forward(self, h, c):
        batch_size = h.size(0)
        eh = self.fc_h(h)
        ec = self.fc_c(c)
        e = F.elu(torch.cat([eh, ec], dim=1))
        e = self.fc_res_block(e)
        z_params = self.fc_params(e)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar



class ConvLatentDecoder(nn.Module):
    def __init__(self, c_dim, z_dim, n_hidden, hidden_dim):
        super(ConvLatentDecoder, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       batch_norm=True)
        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)

    def forward(self, c):
        batch_size = c.size(0)
        ec = self.fc_c(c)
        e = F.elu(ec)
        e = self.fc_res_block(e)
        z_params = self.fc_params(e)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)
        return z_mu, z_logvar



class ConvObservationDecoder(nn.Module):
    def __init__(self, c_dim, z_dim, hidden_dim, n_channels=1):
        super(ConvObservationDecoder, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels

        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_initial = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.fc_linear = nn.Linear(2*self.hidden_dim, 256*4*4)

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=256,
                               kernel_size=2, stride=2),
            Conv2d3x3(in_channels=256, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=2, stride=2),
            Conv2d3x3(in_channels=128, out_channels=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
        ])

        self.conv_mean = nn.Conv2d(64, self.n_channels, kernel_size=1)

    def forward(self, z, c):
        batch_size = z.size(0)
        ez = self.fc_z(z)
        ec = self.fc_c(c)

        # combine z and cs together
        e = F.elu(torch.cat([ez, ec], dim=1))
        e = F.elu(self.fc_initial(e))
        e = self.fc_linear(e)
        e = e.view(-1, 256, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = F.elu(e)

        x_mu = self.conv_mean(e)
        x_mu = torch.sigmoid(x_mu)

        return x_mu


# pre-pooling for statistics network
class PrePool(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PrePool, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # modules
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, h):
        # reshape and affine
        e = h.view(-1, self.input_dim)
        e = self.fc(e)
        if e.size(0) != 1:
            e = self.bn(e)
        e = F.elu(e)

        return e


# post-pooling for statistics network
class PostPool(nn.Module):
    def __init__(self, hidden_dim, c_dim):
        super(PostPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        # modules
        self.fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.Linear(self.hidden_dim, self.hidden_dim)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim)])

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)

    def forward(self, e):
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            e = fc(e)
            if e.size(0) != 1:
                e = bn(e)
            e = F.elu(e)

        # affine transformation to parameters
        e = self.fc_params(e)
        mu, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mu, logvar
