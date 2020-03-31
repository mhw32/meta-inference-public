import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from src.utils import bernoulli_log_pdf, gaussian_log_pdf, normal_init
from src.objectives.elbo import \
    log_bernoulli_marginal_estimate_sets
from src.models.shared import \
    FCResBlock, Conv2d3x3, SharedConvolutionalEncoder, NonLinear


class HomoEncoder_VampPrior(nn.Module):
    # num_components = number of pseudo_inputs
    # choose hyperparameters from 
    # https://github.com/jmtomczak/vae_vampprior/blob/master/experiment.py

    def __init__(self, c_dim, z_dim, device, num_components=500, pseudoinputs_samples=10, 
                 pseudoinputs_mean=0.05, pseudoinputs_std=0.01, hidden_dim=400):
        super(HomoEncoder_VampPrior, self).__init__()
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.num_components = num_components
        self.pseudoinputs_samples = pseudoinputs_samples
        self.pseudoinputs_mean = pseudoinputs_mean
        self.pseudoinputs_std = pseudoinputs_std

        self.input_dim = 784
        self.statistic_net = LinearStatisticNetwork(
            self.input_dim, self.c_dim, hidden_dim=self.hidden_dim)
        self.inference_net = LinearInferenceNetwork(
            self.input_dim, self.c_dim, self.z_dim, 
            hidden_dim=self.hidden_dim)
        self.latent_decoder = LinearLatentDecoder(
            self.input_dim, self.c_dim, self.z_dim, 
            hidden_dim=self.hidden_dim)
        self.observation_decoder = LinearObservationDecoder(
            self.input_dim, self.c_dim, self.z_dim, 
            hidden_dim=self.hidden_dim)

        # initialize weights
        self.apply(self.weights_init)
        self.add_pseudoinputs(device)  # for vamprior

    def forward(self, x, x_dset):
        batch_size = x.size(0)
        n_samples = x_dset.size(1)
        
        x = x.view(batch_size, self.input_dim)
        x_dset = x_dset.view(batch_size, n_samples, self.input_dim)

        c_mu, c_logvar = self.statistic_net(x_dset)
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        qz_mean, qz_logvar = self.inference_net(x, c)
        z = self.reparameterize_gaussian(qz_mean, qz_logvar)
        cz_mean, cz_logvar = self.latent_decoder(c)
        x_mu = self.observation_decoder(z, c)

        outputs = (
            (c, c_mu, c_logvar),
            (qz_mean, qz_logvar, cz_mean, cz_logvar),
            (x, x_mu),
        )

        return outputs

    def bernoulli_elbo(self, outputs, reduce=True):
        return self.elbo(outputs, reduce=reduce)

    def elbo(self, outputs, reduce=True):
        (c, c_mu, c_logvar), (q_mu, q_logvar, p_mu, p_logvar), (x, x_mu) = outputs
        batch_size = x.size(0)
        recon_loss = bernoulli_log_pdf(x.view(batch_size, -1), 
                                       x_mu.view(batch_size, -1))
        log_p_c = self.log_p_c(c)
        log_q_c = gaussian_log_pdf(c, c_mu, c_logvar)
        kl_c = -(log_p_c - log_q_c)
        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=1)

        ELBO = -recon_loss + kl_z + kl_c

        if reduce:
            return torch.mean(ELBO)
        else:
            return ELBO

    def log_p_c(self, c):  # this is a function now thanks to a learned prior
        x_flat = self.means(self.idle_input)
        x_dset = x_flat.view(self.num_components, self.pseudoinputs_samples, self.input_dim)
        c_p_mean, c_p_logvar = self.statistic_net(x_dset)
        c_expand = c.unsqueeze(1)
        means = c_p_mean.unsqueeze(0)
        logvars = c_p_logvar.unsqueeze(0)

        a = gaussian_log_pdf(c_expand, means, logvars) - math.log(self.num_components)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return log_prior

    def add_pseudoinputs(self, device):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.num_components, self.pseudoinputs_samples * np.prod(self.input_dim), 
                               bias=False, activation=nonlinearity)
        normal_init(self.means.linear, self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = torch.eye(self.num_components, self.num_components).to(device)
        self.idle_input = self.idle_input.detach()

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
        x = x.view(batch_size, self.input_dim)
        x_dset = x_dset.view(batch_size, n_samples, self.input_dim)
        c_mu, c_logvar = self.statistic_net(x_dset) 
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        qz_mean, _ = self.inference_net(x, c)
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


class LinearStatisticNetwork(nn.Module):
    def __init__(self, input_dim, c_dim, hidden_dim=400):
        super(LinearStatisticNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)

    def forward(self, h):
        eh = self.fc1(h)
        e_mean = torch.mean(eh, dim=1)
        z_params = self.fc_params(e_mean)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)
        
        return z_mu, z_logvar


class LinearInferenceNetwork(nn.Module):
    """
    Inference network q(z|h, z, c) gives approximate posterior over latent variables.
    """
    def __init__(self, input_dim, c_dim, z_dim, hidden_dim=400):
        super(LinearInferenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc_x = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        
        self.fc1 = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.fc_params = nn.Linear(self.hidden_dim, 2*self.z_dim)

    def forward(self, x, c):
        # combine c and x
        batch_size = x.size(0)
        xh = self.fc_x(x)

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)

        # concatenate
        e = F.elu(torch.cat([xh, ec], dim=1))

        # apply fc layer(s)
        e = F.elu(self.fc1(e))

        # affine transformation to parameters
        z_params = self.fc_params(e)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar


class LinearLatentDecoder(nn.Module):
    def __init__(self, input_dim, c_dim, z_dim, hidden_dim=400):
        super(LinearLatentDecoder, self).__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)

    def forward(self, c):
        batch_size = c.size(0)

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)

        # concat things together
        e = F.elu(ec)

        # apply 3 fc layers
        e = F.elu(self.fc1(e))

        # affine transformation to parameters
        z_params = self.fc_params(e)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

        return z_mu, z_logvar


class LinearObservationDecoder(nn.Module):
    def __init__(self, input_dim, c_dim, z_dim, hidden_dim=400):
        super(LinearObservationDecoder, self).__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        # initial fc layer
        self.fc_initial = nn.Linear(2*self.hidden_dim, 256*4*4)

        # added in final fc layers instead of convolutions
        self.fc3 = nn.Linear(256*4*4, self.input_dim)

    def forward(self, z, c):
        batch_size = z.size(0)
        ez = self.fc_z(z)
        ec = self.fc_c(c)

        # combine z and cs together
        e = F.elu(torch.cat([ez, ec], dim=1))
        e = F.elu(self.fc_initial(e))

        # instead of convolutional layers, add in the fc decoder architecture from before
        x_mu = self.fc3(e)
        x_mu = torch.sigmoid(x_mu)

        return x_mu
