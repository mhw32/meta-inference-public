import os
import sys
import torch
from torch import nn
from torch.nn import functional as F, init
from src.utils import bernoulli_log_pdf
from src.objectives.elbo import \
    log_bernoulli_marginal_estimate_sets
from src.models.shared import \
    FCResBlock, Conv2d3x3, SharedConvolutionalEncoder


class ConvStatistician(nn.Module):
    def __init__(self, c_dim, z_dim, hidden_dim_statistic=3, n_hidden=3, hidden_dim=400, n_channels=1):
        super(ConvStatistician, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim_statistic = hidden_dim_statistic
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels

        self.encoder_net = SharedConvolutionalEncoder(self.n_channels)
        self.feature_dim = 256*4*4
        self.statistic_net = ConvStatisticNetwork(
            self.feature_dim, self.c_dim, self.hidden_dim)
        self.inference_net = ConvInferenceNetwork(
            self.feature_dim , self.c_dim, self.z_dim, self.n_hidden, self.hidden_dim)
        self.latent_decoder = ConvLatentDecoder(
            self.feature_dim, self.c_dim, self.z_dim, self.n_hidden, self.hidden_dim)
        self.observation_decoder = ConvObservationDecoder(
            self.feature_dim, self.c_dim, self.z_dim, self.hidden_dim, n_channels=self.n_channels)

        # initialize weights
        self.apply(self.weights_init)

    def forward(self, x):
        # convolutional encoder
        batch_size = x.size(0)
        sample_size = x.size(1)
        h = self.encoder_net(x)
        h = h.view(batch_size, sample_size, self.feature_dim)
        c_mu, c_logvar = self.statistic_net(h)
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        qz_mu, qz_logvar = self.inference_net(h, c)

        qz_mu = qz_mu.view(batch_size, sample_size, -1)
        qz_logvar = qz_logvar.view(batch_size, sample_size, -1)
        z = self.reparameterize_gaussian(qz_mu, qz_logvar)
        qz_params = [qz_mu, qz_logvar]

        cz_mu, cz_logvar = self.latent_decoder(c)
        pz_params = [cz_mu, cz_logvar]

        # observation decoder
        x_mu = self.observation_decoder(z, c)
        x_mu = x_mu.view(batch_size, sample_size, x.size(2),
                         x.size(3), x.size(4))

        outputs = (
            (cz_mu, cz_logvar),
            (qz_params, pz_params),
            (x, x_mu)
        )

        return outputs

    def bernoulli_elbo_loss_sets(self, outputs, reduce=True):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mu = x_outputs
        n_datasets = x.size(0)
        batch_size = x.size(1)
        recon_loss = bernoulli_log_pdf(x.view(n_datasets * batch_size, -1),
                                       x_mu.view(n_datasets * batch_size, -1))
        recon_loss = recon_loss.view(n_datasets, batch_size)

        # 2. KL Divergence terms
        # a) Context divergence
        c_mu, c_logvar = c_outputs
        kl_c = -0.5 * (1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
        kl_c = torch.sum(kl_c, dim=-1)  # (n_datasets)

        # b) Latent divergences
        qz_params, pz_params = z_outputs

        # this is kl(q_z||p_z)
        p_mu, p_logvar = pz_params
        q_mu, q_logvar = qz_params

        # the dimensions won't line up, so you'll need to broadcast!
        p_mu = p_mu.unsqueeze(1).expand_as(q_mu)
        p_logvar = p_logvar.unsqueeze(1).expand_as(q_logvar)

        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=-1)  # (n_datasets, batch_size)

        ELBO = -recon_loss + kl_z  # these will both be (n_datasets, batch_size)
        ELBO = ELBO.sum(-1) / x.size()[1]  # averaging over (batch_size == self.sample_size)
        ELBO = ELBO + kl_c # now this is (n_datasets,)

        if reduce:
            return torch.mean(ELBO)  # averaging over (n_datasets)
        else:
            return ELBO # (n_datasets)

    def gaussian_elbo_loss_sets(self, outputs, reduce=True):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mu, x_logvar = x_outputs
        n_datasets = x.size(0)
        batch_size = x.size(1)
        recon_loss = gaussian_log_pdf(x.view(n_datasets * batch_size, -1), 
            x_mu.view(n_datasets * batch_size, -1), 
            x_logvar.view(n_datasets * batch_size, -1))
        recon_loss = recon_loss.view(n_datasets, batch_size)

        # 2. KL Divergence terms
        # a) Context divergence
        c_mu, c_logvar = c_outputs
        kl_c = -0.5 * (1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
        kl_c = torch.sum(kl_c, dim=-1)  # (n_datasets)

        # b) Latent divergences
        qz_params, pz_params = z_outputs

        # this is kl(q_z||p_z)
        p_mu, p_logvar = pz_params
        q_mu, q_logvar = qz_params

        # the dimensions won't line up, so you'll need to broadcast!
        p_mu = p_mu.unsqueeze(1).expand_as(q_mu)
        p_logvar = p_logvar.unsqueeze(1).expand_as(q_logvar)

        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=-1)  # (n_datasets, batch_size)

        ELBO = -recon_loss + kl_z  # these will both be (n_datasets, batch_size)
        ELBO = ELBO.sum(-1) / x.size()[1]  # averaging over (batch_size == self.sample_size)
        ELBO = ELBO + kl_c # now this is (n_datasets,)

        if reduce:
            return torch.mean(ELBO)  # averaging over (n_datasets)
        else:
            return ELBO # (n_datasets)

    def estimate_gaussian_marginal(self, x, n_samples=100):
        # need to compute a bunch of outputs
        with torch.no_grad():
            elbo_list = []
            for i in range(n_samples):
                outputs = self.forward(x)
                elbo = self.gaussian_elbo_loss_sets(outputs, reduce=False)
                elbo_list.append(elbo)

            # bernoulli decoder
            log_p_x = log_gaussian_marginal_estimate_sets(elbo_list)

        return log_p_x

    def estimate_bernoulli_marginal(self, x, n_samples=100):
        # need to compute a bunch of outputs
        with torch.no_grad():
            elbo_list = []
            for i in range(n_samples):
                outputs = self.forward(x)
                elbo = self.bernoulli_elbo_loss_sets(outputs, reduce=False)
                elbo_list.append(elbo)

            # bernoulli decoder
            log_p_x = log_bernoulli_marginal_estimate_sets(elbo_list)

        return log_p_x

    def estimate_marginal(self, x, n_samples=100):
        return self.estimate_bernoulli_marginal(x, n_samples=n_samples)

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

    def extract_codes(self, x):
        batch_size, n_samples = x.size(0), x.size(1)
        h = self.encoder_net(x)
        h = h.view(batch_size, n_samples, self.feature_dim)
        c_mu, c_logvar = self.statistic_net(h)
        c = self.reparameterize_gaussian(c_mu, c_logvar)
        z_mu, _ = self.inference_net(h, c)
        return z_mu


class ConvInferenceNetwork(nn.Module):
    def __init__(self, feature_dim, c_dim, z_dim, n_hidden, hidden_dim):
        super(ConvInferenceNetwork, self).__init__()
        self.n_features = feature_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        # modules
        self.fc_h = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=2 * self.hidden_dim, n=2 * self.n_hidden,
                                       batch_norm=True)

        self.fc_params = nn.Linear(2 * self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, h, c):
        batch_size, sample_size = h.size(0), h.size(1)
        eh = h.view(-1, self.n_features)
        eh = self.fc_h(eh)
        eh = eh.view(-1, sample_size, self.hidden_dim)
        sample_size = eh.size(1)

        ec = self.fc_c(c)
        ec = ec.view(-1, 1, self.hidden_dim).expand_as(eh)

        e = torch.cat([eh, ec], dim=2)
        e = e.view(-1, 2 * self.hidden_dim)
        e = F.elu(e)

        e = self.fc_res_block(e)
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mu, logvar = e[:, :self.z_dim].contiguous(), \
                     e[:, self.z_dim:].contiguous()

        return mu, logvar


class ConvStatisticNetwork(nn.Module):
    def __init__(self, n_features, c_dim, hidden_dim):
        super(ConvStatisticNetwork, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        # modules
        self.prepool = PrePool(self.n_features, self.hidden_dim)
        self.postpool = PostPool(self.hidden_dim, self.c_dim)

    def forward(self, h):
        batch_size = h.size(0)
        e = self.prepool(h)
        e = e.view(batch_size, -1, self.hidden_dim)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    def pool(self, e):
        e = e.mean(1).view(-1, self.hidden_dim)
        return e


# latent decoder p(z|c)
class ConvLatentDecoder(nn.Module):
    def __init__(self, n_features, c_dim, z_dim, n_hidden, hidden_dim):
        super(ConvLatentDecoder, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       batch_norm=True)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, c):
        batch_size = c.size(0)
        ec = self.fc_c(c)
        e = F.elu(ec.view(-1, self.hidden_dim))
        e = self.fc_res_block(e)
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)
        mu, logvar = e[:, :self.z_dim].contiguous(), \
                     e[:, self.z_dim:].contiguous()
        return mu, logvar


class ConvObservationDecoder(nn.Module):
    def __init__(self, n_features, c_dim, z_dim, hidden_dim, n_channels=1):
        super(ConvObservationDecoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.fc_zs = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_initial = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.fc_linear = nn.Linear(2 * self.hidden_dim, self.n_features)

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
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=2, stride=2)
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

    def forward(self, zs, c):
        batch_size = zs.size(0)
        sample_size = zs.size(1)
        ezs = self.fc_zs(zs)
        ezs = ezs.view(-1, sample_size, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(-1, 1, self.hidden_dim).expand_as(ezs)

        e = F.elu(torch.cat([ezs, ec], dim=2))
        e = e.view(-1, 2 * self.hidden_dim)

        e = F.elu(self.fc_initial(e))
        e = self.fc_linear(e)
        e = e.view(-1, 256, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = F.elu(e)

        mean = self.conv_mean(e)
        mean = torch.sigmoid(mean)

        return mean


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
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            e = fc(e)
            if e.size(0) != 1:
                e = bn(e)
            e = F.elu(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.c_dim)
        if e.size(0) != 1:
            e = self.bn_params(e)
        # e = self.bn_params(e)
        e = e.view(-1, 2 * self.c_dim)

        mu, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mu, logvar
