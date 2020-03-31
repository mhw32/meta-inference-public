import os
import sys
import torch
from torch import nn
from torch.nn import functional as F, init
from src.utils import bernoulli_log_pdf
from src.objectives.elbo import \
    log_bernoulli_marginal_estimate_sets


class Statistician(nn.Module):
    def __init__(self, c_dim, z_dim, hidden_dim_statistic=3, hidden_dim=400):
        super(Statistician, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim_statistic = hidden_dim_statistic
        self.hidden_dim = hidden_dim

        self.input_dim = 784
        self.statistic_net = LinearStatisticNetwork(
            self.input_dim, self.c_dim, hidden_dim=self.hidden_dim_statistic)
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

    def forward(self, x):
        batch_size, n_samples = x.size(0), x.size(1)
        x = x.view(batch_size, n_samples, self.input_dim)

        c_mean, c_logvar = self.statistic_net(x) 
        c = self.reparameterize_gaussian(c_mean, c_logvar)
        qz_mu, qz_logvar = self.inference_net(x, c)

        qz_mu = qz_mu.view(batch_size, -1, self.z_dim)
        qz_logvar = qz_logvar.view(batch_size, -1, self.z_dim)
        z = self.reparameterize_gaussian(qz_mu, qz_logvar)
        qz_params = [qz_mu, qz_logvar]

        cz_mu, cz_logvar = self.latent_decoder(c)
        pz_params = [cz_mu, cz_logvar]

        x_mu = self.observation_decoder(z, c)

        outputs = (
            (c_mean, c_logvar),
            (qz_params, pz_params),
            (x, x_mu),
        )

        return outputs

    def bernoulli_elbo_loss_sets(self, outputs, reduce=True):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. reconstruction loss
        x, x_mu = x_outputs
        recon_loss = bernoulli_log_pdf(x, x_mu)  # (n_datasets, batch_size) 

        # a) Context divergence: this is the positive D_KL
        c_mu, c_logvar = c_outputs
        kl_c = -0.5 * (1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
        kl_c = torch.sum(kl_c, dim=-1)  # (n_datasets)

        # b) Latent divergence: this is also the positive D_KL
        qz_params, pz_params = z_outputs

        # this is kl(q_z||p_z)
        p_mu, p_logvar = pz_params
        q_mu, q_logvar = qz_params

        # the dimensions won't line up, so you'll need to broadcast!
        p_mu = p_mu.unsqueeze(1).expand_as(q_mu)
        p_logvar = p_logvar.unsqueeze(1).expand_as(q_logvar)

        kl_z = 0.5 * (p_logvar - q_logvar + ((q_mu - p_mu)**2 + q_logvar.exp())/p_logvar.exp() - 1)
        kl_z = torch.sum(kl_z, dim=-1)  # (n_datasets, batch_size)

        # THESE ARE ALSO UNNORMALIZED!!!
        ELBO = -recon_loss + kl_z  # these will both be (n_datasets, batch_size)
        ELBO = ELBO.sum(-1) / x.size()[1]  # averaging over (batch_size == self.sample_size)
        ELBO = ELBO + kl_c # now this is (n_datasets,)

        if reduce:
            return torch.mean(ELBO)  # averaging over (n_datasets)
        else:
            return ELBO # (n_datasets)

    def estimate_marginal(self, x, n_samples=100):
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
        x = x.view(batch_size, n_samples, self.input_dim)
        c_mean, c_logvar = self.statistic_net(x) 
        c = self.reparameterize_gaussian(c_mean, c_logvar)
        z_mu, _ = self.inference_net(x, c)
        return z_mu


class LinearStatisticNetwork(nn.Module):
    def __init__(self, n_features, c_dim, hidden_dim=128):
        super(LinearStatisticNetwork, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
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
        """
        average pooling WITHIN each dataset!
        """
        e = e.mean(1).view(-1, self.hidden_dim)
        return e


class LinearInferenceNetwork(nn.Module):
    def __init__(self, n_features, c_dim, z_dim, hidden_dim=128):
        super(LinearInferenceNetwork, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.fc_h = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
    
        self.fc1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)

    def forward(self, h, c):
        batch_size = h.size(0)
        eh = h.view(-1, self.n_features)  # embed h
        eh = self.fc_h(eh)
        eh = eh.view(batch_size, -1, self.hidden_dim)
        ec = self.fc_c(c)
        ec = ec.view(batch_size, -1, self.hidden_dim).expand_as(eh)

        e = torch.cat([eh, ec], dim=2)
        e = F.elu(e.view(-1, 2 * self.hidden_dim))
        e = F.elu(self.fc1(e))
        e = self.fc_params(e)
        mean, logvar = torch.chunk(e, 2, dim=1)

        return mean, logvar


class LinearLatentDecoder(nn.Module):
    def __init__(self, n_features, c_dim, z_dim, hidden_dim=128):
        super(LinearLatentDecoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)

    def forward(self, c):
        batch_size = c.size(0)
        ec = self.fc_c(c)
        ec = ec.view(batch_size, -1, self.hidden_dim)
        e = F.elu(ec.view(-1, self.hidden_dim))
        e = F.elu(self.fc1(e))
        e = self.fc_params(e)
        mean, logvar = torch.chunk(e, 2, dim=1)

        return mean, logvar


class LinearObservationDecoder(nn.Module):
    def __init__(self, n_features, c_dim, z_dim, hidden_dim=128):
        super(LinearObservationDecoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_initial = nn.Linear(2 * self.hidden_dim, 256 * 4 * 4)
        self.fc3 = nn.Linear(256 * 4 * 4, 784)

    def forward(self, z, c):
        batch_size = z.size(0)
        ez = self.fc_z(z)
        ez = ez.view(batch_size, -1, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(batch_size, -1, self.hidden_dim).expand_as(ez)

        e = torch.cat([ez, ec], dim=2)
        e = F.elu(e)
        e = e.view(-1, 2 * self.hidden_dim)
        e = F.elu(self.fc_initial(e))

        e = self.fc3(e)
        e = e.view(batch_size, -1, 784)
        e = torch.sigmoid(e)

        return e


class PrePool(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(PrePool, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # modules: 1 fc layer
        self.fc = nn.Linear(self.n_features, self.hidden_dim)

    def forward(self, h):
        # reshape and affine
        e = h.view(-1, self.n_features)  # batch_size * sample_size
        e = F.elu(self.fc(e))

        return e


class PostPool(nn.Module):
    def __init__(self, hidden_dim, c_dim):
        super(PostPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)

    def forward(self, e):
        e = self.fc_params(e)
        mean, logvar = torch.chunk(e, 2, dim=1)

        return mean, logvar
