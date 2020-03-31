import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianEncoder(nn.Module):
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
        super(GaussianEncoder, self).__init__()

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


class RNNEncoder(nn.Module):
    r"""Parametrizes h = f(x1,x2,...,xn).
    
    @param input_dim: integer
                      number of input dimension.
    @param hidden_dim: integer
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, hidden_dim):
        super(RNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1,
                          batch_first=True, bidirectional=False)

    def forward(self, x_list):
        _, h = self.rnn(x_list)
        return h


class MeanMLPEncoder(nn.Module):
    r"""Parametrizes f(x1,x2,...,xn) = 1/n sum(f(x1), f(x2), ..., f(xn))
    
    @param input_dim: integer
                      number of input dimension.
    @param hidden_dim: integer
                       number of hidden dimensions.
    """
    def __init__(self, input_dim, hidden_dim):
        super(MeanMLPEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x_list):
        output = 0
        for i in range(x_list.size(1)):
            output += self.mlp(x_list[:, i])
        return output / float(len(x_list))


class Toy_FixedVarianceGaussianDecoder(nn.Module):
    def __init__(self, x_stdev=0.1):
        super(Toy_FixedVarianceGaussianDecoder, self).__init__()
        self.x_stdev = x_stdev

    def forward(self, z):
        logvar = torch.ones_like(z) * 2 * np.log(self.x_stdev)

        return z, logvar


class Toy_FixedVarianceLogNormalDecoder(Toy_FixedVarianceGaussianDecoder):
    pass


class Toy_OneStatisticDecoder(nn.Module):
    def forward(self, z):
        return z, None


class Toy_TwoStatisticDecoder(nn.Module):
    def forward(self, z):
        return z[:, :2], z[:, 2:]


class Toy_GaussianDecoder(Toy_TwoStatisticDecoder):
    pass


class Toy_LogNormalDecoder(Toy_GaussianDecoder):
    pass


class Toy_GammaDecoder(Toy_TwoStatisticDecoder):
    pass


class Toy_ExponentialDecoder(Toy_OneStatisticDecoder):
    pass
