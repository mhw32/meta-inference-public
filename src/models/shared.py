import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedConvolutionalEncoder(nn.Module):
    def __init__(self, n_channels):
        super(SharedConvolutionalEncoder, self).__init__()
        self.n_channels = n_channels

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=self.n_channels, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64, downsample=True),
            # shape is now (-1, 64, 14 , 14)
            Conv2d3x3(in_channels=64, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128, downsample=True),
            # shape is now (-1, 128, 7, 7)
            Conv2d3x3(in_channels=128, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256, downsample=True)
            # shape is now (-1, 256, 4, 4)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
        ])

    def forward(self, x):
        image_size = x.size(-1)
        n_channels = x.size(-3)
        h = x.view(-1, n_channels, image_size, image_size)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = F.elu(h)
        return h


class Conv2d3x3(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, stride=stride)

    def forward(self, x):
        return self.conv(x)


class FCResBlock(nn.Module):
    def __init__(self, dim, n, batch_norm=True):
        super(FCResBlock, self).__init__()
        self.n = n
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.block = nn.ModuleList(
                [nn.ModuleList([nn.Linear(dim, dim), nn.BatchNorm1d(num_features=dim)])
                 for _ in range(self.n)]
            )
        else:
            self.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0

        if self.batch_norm:
            for i, pair in enumerate(self.block):
                fc, bn = pair
                e = fc(e)
                e = bn(e)
                if i < (self.n - 1):
                    e = F.elu(e)

        else:
            for i, layer in enumerate(self.block):
                e = layer(e)
                if i < (self.n - 1):
                    e = F.elu(e)

        return F.elu(e + x)


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


class BernoulliDecoder(nn.Module):
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
        super(BernoulliDecoder, self).__init__()

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


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h
