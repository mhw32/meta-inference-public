import numpy as np
from torch.utils.data import TensorDataset


def build_gaussian_distributions(num_dist, num_samples, mu=None, std=None, prior=5, stdev_dim=2):
    r"""Build a bunch of Gaussian distributions.

    @param num_dist: integer
                     number of distributions
    @param num_samples: integer
                        number of samples 
    @param prior: integer [default:  5]
                  mu ~ Uniform(-5, 5)
                  std ~ Uniform(0.1, 3.1)
                        (or you can hardcode)
    """
    mus, stdevs, clusters, labels = [], [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        if mu is None:
            _mu = np.random.uniform(-prior, prior, size=2)
        else:
            assert mu.shape[0] == 2
            _mu = mu

        if std is None:
            if stdev_dim == 2:
                # randomly sample a standard deviation
                _std = np.random.uniform(STDEV, 1., size=2)
            elif stdev_dim == 1:
                _std = np.random.uniform(STDEV, 1., size=1)
                _std = np.array([_std, _std])[:, 0]
        else:
            _std = np.array([std, std])
        cluster = np.random.multivariate_normal(
            _mu, np.diag(_std), size=num_samples)
        ys = np.ones(num_samples) * i

        # append stuff
        mus.append(_mu)
        stdevs.append(_std)
        clusters.append(cluster)
        labels.append(ys)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, zip(mus, stdevs)


def build_lognormal_distributions(num_dist, num_samples, mu=None, std=None, prior=2, stdev_dim=2):
    r"""Build a bunch of Gaussian distributions.

    @param num_dist: integer
                     number of distributions
    @param num_samples: integer
                        number of samples 
    @param prior: integer [default:  5]
                  mu ~ Uniform(-5, 5)
                  std ~ Uniform(0.1, 3.1)
                        (or you can hardcode)
    """
    mus, stdevs, clusters, labels = [], [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        if mu is None:
            _mu = np.random.uniform(-prior, prior, size=2)
        else:
            assert mu.shape[0] == 2
            _mu = mu

        if std is None:
            if stdev_dim == 2:
                # randomly sample a standard deviation
                _std = np.random.uniform(STDEV, 1., size=2)
            elif stdev_dim == 1:
                _std = np.random.uniform(STDEV, 1., size=1)
                _std = np.array([_std, _std])[:, 0]
        else:
            _std = np.array([std, std])
        cluster = np.random.multivariate_normal(
            _mu, np.diag(_std), size=num_samples)
        # normal --> lognormal
        cluster = np.exp(cluster)
        ys = np.ones(num_samples) * i

        # append stuff
        mus.append(_mu)
        stdevs.append(_std)
        clusters.append(cluster)
        labels.append(ys)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, zip(mus, stdevs)


def build_exponential_distributions(num_dist, num_samples, rate=None, prior=3):
    r"""Build a bunch of Exponential distributions.

    @param num_dist: integer
                     number of distributions
    @param num_samples: integer
                        number of samples 
    @param prior: integer [default: 5]
                  rate ~ Uniform(0, prior)
    """
    rates, clusters, labels = [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        if rate is None:
            _rate = np.random.uniform(0.5, prior, size=2)
        else:
            assert rate.shape[0] == 2
            _rate = rate

        scale = 1. / _rate
        cluster = np.random.exponential(scale, size=(num_samples, 2))
        ys = np.ones(num_samples) * i
        clusters.append(cluster)
        labels.append(ys)
        rates.append(_rate)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, rates


def build_beta_distribution(num_dist, num_samples, alpha, beta):
    alphas, betas, clusters, labels = [], [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        cluster = np.random.beta(alpha, beta, size=(num_samples, 2))
        ys = np.ones(num_samples) * i
        clusters.append(cluster)
        labels.append(ys)
        alphas.append(alpha)
        betas.append(beta)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, (alphas, betas)


def build_chi_squared_distribution(num_dist, num_samples, dof):
    dofs, clusters, labels = [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        cluster = np.random.chisquare(dof, size=(num_samples, 2))
        ys = np.ones(num_samples) * i
        clusters.append(cluster)
        labels.append(ys)
        dofs.append(dof)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, dofs


def build_weibull_distribution(num_dist, num_samples, shape):
    shapes, clusters, labels = [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        cluster = np.random.weibull(shape, size=(num_samples, 2))
        ys = np.ones(num_samples) * i
        clusters.append(cluster)
        labels.append(ys)
        shapes.append(shape)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, shapes


def build_laplace_distribution(num_dist, num_samples, loc, scale):
    locs, scales, clusters, labels = [], [], [], []

    for i in range(num_dist):  # iterating over number of datasets to construct
        cluster = np.random.laplace(loc, scale, size=(num_samples, 2))
        ys = np.ones(num_samples) * i
        clusters.append(cluster)
        labels.append(ys)
        locs.append(loc)
        scales.append(scale)

    split = int(num_samples * 0.8)

    # construct dataset: split train and test! 80-20
    train_dsets = [
        TensorDataset(
            torch.from_numpy(c[:split]).float()
        ) for c in clusters
    ]
    test_dsets  = [
        TensorDataset(
            torch.from_numpy(c[split:]).float()
        ) for c in clusters
    ]

    return train_dsets, test_dsets, labels, (locs, scales)
