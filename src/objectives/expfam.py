import torch
import torch.distributions as dist

from src.utils import log_mean_exp


def log_gaussian_marginal_estimate(x_examples, x_mu_list, x_logvar_list, z_list, z_mu, z_logvar):
    n_samples = len(z_list)
    n_examples = len(x_examples)

    log_w = []
    for i in xrange(n_samples):
        log_p_x_given_z_i = 0
        for j in xrange(n_examples):
            log_p_x_given_z_ij = gaussian_log_pdf(x_examples[j], x_mu_list[i], x_logvar_list[i])
            log_p_x_given_z_i += log_p_x_given_z_ij
        log_p_x_given_z_i /= float(n_examples)

        log_q_z_given_x_i = gaussian_log_pdf(z_list[i], z_mu, z_logvar)
        log_p_z_i = unit_gaussian_log_pdf(z_list[i])

        log_w_i = log_p_x_given_z_i + log_p_z_i - log_q_z_given_x_i
        log_w.append(log_w_i)

    log_w = torch.stack(log_w).t()
    log_marginal = log_mean_exp(log_w, dim=1)
    
    return -torch.mean(log_marginal)


def log_lognormal_marginal_estimate(x_examples, x_mu_list, x_logvar_list, z_list, z_mu, z_logvar):
    n_samples = len(z_list)
    n_examples = len(x_examples)

    log_w = []
    for i in xrange(n_samples):
        log_p_x_given_z_i = 0
        for j in xrange(n_examples):
            log_p_x_given_z_ij = lognormal_log_pdf(x_examples[j], x_mu_list[i], x_logvar_list[i])
            log_p_x_given_z_i += log_p_x_given_z_ij
        log_p_x_given_z_i /= float(n_examples)

        log_q_z_given_x_i = gaussian_log_pdf(z_list[i], z_mu, z_logvar)
        log_p_z_i = unit_gaussian_log_pdf(z_list[i])

        log_w_i = log_p_x_given_z_i + log_p_z_i - log_q_z_given_x_i
        log_w.append(log_w_i)

    log_w = torch.stack(log_w).t()
    log_marginal = log_mean_exp(log_w, dim=1)
    
    return -torch.mean(log_marginal)


def log_exponential_marginal_estimate(x_examples, x_log_rate_list, z_list, z_mu, z_logvar):
    n_samples = len(z_list)
    n_examples = len(x_examples)

    log_w = []
    for i in xrange(n_samples):
        log_p_x_given_z_i = 0
        for j in xrange(n_examples):
            log_p_x_given_z_ij = exponential_log_pdf(x_examples[j], x_log_rate_list[i])
            log_p_x_given_z_i += log_p_x_given_z_ij
        log_p_x_given_z_i /= float(n_examples)

        log_q_z_given_x_i = gaussian_log_pdf(z_list[i], z_mu, z_logvar)
        log_p_z_i = unit_gaussian_log_pdf(z_list[i])

        log_w_i = log_p_x_given_z_i + log_p_z_i - log_q_z_given_x_i
        log_w.append(log_w_i)

    log_w = torch.stack(log_w).t()
    log_marginal = log_mean_exp(log_w, dim=1)
    
    return -torch.mean(log_marginal)


def gaussian_elbo_loss(x_list, x_mu, x_logvar, z, z_mu, z_logvar):
    r"""Evidence lower bound with Gaussian reconstruction loss.
    
    log p(x) >= E_{q(z|x)} [ log p(x,z) - log q(z|x) ]
              = E_{q(z|x)} [ log p(x|z) + log p(z) - log q(z|x) ]

    We use closed form KL divergence.
    """
    n = len(x_list)
    log_p_x_given_z = 0
    for i in xrange(n):
        log_p_x_given_z_i = -gaussian_log_pdf(x_list[i], x_mu, x_logvar)
        log_p_x_given_z += log_p_x_given_z_i
    log_p_x_given_z /= float(n)
    kl_divergence = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_divergence = torch.sum(kl_divergence, dim=1)
    elbo = log_p_x_given_z + kl_divergence
    elbo = torch.mean(elbo)

    return elbo


def lognormal_elbo_loss(x_list, x_mu, x_logvar, z, z_mu, z_logvar):
    r"""Evidence lower bound with LogNormal reconstruction loss.
    
    log p(x) >= E_{q(z|x)} [ log p(x,z) - log q(z|x) ]
              = E_{q(z|x)} [ log p(x|z) + log p(z) - log q(z|x) ]

    We use closed form KL divergence.
    """
    n = len(x_list)
    log_p_x_given_z = 0
    for i in xrange(n):
        log_p_x_given_z_i = -lognormal_log_pdf(x_list[i], x_mu, x_logvar)
        log_p_x_given_z += log_p_x_given_z_i
    log_p_x_given_z /= float(n)
    kl_divergence = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_divergence = torch.sum(kl_divergence, dim=1)
    elbo = log_p_x_given_z + kl_divergence
    elbo = torch.mean(elbo)

    return elbo


def exponential_elbo_loss(x_list, x_log_rate, z, z_mu, z_logvar):
    r"""Evidence lower bound with Exponential reconstruction loss.

    log p(x) >= E_{q(z|x)} [ log p(x,z) - log q(z|x) ]
              = E_{q(z|x)} [ log p(x|z) + log p(z) - log q(z|x) ]

    We use closed form KL divergence.
    """
    n = len(x_list)
    log_p_x_given_z = 0
    for i in xrange(n):
        log_p_x_given_z_i = -exponential_log_pdf(x_list[i], x_log_rate)
        log_p_x_given_z += log_p_x_given_z_i
    # HACK: important to comment this out -->  log_p_x_given_z /= float(n)
    kl_divergence = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_divergence = torch.sum(kl_divergence, dim=1)
    elbo = log_p_x_given_z + kl_divergence
    elbo = torch.mean(elbo)

    return elbo


def exponential_log_pdf(x, log_rate):
    rate = torch.exp(log_rate)
    p_dist = dist.exponential.Exponential(rate)
    logprob = p_dist.log_prob(x)
    return torch.sum(logprob, dim=1)


def gaussian_log_pdf(x, mu, logvar):
    scale = 0.5*torch.exp(logvar)
    p_dist = dist.normal.Normal(mu, scale)
    logprob = p_dist.log_prob(x)
    return torch.sum(logprob, dim=1)


def unit_gaussian_log_pdf(x):
    mu = torch.zeros_like(x)
    scale = torch.ones_like(x)
    p_dist = dist.normal.Normal(mu, scale)
    logprob = p_dist.log_prob(x)
    return torch.sum(logprob, dim=1)


def lognormal_log_pdf(x, mu, logvar):
    scale = 0.5*torch.exp(logvar)
    p_dist = dist.LogNormal(mu, scale)
    logprob = p_dist.log_prob(x)
    return torch.sum(logprob, dim=1)
