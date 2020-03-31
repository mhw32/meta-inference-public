import torch
import torch.nn.functional as F

from math import log
from src.utils import (
    bernoulli_log_pdf,
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    log_mean_exp,
)


def gaussian_elbo_loss(x, x_mu, x_logvar, z, z_mu, z_logvar):
    log_p_x_given_z = -gaussian_log_pdf(x, x_mu, x_logvar)
    kl_divergence = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_divergence = torch.sum(kl_divergence, dim=1)
    elbo = log_p_x_given_z + kl_divergence
    elbo = torch.mean(elbo)

    return elbo


def bernoulli_elbo_loss(x, x_mu, z, z_mu, z_logvar):
    r"""Lower bound on model evidence (average over multiple samples).

    Closed form solution for KL[p(z|x), p(z)]

    Kingma, Diederik P., and Max Welling. "Auto-encoding
    variational bayes." arXiv preprint arXiv:1312.6114 (2013).
    """
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    x_mu = x_mu.view(batch_size, -1)
    log_p_x_given_z = -bernoulli_log_pdf(x, x_mu)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_divergence = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_divergence = torch.sum(kl_divergence, dim=1)

    # lower bound on marginal likelihood
    ELBO = log_p_x_given_z + kl_divergence
    ELBO = torch.mean(ELBO)

    return ELBO


def log_bernoulli_marginal_estimate_sets(elbos_list):
    """
    log_bernoulli_marginal_estimate() function adapted for Neural Statistician
    """
    # 1. grab the list of elbos computed
    # 2. need to do a log mean exp over all the ELBOs over k
    k = len(elbos_list)
    log_w = torch.stack(elbos_list).t() # (n_datasets, k)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p_x = log_mean_exp(log_w, dim=1) 
    
    return -torch.mean(log_p_x) # average over datasets


def log_bernoulli_marginal_estimate(x, x_mu_list, z_list, z_mu, z_logvar):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.

    @param x: torch.Tensor (batch size x input_dim)
              original observed data
    @param x_mu_list: list of torch.Tensor (batch size x input_dim)
                      reconstructed means on bernoulli
    @param z_list: list of torch.Tensor (batch_size x z dim)
                    samples drawn from variational distribution
    @param z_mu: torch.Tensor (batch_size x # samples x z dim)
                 means of variational distribution
    @param z_logvar: torch.Tensor (batch_size x # samples x z dim)
                     log-variance of variational distribution
    """
    k = len(z_list)
    batch_size = x.size(0)

    log_w = []
    for i in range(k):
        log_p_x_given_z_i = bernoulli_log_pdf(x.view(batch_size, -1),   
                                              x_mu_list[i].view(batch_size, -1))
        log_q_z_given_x_i = gaussian_log_pdf(z_list[i], z_mu, z_logvar)
        log_p_z_i = unit_gaussian_log_pdf(z_list[i])
        log_w_i = log_p_x_given_z_i + log_p_z_i - log_q_z_given_x_i
        log_w.append(log_w_i)
    
    log_w = torch.stack(log_w).t()  # (batch_size, k)
    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p_x = log_mean_exp(log_w, dim=1)
    
    return -torch.mean(log_p_x)
