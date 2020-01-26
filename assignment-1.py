import os
import argparse

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, help='the alpha paramter of beta distribution')
parser.add_argument('--beta', type=float, help='the beta parameter of beta distribution')
parser.add_argument('--r', type=int, default=5, help='the r parameter of NB distribution')

parser.add_argument('--num_steps', type=int, default=10, help='the number of discrete steps of HMC')
parser.add_argument('--step_size', type=float, default=1e-3, help='the size of single step of HMC')
parser.add_argument('--num_samples', type=int, default=500, help='the number of samples to be generated')
parser.add_argument('--warm_steps', type=int, default=100, help='the warm up steps of MCMC')
parser.add_argument('--seed', type=int, default=1234, help='the seed to be used when simulation')
args = parser.parse_args()

assert pyro.__version__.startswith('1.1.0')
pyro.set_rng_seed(args.seed) # set the random seed


class NB_Post(object):
    ''' The class for implementing the MCMC of NB distribution
        $X \sim NB(r, p)$, we use MCMC to get the posterior of parameter $p$.
    '''
    def __init__(self, alpha, beta, r):
        '''
        args:
            alpha: the alpha parameter (torch.Tensor) of beta distribution
            beta: the beta parameter (torch.Tensor) of beta distribution
            r: the r parameter of Negative Binomial distribution
        '''
        assert isinstance(alpha, torch.Tensor) and isinstance(beta, torch.Tensor), \
            'please the type torch.Tensor as the input of alpha and beta'
        self.alpha = alpha
        self.beta = beta
        self.r = r

    def model(self, data):
        '''
        The model to be used by Pyro.
        args:
            data: the data (torch.Tensor) to be used by MCMC
        '''
        assert isinstance(data, torch.Tensor), 'Please use torch.Tensor type as the input.'
        p = pyro.sample('p', dist.Beta(self.alpha, self.beta))
        with pyro.plate('data', len(data)):
            pyro.sample('obs', dist.NegativeBinomial(r, p), obs=data)


def plot_density(truth_dist, post_samples):
    x = torch.linspace(1e-2, 1-1e-2, 500)
    truth_density = torch.exp(truth_dist.log_prob(x))

    plt.figure(figsize=(5, 4))
    # sns.set(palette='muted', color_codes=True)
    sns.set()
    
    ax = sns.lineplot(x, truth_density, label='Pred', linewidth=1)
    ax.fill_between(x, truth_density, alpha=0.3)
    sns.distplot(post_samples, hist=False, kde=True, kde_kws={'linewidth':1, 'shade':True}, label='Truth')

    plt.xlim([0.2, 0.9])
    plt.grid(':')
    plt.title('Density Plot')
    plt.xlabel('$p$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('./assets/conjugate_prior.pdf', dpi=600)



if __name__ == '__main__':
    alpha = torch.tensor(args.alpha)
    beta = torch.tensor(args.beta)
    r = torch.tensor(args.r)
    data = torch.tensor([12, 11, 6, 12, 11, 0, 4, 6, 5, 6])

    nb_post = NB_Post(alpha, beta, args.r)
    # create hmc and mcmc object
    hmc_kernel = HMC(nb_post.model, step_size=args.step_size, num_steps=args.num_steps)
    mcmc = MCMC(hmc_kernel, num_samples=args.num_samples, warmup_steps=args.warm_steps)

    # sample the posterior
    mcmc.run(data)
    posterior_samples = mcmc.get_samples()['p']

    # plot the estimated and ground truth density
    true_dist = dist.Beta(alpha + data.sum(), len(data)*r + beta)
    plot_density(true_dist, posterior_samples)







        
    