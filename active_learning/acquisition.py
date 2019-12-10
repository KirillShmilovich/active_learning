from scipy.stats import norm
import numpy as np


def PI(mu, std, **kwargs):
    """
    Probability of improvement acquisition function

    INPUT:
    - mu: mean of predicted point in grid
    - std: sigma (square root of variance) of predicted point in grid
    - fMax: observed or predicted maximum value (depending on noise p.19 [Brochu et al. 2010])
    - epsilon: trade-off parameter (>=0)

    OUTPUT:
    - PI: probability of improvement for candidate point

    As describend in:
    E Brochu, VM Cora, & N de Freitas (2010): 
    A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,
    arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    """
    fMax = kwargs["fmax"]
    epsilon = kwargs["epsilon"]

    Z = (mu - fMax - epsilon) / std

    return norm.cdf(Z)


def EI(mu, std, **kwargs):
    """
    Expected improvement acquisition function

    INPUT:
    - mu: mean of predicted point in grid
    - std: sigma (square root of variance) of predicted point in grid
    - fMax: observed or predicted maximum value (depending on noise p.19 Brochu et al. 2010)
    - epsilon: trade-off parameter (>=0) 
    [Lizotte 2008] suggest setting epsilon = 0.01 (scaled by the signal variance if necessary)  (p.14 [Brochu et al. 2010])

    OUTPUT:
    - EI: expected improvement for candidate point

    As describend in:
    E Brochu, VM Cora, & N de Freitas (2010): 
    A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
    arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    """
    fMax = kwargs["fMax"]
    epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 0.1

    Z = (mu - fMax - epsilon) / std

    return (mu - fMax - epsilon) * norm.cdf(Z) + std * norm.pdf(Z)


def Exploitacquisition(mu, std, **kwargs):
    fMax = kwargs["fmax"]
    epsilon = kwargs["epsilon"]

    Z = (mu - fMax - epsilon) / std

    return norm.cdf(Z)


def UCB(mu, std, **kwargs):
    """
    Upper confidence bound acquisition function

    INPUT:
    - mu: predicted mean
    - std: sigma (square root of variance) of predicted point in grid
    - t: number of iteration
    - d: dimension of optimization space
    - v: hyperparameter v = 1*
    - delta: small constant (prob of regret)

    *These bounds hold for reasonably smooth kernel functions.
    [Srinivas et al., 2010]

    OUTPUT:
    - UCB: upper confidence bound for candidate point

    As describend in:
    E Brochu, VM Cora, & N de Freitas (2010): 
    A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
    arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    """
    t = kwargs["t"]
    d = kwargs["d"]
    v = kwargs["v"] if ("v" in kwargs) else 1
    delta = kwargs["delta"] if ("delta" in kwargs) else 0.1

    Kappa = np.sqrt(
        v * (2 * np.log((t ** (d / 2.0 + 2)) * (np.pi ** 2) / (3.0 * delta)))
    )

    return mu + Kappa * std
