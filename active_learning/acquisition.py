def UCB(mu, std, **kwargs):
    beta = kwargs["beta"]
    return mu + beta * std

