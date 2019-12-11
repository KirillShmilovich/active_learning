"""
BO.py
ctive learning

Handles the primary functions
"""
from active_learning.param_optimizer import cross_validation
from active_learning.sampling import kriging_beliver

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class BayesianOptimizer:
    def __init__(
        self, estimator="GP", kernel=None, param_optimizer="EM", sampling="KB"
    ):

        if param_optimizer == "EM":
            self.param_optimizer = "fmin_l_bfgs_b"
            self.n_restarts = 10
        elif param_optimizer == "CV":
            self.param_optimizer = cross_validation
            self.n_restarts = 0
        else:
            raise NotImplementedError

        self.kernel = kernel  # if None, vanilla C * RBF

        if estimator == "GP":
            self.estimator = GaussianProcessRegressor(
                kernel=self.kernel,
                optimizer=self.param_optimizer,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts,
            )
        else:
            raise NotImplementedError

        if sampling == "KB":
            self.sampling = kriging_beliver
        else:
            raise NotImplementedError

    def fit(self, X, mu, std=None):
        self.X = X
        self.mu = mu

        if std is not None:
            self.std = std
            self.estimator.alpha = std
        else:
            self.std = None

        self.estimator.fit(self.X, self.mu)

    def query(self, X_pool, q, **kwargs):
        self.X_pool = X_pool
        self.q = q
        self.sample_points, self.sample_idxs = self.sampling(
            self.estimator, self.X_pool, self.q, **kwargs
        )
        return self.sample_points

    def teach(self, X_new, mu_new, std_new=None):
        self.X_new = X_new
        self.mu_new = mu_new
        self.std_new = std_new

        self.X = np.concatenate((self.X, self.X_new))
        self.mu = np.concatenate((self.mu, self.mu_new))

        if self.std_new is not None:
            self.std = np.concatenate((self.std, self.std_new))
            self.estimator.alpha = self.std
        else:
            self.std = None

        self.estimator.fit(self.X, self.mu)

    def predict(self, x, return_std=True, return_cov=False):
        return self.estimator.predict(x, return_std, return_cov)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    import active_learning as al
