"""
BO.py
ctive learning

Handles the primary functions
"""
from active_learning.param_optimizer import cross_validation
from active_learning.sampling import kriging_beliver

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


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
        
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = RBF()  # if None, vanilla RBF

        if estimator == "GP":
            self.estimator = GaussianProcessRegressor(
                kernel=self.kernel,
                optimizer=self.param_optimizer,
                normalize_y=False,
                n_restarts_optimizer=self.n_restarts,
            )
        else:
            raise NotImplementedError

        if sampling == "KB":
            self.sampling = kriging_beliver
        else:
            raise NotImplementedError

    def fit(self, X, mu, std=None, normalize_y=True):
        self.X = X
        self.mu = mu
        self.std = std
        self.normalize_y = normalize_y

        if self.normalize_y:
            self.mu, self.std = self._normalize_y(self.mu, self.std)

        if self.std is not None:
            self.estimator.alpha = self.std

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
        
        if self.normalize_y:
            self.mu = self.mu * self.y_std + self.y_mean
            self.std = self.std * self.y_std

        self.X = np.concatenate((self.X, self.X_new))
        self.mu = np.concatenate((self.mu, self.mu_new))
        if self.std_new is not None: self.std = np.concatenate((self.std, self.std_new))

        if self.normalize_y:
            self.mu, self.std = self._normalize_y(self.mu, self.std)

        if self.std is not None:
            self.estimator.alpha = self.std
        else:
            self.std = None
        self.estimator.fit(self.X, self.mu)

    def predict(self, x, return_std=True):
        if return_std:
            mu, std = self.estimator.predict(x, return_std)
            if self.normalize_y:
                mu = self.y_std * mu + self.y_mean
                std = std * self.y_std
            return mu, std
        else:
            mu = self.estimator.predict(x, return_std)
            if self.normalize_y:
                mu = self.y_mean * mu + self.y_std
            return mu
    
    def _normalize_y(self, mu, std=None):
        self.y_mean = np.mean(mu, axis=0)
        self.y_std = np.std(mu, axis=0)
        return (mu - self.y_mean) / self.y_std, std / self.y_std

    @property
    def y(self):
        return self.mu * self.y_std + self.y_mean
    
    @property
    def sigma(self):
        return self.std * self.y_std

        


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    import active_learning as al
