"""
BO.py
ctive learning

Handles the primary functions
"""
from active_learning.param_optimizer import cross_validation
from active_learning.sampling import kriging_beliver

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class Bayesian_Optimizer:
    def __init__(
        self, estimator="GP", kernel="RBF", param_optimizer="CV", sampling="EM"
    ):

        if param_optimizer == "EM":
            self.param_optimizer = "fmin_l_bfgs_b"
        elif param_optimizer == "CV":
            self.param_optimizer = cross_validation
        else:
            raise NotImplementedError

        if kernel == "RBF":
            self.kernel = RBF()
        else:
            raise NotImplementedError

        if estimator == "GP":
            self.estimator = GaussianProcessRegressor(
                kernel=self.kernel, optimizer=self.param_optimizer, normalize_y=True
            )
        else:
            raise NotImplementedError

        if sampling == "CV":
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

        self.estimator.fit(X, mu)

    def query(self, X_pool, q):
        self.X_pool = X_pool
        self.q = q
        self.samples = self.sampling(self.estimator, self.X_pool, self.q)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    import active_learning as al
