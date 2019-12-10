from copy import deepcopy
from active_learning.acquisition import EI
import numpy as np


def kriging_beliver(estimator, X_pool, q, acquisition=EI, **kwargs):
    est = deepcopy(estimator)
    est.optimizer = "fmin_l_bfgs_b"
    est.n_restarts_optimizer = 10
    query = np.zeros(shape=(q, X_pool.shape[1]))
    idxs = np.zeros(shape=q, dtype=np.int)
    for i in range(q):
        mu, std = est.predict(X_pool, return_std=True)
        fMax = est.y_train_.max()
        ac = acquisition(mu=mu, std=std, fMax=fMax, **kwargs)
        ac_idx = ac.argsort()[::-1]

        if i == 0:
            mask = np.ones(shape=(X_pool.shape[0]), dtype=np.bool)
        elif i == 1:
            mask = ~(ac_idx == idxs[0])
        else:
            temp = np.repeat(ac_idx.reshape(-1, 1), i, axis=1)
            mask = ~np.any(temp == idxs[:i], axis=1)

        new_idx = ac_idx[mask][0]
        new_x = X_pool[new_idx]
        new_y = mu[new_idx]
        new_std = std[new_idx]

        idxs[i] = new_idx
        query[i] = new_x

        X_train = np.concatenate((est.X_train_, new_x.reshape(1, -1)))
        y_train = np.concatenate((est.y_train_, new_y.reshape(-1)))
        alpha_train = np.concatenate((est.alpha, new_std.reshape(-1)))

        est.alpha = alpha_train
        est.fit(X_train, y_train)

    return query, idxs
