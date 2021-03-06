import functools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE
from utils import bisection


class TargetRelevance():

    def __init__(self, y, alpha=1.0):
        self.alpha = alpha
        print('TargetRelevance alpha:', self.alpha)

        silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))

        print('Using Silverman Bandwidth', silverman_bandwidth)
        best_bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=best_bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(4096)  # Default precision is 1024
        self.x = x
        
        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()

        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    @functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)
