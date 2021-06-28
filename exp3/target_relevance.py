import functools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0  # Initialize lower
    ju = n-1  # and upper limits.
    while (ju-jl > 1):  # If we are not yet done,
        jm = (ju+jl) >> 1  # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):  # edge cases at bottom
        return 0
    elif (value == array[n-1]):  # and top
        return n-1
    else:
        return jl


class TargetRelevance():

    def __init__(self, y, cv=10, alpha=1.0):
        self.alpha = alpha
        print('TargetRelevance alpha:', self.alpha)

        silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))

        print('Using Silverman Bandwidth')
        best_bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=best_bandwidth).fit(y, weights=None)

        self.y_min = y.min()
        self.y_max = y.max()

        x, y_dens = self.kernel.evaluate(4096)  # Default precision is 1024
        self.x = x

        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens = MinMaxScaler().fit_transform(y_dens.reshape(-1, 1)).flatten()

        self.eps = 1e-6
        w2 = np.maximum(1 - self.alpha * y_dens, self.eps)
        self.mean_w2 = np.mean(w2)
        self.relevances = w2 / self.mean_w2

    def get_density(self, y):
        idx = bisection(self.x, y)
        return self.y_dens[idx]

    @functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w2

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)
