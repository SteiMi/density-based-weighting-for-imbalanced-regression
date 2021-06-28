import os
from os.path import join
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pareto, norm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from utils import init_mpl, plot_histogram, split_data


class Perceptron(torch.nn.Module):
    def __init__(self, different_init: bool = False):
        super(Perceptron, self).__init__()
        self.init = nn.init.normal_
        self.init_a = 0.0
        self.init_b = 1.0
        self.layers = []

        for i in range(3):
            fc = nn.Linear(10, 10)
            if different_init:
                self.init(fc.weight, self.init_a, self.init_b)
            relu = torch.nn.ReLU()
            self.layers.append(fc)
            self.layers.append(relu)

        fc = nn.Linear(10, 1)
        if different_init:
            self.init(fc.weight, self.init_a, self.init_b)
        self.layers.append(fc)

        self.model = nn.ModuleList(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for _, l in enumerate(self.model):
            x = l(x)

        return x


class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(self, n: int = 1000, n_features: int = 10):
        super().__init__()
        gen_mlp = Perceptron(different_init=True)

        self.X = []
        self.Y = []
        for _ in range(n):
            x = torch.Tensor(np.random.randn(n_features))
            y = gen_mlp(x)

            self.X.append(x.numpy())
            self.Y.append(y.item())

        # self.X = torch.stack(X, dim=0)
        # self.Y = torch.unsqueeze(torch.Tensor(Y), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.Y[idx]


def make_uniform(data: pd.DataFrame, size: int = 10000,
                 cut_fraction_low: float = 0., cut_fraction_high: float = 0.) -> pd.DataFrame:
    # silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))
    # kernel = FFTKDE(bw=silverman_bandwidth).fit(y, weights=None)
    # x, y_dens_grid = kernel.evaluate(4096)

    # Cut y_min/max by cut_fraction
    y_min = raw_data['y'].min()
    y_max = raw_data['y'].max()

    cut_val_low = cut_fraction_low * (y_max - y_min)
    cut_val_high = cut_fraction_high * (y_max - y_min)
    y_min += cut_val_low
    y_max -= cut_val_high

    assert y_min < y_max, "y_min is not smaller than y_max anymore."

    # Sample uniformly from y-min to y-max and take closest point
    uni_ys = np.random.uniform(low=y_min, high=y_max, size=size)

    idxs: List[int] = []
    # Find clostest "real" samples to uni_ys
    for uni_y in uni_ys:
        # Find 100 closest idxs
        closest_idxs = data.iloc[(data['y']-uni_y).abs().argsort()[:100]].index.tolist()
        # Find closest idx that is not already in idxs
        for idx in closest_idxs:
            if idx not in idxs:
                idxs.append(idx)
                break

    # If there is no non-duplicate in the closest 100 data points then I just disregard that sample
    # (output size gets smaller)
    if len(idxs) < size:
        print('make_uniform WARNING: output size (%d) is smaller than what was asked for (%d).' % (len(idxs), size))
    return raw_data[raw_data.index.isin(idxs)]


# PDFs

def pareto_pdf(y: float, min_y: float, scale: float = 1., b: float = 1.) -> float:
    return pareto.pdf(y, scale=scale, b=b, loc=min_y-1)


def normal_pdf(y: float, center: float, scale: float = 1.) -> float:
    return norm.pdf(y, loc=center, scale=scale)


# Synthetic Dataset creation

def create_synth_data(create_data: Callable, filename: str, base_data: Optional[pd.DataFrame] = None,
                      random_state: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(filename):
        data = create_data(base_data, random_state)
        data.to_csv(filename, index=False)
    else:
        print(filename, 'exists. Loading..')
        data = pd.read_csv(filename)
        print(filename, 'loaded.')

    return data


def raw(_: None, random_state: Optional[int]) -> pd.DataFrame:
    synth_dataset = SyntheticDataset(n=200000)

    X_raw = np.stack(synth_dataset.X, axis=0)
    Y_raw = np.array(synth_dataset.Y)

    raw_data = pd.DataFrame(X_raw, columns=['feat_' + str(i) for i in range(X_raw.shape[1])])
    raw_data['y'] = Y_raw
    raw_data.sort_values('y', inplace=True)
    return raw_data


def uniform(raw_data: pd.DataFrame, random_state: Optional[int]) -> pd.DataFrame:
    return make_uniform(raw_data, size=10000, cut_fraction_low=0.1, cut_fraction_high=0.4)


def par(uniform_data: pd.DataFrame, random_state: Optional[int]) -> pd.DataFrame:
    y_min = uniform_data['y'].min()
    pareto_dens = np.array([pareto_pdf(y, y_min) for y in uniform_data['y']])
    pareto_data = uniform_data.sample(n=1000, replace=False, random_state=random_state,
                                         weights=pareto_dens/np.sum(pareto_dens))
    return pareto_data


def reverse_pareto(uniform_data: pd.DataFrame, random_state: Optional[int]) -> pd.DataFrame:
    y_min = uniform_data['y'].min()
    y_max = uniform_data['y'].max()
    pareto_dens = np.array([pareto_pdf(-y+y_min+y_max, y_min) for y in uniform_data['y']])
    pareto_data = uniform_data.sample(n=1000, replace=False, random_state=random_state,
                                      weights=pareto_dens/np.sum(pareto_dens))
    return pareto_data


def normal(uniform_data: pd.DataFrame, random_state: Optional[int]) -> pd.DataFrame:
    y_center = uniform_data['y'].median()
    # We define that a standard deviation should be 15% of the data
    norm_std = (uniform_data['y'].max() - uniform_data['y'].min()) * 0.15
    normal_dens = np.array([normal_pdf(y, y_center, norm_std) for y in uniform_data['y']])
    normal_data = uniform_data.sample(n=1000, replace=False, random_state=random_state,
                                      weights=normal_dens/np.sum(normal_dens))
    return normal_data


def double_normal(uniform_data: pd.DataFrame, random_state: Optional[int]) -> pd.DataFrame:
    y_center_left = uniform_data['y'].min()
    y_center_right = uniform_data['y'].max()
    # We define that a standard deviation should be 15% of the data
    norm_std = (uniform_data['y'].max() - uniform_data['y'].min()) * 0.15
    dnormal_dens = np.array([normal_pdf(y, y_center_left, norm_std) + normal_pdf(y, y_center_right, norm_std)
                             for y in uniform_data['y']])
    dnormal_data = uniform_data.sample(n=1000, replace=False, random_state=random_state,
                                       weights=dnormal_dens/np.sum(dnormal_dens))
    plt.plot(uniform_data['y'], dnormal_dens, label='normal densities')
    return dnormal_data


if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Init matplotlib
    init_mpl()

    raw_synth_filename = join('data', 'synthetic', 'raw_synth_data.csv')
    uniform_synth_filename = join('data', 'synthetic', 'uniform_synth_data.csv')

    pareto_synth_filename = join('data', 'synthetic', 'pareto_synth_data.csv')
    rpareto_synth_filename = join('data', 'synthetic', 'rpareto_synth_data.csv')
    normal_synth_filename = join('data', 'synthetic', 'normal_synth_data.csv')
    dnormal_synth_filename = join('data', 'synthetic', 'dnormal_synth_data.csv')

    raw_data = create_synth_data(raw, raw_synth_filename)
    uniform_data = create_synth_data(uniform, uniform_synth_filename, raw_data)
    pareto_data = create_synth_data(par, pareto_synth_filename, uniform_data, random_state=seed)
    rpareto_data = create_synth_data(reverse_pareto, rpareto_synth_filename, uniform_data, random_state=seed)
    normal_data = create_synth_data(normal, normal_synth_filename, uniform_data, random_state=seed)
    dnormal_data = create_synth_data(double_normal, dnormal_synth_filename, uniform_data, random_state=seed)

    plot_histogram(raw_data, 'raw', density=False)
    plot_histogram(raw_data, 'raw', density=True)
    plot_histogram(uniform_data, 'uniform', density=False)
    plot_histogram(uniform_data, 'uniform', density=True)
    plot_histogram(pareto_data, 'pareto', density=False)
    plot_histogram(pareto_data, 'pareto', density=True)
    plot_histogram(rpareto_data, 'rpareto', density=False)
    plot_histogram(rpareto_data, 'rpareto', density=True)
    plot_histogram(normal_data, 'normal', density=False)
    plot_histogram(normal_data, 'normal', density=True)
    plot_histogram(dnormal_data, 'dnormal', density=False)
    plot_histogram(dnormal_data, 'dnormal', density=True)

    split_data(pareto_data, 'pareto', save=True)
    split_data(rpareto_data, 'rpareto', save=True)
    split_data(normal_data, 'normal', save=True)
    split_data(dnormal_data, 'dnormal', save=True)
