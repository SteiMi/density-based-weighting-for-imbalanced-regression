import numbers
from os import listdir
from os.path import join, isdir, isfile
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    From https://stackoverflow.com/a/41856629'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl


def create_error_column(df: pd.DataFrame) -> pd.DataFrame:
    df['error'] = df['y'] - df['estimate']
    df['absolute_error'] = df['error'].abs()
    return df


def create_normalized_absolute_error_column(df: pd.DataFrame) -> pd.DataFrame:
    df = create_error_column(df)
    y_diff = df['y'].max() - df['y'].min()
    df['normalized_absolute_error'] = df['absolute_error'] / y_diff
    return df


def create_normalized_squared_error_column(df: pd.DataFrame) -> pd.DataFrame:
    df['error'] = df['y'] - df['estimate']
    df['squared_error'] = df['error'] ** 2
    y_diff = df['y'].max()**2 - df['y'].min()**2
    df['normalized_squared_error'] = df['squared_error'] / y_diff
    return df


def group_estimates_by_alpha(
    estimates: List[Tuple[Dict[str, Any], pd.DataFrame]]
) -> Dict[float, List[pd.DataFrame]]:
    # Group estimates by alpha-value
    distinct_alpha_values = sorted(list(set([e[0]['alpha'] for e in estimates])))
    est_list_per_alpha: Dict[float, List[pd.DataFrame]] = {}
    for a in distinct_alpha_values:
        est_list_per_alpha[a] = []

    for est in estimates:
        est_list_per_alpha[est[0]['alpha']].append(est[1])

    return est_list_per_alpha


def merge_estimates(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs, axis=1)
    # Calculate the mean of each column with the same name, see: https://stackoverflow.com/a/40312254
    df = df.groupby(by=df.columns, axis=1).apply(
        lambda g: g.mean(axis=1)
        if isinstance(g.iloc[0, 0], numbers.Number)
        else g.iloc[:, 0]
    )
    return df


def parse_name(name: str) -> Dict[str, Any]:
    """Parse a name like denseloss-pareto-wl-0-6-11 accordingly."""
    split = name.split('-')
    dataset = split[1]
    run_number = int(split[-1])
    smogn_dw = 'smogndw' in dataset
    if not smogn_dw:
        smogn = 'smogn' in dataset
    else:
        smogn = False

    if len(split) == 3:
        weighted_loss = False
        alpha = 0.0
    elif len(split) == 6 and split[2] == 'wl':
        weighted_loss = True
        alpha = float(split[3] + split[4])/10.
    else:
        print('ERROR: Cannot parse name:', name)
        sys.exit(1)

    return {
        'name': name,
        'dataset': dataset,
        'weighted_loss': weighted_loss,
        'alpha': alpha,
        'run_number': run_number,
        'smogn': smogn,
        'smogn_dw': smogn_dw
    }


def find_bin_boundaries(b_low: int, b_high: int, data: pd.DataFrame) -> Tuple[float, float]:
    """
    Given bin boundaries in percentages and data, find the boundaries in y-space.
    Example: b_low = 0, b_high = 10, y ranges from 100 to 200 -> y_low = 100, y_high = 110
    """
    assert b_low >= 0, 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low <= 100, 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high) 
    assert b_high >= 0, 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low <= 100, 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low < b_high, 'b_low is not smaller than b_high (b_low: %d, b_high: %d)' % (b_low, b_high)
    y_min = data['y'].min()
    y_max = data['y'].max()
    y_range = y_max - y_min
    y_low = y_min + (b_low / 100.) * y_range
    y_high = y_min + (b_high / 100.) * y_range
    assert y_low >= y_min
    assert y_high <= y_max
    return y_low, y_high


def flatten(l: List) -> List:
    return [item for sublist in l for item in sublist]


def nice_metric_name(metric: str) -> str:
    nice_metric_name = {
        'bias': 'Bias per $\\alpha$',
        'rmse': 'RMSE per $\\alpha$',
        'mean_absolute_error': 'MAE per $\\alpha$',
        'r2_score': '$\\text{R}^2$ per $\\alpha$'
    }
    return nice_metric_name[metric]


def to_bin_string(b_low: int, b_high: int) -> str:
    return str(b_low) + '-' + str(b_high)


def to_bin_bounds(bin_str: str) -> Tuple[int, int]:
    bounds_list = bin_str.split('-')
    return (int(bounds_list[0]), int(bounds_list[1]))


def to_latex_bin_str(bin_str: str) -> str:
    return bin_str.replace('-', '--')


def read_split_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    is_synth_data = dataset_name in ['pareto', 'rpareto', 'normal', 'dnormal',
                                     'pareto_smogn', 'rpareto_smogn', 'normal_smogn', 'dnormal_smogn',
                                     'pareto_smogn_dw', 'rpareto_smogn_dw', 'normal_smogn_dw', 'dnormal_smogn_dw']
    if is_synth_data:
        base_path = join('data', 'synthetic')
    else:
        base_path = join('data', 'real')

    train_file = join(base_path, '%s_train.csv' % dataset_name)
    val_file = join(base_path, '%s_val.csv' % dataset_name)
    test_file = join(base_path, '%s_test.csv' % dataset_name)

    # We use the same val/test csvs for SMOGN data as for "regular" data
    # For smogn-datasets, I have to adjust the path name accordingly
    if 'smogn' in dataset_name:
        val_file = join(base_path, '%s_val.csv' % dataset_name.replace('_smogn_dw', '').replace('_smogn', ''))
        test_file = join(base_path, '%s_test.csv' % dataset_name.replace('_smogn_dw', '').replace('_smogn', ''))

    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)

    if is_synth_data:
        y_col_name = 'y'
    else:
        # The first column of the real world datasets are the target
        y_col_name = train.columns[0]

    return train, val, test, y_col_name


def split_train_val_test(data: pd.DataFrame,
                         random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Solution from https://datascience.stackexchange.com/a/17445
    train, val, test = np.split(data.sample(frac=1, random_state=random_state), [int(.6*len(data)), int(.8*len(data))])
    return train, val, test


def split_data(data: pd.DataFrame, name: str, save: bool = False, base_save_path: str = join('data', 'synthetic'),
               show_dist: bool = False, y_col_name: str = 'y',
               random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # random_states for which I found pleasing splits (pleasing == the range of y is represented as well as possible
    # for each split)
    random_state_dict = {
        # synthetic
        'pareto': 1,
        'rpareto': 11,
        'normal': 3,
        'dnormal': 1,
        # real
        'abalone': 18,
        'concreteStrength': 1,
        'delta_ailerons': 2,
        'boston': 0,
        'available_power': 0,
        'servo': 4,
        'bank8FM': 0,
        'machineCpu': 53,
        'airfoild': 1,
        'a2': 3,
        'a3': 5,
        'a1': 13,
        'cpu_small': 0,
        'acceleration': 4,
        'maximal_torque': 5,
        'a4': 7,
        'a5': 6,
        'fuel_consumption_country': 4,
        'a7': 0,
        'a6': 0
    }

    if random_state is None:
        rs = random_state_dict[name] if name in random_state_dict.keys() else None
    else:
        rs = random_state

    print('Name:', name, 'Seed:', rs)

    train, val, test = split_train_val_test(data, rs)

    texable_name = name.replace('_', ' ')

    if show_dist:
        plt.hist(train[y_col_name], density=True, alpha=0.5, label=texable_name+' train')
        plt.hist(val[y_col_name], density=True, alpha=0.5, label=texable_name+' val')
        plt.hist(test[y_col_name], density=True, alpha=0.5, label=texable_name+' test')
        plt.legend()
        # plt.show()
        plt.savefig(join(base_save_path, name + '.pdf'), format='pdf', bbox_inches='tight')
        plt.clf()

    if save:
        train.to_csv(join(base_save_path, name + '_train.csv'), index=False)
        val.to_csv(join(base_save_path, name + '_val.csv'), index=False)
        test.to_csv(join(base_save_path, name + '_test.csv'), index=False)

    return train, val, test


def set_size(width: float = 347.0, fraction: float = 1., subplot: List[int] = [1, 1]):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
            Default value 347.12354 is textwidth for Springer llncs
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def init_mpl(usetex: bool = True):
    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": usetex,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
    }

    mpl.rcParams.update(nice_fonts)


def plot_histogram(df: pd.DataFrame, name: str, density: bool = False, y_col_name: str = 'y'):
    plt.figure(figsize=set_size(width=516.0, fraction=0.4))
    plt.hist(df[y_col_name], bins=10, density=density)

    # Some hacky optimizations for the paper
    if any([n in name for n in ['normal', 'pareto']]):
        plt.xlim([-35, 80])

    if 'normal' in name:
        plt.ylim([0, 0.025])
    elif 'pareto' in name:
        plt.ylim([0, 0.07])

    if density:
        plt.ylabel('$p(%s)$' % y_col_name)
    else:
        plt.ylabel('Frequency')

    plt.xlabel('$y$')
    dens_str = '_density' if density else ''
    filename = name + '_hist' + dens_str + '.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')
    plt.clf()


def num_features_for_dataset(name: str) -> int:
    """
    Return the number of input features after One-Hot-Encoding for a dataset.
    """

    # smogn and non-smogn datasets have the same number of features
    if '_smogn' in name:
        name = name.replace('_smogn_dw', '').replace('_smogn', '')

    ds_dict = {
        'pareto': 10, 'rpareto': 10, 'normal': 10, 'dnormal': 10,
        'abalone': 10, 'concreteStrength': 8, 'delta_ailerons': 5, 'boston': 13,
        'available_power': 49, 'servo': 12, 'bank8FM': 8, 'machineCpu': 6, 'airfoild': 5,
        'a2': 18, 'a3': 18, 'a1': 18, 'cpu_small': 12, 'acceleration': 22, 'maximal_torque': 95,
        'a4': 18, 'a5': 18, 'a7': 18, 'fuel_consumption_country': 88, 'a6': 18
    }

    return ds_dict[name]


def predict_all_from_dataloader(model: LightningModule, dataloader: DataLoader) -> pd.DataFrame:
    train_data = list(zip(*list(iter(dataloader))))
    train_x = torch.cat(train_data[0])
    train_y = torch.cat(train_data[1])
    train_est = model(train_x)
    data_points = np.concatenate((train_x.numpy(), train_y.numpy(), train_est.detach().numpy()), axis=1)
    col_names = ['feat_'+str(col) for col in range(train_x.shape[1])]
    col_names.append('y')
    col_names.append('estimate')
    return pd.DataFrame(data_points, columns=col_names)


def load_estimates(save_paths: List[str] = ['checkpoints'],
                   split: str = 'test') -> List[Tuple[Dict[str, Any], pd.DataFrame]]:
    """Loads the test_estimates of each run within save_paths."""
    estimates: List[Tuple[Dict[str, Any], pd.DataFrame]] = []
    for sp in save_paths:
        runs = [d for d in listdir(sp) if isdir(join(sp, d))]
        estimates += [(parse_name(r), pd.read_csv(join(sp, r, split + '_estimates.csv'))) for r in tqdm(runs)
                      if isfile(join(sp, r, split + '_estimates.csv'))]
    return estimates
