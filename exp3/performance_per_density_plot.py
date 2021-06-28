import os
import datetime
import glob
import math
import numbers
import pickle
import sys
from typing import Dict, List, Optional

try:
    import configparser as ConfigParser
except ImportError:
    import ConfigParser
import numpy as np
from scipy.signal import convolve
import tensorflow as tf
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib import colors

import prism
from target_relevance import TargetRelevance
from utils import (
    init_mpl,
    set_size,
)


flags = tf.flags
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string(
    'config_file',
    'config.ini',
    'Configuration file with [SRCNN], [Model-%], and [DeepSD] sections.',
)


# parse flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

config = ConfigParser.ConfigParser()
config.read(FLAGS.config_file)

# With OTHER_CONFIG_FILES you can choose other configurations which should also be plotted for comparison to the main
# model.
OTHER_CONFIG_FILES = glob.glob('configs/*.ini')  # Disable to speed up testing
print(OTHER_CONFIG_FILES)
OTHER_CONFIGS = []
for cf in OTHER_CONFIG_FILES:
    cnf = ConfigParser.ConfigParser()
    cnf.read(cf)
    OTHER_CONFIGS.append(cnf)
OTHER_OUTPUT_DIRS = [
    os.path.expanduser(
        os.path.join(
            c.get('SRCNN', 'scratch'), c.get('DeepSD', 'model_name'), 'outputs'
        )
    )
    for c in OTHER_CONFIGS
]

# Base some configurations which we will use for all models on an arbitrary config and hope that this is ok for all models
config = OTHER_CONFIGS[0]
print(
    'Base config from which we take the upscale factor and the PRISM dir (IF NOT ON laptop) for all models:',
    config.get('DeepSD', 'model_name'),
)

if os.uname()[0] == 'Darwin':
    # We are on MacOS and most likely on my laptop where prism data lies in ~/data/prism
    PRISM_DIR = os.path.expanduser(os.path.join('~', 'data', 'prism', 'ppt', 'raw'))
    print('Using local prism files in', PRISM_DIR)
else:
    # We are probably in Kubernetes or something
    PRISM_DIR = os.path.expanduser(
        os.path.join(config.get('Paths', 'prism'), 'ppt', 'raw')
    )

UPSCALE_FACTOR = config.getint('DeepSD', 'upscale_factor')

def create_error_column(df: pd.DataFrame) -> pd.DataFrame:
    df['error'] = df['y'] - df['estimate']
    df['absolute_error'] = df['error'].abs()
    return df


def create_normalized_absolute_error_column(
    df: pd.DataFrame, y_diff=None
) -> pd.DataFrame:
    if y_diff is None:
        y_diff = df['y'].max() - df['y'].min()
    df['normalized_absolute_error'] = (df['y'] - df['estimate']).abs() / y_diff
    return df


def merge_estimates(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs, axis=1)
    # Calculate the mean of each column with the same name, see: https://stackoverflow.com/a/40312254
    df = df.groupby(by=df.columns, axis=1).apply(
        lambda g: g.mean(axis=1)
        if isinstance(g.iloc[0, 0], numbers.Number)
        else g.iloc[:, 0]
    )
    return df


def load_data(year, scale1, n_stacked):
    # read prism dataset
    # resnet parameter will not re-interpolate X
    dataset = prism.PrismSuperRes(
        PRISM_DIR, year, config.get('Paths', 'elevation'), model='srcnn'
    )

    X, elev, Y, lats, lons, times = dataset.make_test(
        scale1=scale1, scale2=1.0 / UPSCALE_FACTOR ** n_stacked
    )

    #  resize x
    n, h, w, c = X.shape

    downscaled = {}

    for i in range(len(OTHER_CONFIGS)):
        ds = xr.open_dataset(
            os.path.join(OTHER_OUTPUT_DIRS[i], 'precip_%s_downscaled.nc' % year)
        )
        ds = ds['precip'].values[:, :, :, np.newaxis]
        downscaled[OTHER_CONFIGS[i].get('DeepSD', 'model_name')] = {
            'ds': ds,
            'wl': OTHER_CONFIGS[i].get('SRCNN', 'weighted_loss'),
            'alpha': OTHER_CONFIGS[i].getfloat('SRCNN', 'alpha'),
        }

    return X, Y, downscaled


def density_to_metric_plot(
    est_per_alpha: Dict[float, pd.DataFrame],
    smogn_est: Optional[pd.DataFrame] = None,
    filename_prefix: str = '',
    metric: str = 'normalized_absolute_error',
    window: int = 300000,
):

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=set_size(fraction=0.8))

    plt.xlabel('$p\'(y)$')
    plt.ylabel('Normalized MAE')
    max_a = 2.0  # 4.0 will get a special color outside of the colormap to improve interpretability of other plots
    as_to_plot = [0.0, 1.0, 2.0, 4.0]

    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.set_prop_cycle(
        color=[scalarMap.to_rgba(a) for a in as_to_plot if a != 4.0] + [(1, 0, 0, 1)]
    )

    for a, df in sorted(est_per_alpha.items()):
        if a in as_to_plot:
            sorted_df = df.sort_values(by=['density'])
            dens_windowed = convolve(
                sorted_df['density'], np.ones(window) / window, mode='valid'
            )
            metric_windowed = convolve(
                sorted_df[metric], np.ones(window) / window, mode='valid'
            )
            ax.plot(
                dens_windowed,
                metric_windowed,
                label='$\\alpha = %.1f$' % a,
                linestyle='--' if a == 4.0 else '-'
            )

    if smogn_est is not None:
        smogn_est = smogn_est.sort_values(by=['density'])
        dens_windowed = convolve(
            smogn_est['density'], np.ones(window) / window, mode='valid'
        )
        metric_windowed = convolve(
            smogn_est[metric], np.ones(window) / window, mode='valid'
        )
        ax.plot(
            dens_windowed, metric_windowed, label='SMOGN', color='#ff7f0e', alpha=0.5
        )

    plt.yscale('log')
    plt.xscale('log')
    ax.legend(fontsize='small')
    if len(filename_prefix) > 0:
        filename = (
            filename_prefix + '_density_to_' + metric + '_w' + str(window) + '_diff_loglog.pdf'
        )
    else:
        filename = 'density_to_' + metric + '_w' + str(window) + '_diff_loglog.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')


def create_density_column(
    df: pd.DataFrame, target_relevance: TargetRelevance
) -> pd.DataFrame:
    df['density'] = df.apply(lambda row: target_relevance.get_density(row['y']), axis=1)
    return df


if __name__ == '__main__':

    # Initialize matplotlib
    init_mpl(usetex=True)

    if os.path.exists('performance_per_density_plot_cache.pickle'):
        print(
            datetime.datetime.now(),
            'Using cached est_per_alpha from performance_per_density_plot_cache.pickle...',
        )
        est_per_alpha = pickle.load(
            open('performance_per_density_plot_cache.pickle', 'rb'), encoding='latin1'
        )

    else:
        highest_resolution = 4.0
        hr_resolution_km = config.getint('DeepSD', 'high_resolution')
        lr_resolution_km = config.getint('DeepSD', 'low_resolution')
        start = highest_resolution / hr_resolution_km
        N = int(math.log(lr_resolution_km / hr_resolution_km, UPSCALE_FACTOR))

        if os.path.exists('performance_per_density_plot_data_cache.pickle'):
            print(
                datetime.datetime.now(),
                'Using cached Y and downscaled from performance_per_density_plot_data_cache.pickle...',
            )
            Y, downscaled = pickle.load(
                open('performance_per_density_plot_data_cache.pickle', 'rb'),
                encoding='latin1',
            )

        else:
            year1 = config.getint('DataOptions', 'max_train_year') + 1
            yearlast = config.getint('DataOptions', 'max_year')
            _, Y, downscaled = load_data(year1, scale1=start, n_stacked=N)
            for y in range(year1 + 1, yearlast + 1):
                _, Y_new, downscaled_new = load_data(y, scale1=start, n_stacked=N)
                Y = np.concatenate([Y, Y_new], axis=0)
                for model_name in downscaled:
                    downscaled[model_name]['ds'] = np.concatenate(
                        [
                            downscaled[model_name]['ds'],
                            downscaled_new[model_name]['ds'],
                        ],
                        axis=0,
                    )
            pickle.dump(
                (Y, downscaled),
                open('performance_per_density_plot_data_cache.pickle', 'wb'),
            )

        # Remove nans and flatten
        not_nan_mask = ~np.isnan(Y)
        Y = Y[not_nan_mask]
        for model_name in downscaled:
            downscaled[model_name]['ds'] = downscaled[model_name]['ds'][not_nan_mask]

        assert (
            Y.shape == downscaled[list(downscaled.keys())[0]]['ds'].shape
        ), 'Y and downscaled should have equal shape'

        Y_df = pd.DataFrame(Y, columns=['y'])
        # Calculate density
        print(datetime.datetime.now(), 'Calculating densities...')
        target_relevance = TargetRelevance(Y_df['y'].to_numpy())
        Y_df = create_density_column(Y_df, target_relevance)
        print(datetime.datetime.now(), 'Calculated densities')

        # Put data into dataframes
        for model_name in downscaled:
            downscaled[model_name]['df'] = pd.concat(
                [
                    pd.DataFrame(downscaled[model_name]['ds'], columns=['estimate']),
                    Y_df,
                ],
                axis=1,
            )
            # Remove xarray datasets from downscaled to save memory
            del downscaled[model_name]['ds']

        print(datetime.datetime.now(), Y_df['y'].to_numpy(), Y_df['y'].to_numpy().shape)
        y_diff = Y_df['y'].max() - Y_df['y'].min()

        for model_name in downscaled:
            downscaled[model_name]['df'] = create_normalized_absolute_error_column(
                downscaled[model_name]['df'], y_diff
            )

        print(datetime.datetime.now(), 'Calculated error columns')

        # Group estimates by alpha-value
        est_list_per_alpha = {}
        model_names = list(downscaled.keys())
        for model_name in model_names:
            cur_a = downscaled[model_name]['alpha']
            if cur_a in est_list_per_alpha:
                est_list_per_alpha[cur_a].append(downscaled[model_name]['df'])
            else:
                est_list_per_alpha[cur_a] = [downscaled[model_name]['df']]
            
            # Remove appended df from downscaled to save memory
            del downscaled[model_name]

        est_per_alpha = {}
        for a in est_list_per_alpha.keys():
            est_per_alpha[a] = merge_estimates(est_list_per_alpha[a])

        print(datetime.datetime.now(), est_per_alpha)
        pickle.dump(
            est_per_alpha, open('performance_per_density_plot_cache.pickle', 'wb')
        )

    # Plot each target value to the mean absolute error
    density_to_metric_plot(est_per_alpha)
