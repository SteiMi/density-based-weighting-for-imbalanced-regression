"""
This script can evaluate the results of DeepSDs runs and compare them against each other.
"""
import sys, os, math, glob
from os.path import join
import itertools
import pickle

try:
    import configparser as ConfigParser
except ImportError:
    import ConfigParser

import numpy as np
import tensorflow as tf
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon

import prism

flags = tf.flags
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('config_file', 'config.ini',
                    'Configuration file with [SRCNN], [Model-%], and [DeepSD] sections.')


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
OTHER_OUTPUT_DIRS = [os.path.expanduser(os.path.join(c.get('SRCNN', 'scratch'), c.get('DeepSD', 'model_name'), 'outputs')) for c in OTHER_CONFIGS]

# Base some configurations which we will use for all models on an arbitrary config and hope that this is ok for all models
config = OTHER_CONFIGS[0]
print('Base config from which we take the upscale factor and the PRISM dir (IF NOT ON laptop) for all models:',
      config.get('DeepSD', 'model_name'))

if os.uname()[0] == 'Darwin':
    # We are on MacOS and most likely on my laptop where prism data lies in ~/data/prism
    PRISM_DIR = os.path.expanduser(os.path.join('~', 'data', 'prism', 'ppt', 'raw'))
    print('Using local prism files in', PRISM_DIR)
else:
    # We are probably in Kubernetes or something
    PRISM_DIR = os.path.expanduser(os.path.join(config.get('Paths', 'prism'), 'ppt', 'raw'))

UPSCALE_FACTOR = config.getint('DeepSD', 'upscale_factor')

COLORS = ['red', 'blue', 'green', 'purple', 'cyan', 'yellow', 'brown']

"""
METRICS
"""


def rmse(y, y_pred):
    return np.sqrt(np.nanmean((y - y_pred)**2, axis=0))


def mae(y, y_pred):
    return np.nanmean(np.abs(y - y_pred), axis=0)


metrics = {
    'mae': mae,
    'rmse': rmse
}

"""
PLOTS
"""


def performance_per_percentiles(Y, y_pred, model_name, color=COLORS[0],
                                percentiles=np.array([0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 99.9])):

    perfs = {}
    for metric in metrics:
        mean = []
        q25 = []
        q75 = []

        for perc in percentiles:
            print('Mean cell Percentile', perc, np.nanmean(np.nanpercentile(Y, perc, axis=0)))
            perc_perf_per_loc = extreme_performance_per_location(Y, y_pred, perc, metrics[metric])
            mean.append(np.nanmean(perc_perf_per_loc.compressed()))

            # We need to use compressed so that it ignores the masked values here
            # See: https://stackoverflow.com/questions/37935954/how-can-i-run-a-numpy-function-percentile-on-a-masked-array
            q25.append(np.nanpercentile(perc_perf_per_loc.compressed(), 25))
            q75.append(np.nanpercentile(perc_perf_per_loc.compressed(), 75))

        perfs[metric] = {
            'mean': mean,
            'q25': q25,
            'q75': q75
        }

    # Return the mean performance per location for each percentile
    return perfs


def plot_performance_per_cell(fig, Y, y_pred, metric, model_name):

    # Later you get a new subplot; change the geometry of the existing
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n+1, 1, i+1)

    # Add the new
    ax = fig.add_subplot(n+1, 1, n+1)
    metric_per_cell = metric(Y, y_pred)
    ax.imshow(metric_per_cell[:, :, 0], vmax=10, cmap='inferno')
    ax.text(5, 35, model_name, bbox={'facecolor': 'white', 'pad': 10})


def load_data(year, scale1, n_stacked):
    # read prism dataset
    # resnet parameter will not re-interpolate X
    dataset = prism.PrismSuperRes(PRISM_DIR, year, config.get('Paths', 'elevation'), model='srcnn')

    X, elev, Y, lats, lons, times = dataset.make_test(scale1=scale1, scale2=1./UPSCALE_FACTOR**n_stacked)

    #  resize x
    n, h, w, c = X.shape

    downscaled = {}

    for i in range(len(OTHER_CONFIGS)):
        ds = xr.open_dataset(join(OTHER_OUTPUT_DIRS[i], 'precip_%s_downscaled.nc' % year))
        ds = ds['precip'].values[:,:,:,np.newaxis]
        downscaled[OTHER_CONFIGS[i].get('DeepSD', 'model_name')] = {
            'ds': ds,
            'wl': OTHER_CONFIGS[i].get('SRCNN', 'weighted_loss'),
            'alpha': OTHER_CONFIGS[i].getfloat('SRCNN', 'alpha')
            }

    return X, Y, downscaled


def extreme_performance_per_location(Y, downscaled, percentile, metric):
    # Remove all values that are below the percentile or are NaN for each cell
    extreme_mask = (Y < np.nanpercentile(Y, percentile, axis=0)) | (np.isnan(Y))
    extreme_Y = np.ma.masked_array(Y, mask=extreme_mask)[:,:,:,0]
    extreme_downscaled = np.ma.masked_array(downscaled, mask=extreme_mask)[:,:,:,0]

    return metric(extreme_Y, extreme_downscaled)


def evaluate_1v1(X, Y, downscaled):
    """
    This works only with python3 since you need matplotlib>=3.1 to have DivergingNorm, which I use for a plot.
    matplotlib>=3.0 is not available for python2.
    """
    from matplotlib.colors import DivergingNorm

    fig = plt.figure(figsize=(9, 13))

    ax = fig.add_subplot(111)
    metric_per_cell = {}
    for i, model_name in enumerate(downscaled):

        metric_per_cell[model_name] = rmse(downscaled[model_name], Y)

    models = list(metric_per_cell.keys())
    if len(models) > 2:
        print('Warning: You compare more than 2 models but use a visualization that works only for two!')

    difference = metric_per_cell[models[0]] - metric_per_cell[models[1]]

    im = ax.imshow(difference[:, :, 0], cmap='bwr_r', norm=DivergingNorm(vcenter=0.))
    ax.text(5, 35, models[0] + ' - ' + models[1], bbox={'facecolor': 'white', 'pad': 10})
    fig.colorbar(im, ax=ax)

    plt.legend()
    plt.show()


def evaluate(X, Y, downscaled):

    # percentile 0 should just yield the average performance over all datapoints
    percentiles = np.array([0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 99, 99.9])

    perf_dict = {}

    difference_criterium = 'alpha'
    distinct_difference_values = list(set([m[difference_criterium] for m in downscaled.values()]))
    print('Comparing', difference_criterium, 'values:', distinct_difference_values)

    print('Calculating performance for models...')
    for i, model_name in enumerate(downscaled):

        overall_rmse = np.sqrt(np.nanmean((downscaled[model_name]['ds'] - Y)**2))
        print('==================')
        print(model_name)
        print('==================')
        print('Overall rmse:', overall_rmse)

        perfs_per_perc = performance_per_percentiles(Y, downscaled[model_name]['ds'], model_name,
                                                     percentiles=percentiles)
        perfs_per_perc['overall_rmse'] = {
            'mean': overall_rmse
            }

        perf_dict[model_name] = perfs_per_perc

    # Do tests for statistical significance
    print('Calculating comparisons between models...')

    # ultimate_perf_dict looks, for example, like this:
    # {
    #   'rmse' = {
    #       perc_0: {
    #           (alpha_0, alpha_0) = 
    #                (mean_rmse_perc_0_left, mean_rmse_perc_0_right, mean_rmse_diff_perc_0,
    #                 mann_p_0, mann_sig?, wilc_p_0, wilc_sig?)
    #           (alpha_0, alpha_1) = ...
    #   }
    # }
    ultimate_perf_dict = {}
    for metric in metrics:
        ultimate_perf_dict[metric] = {}
        for i, perc in enumerate(percentiles):
            print('perc:', perc)
            ultimate_perf_dict[metric][perc] = {}
            # diff_vals_to_compare for alpha might look like: [(0.0, 0.0), (0.0, 0.2), (0.0, 0.4), ...]
            diff_vals_to_compare = list(itertools.product(distinct_difference_values, distinct_difference_values))
            # First, initialize lists per comparison pair in ultimate_perf_dict
            for diff_val_0, diff_val_1 in diff_vals_to_compare:
                ultimate_perf_dict[metric][perc][(diff_val_0, diff_val_1)] = []

            for diff_val_0, diff_val_1 in diff_vals_to_compare:
                model_names_0 = [mn for mn, dct in downscaled.items() if dct[difference_criterium] == diff_val_0]
                model_names_1 = [mn for mn, dct in downscaled.items() if dct[difference_criterium] == diff_val_1]

                # perfs_0/1 = runs x len(percentiles)
                perfs_0 = np.array([perf_dict[mn][metric]['mean'] for mn in model_names_0])
                perfs_1 = np.array([perf_dict[mn][metric]['mean'] for mn in model_names_1])

                perfs_0_at_perc = perfs_0[:, i]
                perfs_1_at_perc = perfs_1[:, i]
                print(metric, '@', perc, ':', diff_val_0, 'vs.', diff_val_1)
                print(perfs_0_at_perc)
                print(perfs_1_at_perc)

                mean_0 = np.mean(perfs_0_at_perc)
                mean_1 = np.mean(perfs_1_at_perc)
                diff = mean_0 - mean_1
                mann_stat, mann_p = mannwhitneyu(perfs_0_at_perc, perfs_1_at_perc)
                if mann_p > 0.05:
                    mann_significant = False
                else:
                    mann_significant = True
                wilc_stat, wilc_p = wilcoxon(perfs_0_at_perc, perfs_1_at_perc)
                if wilc_p > 0.05:
                    wilc_significant = False
                else:
                    wilc_significant = True

                ultimate_perf_dict[metric][perc][(diff_val_0, diff_val_1)].append((mean_0, mean_1, diff,
                                                                                   mann_p, mann_significant,
                                                                                   wilc_p, wilc_significant))

    pickle.dump(ultimate_perf_dict,
                open(os.path.expanduser(os.path.join(config.get('SRCNN', 'scratch'), 'ultimate_perf_dict.pickle')),
                     'wb'))
    print(ultimate_perf_dict)


def find_bin_boundaries(b_low, b_high, data):
    """
    Given bin boundaries in percentages and data, find the boundaries in y-space.
    Example: b_low = 0, b_high = 10, y ranges from 100 to 200 -> y_low = 100, y_high = 110
    """
    assert b_low >= 0, 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low <= 100, 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_high >= 0, 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low <= 100, 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (b_low, b_high)
    assert b_low < b_high, 'b_low is not smaller than b_high (b_low: %d, b_high: %d)' % (b_low, b_high)
    y_min = np.nanmin(data)
    y_max = np.nanmax(data)
    y_range = y_max - y_min
    y_low = y_min + (b_low / 100.) * y_range
    y_high = y_min + (b_high / 100.) * y_range
    assert y_low >= y_min
    assert y_high <= y_max
    return y_low, y_high


def create_bins(Y, bin_percentage):
    bin_infos = []
    for b in range(0 + bin_percentage, 100 + bin_percentage, bin_percentage):
        b_low = b - bin_percentage
        b_high = b
        y_low, y_high = find_bin_boundaries(
            b_low, b_high, Y
        )

        if b_high == 100:
            # Set y_high to infinity for the last bin. Otherwise the last data point is missing since df['y'] < y_high.
            y_high = np.inf

        # Remove all values that are not in the bin or are NaN for each cell
        not_in_bin_mask = ((Y >= y_high) | (Y < y_low) | (np.isnan(Y)))
        bin_Y = np.ma.masked_array(Y, mask=not_in_bin_mask)#[:,:,:,0]
        count = bin_Y.count()

        bin_info = {
            'bin_low': b_low,
            'bin_high': b_high,
            'y_low': y_low,
            'y_high': y_high,
            'count': count
        }
        bin_infos.append(bin_info)

    # Sort bins by count, the index of each bin in the list equals the bin rank
    bin_infos = sorted(bin_infos, key=lambda k: k['count'])

    return bin_infos


def evaluate_bin(dval_dict, bin_Y, metric):
    '''
    Evaluate and compare all distinguishing values for a provided metric with signficance tests.

    dval_dict: Dict[Any, List[np.array]], dictionary with distinguishing value as the key and a list of DataFrames
    containing the model estimates for the runs of a distinguishing value as the value.
    metric: Callable, callable that calculates a metric given the target values and the estimates of a model

    # Returns
    A dictionary with distinguishing value tuples as keys and a tuple containing the performance comparison of the
    corresponding distinguishing values as values.
    '''

    result_dict = {}

    # Find all distinct diff_vals, e.g. all alphas
    distinct_difference_values = list(set(dval_dict.keys()))
    # diff_vals_to_compare for alpha might look like: [(0.0, 0.0), (0.0, 0.2), (0.0, 0.4), ...]
    diff_vals_to_compare = list(
        itertools.product(distinct_difference_values, distinct_difference_values)
    )

    # Check whether there are any data points in this bin
    if len(dval_dict[list(dval_dict.keys())[0]][0]) == 0:
        for diff_val_0, diff_val_1 in diff_vals_to_compare:
            result_dict[(diff_val_0, diff_val_1)] = (
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                False,
                math.nan,
                False,
            )
        return result_dict

    # Calculate the current metric for all dataframes in each diff_val
    metrics_per_diff_val = {}
    # Calculate the current metric for each dataframe per diff_val
    for dval, arrs in dval_dict.items():
        metrics_per_diff_val[dval] = [metric(bin_Y, arr) for arr in arrs]

    for diff_val_0, diff_val_1 in diff_vals_to_compare:
        perfs_0 = metrics_per_diff_val[diff_val_0]
        perfs_1 = metrics_per_diff_val[diff_val_1]

        mean_0 = np.mean(perfs_0)
        mean_1 = np.mean(perfs_1)
        diff = mean_0 - mean_1

        # print(metric.__name__, ':', diff_val_0, 'vs.', diff_val_1, ',', mean_0, 'vs.', mean_1)
        # print(perfs_0)
        # print(perfs_1)

        if (
            diff_val_0 == diff_val_1
            or np.sum(np.array(perfs_0) - np.array(perfs_1)) == 0.0
        ):
            mann_p = 1.0
            mann_significant = False
            wilc_p = 1.0
            wilc_significant = False

        else:
            mann_stat, mann_p = mannwhitneyu(perfs_0, perfs_1)
            if mann_p > 0.05:
                mann_significant = False
            else:
                mann_significant = True
            wilc_stat, wilc_p = wilcoxon(perfs_0, perfs_1)
            if wilc_p > 0.05:
                wilc_significant = False
            else:
                wilc_significant = True

        result_dict[(diff_val_0, diff_val_1)] = (
            mean_0,
            mean_1,
            diff,
            mann_p,
            mann_significant,
            wilc_p,
            wilc_significant,
        )
    return result_dict


def binned_evaluation(
    Y,
    downscaled,
    metrics=[rmse, mae],
    ultimate_perf_dict=None,
):
    '''
    ultimate_perf_dict looks, for example, like this:
    {
        'rmse': {
            0: {
                (0.0, 0.2): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                (0.0, 0.4): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                (0.0, 0.6): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                ...
            }
        }
    }
    '''
    bin_infos = create_bins(Y, 20)
    print(bin_infos)

    if not ultimate_perf_dict:
        ultimate_perf_dict = {}

    for metric in metrics:
        if metric.__name__ not in ultimate_perf_dict.keys():
            ultimate_perf_dict[metric.__name__] = {}
        for bin_rank in range(len(bin_infos)):
            bin_info = bin_infos[bin_rank]

            # Remove all values that are not in the bin or are NaN for each cell
            in_bin_mask = ((Y < bin_info['y_high']) & (Y >= bin_info['y_low']) & (~np.isnan(Y)))
            bin_Y = Y[in_bin_mask]

            dvals = set(v['alpha'] for v in downscaled.values())
            dvals_dict_bin = {dval: [] for dval in dvals}
            for k, v in downscaled.items():
                dvals_dict_bin[v['alpha']] += [v['ds'][in_bin_mask]]

            print('Bin rank', bin_rank)
            print(bin_Y)
            print(dvals_dict_bin[0.0])

            ultimate_perf_dict[metric.__name__][bin_rank] = evaluate_bin(
                dvals_dict_bin, bin_Y, metric
            )

    pickle.dump(ultimate_perf_dict,
                open(os.path.expanduser(os.path.join(config.get('SRCNN', 'scratch'),
                                                     'ultimate_perf_dict_binned.pickle')),
                     'wb'))
    print(ultimate_perf_dict)


if __name__ == '__main__':
    highest_resolution = 4.
    hr_resolution_km = config.getint('DeepSD', 'high_resolution')
    lr_resolution_km = config.getint('DeepSD', 'low_resolution')
    start = highest_resolution / hr_resolution_km
    N = int(math.log(lr_resolution_km / hr_resolution_km, UPSCALE_FACTOR))

    year1 = config.getint('DataOptions', 'max_train_year')+1
    yearlast = config.getint('DataOptions', 'max_year')
    X, Y, downscaled = load_data(year1, scale1=start, n_stacked=N)
    for y in range(year1+1, yearlast+1):
        X_new, Y_new, downscaled_new = load_data(y, scale1=start, n_stacked=N)
        X = np.concatenate([X, X_new], axis=0)
        Y = np.concatenate([Y, Y_new], axis=0)
        for model_name in downscaled:
            downscaled[model_name]['ds'] = np.concatenate([downscaled[model_name]['ds'],
                                                           downscaled_new[model_name]['ds']], axis=0)

    # evaluate(X, Y, downscaled)
    binned_evaluation(Y, downscaled)
