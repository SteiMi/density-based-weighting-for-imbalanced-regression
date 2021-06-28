import argparse
import itertools
import math
from os.path import join, expanduser
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

from utils import (
    create_error_column,
    find_bin_boundaries,
    group_estimates_by_alpha,
    load_estimates,
    merge_estimates,
    nice_metric_name,
    to_bin_bounds,
    to_bin_string,
    to_latex_bin_str,
    init_mpl,
    set_size,
)

UltimatePerfDict = Dict[
    str,
    Dict[
        Tuple[int, int],
        Dict[Tuple[Any, Any], Tuple[float, float, float, float, bool, float, bool]],
    ],
]

'''
METRICS
'''


def rmse(y, y_pred):
    return np.sqrt(np.nanmean((y - y_pred) ** 2, axis=0))


def bias(y, y_pred):
    return np.nanmean(y - y_pred, axis=0)


def corr(y, y_pred):
    '''
    Manually calculate the correlation of each cell in y_pred with each corresponding cell in y.
    '''
    y_pred_hat = np.nanmean(y_pred, axis=0)
    y_hat = np.nanmean(y, axis=0)
    covs = np.nansum((y_pred - y_pred_hat) * (y - y_hat), axis=0) / y.shape[0]
    y_pred_std = np.nanstd(y_pred, axis=0)
    y_std = np.nanstd(y, axis=0)
    return covs / (y_pred_std * y_std)


def skill(y, y_pred, bins=10):

    # The result should be a skill score for each cell i. e. the shape should be (lat_cells, lon_cells)
    res = np.ma.zeros((y.shape[1], y.shape[2]))

    # Iterate over each cell
    for lat in range(y.shape[1]):
        for lon in range(y.shape[2]):

            y_pred_cell = y_pred[:, lat, lon].compressed()
            y_cell = y[:, lat, lon].compressed()

            # Mask the result of this cell when there are no real values
            if len(y_pred_cell) == 0 | len(y_cell) == 0:
                res[lat, lon] = np.ma.masked
                continue

            # Calculate the histogram over the time for the current cell
            y_pred_hist_cell = np.histogram(y_pred_cell, bins=bins, density=True)[0]
            y_hist_cell = np.histogram(y_cell, bins=bins, density=True)[0]

            # Manually normalize since density=True is weird
            # See https://stackoverflow.com/questions/21532667/numpy-histogram-cumulative-density-does-not-sum-to-1
            y_pred_hist_cell = y_pred_hist_cell / y_pred_hist_cell.sum()
            y_hist_cell = y_hist_cell / y_hist_cell.sum()

            # Calculate the skill at the cell according to https://journals.ametsoc.org/doi/full/10.1175/JCLI4253.1
            skill_at_cell = 0.0
            for b in range(bins):
                skill_at_cell += np.minimum([y_pred_hist_cell[b], y_hist_cell[b]])

            res[lat, lon] = skill_at_cell

            assert skill_at_cell <= 1.0, 'Skill cannot be >1. Something is wrong.'
            assert skill_at_cell >= 0.0, 'Skill cannot be <0. Something is wrong.'

    return res


def y_to_mae_plot(
    est_per_alpha: Dict[float, pd.DataFrame],
    smogn_est: Optional[pd.DataFrame] = None,
    filename_prefix: str = '',
):
    # plt.figure(figsize=set_size(fraction=0.75))
    plt.figure(figsize=(8, 8))
    plt.xlabel('$y$')
    plt.ylabel('MAE')
    max_a = np.max(list(est_per_alpha.keys())) + 0.2  # + 0.2 avoids white as line color
    for a, df in est_per_alpha.items():
        if a in [0.0, 0.5, 1.0, 1.5]:
            sorted_df = df.sort_values(by=['y'])
            plt.plot(
                sorted_df['y'],
                sorted_df['absolute_error'],
                label='$\\alpha = %.1f$' % a,
                color=str(a / max_a),
                alpha=0.5,
            )

    if smogn_est is not None:
        smogn_est = smogn_est.sort_values(by=['y'])
        plt.plot(
            smogn_est['y'],
            smogn_est['absolute_error'],
            label='SMOGN',
            color='#ff7f0e',
            alpha=0.5,
        )

    plt.legend(fontsize='small')
    if len(filename_prefix) > 0:
        filename = filename_prefix + '_y_to_mae.pdf'
    else:
        filename = 'y_to_mae.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')


def setBoxColors(bp: plt.boxplot, i: int = 0, color: str = '0.0'):
    plt.setp(bp['boxes'][i], color=color)
    plt.setp(bp['caps'][2 * i], color=color)
    plt.setp(bp['caps'][2 * i + 1], color=color)
    plt.setp(bp['whiskers'][2 * i], color=color)
    plt.setp(bp['whiskers'][2 * i + 1], color=color)
    plt.setp(bp['medians'][i], color=color)


def bin_to_metric_plot(
    est_list_per_bin_per_distval: Dict[Tuple[int, int], Dict[Any, List[pd.DataFrame]]],    bin_infos: pd.DataFrame,
    dvals_to_plot: List[Any] = [a / 10 for a in list(range(0, 21, 2))],
    metric: Callable = rmse,
    filename_prefix: str = '',
    with_smogn: bool = True,
):
    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=set_size(fraction=.5, subplot=[2, 1])
    )
    axs[0].set_title(filename_prefix)
    axs[1].set_xlabel('Bin')
    if filename_prefix in ['normal', 'dnormal']:
        axs[0].set_ylabel('$p$')
        axs[1].set_ylabel('RMSE')

    axs[0].set_ylim([0, 0.043])
    axs[1].set_ylim([3, 28])

    xticks = ['%d--%d' % br for br in est_list_per_bin_per_distval.keys()]

    # Calculate density per bin, see https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/histograms.py#L670-L921
    freq_per_bin = [
        len(b[list(b.keys())[0]][0]) for b in est_list_per_bin_per_distval.values()
    ]
    min_y = min(
        [
            b[list(b.keys())[0]][0]['y'].min()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    max_y = max(
        [
            b[list(b.keys())[0]][0]['y'].max()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    bin_size = (max_y - min_y) / len(xticks)
    axs[0].bar(xticks, freq_per_bin / bin_size / np.sum(freq_per_bin))

    plot_dict: Dict[Any, List[float]] = {}
    for dval in dvals_to_plot:
        plot_dict[dval] = []

    for b_range, dval_dict in est_list_per_bin_per_distval.items():
        for dval, dfs in dval_dict.items():
            if dval in dvals_to_plot:
                scores_in_bin = [metric(df['y'], df['estimate']) for df in dfs]
                plot_dict[dval].append(np.mean(scores_in_bin))

    max_a = 2.0
    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    axs[1].set_prop_cycle(
        color=[scalarMap.to_rgba(a) for a in dvals_to_plot]
    )
    linestyles = ['-', '--', '-.', ':']
    i = 0
    for dval, metric_vals in plot_dict.items():
        axs[1].plot(
            xticks,
            metric_vals,
            label=dval,
            linestyle=linestyles[i % len(linestyles)],
        )
        i += 1

    metric_str = metric.__name__
    smogn_str = '_with_smogn' if with_smogn else ''
    if len(filename_prefix) > 0:
        filename = filename_prefix + '_bin_to_%s%s.pdf' % (metric_str, smogn_str)
    else:
        filename = 'bin_to_%s%s.pdf' % (metric_str, smogn_str)
    fig.savefig(join('plots', filename), format='pdf', bbox_inches='tight')

    figlegend = plt.figure(figsize=(374.0 / 72.27, 1))
    figlegend.legend(
        *axs[1].get_legend_handles_labels(),
        loc='upper center',
        ncol=6,
        fontsize='x-small',
    )
    figlegend.savefig(
        join('plots', 'legend_' + filename), format='pdf', bbox_inches='tight'
    )


def bin_to_metric_plot(
    est_list_per_bin_per_distval: Dict[Tuple[int, int], Dict[Any, List[pd.DataFrame]]],
    bin_infos: pd.DataFrame,
    dvals_to_plot: List[Any] = [a / 10 for a in list(range(0, 21, 2))],
    metric: Callable = rmse,
    filename_prefix: str = '',
    with_smogn: bool = True,
):

    dataset_bin_info = bin_infos.sort_values(by=['count'])

    # Create mappings from bin to bin rank and vice versa
    bin_to_rank = {}
    rank_to_bin = {}
    for i, bin_row in enumerate(dataset_bin_info.iterrows()):
        bin_to_rank[(bin_row[1]['bin_low'], bin_row[1]['bin_high'])] = i
        rank_to_bin[i] = (bin_row[1]['bin_low'], bin_row[1]['bin_high'])

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=set_size(fraction=.5, subplot=[2, 1])
    )
    axs[0].set_title(filename_prefix)
    axs[1].set_xlabel('Bin Rank')
    if filename_prefix in ['normal', 'dnormal']:
        axs[0].set_ylabel('$p$')
        axs[1].set_ylabel('RMSE')

    axs[0].set_ylim([0, 0.044])
    axs[1].set_ylim([3, 28])

    # Calculate density per bin, see https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/histograms.py#L670-L921
    freq_per_bin = [
        len(b[list(b.keys())[0]][0]) for b in est_list_per_bin_per_distval.values()
    ]
    min_y = min(
        [
            b[list(b.keys())[0]][0]['y'].min()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    max_y = max(
        [
            b[list(b.keys())[0]][0]['y'].max()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    bin_size = (max_y - min_y) / len(est_list_per_bin_per_distval.keys())

    xticks = ['%d' % int(bin_to_rank[br] + 1) for br in est_list_per_bin_per_distval.keys()]
    axs[0].bar(xticks, freq_per_bin / bin_size / np.sum(freq_per_bin))

    plot_dict: Dict[Any, List[float]] = {}
    for dval in dvals_to_plot:
        plot_dict[dval] = []

    for b_range, dval_dict in est_list_per_bin_per_distval.items():
        for dval, dfs in dval_dict.items():
            if dval in dvals_to_plot:
                scores_in_bin = [metric(df['y'], df['estimate']) for df in dfs]
                plot_dict[dval].append(np.mean(scores_in_bin))

    max_a = 2.0
    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    axs[1].set_prop_cycle(
        color=[scalarMap.to_rgba(a) for a in dvals_to_plot]
    )
    linestyles = ['-', '--', '-.', ':']
    i = 0
    for dval, metric_vals in plot_dict.items():
        axs[1].plot(
            xticks,
            metric_vals,
            label=dval,
            linestyle=linestyles[i % len(linestyles)],
        )
        i += 1

    metric_str = metric.__name__
    smogn_str = '_with_smogn' if with_smogn else ''
    if len(filename_prefix) > 0:
        filename = filename_prefix + '_bin_to_%s%s.pdf' % (metric_str, smogn_str)
    else:
        filename = 'bin_to_%s%s.pdf' % (metric_str, smogn_str)
    fig.savefig(join('plots', filename), format='pdf', bbox_inches='tight')

    figlegend = plt.figure(figsize=(374.0 / 72.27, 1))
    figlegend.legend(
        *axs[1].get_legend_handles_labels(),
        loc='upper center',
        ncol=6,
        fontsize='x-small',
    )
    figlegend.savefig(
        join('plots', 'legend_' + filename), format='pdf', bbox_inches='tight'
    )


def create_bins(
    est_list_per_alpha: Dict[Any, List[pd.DataFrame]],
    bin_percentage: int,
    y_col_name: str = 'y',
) -> Tuple[Dict[Tuple[int, int], Dict[Any, List[pd.DataFrame]]], pd.DataFrame]:
    est_list_per_bin_per_alpha: Dict[
        Tuple[int, int], Dict[Any, List[pd.DataFrame]]
    ] = {}
    bin_infos = pd.DataFrame(
        columns=['bin_low', 'bin_high', 'y_low', 'y_high', 'count']
    )
    for b in range(0 + bin_percentage, 100 + bin_percentage, bin_percentage):
        b_low = b - bin_percentage
        b_high = b
        y_low, y_high = find_bin_boundaries(
            b_low, b_high, est_list_per_alpha[list(est_list_per_alpha.keys())[0]][0]
        )

        if b_high == 100:
            # Set y_high to infinity for the last bin. Otherwise the last data point is missing since df['y'] < y_high.
            y_high = np.inf

        est_list_per_bin_per_alpha[(b_low, b_high)] = {}
        for a, dfs in est_list_per_alpha.items():
            est_list_per_bin_per_alpha[(b_low, b_high)][a] = [
                df[(df[y_col_name] >= y_low) & (df[y_col_name] < y_high)] for df in dfs
            ]

        some_valid_a = list(est_list_per_alpha.keys())[0]
        # print((b_low, b_high), (y_low, y_high),
        #       len(est_list_per_bin_per_alpha[(b_low, b_high)][some_valid_a][0]), 'elements')
        bin_info = {
            'bin_low': b_low,
            'bin_high': b_high,
            'y_low': y_low,
            'y_high': y_high,
            'count': len(est_list_per_bin_per_alpha[(b_low, b_high)][some_valid_a][0]),
        }
        bin_infos = bin_infos.append(bin_info, ignore_index=True)

    print(bin_infos)

    return est_list_per_bin_per_alpha, bin_infos


def evaluate_bin(
    dval_dict: Dict[Any, List[pd.DataFrame]], metric: Callable
) -> Dict[Tuple[Any, Any], Tuple[float, float, float, float, bool, float, bool]]:
    '''
    Evaluate and compare all distinguishing values for a provided metric with signficance tests.

    dval_dict: Dict[Any, List[pd.DataFrame]], dictionary with distinguishing value as the key and a list of DataFrames
    containing the model estimates for the runs of a distinguishing value as the value.
    metric: Callable, callable that calculates a metric given the target values and the estimates of a model

    # Returns
    A dictionary with distinguishing value tuples as keys and a tuple containing the performance comparison of the
    corresponding distinguishing values as values.
    '''

    result_dict: Dict[
        Tuple[Any, Any], Tuple[float, float, float, float, bool, float, bool]
    ] = {}

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
    metrics_per_diff_val: Dict[Any, List[float]] = {}
    # Calculate the current metric for each dataframe per diff_val
    for dval, dfs in dval_dict.items():
        metrics_per_diff_val[dval] = [metric(df['y'], df['estimate']) for df in dfs]

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
    est_list_per_bin_per_distval: Dict[Tuple[int, int], Dict[Any, List[pd.DataFrame]]],
    metrics: List[Callable] = [rmse, bias, mean_absolute_error, r2_score],
    ultimate_perf_dict: Optional[UltimatePerfDict] = None,
) -> UltimatePerfDict:
    '''
    ultimate_perf_dict looks, for example, like this:
    {
        'rmse': {
            (0, 20): {
                ('SMOGN', 0.0): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                ('SMOGN', 0.1): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                ('SMOGN', 0.2): (mean_rmse_left, mean_rmse_right, mean_rmse_diff,
                                 mann_p, mann_sig?, wilc_p, wilc_sig?),
                ...
            }
        }
    }
    '''
    if not ultimate_perf_dict:
        ultimate_perf_dict = {}

    for metric in metrics:
        if metric.__name__ not in ultimate_perf_dict.keys():
            ultimate_perf_dict[metric.__name__] = {}
        for b_range, dval_dict in est_list_per_bin_per_distval.items():
            ultimate_perf_dict[metric.__name__][b_range] = evaluate_bin(
                dval_dict, metric
            )

    return ultimate_perf_dict


def mean_evaluation(
    est_list_per_distval: Dict[Any, List[pd.DataFrame]],
    metrics: List[Callable] = [rmse, bias, mean_absolute_error, r2_score],
    ultimate_perf_dict: Optional[UltimatePerfDict] = None,
) -> UltimatePerfDict:

    if not ultimate_perf_dict:
        ultimate_perf_dict = {}

    for metric in metrics:
        if metric.__name__ not in ultimate_perf_dict.keys():
            ultimate_perf_dict[metric.__name__] = {}
        ultimate_perf_dict[metric.__name__][(0, 100)] = evaluate_bin(
            est_list_per_distval, metric
        )

    return ultimate_perf_dict


def print_latex_table(
    ultimate_perf_dict: UltimatePerfDict,
    alphas: List[Any] = [0.0, 0.5, 1.0, 1.5, '{SMOGN}'],
    metrics: List[str] = ['rmse', 'mean_absolute_error'],
):
    bin_strings: List[str] = [
        to_bin_string(b_low, b_high)
        for b_low, b_high in ultimate_perf_dict[metrics[0]].keys()
    ]

    # Remove non-existant alphas
    for a in alphas:
        if (a, 0.0) not in ultimate_perf_dict[metrics[0]][
            to_bin_bounds(bin_strings[0])
        ].keys():
            alphas.remove(a)
            print('Alpha', a, 'not found in ultimate_perf_dict.')

    iterables = [np.vectorize(nice_metric_name)(metrics), alphas]

    idx = pd.MultiIndex.from_product(iterables, names=['Metric', 'Alpha'])

    latex_bin_strings = [to_latex_bin_str(bin_str) for bin_str in bin_strings]
    table = pd.DataFrame(0, index=latex_bin_strings, columns=idx)

    table = table.sort_index()

    for bin_str in bin_strings:
        for m in metrics:
            for a in alphas:

                format_str = '%.5f'

                # Print bold when significantly different to alpha=0
                if ultimate_perf_dict[m][to_bin_bounds(bin_str)][(a, 0.0)][-1] is True:
                    format_str = f'\\bfseries {format_str}'

                # Underline when significantly different to SMOGN
                if (
                    '{SMOGN}' in alphas
                    and ultimate_perf_dict[m][to_bin_bounds(bin_str)][(a, '{SMOGN}')][
                        -1
                    ]
                    is True
                ):
                    format_str = '\\Uline{' + format_str + '}'

                table.loc[to_latex_bin_str(bin_str), (nice_metric_name(m), a)] = (
                    format_str
                    % ultimate_perf_dict[m][to_bin_bounds(bin_str)][(a, a)][0]
                )
    print(table)
    col_format = '@{}l' + 'S' * len(table.iloc[0]) + '@{}'
    latex = table.to_latex(
        column_format=col_format,
        escape=False,
        float_format='%.2f',
        multicolumn_format='c',
        multirow=True,
    )

    latex = latex.replace('Metric', '\\multirow{ 2}{*}{Bins}')
    latex = latex.replace('Alpha', '')

    print(latex)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-paths',
        type=str,
        nargs='+',
        help='Base save path for the results.',
        default=['checkpoints'],
    )
    parser.add_argument(
        '--plot-prefix',
        type=str,
        default='',
        help='Prefix for the y_to_mae-plot\'s filename.',
    )
    parser.add_argument(
        '--output-dir', type=str, default='jupyter', help='Directory for outputs.'
    )
    args = parser.parse_args()

    # Initialize matplotlib
    init_mpl(usetex=True)

    print('Loading estimates...')
    test_estimates = load_estimates(save_paths=args.save_paths, split='test')
    print('Estimates loaded')

    # Calculate the error of each individual data point in each DataFrame
    test_estimates = [(d, create_error_column(df)) for d, df in test_estimates]

    # Filter SMOGN runs
    smogn_estimates = [(d, df) for d, df in test_estimates if d['smogn'] is True]
    smogn_dw_estimates = [(d, df) for d, df in test_estimates if d['smogn_dw'] is True]
    test_estimates = [(d, df) for d, df in test_estimates if d['smogn'] is False and d['smogn_dw'] is False]

    # Group estimates by alpha-value
    test_est_list_per_alpha = group_estimates_by_alpha(test_estimates)
    est_list_smogn = [df for d, df in smogn_estimates]
    est_list_smogn_dw = [df for d, df in smogn_dw_estimates]

    # Merge estimates per alpha into a single DataFrame
    est_per_alpha: Dict[float, pd.DataFrame] = {}
    for a, df_list in test_est_list_per_alpha.items():
        est_per_alpha[a] = merge_estimates(df_list)

    # Merge SMOGN runs by averaging
    smogn_est = None
    if len(smogn_estimates) > 0:
        smogn_est = merge_estimates([df for _, df in smogn_estimates])

    smogn_dw_est = None
    if len(smogn_dw_estimates) > 0:
        smogn_dw_est = merge_estimates([df for _, df in smogn_dw_estimates])

    # Plot each target value to the mean absolute error
    # y_to_mae_plot(est_per_alpha, smogn_est=smogn_est, filename_prefix=args.plot_prefix)

    # Add the SMOGN DataFrame to est_per_alpha
    # distval = distinguishing value = Value that distinguishes each entry from the other entries (alpha, SMOGN, ..)
    est_list_per_distval: Dict[Any, List[pd.DataFrame]] = test_est_list_per_alpha
    if len(smogn_estimates) > 0:
        est_list_per_distval['{SMOGN}'] = est_list_smogn
    if len(smogn_dw_estimates) > 0:
        est_list_per_distval['{SMOGN_DW}'] = est_list_smogn_dw

    # Bin the data points by target value
    est_list_per_bin_per_distval, bin_infos = create_bins(est_list_per_distval, 20)
    bin_infos.to_csv(join(args.output_dir, '%sbin_infos.csv' % (args.plot_prefix + '_')), index=False)

    # # Create box plots for each bin
    # # bin_to_metric_boxplot(est_list_per_bin_per_distval, filename_prefix=args.plot_prefix, with_smogn=True)
    # # bin_to_metric_boxplot(est_list_per_bin_per_distval, filename_prefix=args.plot_prefix, with_smogn=False)

    # Create line plots over the bins
    bin_to_metric_plot(est_list_per_bin_per_distval, bin_infos, filename_prefix=args.plot_prefix, with_smogn=False)

    # # Evaluate mean performance
    ultimate_perf_dict = mean_evaluation(est_list_per_distval)

    # # Evaluate by bin
    ultimate_perf_dict = binned_evaluation(est_list_per_bin_per_distval, ultimate_perf_dict=ultimate_perf_dict)

    # print(ultimate_perf_dict)

    pickle.dump(ultimate_perf_dict,
                open(expanduser(join(args.output_dir, '%sultimate_perf_dict.pickle' % (args.plot_prefix + '_'))),
                     'wb'))

    # Print a latex table
    # dvals_to_print: List[Any] = [0.0, 0.5, 1.0, '{SMOGN}']
    # if len(smogn_estimates) > 0:
    #     dvals_to_print.append('{SMOGN}')
    # print_latex_table(ultimate_perf_dict, alphas=dvals_to_print)
