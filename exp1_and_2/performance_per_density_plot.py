from os.path import expanduser, join
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib import colors

from target_relevance import TargetRelevance
from utils import (
    create_normalized_absolute_error_column,
    init_mpl,
    load_estimates,
    merge_estimates,
    set_size,
)


def density_to_metric_plot(
    est_per_alpha: Dict[float, pd.DataFrame],
    smogn_est: Optional[pd.DataFrame] = None,
    smogn_dw_est: Optional[pd.DataFrame] = None,
    filename_prefix: str = '',
    metric: str = 'normalized_absolute_error',
    window: int = 30,
):

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=set_size(fraction=0.8))

    plt.xlabel('$p\'(y)$')
    plt.ylabel('Normalized MAE')
    max_a = 2.0
    as_to_plot = [0.0, 1.0, 2.0]

    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(a) for a in as_to_plot])

    for a, df in est_per_alpha.items():
        if a in as_to_plot:
            sorted_df = df.sort_values(by=['density'])
            dens_windowed = np.convolve(
                sorted_df['density'], np.ones(window) / window, mode='valid'
            )
            metric_windowed = np.convolve(
                sorted_df[metric], np.ones(window) / window, mode='valid'
            )
            ax.plot(
                dens_windowed,
                metric_windowed,
                label='$\\alpha = %.1f$' % a,
            )

    if smogn_est is not None:
        smogn_est = smogn_est.sort_values(by=['density'])
        dens_windowed = np.convolve(
            smogn_est['density'], np.ones(window) / window, mode='valid'
        )
        metric_windowed = np.convolve(
            smogn_est[metric], np.ones(window) / window, mode='valid'
        )
        ax.plot(
            dens_windowed, metric_windowed, label='SMOGN', color='#ff7f0e', alpha=0.5
        )

    if smogn_dw_est is not None:
        smogn_dw_est = smogn_dw_est.sort_values(by=['density'])
        dens_windowed = np.convolve(
            smogn_dw_est['density'], np.ones(window) / window, mode='valid'
        )
        metric_windowed = np.convolve(
            smogn_dw_est[metric], np.ones(window) / window, mode='valid'
        )
        ax.plot(
            dens_windowed,
            metric_windowed,
            label='SMOGN-DW',
            color='red',
            # alpha=1.,
        )

    ax.legend(fontsize='small')
    if len(filename_prefix) > 0:
        filename = (
            filename_prefix + '_density_to_' + metric + '_w' + str(window) + '.pdf'
        )
    else:
        filename = 'density_to_' + metric + '_w' + str(window) + '_diff.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')


def density_to_metric_plot_split(
    est_per_alpha: Dict[float, pd.DataFrame],
    smogn_est: Optional[pd.DataFrame] = None,
    smogn_dw_est: Optional[pd.DataFrame] = None,
    filename_prefix: str = '',
    metric: str = 'normalized_absolute_error',
    window: int = 300,
):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=set_size(fraction=0.8))
    fig.subplots_adjust(hspace=0.1)  # adjust space between axes

    plt.xlabel('$p\'(y)$')
    plt.ylabel('Normalized MAE')
    ax2.yaxis.set_label_coords(-0.09, 1)

    as_to_plot = [0.0, 1.0]
    max_a = 2.0

    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax1.set_prop_cycle(color=[scalarMap.to_rgba(a) for a in as_to_plot])
    ax2.set_prop_cycle(color=[scalarMap.to_rgba(a) for a in as_to_plot])

    for a, df in est_per_alpha.items():
        if a in as_to_plot:
            sorted_df = df.sort_values(by=['density'])
            dens_windowed = np.convolve(
                sorted_df['density'], np.ones(window) / window, mode='valid'
            )
            metric_windowed = np.convolve(
                sorted_df[metric], np.ones(window) / window, mode='valid'
            )
            ax1.plot(
                dens_windowed,
                metric_windowed,
                label='$\\alpha = %.1f$' % a,
            )
            ax2.plot(
                dens_windowed,
                metric_windowed,
                label='$\\alpha = %.1f$' % a,
            )

    if smogn_est is not None:
        smogn_est = smogn_est.sort_values(by=['density'])
        dens_windowed = np.convolve(
            smogn_est['density'], np.ones(window) / window, mode='valid'
        )
        metric_windowed = np.convolve(
            smogn_est[metric], np.ones(window) / window, mode='valid'
        )
        ax1.plot(
            dens_windowed, metric_windowed, label='SMOGN', color='#ff7f0e', alpha=0.5
        )
        ax2.plot(
            dens_windowed, metric_windowed, label='SMOGN', color='#ff7f0e', alpha=0.5
        )

    if smogn_dw_est is not None:
        smogn_dw_est = smogn_dw_est.sort_values(by=['density'])
        dens_windowed = np.convolve(
            smogn_dw_est['density'], np.ones(window) / window, mode='valid'
        )
        metric_windowed = np.convolve(
            smogn_dw_est[metric], np.ones(window) / window, mode='valid'
        )
        ax1.plot(dens_windowed, metric_windowed, label='SMOGN-DW', color='red')
        ax2.plot(dens_windowed, metric_windowed, label='SMOGN-DW', color='red')

    ax1.set_ylim(0.45, 2.3)
    ax2.set_ylim(0.01, 0.22)
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color='k',
        mec='k',
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.legend(fontsize='small')
    if len(filename_prefix) > 0:
        filename = (
            filename_prefix + '_density_to_' + metric + '_w' + str(window) + '.pdf'
        )
    else:
        filename = 'density_to_' + metric + '_w' + str(window) + '_diff.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')


def create_density_column(
    df: pd.DataFrame, target_relevance: TargetRelevance
) -> pd.DataFrame:
    df['density'] = df.apply(lambda row: target_relevance.get_density(row['y']), axis=1)
    return df


if __name__ == '__main__':

    # Change synth depending on which plot you want to plot
    synth = True

    if synth:
        base_path = expanduser(join('~', 'models', 'weighted-loss', 'synthetic'))
        path_suffix = ''

        dataset_names = ['normal', 'dnormal', 'pareto', 'rpareto']
        filename_prefix = 'synthetic'

    else:
        base_path = expanduser(join('~', 'models', 'weighted-loss', 'real'))
        path_suffix = 'units_10_10_10-decay_0_0001-bs_64-lr_0_0001-opt_Adam-bn_False'

        dataset_names = [
            'abalone',
            'concreteStrength',
            'delta_ailerons',
            'boston',
            'available_power',
            'servo',
            'bank8FM',
            'machineCpu',
            'airfoild',
            'a2',
            'a3',
            'a1',
            'cpu_small',
            'acceleration',
            'maximal_torque',
            'a4',
            'a5',
            'a7',
            'fuel_consumption_country',
            'a6',
        ]
        filename_prefix = 'real'

    datasets = [join(base_path, ds, path_suffix) for ds in dataset_names]
    smogn_datasets = [
        join(base_path, ds + '_smogn', path_suffix) for ds in dataset_names
    ]
    smogn_dw_datasets = [
        join(base_path, ds + '_smogn_dw', path_suffix) for ds in dataset_names
    ]

    # Initialize matplotlib
    init_mpl(usetex=True)

    print('Loading estimates...')
    test_estimates = load_estimates(
        save_paths=datasets + smogn_datasets + smogn_dw_datasets, split='test'
    )
    print('Estimates loaded')

    # Calculate the error of each individual data point in each DataFrame
    test_estimates = [
        (d, create_normalized_absolute_error_column(df)) for d, df in test_estimates
    ]

    # Calculate density per dataset
    test_estimates_tmp = []
    for ds in dataset_names:
        # Get some dataframe of that dataset
        df = [
            df
            for d, df in test_estimates
            if d['dataset'].replace('smogndw', '').replace('smogn', '')
            == ds.lower().replace('_', '')
        ][0]
        target_relevance = TargetRelevance(df['y'].to_numpy())
        test_estimates_tmp += [
            (d, create_density_column(df, target_relevance))
            for d, df in test_estimates
            # Unfortunately, there are some naming differences in the dict and in the list
            if d['dataset'].replace('smogndw', '').replace('smogn', '')
            == ds.lower().replace('_', '')
        ]
    test_estimates = test_estimates_tmp

    # Filter SMOGN runs
    smogn_estimates = [(d, df) for d, df in test_estimates if d['smogn'] is True]
    smogn_dw_estimates = [(d, df) for d, df in test_estimates if d['smogn_dw'] is True]
    test_estimates = [
        (d, df)
        for d, df in test_estimates
        if d['smogn'] is False and d['smogn_dw'] is False
    ]

    # Group estimates by alpha-value and dataset
    distinct_alpha_values = sorted(list(set([e[0]['alpha'] for e in test_estimates])))
    est_list_per_alpha_ds: Dict[float, Dict[str, List[pd.DataFrame]]] = {}
    for a in distinct_alpha_values:
        est_list_per_alpha_ds[a] = {}
        for ds in dataset_names:
            est_list_per_alpha_ds[a][ds] = []

    for est in test_estimates:
        for ds in dataset_names:
            if est[0]['dataset'] == ds.lower().replace('_', ''):
                est_list_per_alpha_ds[est[0]['alpha']][ds].append(est[1])

    est_per_alpha_ds: Dict[float, Dict[str, pd.DataFrame]] = {}
    for a in distinct_alpha_values:
        est_per_alpha_ds[a] = {}
        for ds in dataset_names:
            est_per_alpha_ds[a][ds] = merge_estimates(est_list_per_alpha_ds[a][ds])[
                ['y', 'density', 'estimate', 'normalized_absolute_error']
            ]

    est_per_alpha: Dict[float, pd.DataFrame] = {}
    for a in distinct_alpha_values:
        est_per_alpha[a] = pd.concat(est_per_alpha_ds[a].values(), axis=0)

    # Group smogn estimates by dataset
    smogn_est = None
    if len(smogn_estimates) > 0:
        est_list_smogn_ds: Dict[str, List[pd.DataFrame]] = {}
        for ds in dataset_names:
            est_list_smogn_ds[ds] = []
        for est in smogn_estimates:
            for ds in dataset_names:
                if est[0]['dataset'].replace('smogn', '') == ds.lower().replace(
                    '_', ''
                ):
                    est_list_smogn_ds[ds].append(est[1])

        est_smogn_ds: Dict[str, pd.DataFrame] = {}
        for ds in dataset_names:
            est_smogn_ds[ds] = merge_estimates(est_list_smogn_ds[ds])[
                ['y', 'density', 'estimate', 'normalized_absolute_error']
            ]

        smogn_est = pd.concat(est_smogn_ds.values(), axis=0)

    # Group smogn_dw estimates by dataset
    smogn_dw_est = None
    if len(smogn_dw_estimates) > 0:
        est_list_smogn_dw_ds: Dict[str, List[pd.DataFrame]] = {}
        for ds in dataset_names:
            est_list_smogn_dw_ds[ds] = []
        for est in smogn_dw_estimates:
            for ds in dataset_names:
                if est[0]['dataset'].replace('smogndw', '') == ds.lower().replace(
                    '_', ''
                ):
                    est_list_smogn_dw_ds[ds].append(est[1])

        est_smogn_dw_ds: Dict[str, pd.DataFrame] = {}
        for ds in dataset_names:
            est_smogn_dw_ds[ds] = merge_estimates(est_list_smogn_dw_ds[ds])[
                ['y', 'density', 'estimate', 'normalized_absolute_error']
            ]

        smogn_dw_est = pd.concat(est_smogn_dw_ds.values(), axis=0)

    print('data points:', len(est_per_alpha[list(est_per_alpha.keys())[0]]))

    # Plot each target value to the mean absolute error
    if synth:
        density_to_metric_plot(
            est_per_alpha,
            smogn_est=smogn_est,
            smogn_dw_est=smogn_dw_est,
            filename_prefix=filename_prefix,
        )
    else:
        density_to_metric_plot_split(
            est_per_alpha,
            smogn_est=smogn_est,
            smogn_dw_est=smogn_dw_est,
            filename_prefix=filename_prefix,
        )
