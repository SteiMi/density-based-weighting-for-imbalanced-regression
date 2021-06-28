from os.path import join
from typing import Any, Dict, List
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import smogn

from utils import init_mpl, set_size


def load_data(names: List[str] = ['pareto', 'rpareto', 'normal', 'dnormal'],
              base_path: str = join('data', 'synthetic')) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    for name in names:
        filename = join(base_path, '%s_train.csv' % name)
        data = pd.read_csv(filename)
        data_dict[name] = data

    return data_dict


def plot(original: pd.DataFrame, modified: pd.DataFrame, name: str, density: bool = False,
         modified_only: bool = False, y_col_name: str = 'y'):
    plt.figure(figsize=set_size(fraction=0.4))
    plt.hist(modified_df[y_col_name], bins=10, alpha=0.5, density=density, label='SMOGN', color='#ff7f0e')
    if not modified_only:
        plt.hist(df[y_col_name], bins=10, alpha=0.5, density=density, label='Original', color='#1f77b4')
        plt.legend()

    # Some hacky optimizations for the paper
    if any([n in name for n in ['normal', 'pareto']]):
        plt.xlim([-35, 80])

    if 'normal' in name:
        plt.ylim([0, 0.027])
    elif 'pareto' in name:
        plt.ylim([0, 0.07])

    if density:
        plt.ylabel('$p(%s)$' % y_col_name)
    else:
        plt.ylabel('Frequency')

    plt.xlabel('$%s$' % y_col_name)
    dens_str = '_density' if density else ''
    mod_only_str = '_modonly' if modified_only else ''
    filename = name + '_smogn_train_hist' + dens_str + mod_only_str + '.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == "__main__":

    # This script does currently NOT always produce the same data.
    # For some reason, smogn's synthetic y-values still change, despite setting seeds.
    random.seed(8)
    np.random.seed(8)
    init_mpl()

    data = load_data()
    # data = load_data(names=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'abalone', 'acceleration', 'airfoild',
    #                         'available_power', 'bank8FM', 'boston', 'concreteStrength', 'cpu_small', 'delta_ailerons',
    #                         'fuel_consumption_country', 'machineCpu', 'maximal_torque', 'servo'],
    #                  base_path=join('data', 'real'))

    configs: Dict[str, Dict[str, Any]] = {
        # synthetic
        'pareto': {
            'y': 'y',
            'rel_method': 'auto',
            'rel_xtrm_type': 'high',
            'rel_ctrl_pts_rg': None
        },
        'rpareto': {
            'y': 'y',
            'rel_method': 'auto',
            'rel_xtrm_type': 'low',
            'rel_ctrl_pts_rg': None
        },
        'normal': {
            'y': 'y',
            'rel_xtrm_type': 'both',  # Doesn't matter in manual mode anyway
            'rel_method': 'manual',
            'rel_ctrl_pts_rg': [
                [-10, 1, 0],
                [20, 0, 0],
                [50, 1, 0]
            ]
        },
        'dnormal': {
            'y': 'y',
            'rel_method': 'manual',
            'rel_xtrm_type': 'both',  # Doesn't matter in manual mode anyway
            'rel_ctrl_pts_rg': [
                [0, 0, 0],
                [20, 1, 0],
                [50, 0, 0]
            ]
        },
        # real
        # Note: We also always use the extreme type both for these datasets, as the original SMOGN authors did
        'a1': {
            'y': 'a1',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a2': {
            'y': 'a2',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a3': {
            'y': 'a3',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a4': {
            'y': 'a4',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a5': {
            'y': 'a5',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a6': {
            'y': 'a6',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'a7': {
            'y': 'a7',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'abalone': {
            'y': 'Rings',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'acceleration': {
            'y': 'acceleration',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'available_power': {
            'y': 'available.power',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'bank8FM': {
            'y': 'rej',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'cpu_small': {
            'y': 'usr',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'fuel_consumption_country': {
            'y': 'fuel.consumption.country',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'boston': {
            'y': 'HousValue',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'maximal_torque': {
            'y': 'maximal.torque',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'machineCpu': {
            'y': 'class',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'servo': {
            'y': 'class',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'airfoild': {
            'y': 'ScaledSoundPressure',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'concreteStrength': {
            'y': 'ConcreteCompressiveStrength',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        },
        'delta_ailerons': {
            'y': 'Sa',
            'rel_method': 'auto',
            'rel_xtrm_type': 'both',
            'rel_ctrl_pts_rg': None
        }
    }

    for name, df in data.items():
        modified_df = smogn.smoter(data=df,
                                   y=configs[name]['y'],
                                   k=5,
                                   # pert=0.01 seems to be what Branco used, see:
                                   # https://github.com/paobranco/SMOGN-LIDTA17/blob/master/R_Code/ExpsDIBS.R#L12
                                   pert=0.01,
                                   samp_method='balance',
                                   replace=False,
                                   rel_thres=0.8,
                                   rel_method=configs[name]['rel_method'],
                                   rel_ctrl_pts_rg=configs[name]['rel_ctrl_pts_rg'])

        if name in ['pareto', 'rpareto', 'normal', 'dnormal']:
            base_path = join('data', 'synthetic')
        else:
            base_path = join('data', 'real')

        # modified_df = pd.read_csv(join(base_path, '%s_smogn_train.csv' % name))

        print(name)
        plot(df, modified_df, name, density=False, modified_only=False, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=True, modified_only=False, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=False, modified_only=True, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=True, modified_only=True, y_col_name=configs[name]['y'])

        modified_df.to_csv(join(base_path, '%s_smogn_train.csv' % name), index=False)
