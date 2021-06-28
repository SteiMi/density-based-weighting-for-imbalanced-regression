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
    plt.hist(modified_df[y_col_name], bins=10, alpha=0.5, density=density, label='SMOGN-DW', color='#ff7f0e')
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
    filename = name + '_smogn_dw_train_hist' + dens_str + mod_only_str + '.pdf'
    plt.savefig(join('plots', filename), format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == "__main__":

    # This script does currently NOT always produce the same data.
    # For some reason, smogn's synthetic y-values still change, despite setting seeds.
    random.seed(8)
    np.random.seed(8)
    init_mpl()

    # data = load_data()
    data = load_data(names=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'abalone', 'acceleration', 'airfoild',
                            'available_power', 'bank8FM', 'boston', 'concreteStrength', 'cpu_small', 'delta_ailerons',
                            'fuel_consumption_country', 'machineCpu', 'maximal_torque', 'servo'],
                     base_path=join('data', 'real'))

    configs: Dict[str, Dict[str, Any]] = {
        # synthetic
        'pareto': {
            'y': 'y',
        },
        'rpareto': {
            'y': 'y',
        },
        'normal': {
            'y': 'y',
        },
        'dnormal': {
            'y': 'y',
        },
        # real
        # Note: We also always use the extreme type both for these datasets, as the original SMOGN authors did
        'a1': {
            'y': 'a1',
        },
        'a2': {
            'y': 'a2',
        },
        'a3': {
            'y': 'a3',
        },
        'a4': {
            'y': 'a4',
        },
        'a5': {
            'y': 'a5',
        },
        'a6': {
            'y': 'a6',
        },
        'a7': {
            'y': 'a7',
        },
        'abalone': {
            'y': 'Rings',
        },
        'acceleration': {
            'y': 'acceleration',
        },
        'available_power': {
            'y': 'available.power',
        },
        'bank8FM': {
            'y': 'rej',
        },
        'cpu_small': {
            'y': 'usr',
        },
        'fuel_consumption_country': {
            'y': 'fuel.consumption.country',
        },
        'boston': {
            'y': 'HousValue',
        },
        'maximal_torque': {
            'y': 'maximal.torque',
        },
        'machineCpu': {
            'y': 'class',
        },
        'servo': {
            'y': 'class',
        },
        'airfoild': {
            'y': 'ScaledSoundPressure',
        },
        'concreteStrength': {
            'y': 'ConcreteCompressiveStrength',
        },
        'delta_ailerons': {
            'y': 'Sa',
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
                                   rel_method='denseweight',
                                   rel_alpha=1.0)

        if name in ['pareto', 'rpareto', 'normal', 'dnormal']:
            base_path = join('data', 'synthetic')
        else:
            base_path = join('data', 'real')

        # modified_df = pd.read_csv(join(base_path, '%s_smogn_dw_train.csv' % name))

        print(name)
        plot(df, modified_df, name, density=False, modified_only=False, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=True, modified_only=False, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=False, modified_only=True, y_col_name=configs[name]['y'])
        plot(df, modified_df, name, density=True, modified_only=True, y_col_name=configs[name]['y'])

        modified_df.to_csv(join(base_path, '%s_smogn_dw_train.csv' % name), index=False)
