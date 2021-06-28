from os import listdir
from os.path import join
import sys
from typing import List

import numpy as np
import pandas as pd

from utils import init_mpl, split_data


def explore_splits(filenames: List[str],
                   base_path: str = join('data', 'real'),
                   ignored_datasets: List[str] = ['abalone', 'concreteStrength', 'delta_ailerons', 'boston',
                                                  'available_power', 'servo', 'bank8FM', 'machineCpu', 'airfoild',
                                                  'a2', 'a3', 'a1', 'cpu_small', 'acceleration', 'maximal_torque',
                                                  'a4', 'a5', 'a7', 'fuel_consumption_country', 'a6']):
    """
    Interactive helper function for choosing suitable splits.
    """
    selected_seeds = {}
    for f in filenames:
        dataset_name: str = f.split('.')[0]

        if dataset_name in ignored_datasets:
            continue

        data = pd.read_csv(join(base_path, f))

        # The first column is always the target for these datasets
        y_col_name = data.columns[0]

        print('Dataset:', dataset_name, 'Target Variable:', y_col_name)

        for seed in range(9999):
            print('Seed:', seed)
            split_data(data, name=dataset_name, save=False, base_save_path=base_path, show_dist=True,
                       y_col_name=y_col_name, random_state=seed)
            command = input('Try [n]ext seed, [c]ontinue to next dataset, [q] to quit: ')

            if command == 'c':
                selected_seeds[dataset_name] = seed
                break

            if command == 'q':
                print(selected_seeds)
                sys.exit()

    print(selected_seeds)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    # Init matplotlib
    init_mpl()

    base_path = join('data', 'real')
    base_filenames = [f for f in listdir(base_path) if
                      f.split('.')[-1] == 'csv' and
                      '_train' not in f and
                      '_val' not in f and
                      '_test' not in f]

    # explore_splits(base_filenames)

    for f in base_filenames:
        dataset_name: str = f.split('.')[0]

        data = pd.read_csv(join(base_path, f))

        # One-hot-encode categorical features
        data = pd.get_dummies(data)

        # The first column is always the target for these datasets
        y_col_name = data.columns[0]

        split_data(data, name=dataset_name, save=True, base_save_path=base_path, show_dist=True,
                   y_col_name=y_col_name)
