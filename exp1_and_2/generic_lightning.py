"""
Template for DenseLoss experiments
"""
import os
from os.path import join, isdir
from collections import OrderedDict
from typing import Collection, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from argparse import ArgumentParser, Namespace
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl

import matplotlib as mpl

# # Make sure matplotlib works when running in kubernetes cluster without X server
# # See: https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
# if os.uname()[0].lower() != 'darwin':
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import num_features_for_dataset, read_split_data
from target_relevance import TargetRelevance

# from target_relevance_reciprocal import TargetRelevance


class GenericModel(pl.LightningModule):
    """
    Generic Model that works with SKLearn Datasets
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(GenericModel, self).__init__()
        self.hparams = hparams

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        # We cannot use lists for hparams because of the following bug:
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/609#issuecomment-568181791
        # I am sincerely sorry for the following code.
        units_per_layer = [
            self.hparams.layer_1,
            self.hparams.layer_2,
            self.hparams.layer_3,
            self.hparams.layer_4,
            self.hparams.layer_5,
        ]
        # Remove empty layers
        units_per_layer = [u for u in units_per_layer if u > 0]

        # build model
        self.__build_model(
            num_in_features=num_features_for_dataset(self.hparams.dataset),
            batch_normalization=self.hparams.batch_normalization,
            units_per_layer=units_per_layer,
        )

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(
        self,
        num_in_features: int,
        batch_normalization: bool = False,
        units_per_layer: Collection[int] = [10, 10, 10],
    ) -> None:
        """
        Layout model
        :return:
        """

        modules = []
        # in_size = self.hparams.in_features
        in_size = num_in_features
        for i, units in enumerate(units_per_layer):
            linear = nn.Linear(in_size, units)

            if batch_normalization:
                bn = nn.BatchNorm1d(units)  # BatchNorm makes training unstable

            act = ReLU()
            modules.append(('linear' + str(i), linear))

            if batch_normalization:
                modules.append(('bn' + str(i), bn))

            modules.append(('act' + str(i), act))
            in_size = units

        output = nn.Linear(in_size, 1)
        modules.append(('linear' + str(len(units_per_layer)), output))

        self.model = nn.Sequential(OrderedDict(modules))

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

    def loss(self, labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:

        if self.hparams.weighted_loss:

            try:
                relevance = torch.from_numpy(self.target_relevance(labels.numpy()))
            except AttributeError:
                print(
                    'WARNING: self.target_relevance does not exist yet! (This is normal once at the beginning when\
                       lightning tests things)'
                )
                relevance = torch.ones_like(labels)

            err = torch.pow(preds - labels, 2)
            err_weighted = relevance * err
            mse = err_weighted.mean()

            return mse

        else:
            loss = F.mse_loss(preds, labels)
            return loss

    def __common_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: torch.Tensor,
        mode: str,
    ) -> dict:
        x, y = batch

        out = self.forward(x)
        loss = self.loss(y, out)
        rmse = torch.sqrt(
            F.mse_loss(out, y)
        )  # y and out are switched in pytorch's mse_loss

        # Calculate some values for online-r2-score calculation
        squared_errors = torch.pow(out - y.view_as(out), 2)
        sum_of_squared_errors = torch.sum(squared_errors).item()
        squared_deviation_from_mean = torch.pow(y - self.y_mean[mode], 2)
        total_variance = torch.sum(squared_deviation_from_mean).item()

        if self.on_gpu:
            rmse = rmse.cuda(loss.device)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            rmse = rmse.unsqueeze(0)

        output = OrderedDict(
            {
                f'{mode}_loss': loss,
                f'{mode}_rmse': rmse,
                f'{mode}_sse': sum_of_squared_errors,
                f'{mode}_var': total_variance,
            }
        )

        return output

    def __common_end(self, outputs: dict, mode: str):
        loss_mean = 0.0
        rmse_mean = 0.0
        sse_sum = 0.0
        variance_sum = 0.0
        for output in outputs:
            loss = output[f'{mode}_loss']
            rmse = output[f'{mode}_rmse']
            sse = output[f'{mode}_sse']
            variance = output[f'{mode}_var']

            # reduce manually when using dp
            if self.trainer.use_dp:
                loss = torch.mean(loss)
            loss_mean += loss

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                rmse = torch.mean(rmse)
            rmse_mean += rmse

            sse_sum += sse
            variance_sum += variance

        loss_mean /= len(outputs)
        rmse_mean /= len(outputs)
        r2 = 1 - sse_sum / (variance_sum + 1e-8)
        self.log(f'{mode}_loss', loss_mean, prog_bar=True, logger=True)
        self.log(f'{mode}_rmse', rmse_mean, prog_bar=True, logger=True)
        self.log(f'{mode}_r2', r2, prog_bar=True, logger=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: torch.Tensor
    ) -> dict:
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        rmse = torch.sqrt(
            F.mse_loss(y_hat, y)
        )  # y and y_hat are switched in pytorch's mse_loss
        r2 = r2_score(y, y_hat.detach())

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            rmse = rmse.unsqueeze(0)

        self.log('train_rmse', rmse, prog_bar=True, logger=True)
        self.log('train_r2', r2, prog_bar=True, logger=True)
        output = OrderedDict(
            {
                'loss': loss_val,
            }
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: torch.Tensor
    ) -> dict:
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        return self.__common_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs: dict) -> dict:
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        return self.__common_end(outputs, 'val')

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: torch.Tensor
    ) -> dict:
        return self.__common_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs: dict) -> dict:
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        return self.__common_end(outputs, 'test')

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    # it can't find Optimizer so I ignore typing here
    def configure_optimizers(
        self,
    ) -> Tuple[
        List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]  # type: ignore
    ]:
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def __dataloader(
        self, mode: str = 'train', shuffle_override: Optional[bool] = None
    ) -> DataLoader:

        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid dataloader mode:', mode)

        if self.hparams.dataset in [
            'pareto',
            'rpareto',
            'normal',
            'dnormal',
            'pareto_smogn',
            'rpareto_smogn',
            'normal_smogn',
            'dnormal_smogn',
            'pareto_smogn_dw',
            'rpareto_smogn_dw',
            'normal_smogn_dw',
            'dnormal_smogn_dw',
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
            'abalone_smogn',
            'concreteStrength_smogn',
            'delta_ailerons_smogn',
            'boston_smogn',
            'available_power_smogn',
            'servo_smogn',
            'bank8FM_smogn',
            'machineCpu_smogn',
            'airfoild_smogn',
            'a2_smogn',
            'a3_smogn',
            'a1_smogn',
            'cpu_small_smogn',
            'acceleration_smogn',
            'maximal_torque_smogn',
            'a4_smogn',
            'a5_smogn',
            'a7_smogn',
            'fuel_consumption_country_smogn',
            'a6_smogn',
            'abalone_smogn_dw',
            'concreteStrength_smogn_dw',
            'delta_ailerons_smogn_dw',
            'boston_smogn_dw',
            'available_power_smogn_dw',
            'servo_smogn_dw',
            'bank8FM_smogn_dw',
            'machineCpu_smogn_dw',
            'airfoild_smogn_dw',
            'a2_smogn_dw',
            'a3_smogn_dw',
            'a1_smogn_dw',
            'cpu_small_smogn_dw',
            'acceleration_smogn_dw',
            'maximal_torque_smogn_dw',
            'a4_smogn_dw',
            'a5_smogn_dw',
            'a7_smogn_dw',
            'fuel_consumption_country_smogn_dw',
            'a6_smogn_dw',
        ]:

            train, val, test, y_col_name = read_split_data(self.hparams.dataset)

            self.X_train = train[
                [c for c in train.columns if c != y_col_name]
            ].to_numpy()
            self.X_val = val[[c for c in val.columns if c != y_col_name]].to_numpy()
            self.X_test = test[[c for c in test.columns if c != y_col_name]].to_numpy()
            self.y_train = train[y_col_name].to_numpy()
            self.y_val = val[y_col_name].to_numpy()
            self.y_test = test[y_col_name].to_numpy()
            self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], 1))
            self.y_val = np.reshape(self.y_val, (self.y_val.shape[0], 1))
            self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], 1))

        else:
            raise ValueError('Invalid dataset:', self.hparams.dataset)

        if mode == 'train':
            if self.X_train is None:
                # Split into train-validation-test if not split already, see: https://stackoverflow.com/a/38251213
                self.X_train, self.X_val, self.X_test = np.split(
                    X, [int(0.6 * len(X)), int(0.8 * len(X))]
                )
                self.y_train, self.y_val, self.y_test = np.split(
                    y, [int(0.6 * len(y)), int(0.8 * len(y))]
                )

            if self.hparams.weighted_loss:
                # Estimate the kernel density
                self.target_relevance = TargetRelevance(
                    self.y_train, alpha=self.hparams.alpha
                )

                # Plot weights
                sort_y_train = np.sort(self.y_train.flatten(), axis=0)
                y_dens = np.vectorize(self.target_relevance.get_density)(sort_y_train)
                plt.hist(sort_y_train, bins=20, density=True)
                plt.plot(sort_y_train, y_dens, 'r-', label='norm_density')
                plt.plot(
                    sort_y_train,
                    self.target_relevance(sort_y_train),
                    'g-',
                    label='weight',
                )
                plt.legend()
                plot_dir = join('checkpoints', self.hparams.experiment_name)
                if not isdir(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(join(plot_dir, 'weight_distribution.eps'))

            # Standardise the features
            self.scaler = StandardScaler().fit(self.X_train)

        # Also store y-means for R2-score calculation
        self.y_mean = {
            'train': np.mean(self.y_train),
            'val': np.mean(self.y_val),
            'test': np.mean(self.y_test),
        }

        if mode == 'train':
            X_to_use = self.X_train
            Y_to_use = self.y_train
        elif mode == 'val':
            X_to_use = self.X_val
            Y_to_use = self.y_val
        elif mode == 'test':
            X_to_use = self.X_test
            Y_to_use = self.y_test
        else:
            raise ValueError('Invalid dataloader mode:', mode)

        if hasattr(self, 'scaler'):
            X_to_use = self.scaler.transform(X_to_use)
        else:
            print(
                'WARNING: scaler is None! (This is normal once at the beginning since lightning creates a val dataloader at first for sanity checks)'
            )

        dataset = TensorDataset(
            torch.from_numpy(X_to_use).float().clone(),
            torch.from_numpy(Y_to_use).float().clone(),
        )

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size: int = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None and mode == 'train'
        if shuffle_override is not None:
            should_shuffle = shuffle_override
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0,
        )

        return loader

    def train_dataloader(self, shuffle_override: Optional[bool] = None) -> DataLoader:
        print('training data loader called')
        return self.__dataloader(mode='train', shuffle_override=shuffle_override)

    def val_dataloader(self) -> DataLoader:
        print('val data loader called')
        return self.__dataloader(mode='val')

    def test_dataloader(self) -> DataLoader:
        print('test data loader called')
        return self.__dataloader(mode='test')

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser, root_dir: str
    ) -> ArgumentParser:  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument(
            '-bn', '--batch_normalization', action='store_true', default=False
        )
        # We cannot use lists for hparams because of the following bug:
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/609#issuecomment-568181791
        # parser.add_argument('--units_per_layer', default=[10, 10, 10], nargs='+', type=int)
        parser.add_argument('-l1', '--layer-1', default=10, type=int)
        parser.add_argument('-l2', '--layer-2', default=10, type=int)
        parser.add_argument('-l3', '--layer-3', default=10, type=int)
        parser.add_argument('-l4', '--layer-4', default=0, type=int)
        parser.add_argument('-l5', '--layer-5', default=0, type=int)
        parser.add_argument('--learning_rate', default=0.0001, type=float)

        # data
        parser.add_argument('--dataset', default='pareto', type=str)

        # loss params
        parser.add_argument('--weighted_loss', action='store_true', default=False)
        parser.add_argument(
            '--alpha', default=0.0, type=float
        )  # weighting of the relevance function

        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--weight_decay', default=0.0001, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser
