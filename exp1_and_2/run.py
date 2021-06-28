"""
Runs a model on a single node.
"""
import os
import shutil
import numpy as np

from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from generic_lightning import GenericModel

from utils import predict_all_from_dataloader

SEED = 42
np.random.seed(SEED)


def main(hparams: Namespace) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GenericModel(hparams)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.00,
        verbose=False,
        mode='min'
    )

    logger = TensorBoardLogger(os.path.join('checkpoints', hparams.experiment_name), name='lightning_logs')
    
    # I manually initialize the train dataloader so that the feature scaler is initialized.
    # At some point, pytorch lightning decided to first load the val_loader and then the train_loader,
    # which leads to the val_loader not having feature scaling
    model.train_dataloader()

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(#default_save_path=os.path.join('checkpoints', hparams.experiment_name),
                      callbacks=[early_stop_callback],
                      logger=logger,
                      #show_progress_bar=True,
                      max_epochs=100000)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

    # ------------------------
    # 4 TEST MODEL
    # ------------------------
    trainer.test()

    # Avoid shuffling train data so that the output csv is always ordered equally
    train_loader = model.train_dataloader(shuffle_override=False)
    val_loader = model.val_dataloader()
    test_loader = model.test_dataloader()

    for dl, mode in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
        out_data = predict_all_from_dataloader(model, dl)
        path = os.path.join('checkpoints', hparams.experiment_name, mode + '_estimates.csv')
        out_data.to_csv(path, index=False)

    # In order to avoid long, blocking io calls, we first write things to a memory-volume in kubernetes.
    # To persist things, I have to copy it to our distributed filesystem after training (and testing).
    if hparams.k8s:
        # From: https://stackoverflow.com/a/12514470
        for item in os.listdir('checkpoints'):
            s = os.path.join('checkpoints', item)
            d = os.path.join('checkpoints-persistent', item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument('experiment_name', metavar='N', type=str, help='Name of the current experiment.')
    parent_parser.add_argument('--k8s', action='store_true', help='Set flag if this runs in kubernetes.')

    # each LightningModule defines arguments relevant to it
    parser = GenericModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    print(hyperparams)
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
