# DenseLoss

This repository contains code for the paper "Density-based Weighting for Imbalanced Regression".

This folder specifically contains the code for the experiment with synthetic data and the experiment comparing DenseLoss to SMOGN.

The model is trained with the `run.py`. Each run creates a folder in which there is also a CSV-file with the model's test predictions. `evaluate.py` is used to compare models.
