# NYCU Introduction to Machine Learning 2023 Fall HW4 — SVM Kernels

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜)

## Introduction
This homework implements support vector machines with custom linear, polynomial, and RBF kernels. Kernel functions are written from scratch, assembled into Gram matrices, and fed to scikit-learn's `SVC` in precomputed mode. Experiments compare kernel choices and hyperparameters on the heart-disease dataset.

## What I implemented
- Hand-coded kernels and Gram matrix builder for linear, polynomial, and RBF in [110612117_HW4.py](110612117_HW4.py).
- Standardization of features before Gram construction; shared pipeline for train/test.
- Three SVC runs using precomputed kernels, each tunable via `C_` plus global `degree_` (polynomial) and `gamma_` (RBF).
- Accuracy reporting with an assertion on the linear-kernel model (must exceed 0.8).

## How to run
- Environment: Python 3.x with `numpy`, `pandas`, `scikit-learn`.
- Files: place `train.csv` and `test.csv` alongside the script.
- Execute: run `python 110612117_HW4.py` to standardize data, compute Gram matrices, train three SVMs, and print accuracies.
- hyperparameters (editable in the script): `degree_` (polynomial degree), `gamma_` (RBF width), and `C_` for each SVC call (linear/polynomial/RBF).

## Notes
- Gram matrices are built explicitly; dataset size should remain modest to avoid memory issues.
- Standardization is applied separately to train and test in the current script; for stricter consistency you could fit on train then reuse for test.
- Precomputed kernels rely on matching row order between train and test Gram matrices; do not shuffle between fit and predict.
- See `110612117_HW4.pdf` for the detailed report and results.
