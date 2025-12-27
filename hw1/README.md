# NYCU Introduction to Machine Learning 2023 Fall HW1 — Linear Regression

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜)

## Introduction
This homework implements linear regression to predict the "Performance Index" from tabular features. Both the closed-form solution (normal equation) and iterative gradient descent are built from scratch with only NumPy, Pandas, and Matplotlib.

## What I implemented
- Closed-form solver with bias augmentation and weight/intercept extraction in [110612117_HW1.py](110612117_HW1.py).
- Gradient descent trainer with random initialization, MSE loss, and per-epoch weight/bias updates; training losses collected for plotting.
- Prediction and evaluation helpers for both solutions using MSE for consistency across train/test splits.
- Matplotlib learning-curve visualization to show convergence of the gradient descent run.

## How to run
- Environment: Python 3.x with `numpy`, `pandas`, `matplotlib` (no extra packages).
- Steps:
  - Place `train.csv` and `test.csv` in the same folder as the script.
  - Run `python 110612117_HW1.py` to train both solvers, print weights/intercepts, report relative error, and display the learning curve.
- Key training hyperparameters are set in `LR.gradient_descent_fit` call (current: `lr=0.00019273`, `epochs=400000`).

## Notes
- Closed-form and gradient-descent weights should be close; residual differences shrink with a smaller learning rate or more epochs.
- Learning-curve screenshot from `plot_learning_curve` is required for the report; the script displays it after evaluation.
- Gradient descent starts from a random initialization, so minor run-to-run variation in loss is expected.
- See `110612117_HW1.pdf` for detailed analysis, results, and the learning curve screenshot.
