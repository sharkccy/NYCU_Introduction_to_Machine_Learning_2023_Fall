# NYCU Introduction to Machine Learning 2023 Fall HW3 — Decision Tree & AdaBoost

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜)

## Introduction
This homework implements a binary decision tree classifier (with Gini/entropy impurities) and an AdaBoost ensemble of depth-1 decision stumps to predict the heart-disease target. The code builds the tree recursively, evaluates impurity-driven splits, tracks feature usage, and boosts weak learners to improve accuracy.

## What I implemented
- Impurity helpers (Gini, entropy) and weighted versions inside the decision tree; recursive tree construction with max-depth control in [110612117_HW3.py](110612117_HW3.py).
- Prediction via tree traversal plus feature-importance visualization that counts split usage per feature.
- AdaBoost with decision-stump weak learners (max_depth=1), sample-weight updates, custom alpha logic, and signed vote aggregation for final classification.
- Main script to compare criteria (Gini vs. entropy), print accuracies, plot feature importance, and evaluate AdaBoost against the test set.

## How to run
- Environment: Python 3.x with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
- Files: place `train.csv` and `test.csv` alongside the script.
- Execute: run `python 110612117_HW3.py` to train/evaluate decision trees (two criteria, depth=7) and AdaBoost, then show the feature-importance plot from a deeper tree (depth=15, Gini).
- Hyperparameters: random seed set to 0; AdaBoost uses `n_estimators=100` and Gini stumps; decision-tree depths tried at 7 and 15.

## Notes
- Tree impurity stops early when all labels match or max depth is reached; splits that do not reduce impurity fall back to a leaf with majority class.
- Feature-importance plot counts how often a feature is chosen for splits; columns assumed: `age`, `sex`, `cp`, `fbs`, `thalach`, `thal`.
- AdaBoost reweights samples each round; extreme alphas are clipped (0 when error >= 1, large when error <= 0.25) to keep training stable.
- See `110612117_HW3.pdf` for detailed report, results, and plots.
