# NYCU Introduction to Machine Learning 2023 Fall HW2 — Logistic Regression & Fisher's Linear Discriminant

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜)

## Introduction
This homework builds two binary classifiers from scratch on the provided heart-disease dataset: (1) logistic regression trained with gradient descent over all features, and (2) Fisher's Linear Discriminant (FLD) using two selected features to learn a 1D projection and classify by nearest projected mean. Evaluation relies on accuracy, with matplotlib visualizations for the FLD projection line.

## What I implemented
- Logistic regression optimizer with sigmoid activation, gradient-based weight/bias updates, and 0.5 threshold prediction in [110612117_HW2.py](110612117_HW2.py).
- Fisher's Linear Discriminant: compute class means, within-/between-class scatter matrices, dominant eigenvector, projection slope/intercept, and nearest-mean decision rule.
- Projection visualization for FLD that plots the learned line, projected test points, and class-colored scatter for qualitative inspection.
- Accuracy reporting and assertions to ensure performance meets the required thresholds for both methods.

## How to run
- Environment: Python 3.x with `numpy`, `pandas`, `matplotlib`, `scikit-learn` (for `accuracy_score`).
- Steps:
  - Place `train.csv` and `test.csv` in the same folder as the script.
  - Run `python 110612117_HW2.py` to train/evaluate both models; the script prints weights/intercept and accuracy for logistic regression, then shows the FLD projection plot and accuracy.
- Hyperparameters: logistic regression uses `learning_rate=0.00000205`, `iteration=100000`
- FLD trains on features `age` and `thalach` only.

## Notes
- Logistic regression starts from weight=1 and bias=0, so results are deterministic; learning rate strongly affects convergence speed and stability.
- FLD accuracy is bounded by the two-feature subspace; adding features would require a different discriminant approach.
- The projection plot is required for the report; close the matplotlib window to let the script finish if needed.
- See `110612117_HW2`.pdf for the detailed report, results, and visualization screenshots.
