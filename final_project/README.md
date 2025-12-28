# NYCU Introduction to Machine Learning 2023 Fall — Final Project (Bird Image Classification)

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜)

## Introduction
This final project trains an fine-grained image classifier for bird species using fastai and a pretrained ResNet-50 backbone. The pipeline builds fastai `DataLoaders`, applies augmentation/normalization, fine-tunes the network, and exports an inference-ready learner to generate competition-style predictions.

## What I implemented
- Fastai data pipeline with folder-based labels, 20% validation split (seed=50), resize to 256 then augmented to 224, and ImageNet normalization in [final_project/training/main.py](final_project/training/main.py).
- ResNet-50 transfer learning with dropout `ps=0.5`, weight decay `wd=0.3`, early stopping (patience=5), and `fine_tune(epochs=5, freeze_epochs=5)` for staged training; custom recorder patch to plot losses/metrics.
- Post-training analysis: learning-rate vs loss plot and confusion inspection of top misclassified classes.
- Model export to [final_project/training/model/model.pkl](final_project/training/model/model.pkl) and an inference script that loads the learner, runs on the test folder, and writes `predictions.csv` in [final_project/110612117_inference.py](final_project/110612117_inference.py).

## How to run
- Environment: Python 3.x with `torch==2.0.0+cu117`, `torchvision==0.15.0+cu117`, `fastai` (see [final_project/requirements.txt](final_project/requirements.txt)). GPU is recommended.
- Data layout: place training images under `training/data/train/<class>/...` and test images under `training/data/test/`.
- Training: `python training/main.py` (uses `bs=128`, `num_workers=0` for Windows; prints CUDA info if available). Outputs an exported model to `training/model/model.pkl`.
- Inference: ensure `training/model/model.pkl` exists (download link in [final_project/110612117_weights.txt](final_project/110612117_weights.txt) if needed), then run `python 110612117_inference.py` to produce `predictions.csv` beside the script.

## Notes
- Augmentation uses `aug_transforms(size=224)` plus ImageNet stats; adjust `bs` or image size if memory is tight.
- Early stopping keeps training stable; you can lengthen `epochs` or tweak `wd`/dropout for different trade-offs.
- Keep train/test directory ordering unchanged when exporting and during inference so class-to-index mapping stays consistent.
- For reproducibility, the seed is fixed at 50 for the data split; random weight initialization follows fastai defaults.
- See `110612117_report.pdf` for the detailed report, analysis, and plots.