# ML
Progetto Machine Learning 2023-2024

# Abstract
Our project consists in the implementation of a Multi-Layer Perceptron, trained with Gradient Descent and Backpropagation. It was validated through a set of hyperparameter Grid Searches which made use of 5-fold Cross-Validation over a subset of the total data. In particular, The original dataset was split into 80% for training, and 20% as an internal test set for the final assessment. The 10% of the training split was used as a validation set for Early Stopping tuning. The obtained final model was then retrained on the Whole 80% split and evaluated on the internal test set. 