# Single Layer Perceptron — Classification & Regression

A from-scratch NumPy implementation of a Single Layer Perceptron (SLP) neural network for both classification and regression, following a scikit-learn-compatible interface.

---

## Project Overview

This project implements two neural network estimators — `SimpleSLPClassifier` and `SimpleSLPRegressor` — without using any deep learning frameworks. The goal is to demonstrate a full understanding of the mechanics behind neural networks: forward propagation, backpropagation, gradient descent, and adaptive optimization.

Both estimators support:

- Configurable hidden layer sizes (arbitrary depth)
- Multiple activation functions: ReLU, tanh, logistic
- Two optimizers: full-batch SGD and Adam (adaptive moment estimation)
- Early stopping with configurable tolerance and patience
- A scikit-learn-compatible interface (`fit`, `predict`, `score`, `predict_proba`)

Performance is benchmarked against scikit-learn's `MLPClassifier` and `MLPRegressor` across multiple datasets.

---

## Installation and Setup

### Codes and Resources Used

- **Python version:** 3.13
- **Editor:** VS Code

### Python Packages Used

#### General Purpose

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn` — datasets, preprocessing, benchmarking, and metrics

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the test suite with:

```bash
pytest tests/ -v
```

---

## Data

All datasets are loaded directly from scikit-learn — no external downloads required.

### Source Data

| Dataset | Task | Samples | Features | Classes/Outputs |
|---|---|---|---|---|
| `load_diabetes` | Regression | 442 | 10 | 1 (continuous) |
| `fetch_california_housing` | Regression | 20,640 | 8 | 1 (continuous) |
| `load_breast_cancer` | Binary classification | 569 | 30 | 2 |
| `load_digits` | Multiclass classification | 1,797 | 64 | 10 |

### Data Acquisition

All datasets are accessed via `sklearn.datasets`. No API calls or web scraping required.

### Data Preprocessing

- **Features:** standardised with `StandardScaler` (zero mean, unit variance) before fitting
- **Regression targets:** standardised with a second `StandardScaler` on the target column, then inverse-transformed for reporting metrics in original units
- **Classification labels:** integer-encoded; one-hot encoding is applied internally inside `fit()` using `np.eye`

---

## Code Structure

```
neural_network_project-main/
├── activations.py          # Activation functions and their derivatives
├── slp_base.py             # Abstract base class: shared forward/backward/training logic
├── slp_classifier.py       # SimpleSLPClassifier (softmax output, cross-entropy loss)
├── slp_regressor.py        # SimpleSLPRegressor (linear output, MSE loss)
├── report.ipynb            # Full demonstration notebook with plots and metrics
├── tests/
│   ├── test_activations.py # Unit tests for all activation functions
│   ├── test_models.py      # Correctness, learning behaviour, and integration tests
│   └── test_smoke.py       # End-to-end smoke tests
└── neural_network_project.md  # Project specification
```

Key design decisions:

- `BaseSLPEstimator` uses the Template Method pattern — `_run_training_loop`, `_forward_hidden_layers`, and `_backward_propagation` are concrete shared helpers; `_forward_propagation`, `_compute_loss`, `fit`, `predict`, and `score` are abstract and implemented by each subclass.
- Weights are stored as `list[NDArray]` to support arbitrary depth (`hidden_layer_sizes` is a tuple of ints).
- Backward propagation uses the same `delta = y_pred - y` expression for both tasks because softmax + cross-entropy and linear + MSE both simplify to this gradient at the output layer.
- He (Kaiming) initialisation: `W ~ N(0, sqrt(2 / fan_in))` — optimal for ReLU activations.

---

## Results and Evaluation

Full results are in [`report.ipynb`](report.ipynb). All models are trained with `hidden_layer_sizes=(64,)`, `activation='relu'`, `optimizer='adam'`, `learning_rate=0.01`, `max_iter=500` and benchmarked against sklearn's MLP implementations with matching hyperparameters.

### Regression

Both the Diabetes and California Housing datasets are evaluated with training loss curves, predicted vs. actual scatter plots (with a perfect-prediction line and line of best fit), an error metrics table reporting MSE, RMSE, MAE, and R² in original target units, residual scatter plots, residual distributions, and learning curves.

### Classification

The Breast Cancer (binary) and Digits (10-class) datasets are evaluated with training loss curves, confusion matrices, per-class classification reports (precision, recall, F1), log loss, confidence distribution histograms comparing correct vs. misclassified samples, misclassification tables showing the true label, predicted label, and model confidence for each error, ROC curves with AUC, precision-recall curves with average precision, calibration curves, and learning curves.

### Evaluation Methodology

- Regression: R² (coefficient of determination), MSE/RMSE/MAE in original target units, residual scatter plots, residual distributions, and learning curves (Train R² vs Test R² across training set sizes)
- Classification: accuracy, precision, recall, F1-score (per class and macro-averaged), log loss, confusion matrix, ROC/AUC, precision-recall curves, calibration curves, and learning curves (Train vs Test accuracy across training set sizes)
- All results use an 80/20 train/test split with `random_state=42`
