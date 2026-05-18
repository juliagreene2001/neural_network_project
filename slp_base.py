"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the SLP estimator.

        Parameters:
        -----------
        hidden_layer_sizes : tuple of int
            Number of neurons in each hidden layer, e.g. (100,) for one hidden layer
        activation : str
            Activation function ('logistic', 'tanh', 'relu'), default='relu'
        optimizer : str
            Optimization algorithm ('sgd' or 'adam'), default='adam'
        learning_rate : float
            Learning rate (step size) for the optimizer
        max_iter : int
            Maximum number of iterations
        tol : float
            Minimum loss improvement to count as progress (early stopping)
        n_iter_no_change : int
            Iterations without improvement before stopping early
        adam_beta1 : float
            Adam exponential decay rate for the first moment estimate, default=0.9
        adam_beta2 : float
            Adam exponential decay rate for the second moment estimate, default=0.999
        adam_epsilon : float
            Adam numerical stability constant, default=1e-8
        random_state : int or None
            Random seed for reproducibility
        """
        if (
            not isinstance(hidden_layer_sizes, tuple)
            or len(hidden_layer_sizes) == 0
            or not all(isinstance(n, int) and n > 0 for n in hidden_layer_sizes)
        ):
            raise ValueError(
                "hidden_layer_sizes must be a non-empty tuple of positive integers, "
                f"got {hidden_layer_sizes!r}"
            )
        if activation not in {"relu", "tanh", "logistic"}:
            raise ValueError(
                f"activation must be 'relu', 'tanh', or 'logistic', got {activation!r}"
            )
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if n_iter_no_change <= 0:
            raise ValueError(f"n_iter_no_change must be > 0, got {n_iter_no_change}")

        self.hidden_layer_sizes: tuple[int, ...] = hidden_layer_sizes
        self.activation: str = activation
        self.optimizer: str = optimizer
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.n_iter_no_change: int = n_iter_no_change
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon: float = adam_epsilon
        self.random_state: Optional[int] = random_state

        # To be initialized in fit()
        self.weights_: list[NDArray[np.floating]] = []  # weights_[i]: layer i -> layer i+1
        self.biases_: list[NDArray[np.floating]] = []   # biases_[i]: bias at layer i+1
        self.loss_curve_: list[float] = []
        self.n_iter_: int = 0

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(self, n_features: int, n_outputs: int) -> None:
        """
        Initialize weights and biases using He (Kaiming) initialization.
        Variance = 2 / fan_in, which is optimal for ReLU activations.
        Builds one weight matrix per layer transition: input -> hidden_1 -> ... -> output.
        """
        self.weights_ = []
        self.biases_ = []
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights_.append(
                np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            )
            self.biases_.append(np.zeros(fan_out))

    def _forward_hidden_layers(
        self, X: NDArray[np.floating]
    ) -> Tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
        """
        Run forward propagation through all hidden layers.

        Returns activations (including X as activations[0]) and pre-activations
        for every hidden layer. The output layer is handled by each subclass.
        """
        activation_fn, _ = self._get_activation_function()
        activations: list[NDArray[np.floating]] = [X]
        pre_activations: list[NDArray[np.floating]] = []

        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            activations.append(activation_fn(z))

        return activations, pre_activations

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        activations: list[NDArray[np.floating]],
        pre_activations: list[NDArray[np.floating]],
        y_pred: NDArray[np.floating],
    ) -> Tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
        """
        Backpropagate gradients through all layers.

        The output delta (y_pred - y) is valid for both softmax+cross-entropy
        and linear+MSE because both simplify to the same expression.

        Returns dW_list and db_list in forward order (index 0 = input layer weights).
        """
        _, activation_derivative = self._get_activation_function()
        n_samples = X.shape[0]

        dW_list: list[NDArray[np.floating]] = [None] * len(self.weights_)  # type: ignore[list-item]
        db_list: list[NDArray[np.floating]] = [None] * len(self.biases_)   # type: ignore[list-item]

        delta = y_pred - y

        for i in reversed(range(len(self.weights_))):
            dW_list[i] = activations[i].T @ delta / n_samples
            db_list[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * activation_derivative(pre_activations[i - 1])

        return dW_list, db_list

    def _run_training_loop(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> None:
        """
        Run the optimizer training loop with early stopping.

        Supports 'sgd' (plain gradient descent) and 'adam' (adaptive moment
        estimation). Populates loss_curve_ and n_iter_. Called by each subclass
        fit() after weight initialisation and y preprocessing are complete.
        """
        self.loss_curve_ = []
        best_loss = np.inf
        no_improve_count = 0

        # Adam moment accumulators — zero-initialised, same shape as each weight/bias
        if self.optimizer == "adam":
            mW = [np.zeros_like(W) for W in self.weights_]
            vW = [np.zeros_like(W) for W in self.weights_]
            mb = [np.zeros_like(b) for b in self.biases_]
            vb = [np.zeros_like(b) for b in self.biases_]

        for i in range(self.max_iter):
            activations, pre_activations, y_pred = self._forward_propagation(X)
            loss = self._compute_loss(y, y_pred)
            self.loss_curve_.append(loss)

            if best_loss - loss > self.tol:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.n_iter_no_change:
                    self.n_iter_ = i + 1
                    return

            dW_list, db_list = self._backward_propagation(X, y, activations, pre_activations, y_pred)

            if self.optimizer == "adam":
                t = i + 1  # 1-indexed timestep for bias correction
                b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_epsilon
                for j in range(len(self.weights_)):
                    # First moment (mean) update
                    mW[j] = b1 * mW[j] + (1 - b1) * dW_list[j]
                    mb[j] = b1 * mb[j] + (1 - b1) * db_list[j]
                    # Second moment (uncentered variance) update
                    vW[j] = b2 * vW[j] + (1 - b2) * dW_list[j] ** 2
                    vb[j] = b2 * vb[j] + (1 - b2) * db_list[j] ** 2
                    # Bias-corrected estimates
                    mW_hat = mW[j] / (1 - b1 ** t)
                    mb_hat = mb[j] / (1 - b1 ** t)
                    vW_hat = vW[j] / (1 - b2 ** t)
                    vb_hat = vb[j] / (1 - b2 ** t)
                    # Parameter update
                    self.weights_[j] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.biases_[j] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
            else:
                for j in range(len(self.weights_)):
                    self.weights_[j] -= self.learning_rate * dW_list[j]
                    self.biases_[j] -= self.learning_rate * db_list[j]

        self.n_iter_ = self.max_iter

    @abstractmethod
    def _forward_propagation(
        self, X: NDArray[np.floating]
    ) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation through all layers.

        Returns (activations, pre_activations, y_pred) where activations includes
        X as activations[0] and pre_activations covers each hidden layer.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
