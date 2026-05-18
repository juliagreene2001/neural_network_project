"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.
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
        Initialize the SLP regressor.

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
        super().__init__(
            hidden_layer_sizes, activation, optimizer, learning_rate, max_iter,
            tol, n_iter_no_change, adam_beta1, adam_beta2, adam_epsilon, random_state,
        )

    def _forward_propagation(
        self, X: NDArray[np.floating]
    ) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation through hidden layers then linear output.

        Returns (activations, pre_activations, y_pred) where activations[0] is X
        and y_pred is the raw linear output for regression.
        """
        activations, pre_activations = self._forward_hidden_layers(X)
        y_pred = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        return activations, pre_activations, y_pred

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute mean squared error loss.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        return float(np.mean((y_pred - y_true) ** 2))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        y = np.atleast_2d(y).T if y.ndim == 1 else y
        n_outputs = y.shape[1]

        self._initialize_weights(X.shape[1], n_outputs)
        self._run_training_loop(X, y)

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        _, _, y_pred = self._forward_propagation(X)
        return y_pred.squeeze()

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
