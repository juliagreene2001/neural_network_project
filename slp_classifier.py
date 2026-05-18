"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.
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
        Initialize the SLP classifier.

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

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None
        self.n_outputs_: Optional[int] = None

    def _forward_propagation(
        self, X: NDArray[np.floating]
    ) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation through hidden layers then softmax output.

        Returns (activations, pre_activations, y_pred) where activations[0] is X
        and y_pred is the softmax probability distribution.
        """
        activations, pre_activations = self._forward_hidden_layers(X)
        y_pred = softmax(activations[-1] @ self.weights_[-1] + self.biases_[-1])
        return activations, pre_activations, y_pred

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_outputs_ = n_classes
        y_onehot = np.eye(n_classes)[np.searchsorted(self.classes_, y)]

        self._initialize_weights(X.shape[1], n_classes)
        self._run_training_loop(X, y_onehot)

        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        _, _, proba = self._forward_propagation(X)
        return proba

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        y_pred = self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return y_pred

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        return float(np.mean(self.predict(X) == y))
