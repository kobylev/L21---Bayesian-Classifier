"""
Gaussian Naive Bayes - Manual NumPy Implementation

Implements Gaussian Naive Bayes from scratch using NumPy.
Mathematical Foundation: Bayes' Theorem and Gaussian PDF.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class GaussianNaiveBayesNumPy:
    """Gaussian Naive Bayes classifier from scratch with variance smoothing."""

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.

        Args:
            var_smoothing: Portion of largest variance added to all variances
                          for numerical stability (prevents division by zero)
        """
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None
        self.epsilon_: float = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayesNumPy':
        """
        Train the model by calculating priors, means, and variances.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If X and y have incompatible shapes or contain invalid values
            TypeError: If inputs are not numpy arrays

        Example:
            >>> model = GaussianNaiveBayesNumPy()
            >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y_train = np.array([0, 1, 0])
            >>> model.fit(X_train, y_train)
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        if X.shape[0] == 0:
            raise ValueError("Cannot fit model with 0 samples")

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")

        logger.info("Training Gaussian Naive Bayes (NumPy implementation)")
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        logger.info(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.class_priors_ = np.zeros(n_classes)

        for idx, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]
            self.class_priors_[idx] = X_class.shape[0] / n_samples
            self.means_[idx, :] = X_class.mean(axis=0)
            self.variances_[idx, :] = X_class.var(axis=0)
            logger.debug(f"  Class {class_label}: prior={self.class_priors_[idx]:.4f}")

        logger.info("Training completed successfully")
        return self

    def _calculate_gaussian_pdf(self, x: float, mean: float, variance: float) -> float:
        """Calculate Gaussian PDF: (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))"""
        variance = variance + self.epsilon_
        coefficient = 1.0 / np.sqrt(2 * np.pi * variance)
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return coefficient * exponent

    def _calculate_log_posterior(self, x: np.ndarray) -> np.ndarray:
        """Calculate log posterior: log P(y|X) = log P(y) + Σ log P(x_i|y)"""
        log_posteriors = []
        for idx in range(len(self.classes_)):
            log_prior = np.log(self.class_priors_[idx] + self.epsilon_)
            log_likelihood = 0.0
            for feat_idx in range(len(x)):
                pdf = self._calculate_gaussian_pdf(
                    x[feat_idx], self.means_[idx, feat_idx], self.variances_[idx, feat_idx]
                )
                log_likelihood += np.log(pdf + self.epsilon_)
            log_posteriors.append(log_prior + log_likelihood)
        return np.array(log_posteriors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using MAP estimation.

        Args:
            X: Test features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)

        Raises:
            ValueError: If model hasn't been trained or X has wrong shape

        Example:
            >>> X_test = np.array([[2, 3], [4, 5]])
            >>> predictions = model.predict(X_test)
        """
        # Check if model has been trained
        if self.classes_ is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before predict()"
            )

        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.means_.shape[1]} features"
            )

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")

        logger.info(f"Predicting labels for {len(X)} samples")
        predictions = []
        for x in X:
            log_posteriors = self._calculate_log_posterior(x)
            predicted_class = self.classes_[np.argmax(log_posteriors)]
            predictions.append(predicted_class)
        logger.info("Prediction completed")
        return np.array(predictions)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get learned model parameters."""
        return {
            'classes': self.classes_,
            'priors': self.class_priors_,
            'means': self.means_,
            'variances': self.variances_
        }

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk using pickle.

        Args:
            filepath: Path where model will be saved (.pkl extension recommended)

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.classes_ is None:
            raise ValueError("Model must be trained before saving. Call fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            'var_smoothing': self.var_smoothing,
            'classes_': self.classes_,
            'class_priors_': self.class_priors_,
            'means_': self.means_,
            'variances_': self.variances_,
            'epsilon_': self.epsilon_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'GaussianNaiveBayesNumPy':
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded GaussianNaiveBayesNumPy instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        model = cls(var_smoothing=model_state['var_smoothing'])
        model.classes_ = model_state['classes_']
        model.class_priors_ = model_state['class_priors_']
        model.means_ = model_state['means_']
        model.variances_ = model_state['variances_']
        model.epsilon_ = model_state['epsilon_']

        logger.info(f"Model loaded from: {filepath}")
        return model


def visualize_feature_distributions(
    X: np.ndarray, y: np.ndarray, feature_names: list,
    save_path: str = "logs/numpy_feature_distributions.png"
) -> None:
    """Create 2x2 histogram grid showing feature distributions by class."""
    logger.info("Generating feature distribution visualizations")
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['red', 'green', 'blue']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for feature_idx, feature_name in enumerate(feature_names):
        ax = axes[feature_idx]
        for class_id, (class_name, color) in enumerate(zip(class_names, colors)):
            class_data = X[y == class_id, feature_idx]
            ax.hist(class_data, bins=15, alpha=0.6, label=class_name,
                   color=color, edgecolor='black')
        ax.set_xlabel(feature_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature_name} Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Feature distributions saved to: {save_path}")
    plt.close()
