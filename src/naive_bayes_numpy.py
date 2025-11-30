"""
Gaussian Naive Bayes - Manual NumPy Implementation

Implements Gaussian Naive Bayes from scratch using NumPy.
Mathematical Foundation: Bayes' Theorem and Gaussian PDF.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class GaussianNaiveBayesNumPy:
    """Gaussian Naive Bayes classifier from scratch."""

    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
        self.epsilon_ = 1e-9

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayesNumPy':
        """Train the model by calculating priors, means, and variances."""
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
        """Predict class labels using MAP estimation."""
        logger.info(f"Predicting labels for {len(X)} samples")
        predictions = []
        for x in X:
            log_posteriors = self._calculate_log_posterior(x)
            predicted_class = self.classes_[np.argmax(log_posteriors)]
            predictions.append(predicted_class)
        logger.info("Prediction completed")
        return np.array(predictions)

    def get_params(self) -> dict:
        """Get learned model parameters."""
        return {
            'classes': self.classes_,
            'priors': self.class_priors_,
            'means': self.means_,
            'variances': self.variances_
        }


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
