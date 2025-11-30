"""
Gaussian Naive Bayes - Scikit-Learn Implementation

This module provides a wrapper around scikit-learn's GaussianNB classifier
with enhanced logging and parameter inspection for comparison with the
manual NumPy implementation.

Classes:
    GaussianNaiveBayesSklearn: Wrapper around sklearn's GaussianNB
"""

import logging
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple

# Configure logger
logger = logging.getLogger(__name__)


class GaussianNaiveBayesSklearn:
    """
    Wrapper around scikit-learn's GaussianNB classifier.

    Provides the same interface as the manual NumPy implementation
    but uses sklearn's optimized implementation internally. Includes
    enhanced logging and parameter extraction for comparison.

    Attributes:
        model_: Underlying sklearn GaussianNB model
        classes_: Unique class labels
        class_priors_: Prior probabilities P(y) for each class
        means_: Feature means (theta_) for each class
        variances_: Feature variances (var_) for each class
    """

    def __init__(self):
        """Initialize the sklearn Gaussian Naive Bayes wrapper."""
        self.model_ = GaussianNB()
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayesSklearn':
        """
        Train the sklearn Gaussian Naive Bayes model.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            self: Trained classifier instance
        """
        logger.info("Training Gaussian Naive Bayes (Scikit-Learn implementation)")

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        logger.info(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")

        # Train the model
        self.model_.fit(X, y)

        # Extract learned parameters
        self.classes_ = self.model_.classes_
        self.means_ = self.model_.theta_  # Feature means
        self.variances_ = self.model_.var_  # Feature variances

        # Convert log priors to priors (handle both old and new sklearn versions)
        if hasattr(self.model_, 'class_log_prior_'):
            self.class_priors_ = np.exp(self.model_.class_log_prior_)
        else:
            self.class_priors_ = self.model_.class_prior_

        # Log learned parameters
        logger.info("Learned parameters:")
        for idx, class_label in enumerate(self.classes_):
            logger.debug(f"  Class {class_label}: prior={self.class_priors_[idx]:.4f}")

        logger.info("Training completed successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Test features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        logger.info(f"Predicting labels for {len(X)} samples")
        predictions = self.model_.predict(X)
        logger.info("Prediction completed")
        return predictions

    def get_params(self) -> dict:
        """
        Get learned model parameters.

        Returns:
            Dictionary containing priors, means, and variances
        """
        return {
            'classes': self.classes_,
            'priors': self.class_priors_,
            'means': self.means_,
            'variances': self.variances_
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
) -> Tuple[float, np.ndarray, str]:
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for reporting

    Returns:
        accuracy: Overall accuracy score
        conf_matrix: Confusion matrix
        report: Classification report string
    """
    logger.info("Evaluating model performance")

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    logger.info("  Confusion Matrix:")
    for i, row in enumerate(conf_matrix):
        logger.info(f"    Class {i}: {row}")

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("  Classification Report:")
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"    {line}")

    return accuracy, conf_matrix, report


def compare_parameters(
    numpy_params: dict,
    sklearn_params: dict,
    feature_names: list
) -> None:
    """
    Compare learned parameters between NumPy and sklearn implementations.

    Args:
        numpy_params: Parameters from NumPy implementation
        sklearn_params: Parameters from sklearn implementation
        feature_names: List of feature names
    """
    logger.info("Comparing learned parameters:")

    # Compare priors
    logger.info("\nPrior Probabilities:")
    logger.info(f"  {'Class':<10} {'NumPy':<15} {'Sklearn':<15} {'Difference':<15}")
    logger.info(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    for i in range(len(numpy_params['classes'])):
        diff = abs(numpy_params['priors'][i] - sklearn_params['priors'][i])
        logger.info(f"  {i:<10} {numpy_params['priors'][i]:<15.10f} "
                   f"{sklearn_params['priors'][i]:<15.10f} {diff:<15.10e}")

    # Compare means
    logger.info("\nFeature Means (first class):")
    logger.info(f"  {'Feature':<20} {'NumPy':<15} {'Sklearn':<15} {'Difference':<15}")
    logger.info(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    for i, fname in enumerate(feature_names):
        diff = abs(numpy_params['means'][0, i] - sklearn_params['means'][0, i])
        logger.info(f"  {fname:<20} {numpy_params['means'][0, i]:<15.10f} "
                   f"{sklearn_params['means'][0, i]:<15.10f} {diff:<15.10e}")

    # Compare variances
    logger.info("\nFeature Variances (first class):")
    logger.info(f"  {'Feature':<20} {'NumPy':<15} {'Sklearn':<15} {'Difference':<15}")
    logger.info(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    for i, fname in enumerate(feature_names):
        diff = abs(numpy_params['variances'][0, i] - sklearn_params['variances'][0, i])
        logger.info(f"  {fname:<20} {numpy_params['variances'][0, i]:<15.10f} "
                   f"{sklearn_params['variances'][0, i]:<15.10f} {diff:<15.10e}")

    # Calculate maximum differences
    max_prior_diff = np.max(np.abs(numpy_params['priors'] - sklearn_params['priors']))
    max_mean_diff = np.max(np.abs(numpy_params['means'] - sklearn_params['means']))
    max_var_diff = np.max(np.abs(numpy_params['variances'] - sklearn_params['variances']))

    logger.info("\nMaximum Parameter Differences:")
    logger.info(f"  Priors:    {max_prior_diff:.10e}")
    logger.info(f"  Means:     {max_mean_diff:.10e}")
    logger.info(f"  Variances: {max_var_diff:.10e}")
