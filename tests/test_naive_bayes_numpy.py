"""
Unit tests for NumPy Gaussian Naive Bayes implementation.

Tests cover:
- Model initialization
- Training (fit method)
- Prediction accuracy
- Model persistence (save/load)
- Numerical stability
- Edge cases
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from naive_bayes_numpy import GaussianNaiveBayesNumPy


class TestGaussianNaiveBayesNumPy:
    """Test suite for GaussianNaiveBayesNumPy class."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple 2D dataset for testing."""
        np.random.seed(42)
        # Class 0: mean=[0, 0], Class 1: mean=[5, 5]
        X_class0 = np.random.randn(20, 2) + np.array([0, 0])
        X_class1 = np.random.randn(20, 2) + np.array([5, 5])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 20 + [1] * 20)
        return X, y

    @pytest.fixture
    def model(self):
        """Create a fresh model instance."""
        return GaussianNaiveBayesNumPy()

    def test_initialization(self, model):
        """Test model initializes with correct default values."""
        assert model.var_smoothing == 1e-9
        assert model.classes_ is None
        assert model.class_priors_ is None
        assert model.means_ is None
        assert model.variances_ is None

    def test_custom_var_smoothing(self):
        """Test custom variance smoothing parameter."""
        custom_eps = 1e-6
        model = GaussianNaiveBayesNumPy(var_smoothing=custom_eps)
        assert model.var_smoothing == custom_eps
        assert model.epsilon_ == custom_eps

    def test_fit_shape(self, model, simple_dataset):
        """Test that fit method sets correct parameter shapes."""
        X, y = simple_dataset
        model.fit(X, y)

        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        assert model.classes_.shape == (n_classes,)
        assert model.class_priors_.shape == (n_classes,)
        assert model.means_.shape == (n_classes, n_features)
        assert model.variances_.shape == (n_classes, n_features)

    def test_fit_values(self, model, simple_dataset):
        """Test that fit method computes reasonable parameter values."""
        X, y = simple_dataset
        model.fit(X, y)

        # Check priors sum to 1
        assert np.allclose(np.sum(model.class_priors_), 1.0)

        # Check priors are positive
        assert np.all(model.class_priors_ > 0)

        # Check variances are positive
        assert np.all(model.variances_ > 0)

    def test_fit_returns_self(self, model, simple_dataset):
        """Test that fit method returns self for chaining."""
        X, y = simple_dataset
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, model, simple_dataset):
        """Test that predict returns correct shape."""
        X, y = simple_dataset
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (X.shape[0],)
        assert predictions.dtype in [np.int32, np.int64]

    def test_predict_accuracy(self, model, simple_dataset):
        """Test that model achieves reasonable accuracy on simple data."""
        X, y = simple_dataset
        model.fit(X, y)
        predictions = model.predict(X)

        accuracy = np.mean(predictions == y)
        # Should achieve >90% on this simple linearly separable dataset
        assert accuracy > 0.90

    def test_predict_classes(self, model, simple_dataset):
        """Test that predictions are valid class labels."""
        X, y = simple_dataset
        model.fit(X, y)
        predictions = model.predict(X)

        # All predictions should be in the set of known classes
        assert set(predictions).issubset(set(model.classes_))

    def test_gaussian_pdf_calculation(self, model):
        """Test Gaussian PDF calculation."""
        # PDF of N(0,1) at x=0 should be ~0.399
        pdf = model._calculate_gaussian_pdf(0.0, 0.0, 1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert np.isclose(pdf, expected, rtol=1e-5)

    def test_variance_smoothing_prevents_zero_variance(self, model):
        """Test that variance smoothing prevents division by zero."""
        # Create dataset with zero variance in one feature
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y = np.array([0, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        # Should not raise any errors or produce NaN/Inf
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_single_sample_per_class(self, model):
        """Test model behavior with only one sample per class."""
        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        y = np.array([0, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        # Should still produce valid predictions
        assert predictions.shape == (2,)
        assert not np.any(np.isnan(predictions))

    def test_get_params(self, model, simple_dataset):
        """Test get_params returns all learned parameters."""
        X, y = simple_dataset
        model.fit(X, y)
        params = model.get_params()

        assert 'classes' in params
        assert 'priors' in params
        assert 'means' in params
        assert 'variances' in params
        assert np.array_equal(params['classes'], model.classes_)

    def test_save_model_creates_file(self, model, simple_dataset, tmp_path):
        """Test that save_model creates a file."""
        X, y = simple_dataset
        model.fit(X, y)

        filepath = tmp_path / "test_model.pkl"
        model.save_model(filepath)

        assert filepath.exists()

    def test_save_untrained_model_raises_error(self, model, tmp_path):
        """Test that saving untrained model raises ValueError."""
        filepath = tmp_path / "test_model.pkl"

        with pytest.raises(ValueError, match="Model must be trained"):
            model.save_model(filepath)

    def test_load_model_restores_parameters(self, model, simple_dataset, tmp_path):
        """Test that loaded model has same parameters as saved model."""
        X, y = simple_dataset
        model.fit(X, y)

        filepath = tmp_path / "test_model.pkl"
        model.save_model(filepath)

        loaded_model = GaussianNaiveBayesNumPy.load_model(filepath)

        assert np.array_equal(loaded_model.classes_, model.classes_)
        assert np.allclose(loaded_model.class_priors_, model.class_priors_)
        assert np.allclose(loaded_model.means_, model.means_)
        assert np.allclose(loaded_model.variances_, model.variances_)

    def test_load_model_makes_same_predictions(self, model, simple_dataset, tmp_path):
        """Test that loaded model produces same predictions."""
        X, y = simple_dataset
        model.fit(X, y)
        original_predictions = model.predict(X)

        filepath = tmp_path / "test_model.pkl"
        model.save_model(filepath)

        loaded_model = GaussianNaiveBayesNumPy.load_model(filepath)
        loaded_predictions = loaded_model.predict(X)

        assert np.array_equal(original_predictions, loaded_predictions)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GaussianNaiveBayesNumPy.load_model("nonexistent_model.pkl")

    def test_multiclass_classification(self, model):
        """Test model with 3+ classes."""
        np.random.seed(42)
        # Create 3-class dataset
        X_class0 = np.random.randn(15, 2) + np.array([0, 0])
        X_class1 = np.random.randn(15, 2) + np.array([5, 0])
        X_class2 = np.random.randn(15, 2) + np.array([2.5, 5])
        X = np.vstack([X_class0, X_class1, X_class2])
        y = np.array([0] * 15 + [1] * 15 + [2] * 15)

        model.fit(X, y)
        predictions = model.predict(X)

        # Check all classes are represented
        assert len(np.unique(predictions)) <= 3
        assert set(predictions).issubset({0, 1, 2})

    def test_high_dimensional_data(self, model):
        """Test model with higher dimensional data."""
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (n_samples,)
        assert model.means_.shape == (2, n_features)

    def test_log_posterior_numerical_stability(self, model, simple_dataset):
        """Test that log posterior calculation doesn't overflow/underflow."""
        X, y = simple_dataset
        model.fit(X, y)

        # Test with extreme values
        x_extreme = np.array([100.0, 100.0])
        log_post = model._calculate_log_posterior(x_extreme)

        assert not np.any(np.isnan(log_post))
        assert not np.any(np.isinf(log_post))
