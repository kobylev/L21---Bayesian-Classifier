"""
Unit tests for data loading and preprocessing module.

Tests cover:
- CSV loading
- Data validation
- Stratified splitting
- Class mapping
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_iris_data, split_data, get_class_mapping


class TestLoadIrisData:
    """Test suite for load_iris_data function."""

    def test_load_iris_returns_correct_shapes(self):
        """Test that loading Iris.csv returns correct array shapes."""
        X, y, feature_names = load_iris_data("Iris.csv")

        assert X.shape == (150, 4), "Feature matrix should be 150×4"
        assert y.shape == (150,), "Label vector should have 150 elements"
        assert len(feature_names) == 4, "Should have 4 feature names"

    def test_load_iris_returns_correct_types(self):
        """Test that returned arrays have correct data types."""
        X, y, feature_names = load_iris_data("Iris.csv")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert X.dtype in [np.float32, np.float64]
        assert y.dtype in [np.int32, np.int64]

    def test_load_iris_no_missing_values(self):
        """Test that loaded data has no NaN values."""
        X, y, feature_names = load_iris_data("Iris.csv")

        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_load_iris_label_encoding(self):
        """Test that labels are properly encoded as 0, 1, 2."""
        X, y, feature_names = load_iris_data("Iris.csv")

        unique_labels = np.unique(y)
        assert len(unique_labels) == 3
        assert set(unique_labels) == {0, 1, 2}

    def test_load_iris_balanced_classes(self):
        """Test that each class has 50 samples."""
        X, y, feature_names = load_iris_data("Iris.csv")

        for class_label in [0, 1, 2]:
            count = np.sum(y == class_label)
            assert count == 50, f"Class {class_label} should have 50 samples"

    def test_load_iris_feature_names(self):
        """Test that feature names are returned correctly."""
        X, y, feature_names = load_iris_data("Iris.csv")

        expected_features = ['SepalLengthCm', 'SepalWidthCm',
                            'PetalLengthCm', 'PetalWidthCm']
        assert feature_names == expected_features

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_iris_data("nonexistent.csv")


class TestSplitData:
    """Test suite for split_data function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.array([0] * 50 + [1] * 50)
        return X, y

    def test_split_returns_four_arrays(self, simple_data):
        """Test that split returns 4 arrays."""
        X, y = simple_data
        result = split_data(X, y)

        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_split_correct_proportions(self, simple_data):
        """Test that split creates correct train/test proportions."""
        X, y = simple_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

        total_samples = len(y)
        test_samples = len(y_test)
        train_samples = len(y_train)

        assert train_samples + test_samples == total_samples
        assert abs(test_samples / total_samples - 0.25) < 0.05  # Within 5%

    def test_split_stratification(self, simple_data):
        """Test that split maintains class proportions (stratification)."""
        X, y = simple_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

        # Check train set proportions
        train_class0 = np.sum(y_train == 0)
        train_class1 = np.sum(y_train == 1)
        assert abs(train_class0 - train_class1) <= 1  # Almost equal

        # Check test set proportions
        test_class0 = np.sum(y_test == 0)
        test_class1 = np.sum(y_test == 1)
        assert abs(test_class0 - test_class1) <= 1  # Almost equal

    def test_split_reproducibility(self, simple_data):
        """Test that same random_state produces same split."""
        X, y = simple_data

        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)

    def test_split_different_seeds_different_results(self, simple_data):
        """Test that different random_state produces different split."""
        X, y = simple_data

        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=123)

        # At least one should be different
        assert not (np.array_equal(X_train1, X_train2) and
                   np.array_equal(y_train1, y_train2))

    def test_split_no_data_leakage(self, simple_data):
        """Test that train and test sets don't overlap."""
        X, y = simple_data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Convert to sets of tuples for comparison
        train_samples = set(map(tuple, X_train))
        test_samples = set(map(tuple, X_test))

        assert len(train_samples.intersection(test_samples)) == 0

    def test_split_custom_test_size(self, simple_data):
        """Test split with custom test_size parameter."""
        X, y = simple_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

        test_proportion = len(y_test) / len(y)
        assert abs(test_proportion - 0.3) < 0.05  # Within 5%

    def test_split_multiclass(self):
        """Test split with more than 2 classes."""
        X = np.arange(150).reshape(150, 1).astype(float)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)

        # Check all classes present in both sets
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3

    def test_split_preserves_feature_dimensions(self, simple_data):
        """Test that feature dimensions are preserved after split."""
        X, y = simple_data
        X_train, X_test, y_train, y_test = split_data(X, y)

        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]


class TestGetClassMapping:
    """Test suite for get_class_mapping function."""

    def test_mapping_returns_dict(self):
        """Test that function returns a dictionary."""
        mapping = get_class_mapping()
        assert isinstance(mapping, dict)

    def test_mapping_has_three_classes(self):
        """Test that mapping contains 3 classes."""
        mapping = get_class_mapping()
        assert len(mapping) == 3

    def test_mapping_correct_keys(self):
        """Test that mapping has correct integer keys."""
        mapping = get_class_mapping()
        assert set(mapping.keys()) == {0, 1, 2}

    def test_mapping_correct_values(self):
        """Test that mapping has correct species names."""
        mapping = get_class_mapping()

        expected_species = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'}
        assert set(mapping.values()) == expected_species

    def test_mapping_specific_assignments(self):
        """Test specific class-to-species assignments."""
        mapping = get_class_mapping()

        assert mapping[0] == 'Iris-setosa'
        assert mapping[1] == 'Iris-versicolor'
        assert mapping[2] == 'Iris-virginica'
