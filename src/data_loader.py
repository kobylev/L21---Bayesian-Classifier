"""
Data Loading and Preprocessing Module

This module handles loading the Iris dataset from CSV and splitting it into
training and testing sets with stratification to maintain class distribution.

Functions:
    load_iris_data: Load Iris dataset from CSV file
    split_data: Split data into training and testing sets
    get_class_mapping: Get mapping between numeric labels and species names
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Configure logger
logger = logging.getLogger(__name__)


def load_iris_data(filepath: str = "Iris.csv") -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load the Iris dataset from CSV file.

    Args:
        filepath: Path to the Iris CSV file

    Returns:
        X: Feature matrix of shape (150, 4)
        y: Label vector of shape (150,) with integer encoding (0, 1, 2)
        feature_names: List of feature column names

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If data has unexpected shape or missing values
    """
    logger.info(f"Loading Iris dataset from: {filepath}")

    try:
        # Load CSV file
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples")

        # Validate data shape
        if len(df) != 150:
            logger.warning(f"Expected 150 samples, got {len(df)}")

        # Check for missing values
        if df.isnull().any().any():
            missing_count = df.isnull().sum().sum()
            raise ValueError(f"Dataset contains {missing_count} missing values")

        # Extract features (columns 1-4: Sepal/Petal measurements)
        feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        X = df[feature_names].values

        # Extract and encode species labels
        species_mapping = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }
        y = df['Species'].map(species_mapping).values

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Class distribution:")
        for class_id, count in zip(unique, counts):
            species_name = [k for k, v in species_mapping.items() if v == class_id][0]
            logger.info(f"  Class {class_id} ({species_name}): {count} samples")

        # Log feature statistics
        logger.info("Feature statistics:")
        for i, fname in enumerate(feature_names):
            logger.info(f"  {fname}: mean={X[:, i].mean():.2f}, "
                       f"std={X[:, i].std():.2f}, "
                       f"min={X[:, i].min():.2f}, "
                       f"max={X[:, i].max():.2f}")

        return X, y, feature_names

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets with stratification.

    Uses manual stratified splitting to ensure each class is proportionally
    represented in both train and test sets.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for testing (default: 0.25)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, "
                f"{100*test_size:.0f}% test (seed={random_state})")

    # Set random seed
    np.random.seed(random_state)

    # Get unique classes
    classes = np.unique(y)

    # Initialize lists to store indices
    train_indices = []
    test_indices = []

    # Perform stratified split for each class
    for class_id in classes:
        # Get indices for this class
        class_indices = np.where(y == class_id)[0]
        n_samples = len(class_indices)

        # Shuffle indices for this class
        np.random.shuffle(class_indices)

        # Calculate split point
        n_test = int(n_samples * test_size)

        # Split indices
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

        logger.debug(f"  Class {class_id}: {n_samples - n_test} train, {n_test} test")

    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Create train/test splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Log split statistics
    logger.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Log class distribution in splits
    logger.info("Training set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_id, count in zip(unique, counts):
        logger.info(f"  Class {class_id}: {count} samples")

    logger.info("Test set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for class_id, count in zip(unique, counts):
        logger.info(f"  Class {class_id}: {count} samples")

    return X_train, X_test, y_train, y_test


def get_class_mapping() -> Dict[int, str]:
    """
    Get mapping from numeric class labels to species names.

    Returns:
        Dictionary mapping class IDs (0, 1, 2) to species names
    """
    return {
        0: 'Iris-setosa',
        1: 'Iris-versicolor',
        2: 'Iris-virginica'
    }
