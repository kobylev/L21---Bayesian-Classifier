"""
Constants and Configuration

Central location for all magic numbers and configuration values used across the project.
"""

# Numerical Stability
DEFAULT_VAR_SMOOTHING = 1e-9  # Default variance smoothing for numerical stability
MIN_VARIANCE_THRESHOLD = 1e-10  # Minimum variance before smoothing is critical

# Data Splitting
DEFAULT_TEST_SIZE = 0.25  # Default train/test split ratio (25% test)
DEFAULT_RANDOM_SEED = 42  # Default random seed for reproducibility

# Visualization
DEFAULT_DPI = 150  # Default resolution for saved figures
DEFAULT_FIGURE_SIZE = (12, 10)  # Default figure size for plots
CONFUSION_MATRIX_SIZE = (14, 6)  # Size for confusion matrix side-by-side plots

# Class Mapping
CLASS_LABELS = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

# Feature Names
FEATURE_NAMES = [
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_FILE = 'logs/iris_classification.log'

# Model Persistence
DEFAULT_MODEL_EXTENSION = '.pkl'  # Default file extension for saved models

# Success Criteria
MIN_ACCURACY_THRESHOLD = 0.90  # Minimum accuracy for success (90%)
MIN_AGREEMENT_THRESHOLD = 0.95  # Minimum prediction agreement (95%)

# Dataset
EXPECTED_N_SAMPLES = 150  # Expected number of samples in Iris dataset
EXPECTED_N_FEATURES = 4  # Expected number of features in Iris dataset
EXPECTED_N_CLASSES = 3  # Expected number of classes in Iris dataset
