"""
Main Pipeline for Iris Classification with Naive Bayes

Orchestrates 8-step workflow comparing NumPy and sklearn implementations.
"""

import logging
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_iris_data, split_data, get_class_mapping
from naive_bayes_numpy import GaussianNaiveBayesNumPy, visualize_feature_distributions
from naive_bayes_sklearn import (
    GaussianNaiveBayesSklearn,
    evaluate_model,
    compare_parameters
)
from comparison import (
    compare_predictions,
    visualize_confusion_matrices,
    generate_comparison_report
)


def setup_logging(log_file: str = "logs/iris_classification.log") -> None:
    """Configure dual logging to file and console."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main():
    """Main pipeline execution."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("IRIS CLASSIFICATION - GAUSSIAN NAIVE BAYES COMPARISON")
    logger.info("="*70)
    logger.info("Comparing manual NumPy implementation vs. Scikit-Learn")
    logger.info("")

    try:
        # STEP 1: Load and Split Data
        logger.info("\n" + "="*70)
        logger.info("STEP 1: LOAD AND SPLIT DATA")
        logger.info("="*70)
        X, y, feature_names = load_iris_data("Iris.csv")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
        class_names = list(get_class_mapping().values())
        logger.info("Data loading and splitting completed successfully\n")

        # STEP 2: Train NumPy Model
        logger.info("\n" + "="*70)
        logger.info("STEP 2: TRAIN NUMPY IMPLEMENTATION")
        logger.info("="*70)
        numpy_model = GaussianNaiveBayesNumPy()
        numpy_model.fit(X_train, y_train)
        numpy_params = numpy_model.get_params()
        logger.info("NumPy model training completed\n")

        # STEP 3: Visualize Feature Distributions
        logger.info("\n" + "="*70)
        logger.info("STEP 3: VISUALIZE FEATURE DISTRIBUTIONS")
        logger.info("="*70)
        visualize_feature_distributions(X_train, y_train, feature_names)
        logger.info("Feature distribution visualization completed\n")

        # STEP 4: Test NumPy Model
        logger.info("\n" + "="*70)
        logger.info("STEP 4: TEST NUMPY IMPLEMENTATION")
        logger.info("="*70)
        y_pred_numpy = numpy_model.predict(X_test)
        accuracy_numpy, cm_numpy, report_numpy = evaluate_model(
            y_test, y_pred_numpy, class_names
        )
        logger.info("NumPy model testing completed\n")

        # STEP 5: Train Sklearn Model
        logger.info("\n" + "="*70)
        logger.info("STEP 5: TRAIN SKLEARN IMPLEMENTATION")
        logger.info("="*70)
        sklearn_model = GaussianNaiveBayesSklearn()
        sklearn_model.fit(X_train, y_train)
        sklearn_params = sklearn_model.get_params()
        logger.info("Sklearn model training completed\n")

        # STEP 6: Test Sklearn Model
        logger.info("\n" + "="*70)
        logger.info("STEP 6: TEST SKLEARN IMPLEMENTATION")
        logger.info("="*70)
        y_pred_sklearn = sklearn_model.predict(X_test)
        accuracy_sklearn, cm_sklearn, report_sklearn = evaluate_model(
            y_test, y_pred_sklearn, class_names
        )
        logger.info("Sklearn model testing completed\n")

        # STEP 7: Compare Results
        logger.info("\n" + "="*70)
        logger.info("STEP 7: COMPARE IMPLEMENTATIONS")
        logger.info("="*70)
        comparison_metrics = compare_predictions(y_test, y_pred_numpy, y_pred_sklearn)
        logger.info("\nComparing learned parameters:")
        compare_parameters(numpy_params, sklearn_params, feature_names)
        logger.info("Comparison completed\n")

        # STEP 8: Generate Visualizations
        logger.info("\n" + "="*70)
        logger.info("STEP 8: GENERATE VISUALIZATIONS")
        logger.info("="*70)
        visualize_confusion_matrices(y_test, y_pred_numpy, y_pred_sklearn, class_names)
        logger.info("Visualization generation completed\n")

        # FINAL REPORT
        generate_comparison_report(comparison_metrics, numpy_params, sklearn_params)

        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("Generated outputs:")
        logger.info("  - logs/iris_classification.log")
        logger.info("  - logs/numpy_feature_distributions.png")
        logger.info("  - logs/confusion_matrices.png")
        logger.info("")

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
