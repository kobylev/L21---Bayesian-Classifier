"""
Model Comparison and Visualization Module

Compare predictions and visualize results from NumPy and sklearn implementations.
"""

import logging
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)


def compare_predictions(
    y_true: np.ndarray,
    y_pred_numpy: np.ndarray,
    y_pred_sklearn: np.ndarray
) -> Dict[str, float]:
    """
    Compare predictions between NumPy and sklearn implementations.

    Args:
        y_true: True labels
        y_pred_numpy: Predictions from NumPy implementation
        y_pred_sklearn: Predictions from sklearn implementation

    Returns:
        Dictionary containing comparison metrics
    """
    logger.info("Comparing predictions between implementations")

    # Calculate agreement
    agreement = np.sum(y_pred_numpy == y_pred_sklearn)
    agreement_rate = agreement / len(y_true)

    logger.info(f"  Prediction agreement: {agreement}/{len(y_true)} ({agreement_rate*100:.2f}%)")

    # Find disagreements
    disagreement_indices = np.where(y_pred_numpy != y_pred_sklearn)[0]

    if len(disagreement_indices) > 0:
        logger.info(f"  Found {len(disagreement_indices)} disagreements at indices:")
        for idx in disagreement_indices:
            logger.info(f"    Sample {idx}: NumPy={y_pred_numpy[idx]}, "
                       f"Sklearn={y_pred_sklearn[idx]}, True={y_true[idx]}")
    else:
        logger.info("  Perfect agreement: All predictions match!")

    # Calculate individual accuracies
    accuracy_numpy = np.sum(y_pred_numpy == y_true) / len(y_true)
    accuracy_sklearn = np.sum(y_pred_sklearn == y_true) / len(y_true)

    logger.info(f"  NumPy accuracy:   {accuracy_numpy:.4f} ({accuracy_numpy*100:.2f}%)")
    logger.info(f"  Sklearn accuracy: {accuracy_sklearn:.4f} ({accuracy_sklearn*100:.2f}%)")

    return {
        'agreement_count': agreement,
        'agreement_rate': agreement_rate,
        'disagreement_indices': disagreement_indices,
        'accuracy_numpy': accuracy_numpy,
        'accuracy_sklearn': accuracy_sklearn
    }


def visualize_confusion_matrices(
    y_true: np.ndarray,
    y_pred_numpy: np.ndarray,
    y_pred_sklearn: np.ndarray,
    class_names: List[str],
    save_path: str = "logs/confusion_matrices.png"
) -> None:
    """
    Create side-by-side confusion matrix visualizations.

    Args:
        y_true: True labels
        y_pred_numpy: Predictions from NumPy implementation
        y_pred_sklearn: Predictions from sklearn implementation
        class_names: List of class names
        save_path: Path to save the figure
    """
    logger.info("Generating confusion matrix visualizations")

    # Calculate confusion matrices
    from sklearn.metrics import confusion_matrix

    cm_numpy = confusion_matrix(y_true, y_pred_numpy)
    cm_sklearn = confusion_matrix(y_true, y_pred_sklearn)

    # Calculate accuracies for titles
    acc_numpy = np.sum(y_pred_numpy == y_true) / len(y_true)
    acc_sklearn = np.sum(y_pred_sklearn == y_true) / len(y_true)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot NumPy confusion matrix
    sns.heatmap(cm_numpy, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'NumPy Implementation\nAccuracy: {acc_numpy:.4f} ({acc_numpy*100:.2f}%)',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)

    # Plot sklearn confusion matrix
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title(f'Scikit-Learn Implementation\nAccuracy: {acc_sklearn:.4f} ({acc_sklearn*100:.2f}%)',
                      fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrices saved to: {save_path}")
    plt.close()


def generate_comparison_report(
    comparison_metrics: Dict[str, float],
    numpy_params: Dict[str, np.ndarray],
    sklearn_params: Dict[str, np.ndarray]
) -> None:
    """
    Generate comprehensive comparison report.

    Args:
        comparison_metrics: Dictionary with comparison metrics
        numpy_params: Parameters from NumPy implementation
        sklearn_params: Parameters from sklearn implementation
    """
    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON REPORT")
    logger.info("="*70)

    # Prediction comparison
    logger.info("\n1. PREDICTION COMPARISON:")
    logger.info(f"   Agreement Rate: {comparison_metrics['agreement_rate']*100:.2f}% "
                f"({comparison_metrics['agreement_count']}/{comparison_metrics['agreement_count'] + len(comparison_metrics['disagreement_indices'])})")
    logger.info(f"   NumPy Accuracy:   {comparison_metrics['accuracy_numpy']*100:.2f}%")
    logger.info(f"   Sklearn Accuracy: {comparison_metrics['accuracy_sklearn']*100:.2f}%")
    logger.info(f"   Accuracy Difference: {abs(comparison_metrics['accuracy_numpy'] - comparison_metrics['accuracy_sklearn'])*100:.2f}%")

    # Parameter comparison
    logger.info("\n2. PARAMETER COMPARISON:")

    max_prior_diff = np.max(np.abs(numpy_params['priors'] - sklearn_params['priors']))
    max_mean_diff = np.max(np.abs(numpy_params['means'] - sklearn_params['means']))
    max_var_diff = np.max(np.abs(numpy_params['variances'] - sklearn_params['variances']))

    logger.info(f"   Maximum Prior Difference:    {max_prior_diff:.10e}")
    logger.info(f"   Maximum Mean Difference:     {max_mean_diff:.10e}")
    logger.info(f"   Maximum Variance Difference: {max_var_diff:.10e}")

    # Interpretation
    logger.info("\n3. INTERPRETATION:")

    if comparison_metrics['agreement_rate'] == 1.0:
        logger.info("   ✓ PERFECT MATCH: Both implementations produce identical predictions!")
    elif comparison_metrics['agreement_rate'] >= 0.95:
        logger.info("   ✓ EXCELLENT AGREEMENT: Implementations are highly consistent (≥95%)")
    elif comparison_metrics['agreement_rate'] >= 0.90:
        logger.info("   ~ GOOD AGREEMENT: Minor differences exist but results are similar (≥90%)")
    else:
        logger.info("   ✗ SIGNIFICANT DIFFERENCES: Implementations diverge substantially (<90%)")

    # Success criteria check
    logger.info("\n4. SUCCESS CRITERIA:")
    logger.info(f"   {'Metric':<30} {'Status':<10} {'Result'}")
    logger.info(f"   {'-'*30} {'-'*10} {'-'*30}")

    # Accuracy check
    numpy_acc_pass = comparison_metrics['accuracy_numpy'] > 0.90
    sklearn_acc_pass = comparison_metrics['accuracy_sklearn'] > 0.90
    logger.info(f"   {'NumPy accuracy > 90%':<30} {'✓ PASS' if numpy_acc_pass else '✗ FAIL':<10} "
                f"{comparison_metrics['accuracy_numpy']*100:.2f}%")
    logger.info(f"   {'Sklearn accuracy > 90%':<30} {'✓ PASS' if sklearn_acc_pass else '✗ FAIL':<10} "
                f"{comparison_metrics['accuracy_sklearn']*100:.2f}%")

    # Agreement check
    agreement_pass = comparison_metrics['agreement_rate'] >= 0.95
    logger.info(f"   {'Agreement rate ≥ 95%':<30} {'✓ PASS' if agreement_pass else '✗ FAIL':<10} "
                f"{comparison_metrics['agreement_rate']*100:.2f}%")

    # Overall result
    all_pass = numpy_acc_pass and sklearn_acc_pass and agreement_pass
    logger.info(f"\n   {'OVERALL RESULT:':<30} {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")

    logger.info("\n" + "="*70)
