"""
Evaluation Module
Computes metrics and generates visualizations for model evaluation.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    prefix: str = "",
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for ROC-AUC)
        prefix: Prefix for metric names
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    metrics[f"{prefix}_accuracy"] = float(accuracy)
    metrics[f"{prefix}_precision"] = float(precision)
    metrics[f"{prefix}_recall"] = float(recall)
    metrics[f"{prefix}_f1"] = float(f1)

    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        try:
            # Multi-class ROC-AUC
            n_classes = y_proba.shape[1]

            # One-hot encode true labels
            y_true_onehot = np.eye(n_classes)[y_true]

            # Compute ROC-AUC
            roc_auc = roc_auc_score(y_true_onehot, y_proba, average=average, multi_class='ovr')
            metrics[f"{prefix}_roc_auc"] = float(roc_auc)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: Dict[str, int],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = False,
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_mapping: Mapping from class names to labels
        title: Plot title
        save_path: Path to save the figure
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Get class names in label order
    class_names = [name for name, label in sorted(class_mapping.items(), key=lambda x: x[1])]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history,
    title: str = "Training History",
    save_path: Optional[Path] = None,
):
    """
    Plot training history (loss and accuracy).

    Args:
        history: Training history (Keras History object or dict)
        title: Plot title
        save_path: Path to save the figure
    """
    # Convert to dict if Keras History object
    if hasattr(history, 'history'):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[Path] = None,
):
    """
    Plot ROC curves for multi-class classification.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities (n_samples, n_classes)
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])

    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})',
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_mapping: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
):
    """
    Plot class distribution.

    Args:
        labels: Label array
        class_mapping: Mapping from class names to labels
        title: Plot title
        save_path: Path to save the figure
    """
    # Count samples per class
    class_counts = {}
    for label in labels:
        class_name = next(name for name, l in class_mapping.items() if l == label)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Sort by label order
    classes = [name for name, label in sorted(class_mapping.items(), key=lambda x: x[1])]
    counts = [class_counts[name] for name in classes]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(classes, counts, color='steelblue', edgecolor='black', alpha=0.7)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=11,
        )

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution saved to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_models(
    results: Dict[str, Dict],
    metrics: List[str] = ["test_accuracy", "test_f1", "test_precision", "test_recall"],
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
):
    """
    Compare models using bar plots.

    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save the figure
    """
    # Extract model names and metrics
    model_names = list(results.keys())

    # Separate ML and DL models
    ml_models = [name for name in model_names if results[name].get('type') == 'ml']
    dl_models = [name for name in model_names if results[name].get('type') == 'dl']

    # Create figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Collect values for this metric
        ml_values = [results[name].get(metric, 0) for name in ml_models]
        dl_values = [results[name].get(metric, 0) for name in dl_models]

        # Plot ML models
        if ml_models:
            x_ml = np.arange(len(ml_models))
            ax.bar(x_ml, ml_values, width=0.35, label='ML Models', color=colors[0], alpha=0.8)

        # Plot DL models
        if dl_models:
            x_dl = np.arange(len(dl_models)) + (0.35 if ml_models else 0)
            ax.bar(x_dl, dl_values, width=0.35, label='DL Models', color=colors[1], alpha=0.8)

        # Labels and formatting
        all_models = ml_models + dl_models
        if all_models:
            ax.set_xticks(np.arange(len(all_models)))
            ax.set_xticklabels(all_models, rotation=45, ha='right')

        ax.set_ylabel(metric.replace('test_', '').title(), fontsize=11)
        ax.set_title(metric.replace('test_', '').title(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Set y-axis range
        ax.set_ylim([0, 1.0])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: Dict[str, int],
):
    """
    Print classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_mapping: Mapping from class names to labels
    """
    # Get class names in label order
    target_names = [name for name, label in sorted(class_mapping.items(), key=lambda x: x[1])]

    # Generate report
    report = classification_report(y_true, y_pred, target_names=target_names)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_mapping: Mapping from class names to labels

    Returns:
        Dictionary of per-class metrics
    """
    # Get class names
    class_names = [name for name, label in sorted(class_mapping.items(), key=lambda x: x[1])]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute per-class metrics
    per_class_metrics = {}

    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "support": int(cm[i, :].sum()),
        }

    return per_class_metrics