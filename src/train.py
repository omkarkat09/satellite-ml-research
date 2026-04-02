"""
Training Script
Main training pipeline for classical ML and deep learning models.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import json
from datetime import datetime

from config import (
    EXPERIMENTS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    DEFAULT_CLASS_MAPPING,
)

from data_loader import SatelliteDataset, create_sample_dataset
from preprocessing import Preprocessor, preprocess_pipeline, NormalizationStats
from features import FeatureExtractor, extract_features_pipeline
from models_ml import MLModelTrainer, train_baseline_models
from models_dl import DLModelFactory, train_dl_models
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history,
    compare_models,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seeds():
    """Set random seeds for reproducibility."""
    import random
    import os

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
    except ImportError:
        pass

    logger.info(f"Random seeds set to {RANDOM_SEED}")


class TrainingPipeline:
    """Complete training pipeline for satellite image classification."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        create_sample: bool = False,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize training pipeline.

        Args:
            data_dir: Directory containing the dataset
            create_sample: Whether to create a sample dataset
            class_mapping: Mapping from class names to labels
        """
        self.data_dir = data_dir
        self.create_sample = create_sample
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        self.n_classes = len(self.class_mapping)

        self.experiment_dir = EXPERIMENTS_DIR / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_feat = None
        self.X_val_feat = None
        self.X_test_feat = None

        # Models
        self.ml_models = {}
        self.dl_models = {}

        # Results
        self.results = {}

        logger.info(f"Experiment directory: {self.experiment_dir}")

    def load_data(self):
        """Load and split the dataset."""
        logger.info("="*60)
        logger.info("DATA LOADING")
        logger.info("="*60)

        # Create sample dataset if requested
        if self.create_sample:
            logger.info("Creating sample dataset...")
            create_sample_dataset()
            self.data_dir = Path("data/raw/sample")

        # Load dataset
        dataset = SatelliteDataset(
            data_dir=self.data_dir,
            class_mapping=self.class_mapping,
        )

        images, labels = dataset.load_from_directory_structure()

        # Train/val/test split
        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = dataset.train_val_test_split(images, labels)

        logger.info(f"Data loaded successfully")
        logger.info(f"Train: {len(self.X_train)} samples")
        logger.info(f"Val: {len(self.X_val)} samples")
        logger.info(f"Test: {len(self.X_test)} samples")

    def preprocess_data(self, normalize: bool = True, augment: bool = False):
        """
        Preprocess the data.

        Args:
            normalize: Whether to normalize images
            augment: Whether to augment training data
        """
        logger.info("="*60)
        logger.info("PREPROCESSING")
        logger.info("="*60)

        self.X_train, self.X_val, self.X_test, norm_stats = preprocess_pipeline(
            self.X_train,
            self.X_val,
            self.X_test,
            normalize=normalize,
            augment=augment,
        )

        # Save normalization stats
        if norm_stats:
            stats_path = self.experiment_dir / "normalization_stats.npz"
            norm_stats.save(str(stats_path))

        logger.info("Preprocessing complete")

    def extract_features(self, use_spectral_indices: bool = True):
        """
        Extract features for classical ML models.

        Args:
            use_spectral_indices: Whether to extract spectral indices
        """
        logger.info("="*60)
        logger.info("FEATURE EXTRACTION")
        logger.info("="*60)

        self.X_train_feat, self.X_val_feat, self.X_test_feat = extract_features_pipeline(
            self.X_train,
            self.X_val,
            self.X_test,
            use_spectral_indices=use_spectral_indices,
        )

        logger.info("Feature extraction complete")

    def train_ml_models(
        self,
        models: Optional[List[str]] = None,
        use_hyperparameter_tuning: bool = False,
    ):
        """
        Train classical ML models.

        Args:
            models: List of model names to train
            use_hyperparameter_tuning: Whether to tune hyperparameters
        """
        logger.info("="*60)
        logger.info("TRAINING CLASSICAL ML MODELS")
        logger.info("="*60)

        if self.X_train_feat is None:
            self.extract_features()

        self.ml_models = train_baseline_models(
            self.X_train_feat,
            self.y_train,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
        )

        logger.info(f"Trained {len(self.ml_models)} ML models")

    def train_dl_models(
        self,
        models: Optional[List[str]] = None,
        epochs: int = 50,
    ):
        """
        Train deep learning models.

        Args:
            models: List of model names to train
            epochs: Number of training epochs
        """
        logger.info("="*60)
        logger.info("TRAINING DEEP LEARNING MODELS")
        logger.info("="*60)

        self.dl_models = train_dl_models(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.n_classes,
            model_names=models,
        )

        logger.info(f"Trained {len(self.dl_models)} DL models")

    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("="*60)
        logger.info("EVALUATION")
        logger.info("="*60)

        all_results = {}

        # Evaluate ML models
        if self.ml_models:
            logger.info("\nEvaluating ML models...")
            ml_results = self._evaluate_ml_models()
            all_results.update(ml_results)

        # Evaluate DL models
        if self.dl_models:
            logger.info("\nEvaluating DL models...")
            dl_results = self._evaluate_dl_models()
            all_results.update(dl_results)

        self.results = all_results

        # Save results
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_json(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            json.dump(all_results, f, default=convert_to_json, indent=2)

        logger.info(f"Results saved to {results_path}")

        # Generate comparison plots
        self._plot_comparison()

    def _evaluate_ml_models(self) -> Dict[str, Dict]:
        """Evaluate ML models."""
        results = {}

        for name, model in self.ml_models.items():
            logger.info(f"\nEvaluating {name}...")

            # Predictions
            y_train_pred = model.predict(self.X_train_feat)
            y_val_pred = model.predict(self.X_val_feat)
            y_test_pred = model.predict(self.X_test_feat)

            # Metrics
            train_metrics = compute_metrics(self.y_train, y_train_pred, prefix="train")
            val_metrics = compute_metrics(self.y_val, y_val_pred, prefix="val")
            test_metrics = compute_metrics(self.y_test, y_test_pred, prefix="test")

            results[name] = {
                **train_metrics,
                **val_metrics,
                **test_metrics,
                "type": "ml",
                "training_time": model.training_time,
            }

            logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")

        return results

    def _evaluate_dl_models(self) -> Dict[str, Dict]:
        """Evaluate DL models."""
        results = {}

        for name, model in self.dl_models.items():
            logger.info(f"\nEvaluating {name}...")

            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)

            # Metrics
            train_metrics = compute_metrics(self.y_train, y_train_pred, prefix="train")
            val_metrics = compute_metrics(self.y_val, y_val_pred, prefix="val")
            test_metrics = compute_metrics(self.y_test, y_test_pred, prefix="test")

            # Evaluation from model
            eval_results = model.evaluate(self.X_test, self.y_test)

            results[name] = {
                **train_metrics,
                **val_metrics,
                **test_metrics,
                "test_loss": eval_results['loss'],
                "type": "dl",
                "training_time": model.training_time,
            }

            logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")

        return results

    def _plot_comparison(self):
        """Generate comparison plots."""
        logger.info("\nGenerating comparison plots...")

        # Compare models
        if self.results:
            compare_models(
                self.results,
                save_path=self.experiment_dir / "model_comparison.png",
            )

        # Plot confusion matrices for best models
        self._plot_confusion_matrices()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for each model."""
        from evaluate import plot_confusion_matrix

        for name, model in self.ml_models.items():
            y_pred = model.predict(self.X_test_feat)
            plot_confusion_matrix(
                self.y_test,
                y_pred,
                self.class_mapping,
                title=f"{name} - Confusion Matrix",
                save_path=self.experiment_dir / f"{name}_confusion_matrix.png",
            )

        for name, model in self.dl_models.items():
            y_pred = model.predict(self.X_test)
            plot_confusion_matrix(
                self.y_test,
                y_pred,
                self.class_mapping,
                title=f"{name} - Confusion Matrix",
                save_path=self.experiment_dir / f"{name}_confusion_matrix.png",
            )

    def save_models(self):
        """Save trained models."""
        logger.info("\nSaving models...")

        models_dir = MODELS_DIR / self.experiment_dir.name
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save ML models (using joblib)
        try:
            import joblib

            for name, model in self.ml_models.items():
                model_path = models_dir / f"{name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} to {model_path}")
        except ImportError:
            logger.warning("joblib not available, skipping ML model saving")

        # Save DL models
        for name, model in self.dl_models.items():
            model_path = models_dir / f"{name}.keras"
            model.save(str(model_path))
            logger.info(f"Saved {name} to {model_path}")

    def run_full_pipeline(
        self,
        normalize: bool = True,
        augment: bool = False,
        train_ml: bool = True,
        train_dl: bool = True,
        save_models: bool = True,
    ):
        """
        Run the complete training pipeline.

        Args:
            normalize: Whether to normalize images
            augment: Whether to augment training data
            train_ml: Whether to train ML models
            train_dl: Whether to train DL models
            save_models: Whether to save trained models
        """
        # Setup
        set_random_seeds()

        # Load data
        self.load_data()

        # Preprocess
        self.preprocess_data(normalize=normalize, augment=augment)

        # Train models
        if train_ml:
            self.train_ml_models()

        if train_dl:
            self.train_dl_models()

        # Evaluate
        self.evaluate_models()

        # Save models
        if save_models:
            self.save_models()

        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Experiment saved to: {self.experiment_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train satellite image classification models")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--no_normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--ml_only", action="store_true", help="Train only ML models")
    parser.add_argument("--dl_only", action="store_true", help="Train only DL models")
    parser.add_argument("--no_save", action="store_true", help="Don't save models")

    args = parser.parse_args()

    # Run pipeline
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        create_sample=args.create_sample,
    )

    pipeline.run_full_pipeline(
        normalize=not args.no_normalize,
        augment=args.augment,
        train_ml=not args.dl_only,
        train_dl=not args.ml_only,
        save_models=not args.no_save,
    )


if __name__ == "__main__":
    main()