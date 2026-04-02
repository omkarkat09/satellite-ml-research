"""
Classical Machine Learning Models
Implements baseline ML models for satellite image classification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
import time

from config import ML_MODELS, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModel:
    """Base class for classical ML models."""

    def __init__(self, model_name: str, random_state: int = RANDOM_SEED):
        """
        Initialize ML model.

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        logger.info(f"{self.model_name} trained in {self.training_time:.2f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            return np.eye(n_classes)[predictions]


class LogisticRegressionModel(MLModel):
    """Logistic Regression classifier."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
    ):
        super().__init__("Logistic Regression", random_state)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            ))
        ])


class SVMModel(MLModel):
    """Support Vector Machine classifier."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        C: float = 1.0,
        kernel: str = 'rbf',
        class_weight: str = 'balanced',
    ):
        super().__init__("SVM", random_state)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                C=C,
                kernel=kernel,
                class_weight=class_weight,
                probability=True,  # Enable predict_proba
                random_state=random_state,
            ))
        ])


class RandomForestModel(MLModel):
    """Random Forest classifier."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        class_weight: str = 'balanced',
    ):
        super().__init__("Random Forest", random_state)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )


class GradientBoostingModel(MLModel):
    """Gradient Boosting classifier."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ):
        super().__init__("Gradient Boosting", random_state)

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )


class MLModelTrainer:
    """Train and evaluate multiple ML models."""

    def __init__(self, model_names: Optional[List[str]] = None):
        """
        Initialize model trainer.

        Args:
            model_names: List of model names to train
        """
        self.model_names = model_names or ML_MODELS
        self.models = {}
        self.results = {}

    def create_models(self) -> Dict[str, MLModel]:
        """Create model instances."""
        models = {}

        if "logistic_regression" in self.model_names:
            models["logistic_regression"] = LogisticRegressionModel()

        if "svm" in self.model_names:
            models["svm"] = SVMModel()

        if "random_forest" in self.model_names:
            models["random_forest"] = RandomForestModel()

        if "gradient_boosting" in self.model_names:
            models["gradient_boosting"] = GradientBoostingModel()

        self.models = models
        logger.info(f"Created models: {list(models.keys())}")

        return models

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, MLModel]:
        """
        Train all models.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.create_models()

        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                logger.info(f"✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"✗ {name} failed: {e}")

        return self.models

    def train_with_hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted',
    ) -> Dict[str, MLModel]:
        """
        Train models with hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.create_models()

        param_grids = {
            "logistic_regression": {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__max_iter': [1000, 2000],
            },
            "svm": {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['rbf', 'linear'],
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            },
            "gradient_boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            },
        }

        for name, model in self.models.items():
            logger.info(f"Tuning hyperparameters for {name}...")

            try:
                # For Pipeline models, access the classifier
                estimator = model.model

                if name in param_grids:
                    grid_search = GridSearchCV(
                        estimator,
                        param_grids[name],
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1,
                    )

                    grid_search.fit(X_train, y_train)

                    model.model = grid_search.best_estimator_
                    logger.info(f"Best params for {name}: {grid_search.best_params_}")
                    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

            except Exception as e:
                logger.error(f"Hyperparameter tuning failed for {name}: {e}")

        return self.models

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted',
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on models.

        Args:
            X: Features
            y: Labels
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Dictionary of CV scores
        """
        if not self.models:
            raise ValueError("No models created. Call create_models() first.")

        cv_scores = {}

        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")

            try:
                scores = cross_val_score(
                    model.model,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                )

                cv_scores[name] = scores
                logger.info(f"{name} CV scores: {scores}")
                logger.info(f"{name} mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

            except Exception as e:
                logger.error(f"Cross-validation failed for {name}: {e}")

        return cv_scores


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_hyperparameter_tuning: bool = False,
) -> Dict[str, MLModel]:
    """
    Train baseline ML models.

    Args:
        X_train: Training features
        y_train: Training labels
        use_hyperparameter_tuning: Whether to tune hyperparameters

    Returns:
        Dictionary of trained models
    """
    trainer = MLModelTrainer()

    if use_hyperparameter_tuning:
        trainer.train_with_hyperparameter_tuning(X_train, y_train)
    else:
        trainer.train_all(X_train, y_train)

    return trainer.models


def get_model_predictions(
    models: Dict[str, MLModel],
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Get predictions from all models.

    Args:
        models: Dictionary of trained models
        X: Input features

    Returns:
        Dictionary of predictions
    """
    predictions = {}

    for name, model in models.items():
        if model.is_fitted:
            predictions[name] = model.predict(X)

    return predictions


def get_model_probabilities(
    models: Dict[str, MLModel],
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Get prediction probabilities from all models.

    Args:
        models: Dictionary of trained models
        X: Input features

    Returns:
        Dictionary of probabilities
    """
    probabilities = {}

    for name, model in models.items():
        if model.is_fitted:
            probabilities[name] = model.predict_proba(X)

    return probabilities