"""
Deep Learning Models
Implements CNN-based models for satellite image classification.
Supports both TensorFlow/Keras and PyTorch backends.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import time

from config import (
    DL_MODELS,
    IMAGE_SIZE,
    N_CHANNELS,
    CNN_FILTERS,
    KERNEL_SIZE,
    DROPOUT_RATE,
    RANDOM_SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    N_EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TensorFlow first, then PyTorch
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow/Keras backend available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch backend available")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


# ==================== TensorFlow/Keras Models ====================

if TENSORFLOW_AVAILABLE:
    class SimpleCNN(keras.Model):
        """Simple CNN for satellite image classification."""

        def __init__(
            self,
            n_classes: int,
            input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, N_CHANNELS),
            filters: List[int] = CNN_FILTERS,
            dropout_rate: float = DROPOUT_RATE,
        ):
            super().__init__()

            self.n_classes = n_classes

            # Feature extraction layers
            self.conv_blocks = []

            in_channels = input_shape[-1]
            for n_filters in filters:
                self.conv_blocks.append([
                    layers.Conv2D(n_filters, KERNEL_SIZE, padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.MaxPooling2D((2, 2)),
                    layers.Dropout(dropout_rate),
                ])

            # Global pooling and classifier
            self.global_pool = layers.GlobalAveragePooling2D()
            self.dense1 = layers.Dense(512, activation='relu')
            self.dropout = layers.Dropout(dropout_rate)
            self.dense2 = layers.Dense(n_classes, activation='softmax')

        def call(self, inputs, training=False):
            x = inputs

            # Apply convolutional blocks
            for block in self.conv_blocks:
                for layer in block:
                    x = layer(x)

            # Classification head
            x = self.global_pool(x)
            x = self.dense1(x)
            x = self.dropout(x, training=training)
            x = self.dense2(x)

            return x


    class DeepCNN(keras.Model):
        """Deeper CNN with residual connections."""

        def __init__(
            self,
            n_classes: int,
            input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, N_CHANNELS),
            filters: List[int] = [64, 128, 256, 512],
            dropout_rate: float = DROPOUT_RATE,
        ):
            super().__init__()

            self.n_classes = n_classes

            # Initial convolution
            self.conv_init = layers.Conv2D(filters[0], KERNEL_SIZE, padding='same')
            self.bn_init = layers.BatchNormalization()
            self.relu_init = layers.ReLU()

            # Residual blocks
            self.residual_blocks = []

            for i in range(len(filters) - 1):
                in_filters = filters[i]
                out_filters = filters[i + 1]

                block = [
                    layers.Conv2D(in_filters, KERNEL_SIZE, padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(out_filters, KERNEL_SIZE, padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                ]
                self.residual_blocks.append(block)

            # Global pooling and classifier
            self.global_pool = layers.GlobalAveragePooling2D()
            self.dense1 = layers.Dense(1024, activation='relu')
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dense2 = layers.Dense(512, activation='relu')
            self.dropout2 = layers.Dropout(dropout_rate)
            self.dense3 = layers.Dense(n_classes, activation='softmax')

        def call(self, inputs, training=False):
            x = inputs

            # Initial convolution
            x = self.conv_init(x)
            x = self.bn_init(x, training=training)
            x = self.relu_init(x)

            # Residual blocks with downsampling
            for block in self.residual_blocks:
                residual = x

                for layer in block:
                    if isinstance(layer, layers.BatchNormalization):
                        x = layer(x, training=training)
                    else:
                        x = layer(x)

                # Add residual connection
                if x.shape[-1] == residual.shape[-1]:
                    x = layers.Add()([x, residual])

                # Max pooling
                x = layers.MaxPooling2D((2, 2))(x)

            # Classification head
            x = self.global_pool(x)
            x = self.dense1(x)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            x = self.dense3(x)

            return x


    class ResNetLike(keras.Model):
        """ResNet-inspired architecture."""

        def __init__(
            self,
            n_classes: int,
            input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, N_CHANNELS),
            filters: List[int] = [64, 128, 256],
            dropout_rate: float = DROPOUT_RATE,
        ):
            super().__init__()

            self.n_classes = n_classes

            # Initial layers
            self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same')
            self.bn1 = layers.BatchNormalization()
            self.relu1 = layers.ReLU()
            self.maxpool1 = layers.MaxPooling2D(3, strides=2, padding='same')

            # Residual blocks
            self.res_blocks = []
            for n_filters in filters:
                self.res_blocks.append([
                    layers.Conv2D(n_filters, 3, padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(n_filters, 3, padding='same'),
                    layers.BatchNormalization(),
                ])

            # Final layers
            self.global_pool = layers.GlobalAveragePooling2D()
            self.dropout = layers.Dropout(dropout_rate)
            self.dense = layers.Dense(n_classes, activation='softmax')

        def call(self, inputs, training=False):
            x = inputs

            # Initial layers
            x = self.conv1(x)
            x = self.bn1(x, training=training)
            x = self.relu1(x)
            x = self.maxpool1(x)

            # Residual blocks
            for block in self.res_blocks:
                shortcut = x

                for layer in block:
                    if isinstance(layer, layers.BatchNormalization):
                        x = layer(x, training=training)
                    else:
                        x = layer(x)

                # Add shortcut
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)

                # Downsampling
                x = layers.MaxPooling2D((2, 2))(x)

            # Classification
            x = self.global_pool(x)
            x = self.dropout(x, training=training)
            x = self.dense(x)

            return x


    class DLModel:
        """Wrapper for deep learning models."""

        def __init__(
            self,
            model_name: str,
            n_classes: int,
            learning_rate: float = LEARNING_RATE,
            weight_decay: float = WEIGHT_DECAY,
        ):
            """
            Initialize DL model.

            Args:
                model_name: Name of the model architecture
                n_classes: Number of classes
                learning_rate: Learning rate
                weight_decay: Weight decay (L2 regularization)
            """
            self.model_name = model_name
            self.n_classes = n_classes
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            self.model = None
            self.history = None
            self.is_compiled = False
            self.training_time = 0

        def build_model(self):
            """Build the model architecture."""
            input_shape = (*IMAGE_SIZE, N_CHANNELS)

            if self.model_name == "cnn_simple":
                self.model = SimpleCNN(self.n_classes, input_shape)
            elif self.model_name == "cnn_deep":
                self.model = DeepCNN(self.n_classes, input_shape)
            elif self.model_name == "resnet_like":
                self.model = ResNetLike(self.n_classes, input_shape)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            logger.info(f"Built {self.model_name} model")

        def compile_model(self):
            """Compile the model."""
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                ),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )

            self.is_compiled = True
            logger.info("Model compiled")

        def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = N_EPOCHS,
            batch_size: int = BATCH_SIZE,
            early_stopping: bool = True,
            verbose: int = 1,
        ):
            """
            Train the model.

            Args:
                X_train: Training images
                y_train: Training labels
                X_val: Validation images
                y_val: Validation labels
                epochs: Number of epochs
                batch_size: Batch size
                early_stopping: Whether to use early stopping
                verbose: Verbosity level
            """
            if self.model is None:
                self.build_model()

            if not self.is_compiled:
                self.compile_model()

            # Callbacks
            callbacks_list = []

            if early_stopping and X_val is not None:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=EARLY_STOPPING_PATIENCE,
                        restore_best_weights=True,
                    )
                )

            callbacks_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                )
            )

            # Training
            validation_data = (X_val, y_val) if X_val is not None else None

            logger.info(f"Training {self.model_name}...")
            start_time = time.time()

            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose,
            )

            self.training_time = time.time() - start_time
            logger.info(f"{self.model_name} trained in {self.training_time:.2f}s")

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions."""
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")

            y_pred_proba = self.model.predict(X)
            y_pred = np.argmax(y_pred_proba, axis=1)

            return y_pred

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Get prediction probabilities."""
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")

            return self.model.predict(X)

        def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
            """Evaluate the model."""
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")

            results = self.model.evaluate(X_test, y_test, verbose=0)

            metrics = {
                'loss': results[0],
                'accuracy': results[1],
            }

            return metrics

        def save(self, path: str):
            """Save the model."""
            if self.model is None:
                raise ValueError("No model to save")

            self.model.save(path)
            logger.info(f"Model saved to {path}")

        def load(self, path: str):
            """Load a model."""
            self.model = keras.models.load_model(path)
            self.is_compiled = True
            logger.info(f"Model loaded from {path}")


# ==================== PyTorch Models (Optional) ====================

if PYTORCH_AVAILABLE:
    class PyTorchDataset(Dataset):
        """PyTorch dataset for satellite images."""

        def __init__(self, images: np.ndarray, labels: np.ndarray):
            self.images = torch.FloatTensor(images.transpose(0, 3, 1, 2))  # NHWC -> NCHW
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    class PyTorchCNN(nn.Module):
        """Simple CNN in PyTorch."""

        def __init__(self, n_classes: int, input_channels: int = N_CHANNELS):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


# ==================== Model Factory ====================

class DLModelFactory:
    """Factory for creating deep learning models."""

    @staticmethod
    def create_model(
        model_name: str,
        n_classes: int,
        backend: str = 'tensorflow',
    ) -> DLModel:
        """
        Create a deep learning model.

        Args:
            model_name: Name of the model architecture
            n_classes: Number of classes
            backend: Backend to use ('tensorflow' or 'pytorch')

        Returns:
            Model instance
        """
        if backend == 'tensorflow':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available")
            return DLModel(model_name, n_classes)
        elif backend == 'pytorch':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            raise NotImplementedError("PyTorch models not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {backend}")


def train_dl_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    model_names: Optional[List[str]] = None,
) -> Dict[str, DLModel]:
    """
    Train deep learning models.

    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        n_classes: Number of classes
        model_names: List of model names to train

    Returns:
        Dictionary of trained models
    """
    model_names = model_names or DL_MODELS
    models = {}

    for name in model_names:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {name}")
        logger.info(f"{'='*50}")

        try:
            model = DLModelFactory.create_model(name, n_classes)
            model.fit(X_train, y_train, X_val, y_val)
            models[name] = model
            logger.info(f"✓ {name} trained successfully")
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")

    return models