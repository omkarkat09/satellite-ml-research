"""
Data Loader Module
Handles loading and preprocessing of satellite imagery datasets.
Supports various formats including GeoTIFF, PNG, JPG, and NumPy arrays.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import train_test_split
import logging

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    IMAGE_SIZE,
    BANDS,
    SUPPORTED_FORMATS,
    DEFAULT_CLASS_MAPPING,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDataset:
    """Class to handle satellite image datasets."""

    def __init__(
        self,
        data_dir: Union[str, Path] = RAW_DATA_DIR,
        class_mapping: Optional[Dict[str, int]] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            class_mapping: Mapping from class names to integer labels
            image_size: Target size for images (height, width)
        """
        self.data_dir = Path(data_dir)
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.file_paths = []
        self.classes = sorted(self.class_mapping.keys())

        logger.info(f"Initialized SatelliteDataset with classes: {self.classes}")

    def load_from_directory_structure(
        self, structure_type: str = "by_class"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from directory structure.

        Args:
            structure_type: Either "by_class" (separate folders per class) or "flat" (all in one folder)

        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        logger.info(f"Loading dataset from {self.data_dir}")

        if structure_type == "by_class":
            self._load_by_class()
        elif structure_type == "flat":
            self._load_flat()
        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

        if len(self.images) == 0:
            raise ValueError("No images found in the dataset directory")

        images = np.array(self.images)
        labels = np.array(self.labels)

        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        logger.info(f"Class distribution: {self._get_class_distribution()}")

        return images, labels

    def _load_by_class(self):
        """Load images organized by class in subdirectories."""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            image_files = self._find_image_files(class_dir)
            logger.info(f"Found {len(image_files)} images for class '{class_name}'")

            for img_path in image_files:
                try:
                    img = self._load_single_image(img_path)
                    self.images.append(img)
                    self.labels.append(self.class_mapping[class_name])
                    self.file_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

    def _load_flat(self):
        """Load images from a flat directory structure."""
        # This would require filename-based class extraction
        # For now, raise an error
        raise NotImplementedError(
            "Flat directory structure not yet implemented. Use 'by_class' structure."
        )

    def _find_image_files(self, directory: Path) -> List[Path]:
        """Find all supported image files in a directory."""
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        return image_files

    def _load_single_image(self, img_path: Path) -> np.ndarray:
        """
        Load a single image file.

        Args:
            img_path: Path to the image file

        Returns:
            Image as numpy array
        """
        suffix = img_path.suffix.lower()

        if suffix in ['.tif', '.tiff']:
            return self._load_geotiff(img_path)
        elif suffix in ['.png', '.jpg', '.jpeg']:
            return self._load_rgb_image(img_path)
        elif suffix == '.npy':
            return np.load(img_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_geotiff(self, img_path: Path) -> np.ndarray:
        """Load a GeoTIFF file using rasterio."""
        with rasterio.open(img_path) as src:
            # Read all bands
            img = src.read()

            # Handle multi-band images (Sentinel-2 style)
            if img.shape[0] >= len(BANDS):
                # Select specified bands (assuming standard band ordering)
                img = img[: len(BANDS), :, :]
            else:
                # Pad if fewer bands than expected
                padding = np.zeros((len(BANDS) - img.shape[0], *img.shape[1:]))
                img = np.vstack([img, padding])

            # Transpose from (bands, height, width) to (height, width, bands)
            img = np.transpose(img, (1, 2, 0))

        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = self._resize_image(img)

        return img

    def _load_rgb_image(self, img_path: Path) -> np.ndarray:
        """Load a standard RGB image (PNG, JPG)."""
        from PIL import Image

        img = Image.open(img_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img)

        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = self._resize_image(img)

        return img

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        from skimage.transform import resize

        # Normalize to [0, 1] for resize
        img_normalized = img.astype(np.float32) / img.max()

        # Resize
        img_resized = resize(
            img_normalized,
            (*self.image_size, img.shape[-1]),
            mode='reflect',
            anti_aliasing=True,
        )

        return img_resized

    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for label in self.labels:
            class_name = self.reverse_class_mapping[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def train_val_test_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
    ) -> Tuple:
        """
        Split dataset into train, validation, and test sets.

        Args:
            images: Image array
            labels: Label array
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
        )

        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=RANDOM_SEED,
            stratify=y_train_val,
        )

        logger.info(f"Train set: {len(X_train)} images")
        logger.info(f"Validation set: {len(X_val)} images")
        logger.info(f"Test set: {len(X_test)} images")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(
        self, X_train, X_val, X_test, y_train, y_val, y_test
    ):
        """Save processed datasets to disk."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        for name, data in splits.items():
            path = PROCESSED_DATA_DIR / f"{name}.npy"
            np.save(path, data)
            logger.info(f"Saved {name} to {path}")

    def load_processed_data(self) -> Tuple:
        """Load previously saved processed datasets."""
        splits = {}
        for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            path = PROCESSED_DATA_DIR / f"{name}.npy"
            if path.exists():
                splits[name] = np.load(path)
                logger.info(f"Loaded {name} from {path}")
            else:
                raise FileNotFoundError(f"Processed data not found: {path}")

        return (
            splits["X_train"],
            splits["X_val"],
            splits["X_test"],
            splits["y_train"],
            splits["y_val"],
            splits["y_test"],
        )


def create_sample_dataset():
    """
    Create a sample dataset structure for testing.

    This function creates dummy data to test the pipeline without
    requiring actual satellite imagery.
    """
    logger.info("Creating sample dataset for testing...")

    SAMPLE_DIR = RAW_DATA_DIR / "sample"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in DEFAULT_CLASS_MAPPING.keys():
        class_dir = SAMPLE_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        # Create 10 sample images per class
        for i in range(10):
            # Create random multispectral image (4 bands: B, G, R, NIR)
            img = np.random.rand(*IMAGE_SIZE, 4) * 255
            img = img.astype(np.uint8)

            # Save as numpy array for simplicity
            img_path = class_dir / f"sample_{i:04d}.npy"
            np.save(img_path, img)

    logger.info(f"Sample dataset created in {SAMPLE_DIR}")
    logger.info("Run: python -m src.data_loader to test loading")