"""
Preprocessing Module
Handles normalization, augmentation, and spectral index computation for satellite imagery.
"""

import numpy as np
from typing import Optional, Tuple
from skimage import exposure
import logging

from config import (
    IMAGE_SIZE,
    BANDS,
    BAND_WAVELENGTHS,
    NORMALIZE,
    CLIP_PERCENTILES,
    APPLY_HISTOGRAM_EQUALIZATION,
    COMPUTE_NDVI,
    COMPUTE_NDWI,
    COMPUTE_NDBI,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessing pipeline for satellite imagery."""

    def __init__(
        self,
        normalize: bool = NORMALIZE,
        clip_percentiles: Tuple[float, float] = CLIP_PERCENTILES,
        apply_histogram_equalization: bool = APPLY_HISTOGRAM_EQUALIZATION,
        compute_ndvi: bool = COMPUTE_NDVI,
        compute_ndwi: bool = COMPUTE_NDWI,
        compute_ndbi: bool = COMPUTE_NDBI,
    ):
        """
        Initialize preprocessor.

        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            clip_percentiles: Lower and upper percentiles for contrast clipping
            apply_histogram_equalization: Whether to apply histogram equalization
            compute_ndvi: Whether to compute NDVI index
            compute_ndwi: Whether to compute NDWI index
            compute_ndbi: Whether to compute NDBI index
        """
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles
        self.apply_histogram_equalization = apply_histogram_equalization
        self.compute_ndvi = compute_ndvi
        self.compute_ndwi = compute_ndwi
        self.compute_ndbi = compute_ndbi

    def process_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            images: Array of shape (n_samples, height, width, n_channels)

        Returns:
            Processed images
        """
        processed = np.array([self.process_single(img) for img in images])
        logger.info(f"Processed batch of {len(processed)} images")
        return processed

    def process_single(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image.

        Args:
            image: Array of shape (height, width, n_channels)

        Returns:
            Processed image
        """
        # Clip extreme values for contrast stretching
        image = self._clip_values(image)

        # Apply histogram equalization if requested
        if self.apply_histogram_equalization:
            image = self._histogram_equalization(image)

        # Normalize to [0, 1]
        if self.normalize:
            image = self._normalize(image)

        return image

    def _clip_values(self, image: np.ndarray) -> np.ndarray:
        """Clip values at specified percentiles."""
        p_low, p_high = self.clip_percentiles
        p_low_val = np.percentile(image, p_low)
        p_high_val = np.percentile(image, p_high)

        image = np.clip(image, p_low_val, p_high_val)
        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        img_min = image.min()
        img_max = image.max()

        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)

        return image

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to each band."""
        equalized = np.zeros_like(image)

        for band in range(image.shape[-1]):
            equalized[..., band] = exposure.equalize_hist(image[..., band])

        return equalized

    def compute_spectral_indices(self, images: np.ndarray) -> np.ndarray:
        """
        Compute spectral indices (NDVI, NDWI, NDBI) from multispectral images.

        Args:
            images: Array of shape (n_samples, height, width, n_channels)
                    Assuming bands are ordered as [B2, B3, B4, B8] for Sentinel-2

        Returns:
            Array of spectral indices with shape (n_samples, height, width, n_indices)
        """
        indices_list = []

        for img in images:
            indices = []

            # Extract bands
            # Standard Sentinel-2 ordering: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
            if img.shape[-1] >= 4:
                red = img[..., 2]  # B4
                nir = img[..., 3]  # B8
                green = img[..., 1]  # B3

                # NDVI: (NIR - Red) / (NIR + Red)
                if self.compute_ndvi:
                    ndvi = self._safe_divide(nir - red, nir + red)
                    indices.append(ndvi[..., np.newaxis])

                # NDWI: (Green - NIR) / (Green + NIR)
                if self.compute_ndwi:
                    ndwi = self._safe_divide(green - nir, green + nir)
                    indices.append(ndwi[..., np.newaxis])

                # NDBI: (SWIR - NIR) / (SWIR + NIR)
                # Note: SWIR not in standard 4-band setup, using red as proxy
                if self.compute_ndbi:
                    ndbi = self._safe_divide(red - nir, red + nir)
                    indices.append(ndbi[..., np.newaxis])

            if indices:
                img_indices = np.concatenate(indices, axis=-1)
                indices_list.append(img_indices)
            else:
                indices_list.append(np.zeros((*img.shape[:2], 1)))

        return np.array(indices_list)

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division that handles division by zero."""
        # Add small epsilon to avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        return numerator / denominator

    def augment_images(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to training images.

        Args:
            images: Array of images
            labels: Array of labels

        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []

        for img, label in zip(images, labels):
            # Original image
            augmented_images.append(img)
            augmented_labels.append(label)

            # Horizontal flip
            flipped_h = np.fliplr(img)
            augmented_images.append(flipped_h)
            augmented_labels.append(label)

            # Vertical flip
            flipped_v = np.flipud(img)
            augmented_images.append(flipped_v)
            augmented_labels.append(label)

            # 90-degree rotation
            rotated = np.rot90(img, k=1)
            augmented_images.append(rotated)
            augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)


class NormalizationStats:
    """Compute and store normalization statistics for the dataset."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, images: np.ndarray):
        """
        Compute normalization statistics.

        Args:
            images: Array of images
        """
        logger.info("Computing normalization statistics...")
        self.mean = np.mean(images, axis=(0, 1, 2))
        self.std = np.std(images, axis=(0, 1, 2))
        self.min = np.min(images, axis=(0, 1, 2))
        self.max = np.max(images, axis=(0, 1, 2))

        logger.info(f"Mean: {self.mean}")
        logger.info(f"Std: {self.std}")
        logger.info(f"Min: {self.min}")
        logger.info(f"Max: {self.max}")

    def standardize(self, images: np.ndarray) -> np.ndarray:
        """
        Standardize images using z-score normalization.

        Args:
            images: Array of images

        Returns:
            Standardized images
        """
        if self.mean is None or self.std is None:
            raise ValueError("NormalizationStats not fitted. Call fit() first.")

        standardized = (images - self.mean) / (self.std + 1e-10)
        return standardized

    def save(self, path: str):
        """Save normalization statistics."""
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            min=self.min,
            max=self.max,
        )
        logger.info(f"Saved normalization stats to {path}")

    def load(self, path: str):
        """Load normalization statistics."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.min = data['min']
        self.max = data['max']
        logger.info(f"Loaded normalization stats from {path}")


def preprocess_pipeline(
    X_train,
    X_val,
    X_test,
    normalize: bool = True,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[NormalizationStats]]:
    """
    Complete preprocessing pipeline.

    Args:
        X_train: Training images
        X_val: Validation images
        X_test: Test images
        normalize: Whether to normalize images
        augment: Whether to augment training data

    Returns:
        Tuple of (X_train_processed, X_val_processed, X_test_processed, norm_stats)
    """
    preprocessor = Preprocessor(normalize=normalize)

    # Process all sets
    X_train_proc = preprocessor.process_batch(X_train)
    X_val_proc = preprocessor.process_batch(X_val)
    X_test_proc = preprocessor.process_batch(X_test)

    # Optionally compute standardization stats
    norm_stats = None
    if normalize:
        norm_stats = NormalizationStats()
        norm_stats.fit(X_train_proc)
        X_train_proc = norm_stats.standardize(X_train_proc)
        X_val_proc = norm_stats.standardize(X_val_proc)
        X_test_proc = norm_stats.standardize(X_test_proc)

    # Optionally augment training data
    if augment:
        X_train_proc, _ = preprocessor.augment_images(X_train_proc, np.zeros(len(X_train_proc)))

    return X_train_proc, X_val_proc, X_test_proc, norm_stats