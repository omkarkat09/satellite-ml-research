"""
Features Module
Extracts handcrafted features from satellite imagery for classical ML models.
"""

import numpy as np
from typing import List, Optional, Tuple
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel
import logging

from config import COMPUTE_TEXTURE_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract various features from satellite imagery."""

    def __init__(
        self,
        compute_spectral: bool = True,
        compute_texture: bool = COMPUTE_TEXTURE_FEATURES,
        compute_edge: bool = True,
        compute_statistical: bool = True,
    ):
        """
        Initialize feature extractor.

        Args:
            compute_spectral: Compute spectral features (mean, std per band)
            compute_texture: Compute texture features (GLCM)
            compute_edge: Compute edge features
            compute_statistical: Compute statistical features
        """
        self.compute_spectral = compute_spectral
        self.compute_texture = compute_texture
        self.compute_edge = compute_edge
        self.compute_statistical = compute_statistical

        self.feature_names = []

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of images.

        Args:
            images: Array of shape (n_samples, height, width, n_channels)

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        features_list = []

        for i, img in enumerate(images):
            if i % 100 == 0:
                logger.info(f"Extracting features from image {i}/{len(images)}")

            features = self.extract_single(img)
            features_list.append(features)

        features_matrix = np.array(features_list)
        logger.info(f"Extracted features: shape {features_matrix.shape}")

        return features_matrix

    def extract_single(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: Array of shape (height, width, n_channels)

        Returns:
            Feature vector
        """
        features = []

        # Spectral features
        if self.compute_spectral:
            spectral_features = self._extract_spectral_features(image)
            features.extend(spectral_features)

        # Texture features
        if self.compute_texture:
            texture_features = self._extract_texture_features(image)
            features.extend(texture_features)

        # Edge features
        if self.compute_edge:
            edge_features = self._extract_edge_features(image)
            features.extend(edge_features)

        # Statistical features
        if self.compute_statistical:
            stat_features = self._extract_statistical_features(image)
            features.extend(stat_features)

        return np.array(features)

    def _extract_spectral_features(self, image: np.ndarray) -> List[float]:
        """Extract spectral features (mean, std per band)."""
        features = []

        for band in range(image.shape[-1]):
            band_data = image[..., band]

            # Mean and standard deviation
            features.append(np.mean(band_data))
            features.append(np.std(band_data))

            # Percentiles
            features.append(np.percentile(band_data, 25))
            features.append(np.percentile(band_data, 50))
            features.append(np.percentile(band_data, 75))

        return features

    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using GLCM."""
        features = []

        # Convert to grayscale by averaging bands
        if len(image.shape) == 3 and image.shape[-1] > 1:
            gray = np.mean(image, axis=-1)
        else:
            gray = image.squeeze()

        # Ensure 8-bit for GLCM
        gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)

        # Compute GLCM
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        try:
            glcm = graycomatrix(
                gray,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True
            )

            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation']

            for prop in properties:
                glcm_prop = graycoprops(glcm, prop)
                features.append(np.mean(glcm_prop))
                features.append(np.std(glcm_prop))

        except Exception as e:
            logger.warning(f"Failed to compute texture features: {e}")
            # Add zeros if texture computation fails
            n_features = len(distances) * len(angles) * len(properties) * 2
            features.extend([0.0] * n_features)

        return features

    def _extract_edge_features(self, image: np.ndarray) -> List[float]:
        """Extract edge features."""
        features = []

        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[-1] > 1:
            gray = np.mean(image, axis=-1)
        else:
            gray = image.squeeze()

        # Compute Sobel edges
        edges = sobel(gray)

        # Edge statistics
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.percentile(edges, 90))
        features.append(np.percentile(edges, 95))
        features.append(np.percentile(edges, 99))

        # Edge density (percentage of strong edges)
        edge_threshold = np.percentile(edges, 75)
        edge_density = np.sum(edges > edge_threshold) / edges.size
        features.append(edge_density)

        return features

    def _extract_statistical_features(self, image: np.ndarray) -> List[float]:
        """Extract statistical features."""
        features = []

        # Flatten the image
        flat = image.flatten()

        # Basic statistics
        features.append(np.mean(flat))
        features.append(np.std(flat))
        features.append(np.min(flat))
        features.append(np.max(flat))

        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        features.append(skew(flat))
        features.append(kurtosis(flat))

        # Entropy approximation
        hist, _ = np.histogram(flat, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        features.append(entropy)

        return features


class SpectralIndexExtractor:
    """Extract spectral indices from multispectral images."""

    def __init__(self, indices: Optional[List[str]] = None):
        """
        Initialize spectral index extractor.

        Args:
            indices: List of indices to compute (e.g., ['NDVI', 'NDWI', 'NDBI'])
        """
        self.indices = indices or ['NDVI', 'NDWI', 'NDBI']

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract spectral indices from batch of images.

        Args:
            images: Array of shape (n_samples, height, width, n_channels)
                    Assuming bands: [Blue, Green, Red, NIR]

        Returns:
            Array of shape (n_samples, n_indices)
        """
        indices_list = []

        for img in images:
            index_features = self.extract_single(img)
            indices_list.append(index_features)

        return np.array(indices_list)

    def extract_single(self, image: np.ndarray) -> List[float]:
        """Extract spectral indices from single image."""
        features = []

        if image.shape[-1] < 4:
            # Not enough bands for indices
            return [0.0] * len(self.indices)

        # Extract bands (standard Sentinel-2 order: B2, B3, B4, B8)
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]
        nir = image[..., 3]

        # Safe division helper
        def safe_div(a, b):
            return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        # NDVI: (NIR - Red) / (NIR + Red)
        if 'NDVI' in self.indices:
            ndvi = safe_div(nir - red, nir + red)
            features.append(np.mean(ndvi))
            features.append(np.std(ndvi))
            features.append(np.percentile(ndvi, 25))
            features.append(np.percentile(ndvi, 75))

        # NDWI: (Green - NIR) / (Green + NIR)
        if 'NDWI' in self.indices:
            ndwi = safe_div(green - nir, green + nir)
            features.append(np.mean(ndwi))
            features.append(np.std(ndwi))
            features.append(np.percentile(ndwi, 25))
            features.append(np.percentile(ndwi, 75))

        # NDBI: (SWIR - NIR) / (SWIR + NIR)
        # Using red as proxy for SWIR if not available
        if 'NDBI' in self.indices:
            ndbi = safe_div(red - nir, red + nir)
            features.append(np.mean(ndbi))
            features.append(np.std(ndbi))
            features.append(np.percentile(ndbi, 25))
            features.append(np.percentile(ndbi, 75))

        return features


def extract_features_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    use_spectral_indices: bool = True,
    use_texture: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete feature extraction pipeline.

    Args:
        X_train: Training images
        X_val: Validation images
        X_test: Test images
        use_spectral_indices: Whether to extract spectral indices
        use_texture: Whether to extract texture features

    Returns:
        Tuple of (X_train_features, X_val_features, X_test_features)
    """
    logger.info("Starting feature extraction...")

    # Extract basic features
    feature_extractor = FeatureExtractor(
        compute_spectral=True,
        compute_texture=use_texture,
        compute_edge=True,
        compute_statistical=True,
    )

    X_train_feat = feature_extractor.extract_batch(X_train)
    X_val_feat = feature_extractor.extract_batch(X_val)
    X_test_feat = feature_extractor.extract_batch(X_test)

    # Extract spectral indices
    if use_spectral_indices:
        logger.info("Extracting spectral indices...")
        index_extractor = SpectralIndexExtractor()

        X_train_idx = index_extractor.extract_batch(X_train)
        X_val_idx = index_extractor.extract_batch(X_val)
        X_test_idx = index_extractor.extract_batch(X_test)

        # Concatenate features
        X_train_feat = np.hstack([X_train_feat, X_train_idx])
        X_val_feat = np.hstack([X_val_feat, X_val_idx])
        X_test_feat = np.hstack([X_test_feat, X_test_idx])

    logger.info(f"Feature extraction complete:")
    logger.info(f"  Train: {X_train_feat.shape}")
    logger.info(f"  Val: {X_val_feat.shape}")
    logger.info(f"  Test: {X_test_feat.shape}")

    return X_train_feat, X_val_feat, X_test_feat