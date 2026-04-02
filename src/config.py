import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data settings
IMAGE_SIZE = (256, 256)
BANDS = ["B2", "B3", "B4", "B8"]  # Sentinel-2 RGB+NIR bands
N_CHANNELS = len(BANDS)

# Sentinel-2 band wavelengths (nm) for spectral indices
BAND_WAVELENGTHS = {
    "B2": 490,  # Blue
    "B3": 560,  # Green
    "B4": 665,  # Red
    "B8": 842,  # NIR
}

# Dataset settings
SUPPORTED_FORMATS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.npy']
DEFAULT_CLASS_MAPPING = {
    "urban": 0,
    "agriculture": 1,
    "forest": 2,
    "water": 3,
    "barren": 4,
    "grassland": 5,
}

# Preprocessing settings
NORMALIZE = True
CLIP_PERCENTILES = (1, 99)  # For contrast stretching
APPLY_HISTOGRAM_EQUALIZATION = False

# Feature extraction settings
COMPUTE_NDVI = True
COMPUTE_NDWI = True
COMPUTE_NDBI = True
COMPUTE_TEXTURE_FEATURES = False  # GLCM features (computationally expensive)

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training settings
N_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Classical ML settings
ML_MODELS = ["logistic_regression", "svm", "random_forest", "gradient_boosting"]

# Deep Learning settings
DL_MODELS = ["cnn_simple", "cnn_deep", "resnet_like"]
DROPOUT_RATE = 0.5
CNN_FILTERS = [32, 64, 128, 256]
KERNEL_SIZE = (3, 3)

# Evaluation settings
METRICS = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]
SAVE_PREDICTIONS = True

# Reproducibility
FIXED_SEEDS = {
    "numpy": 42,
    "tensorflow": 42,
    "torch": 42,
    "random": 42,
}

# Logging
LOG_LEVEL = "INFO"
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQ = 5
