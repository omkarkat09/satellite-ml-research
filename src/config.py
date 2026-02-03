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

# Data settings
IMAGE_SIZE = (256, 256)
BANDS = ["B2", "B3", "B4", "B8"]  # Example: Sentinel-2 bands

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
