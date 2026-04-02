# Satellite ML Research - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your dataset with one folder per class:

```
data/raw/
├── urban/
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
├── agriculture/
│   ├── image_001.tif
│   └── ...
├── forest/
│   └── ...
├── water/
│   └── ...
├── barren/
│   └── ...
└── grassland/
    └── ...
```

Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.npy`

### 3. Train Models

#### Train with Your Dataset

```bash
python -m src.train --data_dir data/raw
```

#### Train with Sample Data (for testing)

```bash
python -m src.train --create_sample
```

#### Training Options

```bash
# Train only ML models
python -m src.train --data_dir data/raw --ml_only

# Train only DL models
python -m src.train --data_dir data/raw --dl_only

# Enable data augmentation
python -m src.train --data_dir data/raw --augment

# Disable normalization
python -m src.train --data_dir data/raw --no_normalize
```

## Project Structure

```
satellite-ml-research/
├── data/
│   ├── raw/              # Raw satellite images
│   ├── processed/        # Preprocessed data (auto-generated)
│   └── sample/           # Sample dataset for testing
├── src/
│   ├── config.py         # Configuration settings
│   ├── data_loader.py    # Dataset loading and splitting
│   ├── preprocessing.py  # Normalization and augmentation
│   ├── features.py       # Feature extraction for ML
│   ├── models_ml.py      # Classical ML models
│   ├── models_dl.py      # Deep learning models
│   ├── train.py          # Training pipeline
│   └── evaluate.py       # Evaluation metrics and visualization
├── experiments/          # Experiment logs and results
│   └── exp_YYYYMMDD_HHMMSS/
│       ├── results.json
│       ├── normalization_stats.npz
│       └── *.png
├── results/
│   ├── models/           # Saved models
│   ├── figures/          # Visualization plots
│   └── metrics/          # Metric summaries
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_baseline_ml.ipynb
├── report/               # Research notes and documentation
│   └── notes.md
└── requirements.txt      # Python dependencies
```

## Available Models

### Classical ML Models

- **Logistic Regression**: Simple baseline with regularization
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree boosting

### Deep Learning Models

- **CNN Simple**: Basic 4-layer CNN
- **CNN Deep**: Deeper CNN with residual connections
- **ResNet-Like**: ResNet-inspired architecture

## Features

### Data Loading

- Support for multiple image formats (GeoTIFF, PNG, JPG, NumPy)
- Automatic train/validation/test splitting
- Class-stratified splits

### Preprocessing

- Normalization (min-max, z-score)
- Contrast stretching
- Histogram equalization
- Data augmentation (flips, rotations)

### Feature Extraction

- Spectral features (mean, std, percentiles per band)
- Texture features (GLCM)
- Edge features (Sobel)
- Statistical features (skewness, kurtosis, entropy)
- Spectral indices (NDVI, NDWI, NDBI)

### Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC (multi-class)
- Per-class metrics

## Advanced Usage

### Custom Configuration

Edit `src/config.py` to modify:

- Image size and number of channels
- Model hyperparameters
- Training settings (epochs, batch size, learning rate)
- Class mapping
- Preprocessing options

### Using in Python Scripts

```python
from src.train import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(
    data_dir="data/raw",
    create_sample=False,
)

# Load data
pipeline.load_data()

# Preprocess
pipeline.preprocess_data(normalize=True, augment=True)

# Train ML models only
pipeline.train_ml_models()

# Evaluate
pipeline.evaluate_models()

# Save models
pipeline.save_models()
```

### Loading Trained Models

```python
import joblib
from tensorflow import keras

# Load ML model
model = joblib.load("results/models/exp_20240101_120000/logistic_regression.joblib")

# Load DL model
model = keras.models.load_model("results/models/exp_20240101_120000/cnn_simple.keras")

# Make predictions
predictions = model.predict(X_test)
```

## Experiment Tracking

Each training run creates a timestamped experiment directory in `experiments/`:

```
experiments/exp_20240101_120000/
├── results.json                    # All metrics
├── normalization_stats.npz         # Preprocessing stats
├── model_comparison.png            # Model comparison plot
├── *_confusion_matrix.png          # Confusion matrices
└── training_history.png            # Training curves (DL models)
```

### Results JSON Format

```json
{
  "logistic_regression": {
    "train_accuracy": 0.95,
    "val_accuracy": 0.88,
    "test_accuracy": 0.86,
    "test_f1": 0.85,
    "training_time": 2.5,
    "type": "ml"
  },
  "cnn_simple": {
    "train_accuracy": 0.98,
    "val_accuracy": 0.92,
    "test_accuracy": 0.90,
    "test_f1": 0.89,
    "test_loss": 0.35,
    "type": "dl",
    "training_time": 45.2
  }
}
```

## Troubleshooting

### Memory Issues

- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE`
- Use fewer deep learning models (`--ml_only`)

### Slow Training

- Use fewer epochs (modify `N_EPOCHS` in `config.py`)
- Use `--ml_only` for faster classical ML baseline
- Reduce number of models in `ML_MODELS` or `DL_MODELS`

### Dataset Issues

- Ensure images are organized by class in separate folders
- Check that image formats are supported
- Verify that all images have the same number of bands

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For geospatial packages, install GDAL separately if needed
- For GPU support with TensorFlow, install `tensorflow-gpu` instead

## Next Steps

1. **Exploratory Analysis**: Use `notebooks/01_exploration.ipynb` to understand your data
2. **Baseline Models**: Train classical ML models first for quick insights
3. **Deep Learning**: Compare CNNs against baselines
4. **Hyperparameter Tuning**: Use `train_with_hyperparameter_tuning=True`
5. **Model Interpretation**: Analyze confusion matrices and per-class metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@software{satellite_ml_research,
  title={Satellite ML Research: Land Use Classification Pipeline},
  author={Katkar, Omkar M.},
  year={2026},
  url={https://github.com/omkarkat09/satellite-ml-research}
}
```

## License

This project is for academic and research purposes. Dataset licensing depends on the selected satellite dataset.

## Contact

Omkar M. Katkar
BSc Computer Science
Focus: Machine Learning, Data Systems, Applied AI

GitHub: https://github.com/omkarkat09