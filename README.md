# Satellite ML Research Project
# Fundamental Machine Learning for Land Use Classification in Satellite Imagery

## 1. Project Overview

This project investigates fundamental machine learning and deep learning approaches for Land Use and Land Cover (LULC) classification using satellite imagery.

The primary goal is to develop a clean, reproducible, and research-oriented machine learning pipeline starting from first principles. Instead of relying on pre-trained models or complex architectures, the focus is placed on:

- Strong data preprocessing
- Classical machine learning baselines
- Careful evaluation and comparison
- Incremental development towards deep learning models

The project is designed to serve both academic research purposes and real-world ML engineering applications.

---

## 2. Motivation

Satellite image classification plays a critical role in:

- Urban planning
- Environmental monitoring
- Climate research
- Agricultural assessment
- Disaster management

Recent advances in deep learning have significantly improved classification performance. However, understanding the contribution of preprocessing, feature engineering, and classical machine learning baselines remains essential.

This project emphasizes building foundational understanding before progressing to more advanced architectures.

---

## 3. Objectives

The key objectives of this project are:

1. Develop a clean and modular ML pipeline for satellite image classification.
2. Compare classical ML methods (e.g., Logistic Regression, SVM, Random Forest) against CNN-based deep learning models.
3. Evaluate the impact of preprocessing techniques on classification performance.
4. Maintain reproducibility and clear experimental documentation.
5. Produce research-style analysis suitable for academic or industrial evaluation.

---

## 4. Project Structure
satellite-ml-research/
│
├── data/ # Raw and processed datasets
├── notebooks/ # Exploratory analysis
├── src/ # Core implementation
├── experiments/ # Experiment logs
├── results/ # Metrics and visualizations
└── report/ # Research notes and documentation

### Key Components

- `data_loader.py` → Handles structured loading of image datasets.
- `preprocessing.py` → Image normalization and standardization.
- `models_ml.py` → Classical ML baselines.
- `models_dl.py` → CNN-based models (later stage).
- `train.py` → Training pipeline.
- `evaluate.py` → Performance metrics and analysis.

---

## 5. Methodology

The development follows a staged approach:

### Stage 1 – Data Pipeline
- Image loading
- Resizing
- Normalization
- Train/validation/test split

### Stage 2 – Classical Machine Learning
- Feature extraction
- Logistic Regression
- Support Vector Machines
- Random Forest

### Stage 3 – Deep Learning
- Convolutional Neural Networks
- Regularization techniques
- Performance comparison

Each stage builds upon the previous one to ensure conceptual clarity and incremental learning.

---

## 6. Reproducibility

This project emphasizes reproducibility:

- Fixed random seeds
- Modular configuration file
- Structured experiment logging
- Clean separation between raw and processed data

All experiments are documented in the `experiments/` directory.

---

## 10. Tools and Technologies

### Core
- Python 3.8+
- NumPy, Pandas
- Scikit-learn

### Deep Learning
- TensorFlow / Keras
- PyTorch (optional support)

### Geospatial & Image Processing
- Rasterio (GeoTIFF support)
- GDAL
- OpenCV
- scikit-image
- Pillow

### Geospatial Data
- GeoPandas
- Shapely

### Visualization
- Matplotlib
- Seaborn

### Development
- Jupyter
- Black (code formatting)
- Flake8 (linting)

---

## 8. Current Status

✅ **COMPLETE** - Full end-to-end pipeline implemented:

- ✅ Data loading (multi-format support: GeoTIFF, PNG, JPG, NumPy)
- ✅ Preprocessing (normalization, augmentation, spectral indices)
- ✅ Feature extraction (spectral, texture, edge, statistical)
- ✅ Classical ML models (Logistic Regression, SVM, Random Forest, Gradient Boosting)
- ✅ Deep learning models (Simple CNN, Deep CNN, ResNet-like)
- ✅ Training pipeline with hyperparameter tuning
- ✅ Comprehensive evaluation (accuracy, F1, confusion matrices, ROC curves)
- ✅ Experiment tracking and visualization
- ✅ Model saving and loading

## 9. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo with sample data
python quick_start.py

# Or train with your own dataset
python -m src.train --data_dir path/to/your/data
```

See [USAGE.md](USAGE.md) for detailed documentation.

- Scaling to larger satellite datasets
- Transfer learning experiments
- Semi-supervised approaches
- Model interpretability analysis
- Deployment-ready inference pipeline

## 11. Future Work

- Scaling to larger satellite datasets
- Transfer learning experiments
- Semi-supervised approaches
- Model interpretability analysis
- Deployment-ready inference pipeline

---

## 12. License

This project is for academic and research purposes. Dataset licensing depends on the selected satellite dataset.

---

## Author

Omkar M. Katkar
BSc Computer Science  
Focus: Machine Learning, Data Systems, Applied AI
