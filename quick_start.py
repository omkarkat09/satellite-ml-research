#!/usr/bin/env python3
"""
Quick Start Script
Runs a complete training pipeline with sample data for demonstration.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from train import TrainingPipeline, set_random_seeds

def main():
    """Run quick start demo."""
    print("="*60)
    print("SATELLITE ML RESEARCH - QUICK START DEMO")
    print("="*60)
    print()

    # Set seeds for reproducibility
    set_random_seeds()

    # Initialize pipeline with sample data
    print("1. Initializing pipeline with sample dataset...")
    pipeline = TrainingPipeline(create_sample=True)
    print("✓ Pipeline initialized\n")

    # Load data
    print("2. Loading dataset...")
    pipeline.load_data()
    print("✓ Dataset loaded\n")

    # Preprocess
    print("3. Preprocessing images...")
    pipeline.preprocess_data(normalize=True, augment=False)
    print("✓ Preprocessing complete\n")

    # Train ML models (faster for demo)
    print("4. Training classical ML models...")
    pipeline.train_ml_models()
    print("✓ ML models trained\n")

    # Evaluate
    print("5. Evaluating models...")
    pipeline.evaluate_models()
    print("✓ Evaluation complete\n")

    # Print summary
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Results saved to: {pipeline.experiment_dir}")
    print()

    # Print model comparison
    print("MODEL COMPARISON:")
    print("-" * 60)
    for model_name, metrics in pipeline.results.items():
        acc = metrics.get('test_accuracy', 0)
        f1 = metrics.get('test_f1', 0)
        time_sec = metrics.get('training_time', 0)
        print(f"{model_name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {time_sec:.1f}s")
    print()

    print("="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. View results in experiments/exp_*/")
    print("2. Try with your own data: python -m src.train --data_dir <path>")
    print("3. Enable deep learning: python -m src.train --create_sample")
    print("4. See USAGE.md for more options")
    print()

if __name__ == "__main__":
    main()