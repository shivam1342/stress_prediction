"""
Test script to verify models work correctly
"""

import numpy as np
from data_processor import WESADDataProcessor
from models import StressPredictionModels

def test_synthetic_data():
    """Test models with synthetic data"""
    print("=" * 60)
    print("Testing Stress Prediction Models")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Generating synthetic data...")
    processor = WESADDataProcessor()
    X, y = processor.create_sample_data(n_samples=500)
    print(f"   ✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   ✓ Stress cases: {np.sum(y == 1)}, No stress: {np.sum(y == 0)}")
    
    # Train models
    print("\n2. Training models...")
    models = StressPredictionModels()
    models.train_models(X, y, test_size=0.2)
    print("   ✓ All models trained successfully")
    
    # Evaluate models
    print("\n3. Evaluating models...")
    evaluations = models.evaluate_models()
    
    print("\n4. Model Performance:")
    print("-" * 60)
    for model_name, eval_data in evaluations.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {eval_data['accuracy']:.4f}")
        print(f"  Precision: {eval_data['precision']:.4f}")
        print(f"  Recall:    {eval_data['recall']:.4f}")
        print(f"  F1 Score:  {eval_data['f1_score']:.4f}")
    
    # Test prediction
    print("\n5. Testing predictions...")
    test_samples = X[:10]
    predictions, probabilities = models.predict(test_samples, model_name='RandomForest')
    print(f"   ✓ Predictions made: {predictions}")
    print(f"   ✓ Probabilities: {probabilities}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_synthetic_data()

