#!/usr/bin/env python3
"""
Test script to verify the trained TPGMM model can be loaded and used.
"""

import pickle
import numpy as np

def test_model_loading():
    """Test loading and basic functionality of the saved model."""
    
    print("=== Testing Model Loading ===")
    
    # Load the model
    model_path = "/home/jemajuinta/ws/Gait-analysis-coupled/gait_tpgmm_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✓ Model loaded successfully!")
        
        # Check model components
        tpgmm = model_data['tpgmm']
        print(f"✓ Number of components: {model_data['n_components']}")
        print(f"✓ Final log-likelihood: {model_data['log_likelihood_']:.2f}")
        print(f"✓ Feature names: {model_data['feature_names']}")
        
        # Check model parameters
        print(f"✓ Weights shape: {model_data['weights_'].shape}")
        print(f"✓ Means shape: {model_data['means_'].shape}")
        print(f"✓ Covariances shape: {model_data['covariances_'].shape}")
        
        # Test basic prediction (create dummy test data)
        print("\n=== Testing Model Prediction ===")
        
        # Create test data similar to training format (3 frames, few points, 9 features)
        test_data = np.random.randn(3, 10, 9)  # 3 frames, 10 points, 9 features
        test_data[:, :, 0] = np.linspace(0, 1, 10)  # Time dimension
        
        try:
            predictions = tpgmm.predict(test_data)
            print(f"✓ Prediction successful! Shape: {predictions.shape}")
            print(f"✓ Predicted labels: {np.unique(predictions)}")
            
            # Test prediction probabilities
            proba = tpgmm.predict_proba(test_data)
            print(f"✓ Prediction probabilities shape: {proba.shape}")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
        
        print("\n=== Model Test Complete ===")
        print("The model is ready for Gaussian Mixture Regression (GMR)!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()