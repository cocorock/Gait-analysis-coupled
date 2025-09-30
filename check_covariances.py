#!/usr/bin/env python3
"""
Script to analyze the covariance matrices and ellipse sizes in the TPGMM model.
"""

import pickle
import numpy as np
import sys

# Add TaskParameterizedGaussianMixtureModels to Python path
sys.path.append('TaskParameterizedGaussianMixtureModels')

def load_and_analyze_covariances():
    """Load the model and analyze covariance matrices."""
    
    # Load the trained model
    model_path = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls/gait_tpgmm_model_final.pkl"
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    print("=== Covariance Analysis ===")
    print(f"Feature names: {feature_names}")
    print(f"Number of components: {tpgmm._n_components}")
    print(f"Number of frames: {len(tpgmm.means_)}")
    print()
    
    # Use the first frame (FR1)
    frame_idx = 0
    means = tpgmm.means_[frame_idx]
    covariances = tpgmm.covariances_[frame_idx]
    weights = tpgmm.weights_
    
    print("=== Frame 0 Analysis ===")
    print(f"Means shape: {means.shape}")
    print(f"Covariances shape: {covariances.shape}")
    print(f"Weights: {weights}")
    print()
    
    # Analyze different feature pairs
    feature_pairs = [
        ([1, 2], "Right Ankle Position (X, Y)"),
        ([5, 6], "Left Ankle Position (X, Y)"),
        ([3, 4], "Right Ankle Velocity (X, Y)"),
        ([7, 8], "Left Ankle Velocity (X, Y)")
    ]
    
    for dims, name in feature_pairs:
        print(f"=== {name} ===")
        print(f"Dimensions: {dims}")
        
        for k in range(tpgmm._n_components):
            mean_vals = means[k, dims]
            cov_matrix = covariances[k][np.ix_(dims, dims)]
            
            # Calculate eigenvalues for ellipse sizing
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            width, height = 2 * np.sqrt(eigenvals)
            
            print(f"  Component {k+1}:")
            print(f"    Weight: {weights[k]:.4f}")
            print(f"    Mean: [{mean_vals[0]:.4f}, {mean_vals[1]:.4f}]")
            print(f"    Covariance matrix:")
            print(f"      [[{cov_matrix[0,0]:.6f}, {cov_matrix[0,1]:.6f}],")
            print(f"       [{cov_matrix[1,0]:.6f}, {cov_matrix[1,1]:.6f}]]")
            print(f"    Eigenvalues: [{eigenvals[0]:.6f}, {eigenvals[1]:.6f}]")
            print(f"    Ellipse size (width, height): ({width:.4f}, {height:.4f})")
            print(f"    Determinant: {np.linalg.det(cov_matrix):.8f}")
            print(f"    Condition number: {np.linalg.cond(cov_matrix):.2f}")
            print()
        
        print(f"  Summary for {name}:")
        all_eigenvals = []
        all_dets = []
        for k in range(tpgmm._n_components):
            cov_matrix = covariances[k][np.ix_(dims, dims)]
            eigenvals, _ = np.linalg.eigh(cov_matrix)
            all_eigenvals.extend(eigenvals)
            all_dets.append(np.linalg.det(cov_matrix))
        
        print(f"    Eigenvalue range: {np.min(all_eigenvals):.6f} to {np.max(all_eigenvals):.6f}")
        print(f"    Eigenvalue ratio (max/min): {np.max(all_eigenvals)/np.min(all_eigenvals):.2f}")
        print(f"    Ellipse size range: {2*np.sqrt(np.min(all_eigenvals)):.4f} to {2*np.sqrt(np.max(all_eigenvals)):.4f}")
        print(f"    Determinant range: {np.min(all_dets):.8f} to {np.max(all_dets):.8f}")
        print("=" * 50)

if __name__ == "__main__":
    load_and_analyze_covariances()