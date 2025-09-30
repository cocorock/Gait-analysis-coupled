#!/usr/bin/env python3
"""
Improved Gaussian visualization with better ellipse scaling and regularization.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import sys

# Add TaskParameterizedGaussianMixtureModels to Python path
sys.path.append('TaskParameterizedGaussianMixtureModels')

def regularize_covariance(cov_matrix, min_eigenval=1e-6, max_condition=100):
    """Regularize covariance matrix to avoid numerical issues."""
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Clamp eigenvalues to avoid too small or too large values
    eigenvals = np.maximum(eigenvals, min_eigenval)
    
    # Limit condition number
    max_eigenval = eigenvals.max()
    min_allowed = max_eigenval / max_condition
    eigenvals = np.maximum(eigenvals, min_allowed)
    
    # Reconstruct matrix
    regularized_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    return regularized_cov, eigenvals, eigenvecs

def plot_improved_gaussian_models(model_data, predicted_trajectory=None, predicted_covariance=None, save_dir="plots"):
    """Plot Gaussian models with improved ellipse scaling and regularization."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Use the first frame (FR1)
    frame_idx = 0
    means = tpgmm.means_[frame_idx]
    covariances = tpgmm.covariances_[frame_idx]
    weights = tpgmm.weights_
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for each Gaussian component
    colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
    
    # Feature pairs to plot
    feature_configs = [
        ([5, 6], ax1, "Left Ankle Position", "Left Ankle X Position (m)", "Left Ankle Y Position (m)", [4, 5]),
        ([1, 2], ax2, "Right Ankle Position", "Right Ankle X Position (m)", "Right Ankle Y Position (m)", [0, 1]),
        ([7, 8], ax3, "Left Ankle Velocity", "Left Ankle X Velocity (m/s)", "Left Ankle Y Velocity (m/s)", [6, 7]),
        ([3, 4], ax4, "Right Ankle Velocity", "Right Ankle X Velocity (m/s)", "Right Ankle Y Velocity (m/s)", [2, 3])
    ]
    
    for dims, ax, title, xlabel, ylabel, traj_dims in feature_configs:
        # Plot recovered trajectory with uncertainty if provided
        if predicted_trajectory is not None and predicted_covariance is not None:
            x_data = predicted_trajectory[:, traj_dims[0]]
            y_data = predicted_trajectory[:, traj_dims[1]]
            x_std = np.sqrt(predicted_covariance[:, traj_dims[0], traj_dims[0]])
            y_std = np.sqrt(predicted_covariance[:, traj_dims[1], traj_dims[1]])
            
            # Plot trajectory
            ax.plot(x_data, y_data, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
            
            # Plot uncertainty as shaded area (every 10th point to avoid clutter)
            step = max(1, len(x_data) // 20)
            for i in range(0, len(x_data), step):
                ellipse_unc = Ellipse((x_data[i], y_data[i]), 2*x_std[i], 2*y_std[i], 
                                    alpha=0.1, color='red', zorder=3)
                ax.add_patch(ellipse_unc)
            
            ax.plot(x_data[0], y_data[0], 'ko', markersize=8, label='Start', zorder=11)
            ax.plot(x_data[-1], y_data[-1], 'ks', markersize=8, label='End', zorder=11)
        
        # Collect all ellipse sizes for scaling
        all_sizes = []
        valid_components = []
        
        for k in range(tpgmm._n_components):
            mean_vals = means[k, dims]
            cov_matrix = covariances[k][np.ix_(dims, dims)]
            
            # Regularize covariance
            reg_cov, eigenvals, eigenvecs = regularize_covariance(cov_matrix)
            
            # Calculate ellipse parameters
            width, height = 2 * np.sqrt(eigenvals)
            all_sizes.extend([width, height])
            
            valid_components.append((k, mean_vals, reg_cov, eigenvals, eigenvecs))
        
        # Determine reasonable scaling
        median_size = np.median(all_sizes)
        max_reasonable_size = median_size * 5  # Cap at 5x median
        min_reasonable_size = median_size * 0.2  # Floor at 0.2x median
        
        print(f"{title}: Median ellipse size: {median_size:.4f}, Range: [{min_reasonable_size:.4f}, {max_reasonable_size:.4f}]")
        
        # Plot Gaussian components
        for k, mean_vals, reg_cov, eigenvals, eigenvecs in valid_components:
            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals)
            
            # Apply size constraints
            width = np.clip(width, min_reasonable_size, max_reasonable_size)
            height = np.clip(height, min_reasonable_size, max_reasonable_size)
            
            # Plot mean as point (size proportional to weight)
            point_size = 50 + 200 * weights[k]
            ax.scatter(mean_vals[0], mean_vals[1], c=[colors[k]], s=point_size, 
                      alpha=0.9, edgecolors='black', linewidth=1.5, 
                      label=f'Component {k+1}' if k < 3 else "", zorder=6)
            
            # Plot covariance as ellipse
            ellipse = Ellipse(mean_vals, width, height, angle=angle, 
                            facecolor=colors[k], alpha=0.25, edgecolor=colors[k], 
                            linewidth=2, zorder=2)
            ax.add_patch(ellipse)
            
            # Add component number annotation
            ax.annotate(f'{k+1}', (mean_vals[0], mean_vals[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='white', zorder=7)
        
        ax.set_title(f'{title} Gaussians (Regularized)\\n(Dimensions {dims[0]}, {dims[1]})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if dims == [5, 6]:  # Only show legend on first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.axis('equal')
        
        # Set reasonable axis limits based on data
        if predicted_trajectory is not None:
            x_data = predicted_trajectory[:, traj_dims[0]]
            y_data = predicted_trajectory[:, traj_dims[1]]
            x_margin = (x_data.max() - x_data.min()) * 0.1
            y_margin = (y_data.max() - y_data.min()) * 0.1
            ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
            ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improved_gaussian_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improved Gaussian models visualization saved to {save_dir}/improved_gaussian_models.png")

def main():
    """Load model and create improved visualization."""
    
    # Load the trained model
    model_path = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls/gait_tpgmm_model_final.pkl"
    recovery_path = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls/gait_simplified_recovery.pkl"
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(recovery_path, 'rb') as f:
        recovery_data = pickle.load(f)
    
    # Create improved visualization
    plot_improved_gaussian_models(
        model_data, 
        recovery_data['predicted_trajectory'],
        recovery_data['prediction_covariance']
    )

if __name__ == "__main__":
    main()