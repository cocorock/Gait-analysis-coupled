#!/usr/bin/env python3
"""
Theoretical Gait Analysis TP-GMR Script

This script implements the proper Task-Parameterized Gaussian Mixture Regression
following equations (5) and (6) from Calinon-ISRR2015.pdf chapter 5.1.

The implementation follows the theoretical framework where multiple frames are combined
using Gaussian products as described in the TP-GMM methodology.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import os
import sys

# Add TaskPaGMMM to Python path
sys.path.append('TaskPaGMMM')


def load_trained_model(model_path):
    """Load the trained TPGMM model from pickle file."""
    print(f"Loading trained model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("Model loaded successfully!")
    print(f"Number of frames: {model_data['num_frames']}")
    print(f"Number of components: {model_data['n_components']}")
    print(f"Feature dimensions: {model_data['feature_dims']}")
    print(f"Feature names: {model_data['feature_names']}")
    
    return model_data


def extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0):
    """Extract a sample trajectory from the training data."""
    
    # Get the sample trajectory
    sample_trajectory = model_data['all_trajectories'][frame_idx][trajectory_idx]
    
    print(f"Sample trajectory shape: {sample_trajectory.shape}")
    print(f"Time range: {sample_trajectory[:, 0].min():.3f} to {sample_trajectory[:, 0].max():.3f}")
    
    return sample_trajectory


def create_frame_transformations(model_data, time_input):
    """
    Create frame transformations {A_t,j, b_t,j} for each time step.
    
    For gait analysis, we use identity transformations as a starting point,
    but this can be extended to include actual coordinate frame transformations.
    """
    num_frames = model_data['num_frames']
    num_features = model_data['feature_dims']
    num_time_steps = len(time_input)
    
    # Initialize frame transformations
    # A_t,j: transformation matrices (identity for simplicity)
    # b_t,j: translation vectors (zero for simplicity)
    
    A_frames = np.zeros((num_time_steps, num_frames, num_features, num_features))
    b_frames = np.zeros((num_time_steps, num_frames, num_features))
    
    for t in range(num_time_steps):
        for j in range(num_frames):
            # Use identity transformation (can be extended for actual frame transformations)
            A_frames[t, j] = np.eye(num_features)
            b_frames[t, j] = np.zeros(num_features)
    
    return A_frames, b_frames


def compute_gaussian_product(means_list, covariances_list):
    """
    Compute the product of multiple Gaussians using equation (6) from Calinon paper.
    
    Σ̂_t,i = (Σ_{j=1}^P Σ̂^{(j)}_t,i^{-1})^{-1}
    ξ̂_t,i = Σ̂_t,i Σ_{j=1}^P Σ̂^{(j)}_t,i^{-1} ξ̂^{(j)}_t,i
    
    Args:
        means_list: List of means from different frames
        covariances_list: List of covariances from different frames
    
    Returns:
        combined_mean, combined_covariance
    """
    
    # Compute precision matrices (inverse covariances)
    precision_sum = np.zeros_like(covariances_list[0])
    weighted_mean_sum = np.zeros_like(means_list[0])
    
    for mean, cov in zip(means_list, covariances_list):
        # Add small regularization for numerical stability
        regularized_cov = cov + 1e-6 * np.eye(cov.shape[0])
        precision = np.linalg.inv(regularized_cov)
        
        precision_sum += precision
        weighted_mean_sum += precision @ mean
    
    # Combined covariance (inverse of sum of precisions)
    combined_covariance = np.linalg.inv(precision_sum)
    
    # Combined mean
    combined_mean = combined_covariance @ weighted_mean_sum
    
    return combined_mean, combined_covariance


def predict_using_tpgmm_theory(tpgmm, time_input, feature_idx_to_predict, A_frames, b_frames):
    """
    Theoretical TP-GMR prediction following equations (5) and (6) from Calinon paper.
    
    This implements the complete TP-GMM framework with multiple frames and 
    Gaussian products as described in the theoretical paper.
    """
    
    num_frames = A_frames.shape[1]
    n_components = tpgmm._n_components
    n_points = len(time_input)
    n_output_features = len(feature_idx_to_predict)
    
    predicted_output = np.zeros((n_points, n_output_features))
    predicted_covariance = np.zeros((n_points, n_output_features, n_output_features))
    
    # Time index (input dimension)
    time_idx = 0
    
    print(f"Processing {n_points} time steps with {num_frames} frames and {n_components} components...")
    
    for t_idx, t in enumerate(time_input):
        if t_idx % 10 == 0:
            print(f"Processing time step {t_idx}/{n_points}")
        
        # Step 1: Compute responsibilities across all frames and components
        all_responsibilities = np.zeros((num_frames, n_components))
        
        for j in range(num_frames):
            means = tpgmm.means_[j]  # Shape: (n_components, n_features)
            covariances = tpgmm.covariances_[j]  # Shape: (n_components, n_features, n_features)
            weights = tpgmm.weights_  # Shape: (n_components,)
            
            for k in range(n_components):
                # Transform Gaussian parameters using frame transformations (Eq. 5)
                # ξ̂^{(j)}_t,i = A_t,j μ^{(j)}_i + b_t,j
                # Σ̂^{(j)}_t,i = A_t,j Σ^{(j)}_i A^T_t,j
                
                A_t_j = A_frames[t_idx, j]
                b_t_j = b_frames[t_idx, j]
                
                transformed_mean = A_t_j @ means[k] + b_t_j
                transformed_cov = A_t_j @ covariances[k] @ A_t_j.T
                
                # Evaluate Gaussian at current time
                mean_time = transformed_mean[time_idx]
                var_time = transformed_cov[time_idx, time_idx]
                
                # Gaussian probability at time t
                prob = np.exp(-0.5 * ((t - mean_time) ** 2) / var_time) / np.sqrt(2 * np.pi * var_time)
                all_responsibilities[j, k] = weights[k] * prob
        
        # Normalize responsibilities
        total_responsibility = np.sum(all_responsibilities) + 1e-10
        all_responsibilities /= total_responsibility
        
        # Step 2: For each component, combine predictions from all frames using Gaussian products
        component_predictions = []
        component_covariances = []
        component_weights = []
        
        for k in range(n_components):
            # Collect frame-specific predictions for component k
            frame_means = []
            frame_covariances = []
            frame_weights = []
            
            for j in range(num_frames):
                means = tpgmm.means_[j]
                covariances = tpgmm.covariances_[j]
                
                # Apply frame transformations
                A_t_j = A_frames[t_idx, j]
                b_t_j = b_frames[t_idx, j]
                
                transformed_mean = A_t_j @ means[k] + b_t_j
                transformed_cov = A_t_j @ covariances[k] @ A_t_j.T
                
                # GMR prediction for this frame and component
                mu_i = transformed_mean[time_idx]  # input mean
                mu_o = transformed_mean[feature_idx_to_predict]  # output mean
                
                sigma_ii = transformed_cov[time_idx, time_idx]  # input-input covariance
                sigma_io = transformed_cov[time_idx, feature_idx_to_predict]  # input-output covariance
                sigma_oo = transformed_cov[np.ix_(feature_idx_to_predict, feature_idx_to_predict)]  # output-output
                
                # GMR conditional prediction
                pred_mean_frame = mu_o + (sigma_io / sigma_ii) * (t - mu_i)
                pred_cov_frame = sigma_oo - np.outer(sigma_io, sigma_io) / sigma_ii
                
                frame_means.append(pred_mean_frame)
                frame_covariances.append(pred_cov_frame)
                frame_weights.append(all_responsibilities[j, k])
            
            # Skip if no significant weight
            total_weight = sum(frame_weights)
            if total_weight < 1e-10:
                continue
            
            # Apply Gaussian product across frames (Eq. 6)
            if len(frame_means) > 1:
                combined_mean, combined_cov = compute_gaussian_product(frame_means, frame_covariances)
            else:
                combined_mean = frame_means[0]
                combined_cov = frame_covariances[0]
            
            component_predictions.append(combined_mean)
            component_covariances.append(combined_cov)
            component_weights.append(total_weight)
        
        # Step 3: Combine predictions across components
        if len(component_predictions) > 0:
            component_weights = np.array(component_weights)
            component_weights /= (np.sum(component_weights) + 1e-10)
            
            final_mean = np.zeros(n_output_features)
            final_cov = np.zeros((n_output_features, n_output_features))
            
            for pred_mean, pred_cov, weight in zip(component_predictions, component_covariances, component_weights):
                final_mean += weight * pred_mean
                final_cov += weight * (pred_cov + np.outer(pred_mean, pred_mean))
            
            # Correct covariance calculation
            final_cov -= np.outer(final_mean, final_mean)
            
            predicted_output[t_idx] = final_mean
            predicted_covariance[t_idx] = final_cov
        else:
            # Fallback to zero prediction
            predicted_output[t_idx] = np.zeros(n_output_features)
            predicted_covariance[t_idx] = np.eye(n_output_features) * 1e-3
    
    return predicted_output, predicted_covariance


def apply_pca_to_features(trajectories, feature_names, n_components=2):
    """Apply PCA to extract principal components from high-dimensional features."""
    
    # Separate position and velocity features (exclude time)
    pos_indices = [i for i, name in enumerate(feature_names) if 'pos' in name]
    vel_indices = [i for i, name in enumerate(feature_names) if 'vel' in name]
    
    print(f"Position feature indices: {pos_indices}")
    print(f"Velocity feature indices: {vel_indices}")
    
    results = {}
    
    # Apply PCA to position features
    if len(pos_indices) > 0:
        pos_data = trajectories[:, pos_indices]
        pca_pos = PCA(n_components=min(n_components, pos_data.shape[1]))
        pos_pca = pca_pos.fit_transform(pos_data)
        
        results['position'] = {
            'pca_data': pos_pca,
            'pca_model': pca_pos,
            'explained_variance': pca_pos.explained_variance_ratio_,
            'feature_indices': pos_indices,
            'original_data': pos_data
        }
        
        print(f"Position PCA explained variance: {pca_pos.explained_variance_ratio_}")
    
    # Apply PCA to velocity features  
    if len(vel_indices) > 0:
        vel_data = trajectories[:, vel_indices]
        pca_vel = PCA(n_components=min(n_components, vel_data.shape[1]))
        vel_pca = pca_vel.fit_transform(vel_data)
        
        results['velocity'] = {
            'pca_data': vel_pca,
            'pca_model': pca_vel,
            'explained_variance': pca_vel.explained_variance_ratio_,
            'feature_indices': vel_indices,
            'original_data': vel_data
        }
        
        print(f"Velocity PCA explained variance: {pca_vel.explained_variance_ratio_}")
    
    return results


def plot_theoretical_comparison(model_data, predicted_trajectory, predicted_covariance, save_dir="plots"):
    """Plot theoretical TP-GMR results with multi-frame analysis."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    num_frames = model_data['num_frames']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define colors for each frame
    frame_colors = plt.cm.Set1(np.linspace(0, 1, num_frames))
    component_colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
    
    # Plot 1: Multi-frame Gaussian components (Left ankle positions)
    ax = axes[0, 0]
    for j in range(num_frames):
        means = tpgmm.means_[j]
        covariances = tpgmm.covariances_[j]
        weights = tpgmm.weights_
        
        for k in range(tpgmm._n_components):
            mean_pos = means[k, [5, 6]]  # [left_ankle_pos_x, left_ankle_pos_y]
            cov_pos = covariances[k][np.ix_([5, 6], [5, 6])]
            
            # Plot with frame-specific transparency
            alpha = 0.3 + 0.4 * (j / max(1, num_frames - 1))
            
            # Plot covariance ellipse
            eigenvals, eigenvecs = np.linalg.eigh(cov_pos)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals)
            
            ellipse = Ellipse(mean_pos, width, height, angle=angle,
                            facecolor=frame_colors[j], alpha=alpha, 
                            edgecolor=frame_colors[j], linewidth=2,
                            label=f'Frame {j+1}' if k == 0 else "")
            ax.add_patch(ellipse)
    
    # Plot recovered trajectory
    if predicted_trajectory is not None:
        x_pos = predicted_trajectory[:, 4]  # Left ankle X
        y_pos = predicted_trajectory[:, 5]  # Left ankle Y
        ax.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label='TP-GMR Recovery', zorder=10)
        ax.plot(x_pos[0], y_pos[0], 'ko', markersize=8, label='Start', zorder=11)
        ax.plot(x_pos[-1], y_pos[-1], 'ks', markersize=8, label='End', zorder=11)
    
    ax.set_title('Multi-Frame Left Ankle Position\n(Theoretical TP-GMM)', fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Multi-frame Gaussian components (Right ankle positions)
    ax = axes[0, 1]
    for j in range(num_frames):
        means = tpgmm.means_[j]
        covariances = tpgmm.covariances_[j]
        
        for k in range(tpgmm._n_components):
            mean_pos = means[k, [1, 2]]  # [right_ankle_pos_x, right_ankle_pos_y]
            cov_pos = covariances[k][np.ix_([1, 2], [1, 2])]
            
            alpha = 0.3 + 0.4 * (j / max(1, num_frames - 1))
            
            eigenvals, eigenvecs = np.linalg.eigh(cov_pos)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(eigenvals)
            
            ellipse = Ellipse(mean_pos, width, height, angle=angle,
                            facecolor=frame_colors[j], alpha=alpha,
                            edgecolor=frame_colors[j], linewidth=2)
            ax.add_patch(ellipse)
    
    if predicted_trajectory is not None:
        x_pos = predicted_trajectory[:, 0]  # Right ankle X
        y_pos = predicted_trajectory[:, 1]  # Right ankle Y
        ax.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label='TP-GMR Recovery', zorder=10)
        ax.plot(x_pos[0], y_pos[0], 'ko', markersize=8, label='Start', zorder=11)
        ax.plot(x_pos[-1], y_pos[-1], 'ks', markersize=8, label='End', zorder=11)
    
    ax.set_title('Multi-Frame Right Ankle Position\n(Theoretical TP-GMM)', fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Uncertainty visualization
    ax = axes[0, 2]
    if predicted_trajectory is not None and predicted_covariance is not None:
        time_steps = np.arange(len(predicted_trajectory))
        
        # Plot uncertainty for different features
        features = [(0, 'Right Ankle X'), (1, 'Right Ankle Y'), (4, 'Left Ankle X'), (5, 'Left Ankle Y')]
        colors_uncert = ['red', 'blue', 'green', 'orange']
        
        for i, (feat_idx, feat_name) in enumerate(features):
            uncertainty = np.sqrt(predicted_covariance[:, feat_idx, feat_idx])
            ax.plot(time_steps, uncertainty, color=colors_uncert[i], 
                   linewidth=2, label=f'{feat_name} Uncertainty')
        
        ax.set_title('Prediction Uncertainty Over Time\n(Theoretical TP-GMM)', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Velocity components
    for ax_idx, (vel_dims, title) in enumerate([
        ([6, 7], 'Left Ankle Velocity'),
        ([2, 3], 'Right Ankle Velocity'),
        (None, 'Frame Weights Analysis')
    ]):
        ax = axes[1, ax_idx]
        
        if vel_dims is not None:
            # Plot velocity Gaussians
            for j in range(num_frames):
                means = tpgmm.means_[j]
                covariances = tpgmm.covariances_[j]
                
                for k in range(tpgmm._n_components):
                    mean_vel = means[k, vel_dims]
                    cov_vel = covariances[k][np.ix_(vel_dims, vel_dims)]
                    
                    alpha = 0.3 + 0.4 * (j / max(1, num_frames - 1))
                    
                    eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigenvals)
                    
                    ellipse = Ellipse(mean_vel, width, height, angle=angle,
                                    facecolor=frame_colors[j], alpha=alpha,
                                    edgecolor=frame_colors[j], linewidth=2)
                    ax.add_patch(ellipse)
            
            if predicted_trajectory is not None:
                x_vel = predicted_trajectory[:, vel_dims[0]]
                y_vel = predicted_trajectory[:, vel_dims[1]]
                ax.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label='TP-GMR Recovery')
                ax.plot(x_vel[0], y_vel[0], 'ko', markersize=8, label='Start')
                ax.plot(x_vel[-1], y_vel[-1], 'ks', markersize=8, label='End')
            
            ax.set_title(f'{title}\n(Theoretical TP-GMM)', fontweight='bold')
            ax.set_xlabel('X Velocity (m/s)')
            ax.set_ylabel('Y Velocity (m/s)')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        else:
            # Frame weights analysis placeholder
            ax.text(0.5, 0.5, f'Multi-Frame Analysis\n{num_frames} frames combined\nusing Gaussian products\n(Equations 5 & 6)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax.set_title('Theoretical Framework Info', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/theoretical_tpgmm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Theoretical TP-GMM analysis saved to {save_dir}/theoretical_tpgmm_analysis.png")


def main():
    """Main theoretical TP-GMR trajectory recovery pipeline."""
    print("=== Theoretical Gait Analysis TP-GMR (Equations 5 & 6) ===")
    
    # Configuration
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    model_path = os.path.join(pkls_dir, "gait_tpgmm_model_final.pkl")
    
    # Step 1: Load trained model
    print("\nStep 1: Loading trained TPGMM model...")
    model_data = load_trained_model(model_path)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Step 2: Extract sample trajectory
    print("\nStep 2: Extracting sample trajectory...")
    sample_trajectory = extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0)
    
    # Step 3: Prepare input data (time vector)
    print("\nStep 3: Preparing time input...")
    time_input = sample_trajectory[:, 0]  # Time column
    print(f"Time input shape: {time_input.shape}")
    
    # Step 4: Create frame transformations
    print("\nStep 4: Creating frame transformations...")
    A_frames, b_frames = create_frame_transformations(model_data, time_input)
    print(f"Frame transformations shape: A={A_frames.shape}, b={b_frames.shape}")
    
    # Step 5: Define output features to predict (exclude time)
    output_feature_indices = list(range(1, len(feature_names)))  # [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Output feature indices: {output_feature_indices}")
    print(f"Output features: {[feature_names[i] for i in output_feature_indices]}")
    
    # Step 6: Perform theoretical TP-GMR prediction
    print("\nStep 6: Performing theoretical TP-GMR prediction (Equations 5 & 6)...")
    predicted_output, predicted_covariance = predict_using_tpgmm_theory(
        tpgmm, time_input, output_feature_indices, A_frames, b_frames
    )
    
    print(f"Predicted output shape: {predicted_output.shape}")
    print(f"Predicted covariance shape: {predicted_covariance.shape}")
    
    # Step 7: Apply PCA to extract principal components
    print("\nStep 7: Applying PCA for dimensionality reduction...")
    pca_results = apply_pca_to_features(predicted_output, feature_names[1:])  # Exclude time from PCA
    
    # Step 8: Plot theoretical analysis
    print("\nStep 8: Creating theoretical TP-GMM visualization...")
    plot_theoretical_comparison(model_data, predicted_output, predicted_covariance)
    
    # Step 9: Save theoretical recovery results
    print("\nStep 9: Saving theoretical recovery data...")
    recovery_data = {
        'original_trajectory': sample_trajectory,
        'predicted_trajectory': predicted_output,
        'prediction_covariance': predicted_covariance,
        'pca_results': pca_results,
        'feature_names': feature_names,
        'time_input': time_input,
        'output_feature_indices': output_feature_indices,
        'frame_transformations': {
            'A_frames': A_frames,
            'b_frames': b_frames
        },
        'method': 'theoretical_tpgmr_equations_5_6'
    }
    
    recovery_save_path = os.path.join(pkls_dir, "gait_theoretical_recovery.pkl")
    with open(recovery_save_path, 'wb') as f:
        pickle.dump(recovery_data, f)
        
    print(f"Theoretical recovery data saved to: {recovery_save_path}")
    print("\n=== Theoretical TP-GMR Recovery Complete ===")
    print("\nKey theoretical improvements:")
    print("✓ Multi-frame Gaussian transformations (Equation 5)")
    print("✓ Gaussian product combinations (Equation 6)")
    print("✓ Proper precision matrix calculations")
    print("✓ Frame-aware responsibility computation")


if __name__ == "__main__":
    main()