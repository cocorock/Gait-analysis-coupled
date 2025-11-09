#!/usr/bin/env python3
"""
Simplified Gait Analysis GMR Trajectory Recovery Script - 5D Version

This script loads a trained 5D TPGMM model and performs simplified trajectory recovery
for right ankle position and velocity data only (5D feature space).
Based on gait_gmr.py but adapted for the 5D model from final_gait_trainer_5d.py.
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
from tpgmm.gmr.gmr import GaussianMixtureRegression


def load_trained_model(model_path):
    """Load the trained 5D TPGMM model from pickle file."""
    print(f"Loading trained 5D model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("5D Model loaded successfully!")
    print(f"Number of frames: {model_data['num_frames']}")
    print(f"Number of components: {model_data['n_components']}")
    print(f"Feature dimensions: {model_data['feature_dims']}")
    print(f"Feature names: {model_data['feature_names']}")
    
    return model_data


def extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0):
    """Extract a sample trajectory from the 5D training data."""
    
    # Get the sample trajectory
    sample_trajectory = model_data['all_trajectories'][frame_idx][trajectory_idx]
    
    print(f"Sample trajectory shape: {sample_trajectory.shape}")
    print(f"Time range: {sample_trajectory[:, 0].min():.3f} to {sample_trajectory[:, 0].max():.3f}")
    
    return sample_trajectory


def predict_using_tpgmm_gmr_5d(tpgmm, time_input, feature_idx_to_predict):
    """
    Prediction using TaskPaGMMM GaussianMixtureRegression with 2 frames (5D version).
    Simplified for right ankle data only without complex frame transformations.
    """
    
    # Create GMR instance with time as input (index 0)
    gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0])
    
    num_output_features = len(feature_idx_to_predict)
    
    # Create simple translation matrices for 2 frames (FR1, FR2)
    # For 5D model, we use minimal transformations since we're only dealing with right ankle
    translation = np.zeros((2, num_output_features))
    
    # Create rotation matrices for 2 frames (identity matrices for simplicity)
    rotation_matrix = np.eye(num_output_features)[None].repeat(2, axis=0)
    
    print(f"5D GMR setup:")
    print(f"  Translation shape: {translation.shape}")
    print(f"  Rotation shape: {rotation_matrix.shape}")
    print(f"  Frames: FR1, FR2")
    
    # Fit the GMR with the 2-frame transformations
    gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
    
    # Prepare input data for prediction (time points)
    time_input_reshaped = time_input.reshape(-1, 1)
    
    # Predict using GMR
    predicted_output, predicted_covariance = gmr.predict(time_input_reshaped)
    
    return predicted_output, predicted_covariance


def apply_pca_to_features_5d(trajectories, feature_names, n_components=2):
    """Apply PCA to extract principal components from 5D features (right ankle only)."""
    
    # Separate position and velocity features (exclude time)
    pos_indices = [i for i, name in enumerate(feature_names) if 'pos' in name]
    vel_indices = [i for i, name in enumerate(feature_names) if 'vel' in name]
    
    print(f"Position feature indices: {pos_indices}")
    print(f"Velocity feature indices: {vel_indices}")
    
    results = {}
    
    # Apply PCA to position features (right ankle only)
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
    
    # Apply PCA to velocity features (right ankle only)
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


def plot_gaussian_models_5d(model_data, predicted_trajectory=None, predicted_covariance=None, save_dir="plots"):
    """Plot Gaussian models for 5D right ankle data with trajectory deviations."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Use the first frame (FR1)
    frame_idx = 0
    means = tpgmm.means_[frame_idx]  # Shape: (n_components, n_features)
    covariances = tpgmm.covariances_[frame_idx]  # Shape: (n_components, n_features, n_features)
    weights = tpgmm.weights_  # Shape: (n_components,)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for each Gaussian component
    colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle positions trajectory with uncertainty (dims 0, 1 in predicted_trajectory correspond to dims 1, 2)
        x_pos = predicted_trajectory[:, 0]
        y_pos = predicted_trajectory[:, 1]
        x_std = np.sqrt(predicted_covariance[:, 0, 0])
        y_std = np.sqrt(predicted_covariance[:, 1, 1])
        
        ax1.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
        ax1.fill_between(x_pos, y_pos - y_std, y_pos + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax1.plot(x_pos[0], y_pos[0], 'ko', markersize=8, label='Start', zorder=11)
        ax1.plot(x_pos[-1], y_pos[-1], 'ks', markersize=8, label='End', zorder=11)
    
    # Plot position Gaussians (right ankle positions - dims 1, 2)
    for k in range(tpgmm._n_components):
        mean_pos = means[k, [1, 2]]  # [right_ankle_pos_x, right_ankle_pos_y]
        cov_pos = covariances[k][np.ix_([1, 2], [1, 2])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax1.scatter(mean_pos[0], mean_pos[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, label=f'Component {k+1}', zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_pos)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_pos, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax1.add_patch(ellipse)
    
    ax1.set_title('Right Ankle Position Gaussians - 5D Version\\n(Dimensions 1, 2)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Right Ankle X Position (m)')
    ax1.set_ylabel('Right Ankle Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axis('equal')
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle velocities trajectory with uncertainty (dims 2, 3 in predicted_trajectory correspond to dims 3, 4)
        x_vel = predicted_trajectory[:, 2]
        y_vel = predicted_trajectory[:, 3]
        x_std = np.sqrt(predicted_covariance[:, 2, 2])
        y_std = np.sqrt(predicted_covariance[:, 3, 3])
        
        ax2.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
        ax2.fill_between(x_vel, y_vel - y_std, y_vel + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax2.plot(x_vel[0], y_vel[0], 'ko', markersize=8, label='Start', zorder=11)
        ax2.plot(x_vel[-1], y_vel[-1], 'ks', markersize=8, label='End', zorder=11)
    
    # Plot velocity Gaussians (right ankle velocities - dims 3, 4)
    for k in range(tpgmm._n_components):
        mean_vel = means[k, [3, 4]]  # [right_ankle_vel_x, right_ankle_vel_y]
        cov_vel = covariances[k][np.ix_([3, 4], [3, 4])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax2.scatter(mean_vel[0], mean_vel[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_vel, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax2.add_patch(ellipse)
    
    ax2.set_title('Right Ankle Velocity Gaussians - 5D Version\\n(Dimensions 3, 4)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Right Ankle X Velocity (m/s)')
    ax2.set_ylabel('Right Ankle Y Velocity (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gaussian_models_with_deviations_5d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"5D Gaussian models with deviations visualization saved to {save_dir}/tpgmm_gaussian_models_with_deviations_5d.png")


def plot_recovery_results_5d(original_trajectory, predicted_trajectory, pca_results, feature_names, save_dir="plots"):
    """Plot trajectory recovery results for 5D right ankle data."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot time series comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    time_data = original_trajectory[:, 0]
    
    # Plot key features over time (right ankle only)
    feature_plots = [
        (1, 'Right Ankle X Pos'),
        (2, 'Right Ankle Y Pos'), 
        (3, 'Right Ankle X Vel'),
        (4, 'Right Ankle Y Vel')
    ]
    
    for idx, (feat_idx, title) in enumerate(feature_plots):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].plot(time_data, original_trajectory[:, feat_idx], 'b-', label='Original', linewidth=2)
        axes[row, col].plot(time_data, predicted_trajectory[:, feat_idx-1], 'r--', label='Predicted', linewidth=2)
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Value')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # PCA plots for 5D data
    if 'position' in pca_results:
        pca_pos = pca_results['position']['pca_data']
        axes[0, 2].plot(pca_pos[:, 0], pca_pos[:, 1], 'g-', linewidth=2)
        axes[0, 2].set_title('Position PCA - 5D Version\\n(Right Ankle Only)')
        axes[0, 2].set_xlabel(f'PC1 ({pca_results["position"]["explained_variance"][0]:.2%} variance)')
        axes[0, 2].set_ylabel(f'PC2 ({pca_results["position"]["explained_variance"][1]:.2%} variance)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')
    
    if 'velocity' in pca_results:
        pca_vel = pca_results['velocity']['pca_data']
        axes[1, 2].plot(pca_vel[:, 0], pca_vel[:, 1], 'm-', linewidth=2)
        axes[1, 2].set_title('Velocity PCA - 5D Version\\n(Right Ankle Only)')
        axes[1, 2].set_xlabel(f'PC1 ({pca_results["velocity"]["explained_variance"][0]:.2%} variance)')
        axes[1, 2].set_ylabel(f'PC2 ({pca_results["velocity"]["explained_variance"][1]:.2%} variance)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gmr_recovery_results_5d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"5D Recovery results plots saved to {save_dir}/ directory")


def main():
    """Main simplified GMR trajectory recovery pipeline for 5D model."""
    print("=== Simplified Gait Analysis GMR Trajectory Recovery - 5D Version ===")
    
    # Configuration
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    model_path = os.path.join(pkls_dir, "gait_tpgmm_model_5d.pkl")
    
    # Step 1: Load trained 5D model
    print("\\nStep 1: Loading trained 5D TPGMM model...")
    model_data = load_trained_model(model_path)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Step 2: Extract sample trajectory
    print("\\nStep 2: Extracting sample trajectory...")
    sample_trajectory = extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0)
    
    # Step 3: Prepare input data (time vector)
    print("\\nStep 3: Preparing time input...")
    time_input = sample_trajectory[:, 0]  # Time column
    print(f"Time input shape: {time_input.shape}")
    
    # Step 4: Define output features to predict (exclude time) - 5D version
    output_feature_indices = list(range(1, len(feature_names)))  # [1, 2, 3, 4] for 5D
    print(f"Output feature indices: {output_feature_indices}")
    print(f"Output features: {[feature_names[i] for i in output_feature_indices]}")
    
    # Step 5: Perform GMR prediction using TaskPaGMMM (5D version)
    print("\\nStep 5: Performing 5D GMR prediction using TaskPaGMMM...")
    predicted_output, predicted_covariance = predict_using_tpgmm_gmr_5d(
        tpgmm, time_input, output_feature_indices
    )
    
    print(f"Predicted output shape: {predicted_output.shape}")
    print(f"Predicted covariance shape: {predicted_covariance.shape}")
    
    # Step 6: Apply PCA to extract principal components (5D version)
    print("\\nStep 6: Applying PCA for dimensionality reduction (5D version)...")
    pca_results = apply_pca_to_features_5d(predicted_output, feature_names[1:])  # Exclude time from PCA
    
    # Step 7: Plot Gaussian models visualization (5D version)
    print("\\nStep 7: Creating 5D Gaussian models visualization...")
    plot_gaussian_models_5d(model_data, predicted_output, predicted_covariance)
    
    # Step 8: Plot recovery results (5D version)
    print("\\nStep 8: Creating 5D recovery results visualizations...")
    plot_recovery_results_5d(sample_trajectory, predicted_output, pca_results, feature_names)
    
    # Step 9: Save recovered trajectory
    print("\\nStep 9: Saving 5D recovered trajectory...")
    recovery_data = {
        'original_trajectory': sample_trajectory,
        'predicted_trajectory': predicted_output,
        'prediction_covariance': predicted_covariance,
        'pca_results': pca_results,
        'feature_names': feature_names,
        'time_input': time_input,
        'output_feature_indices': output_feature_indices,
        'model_version': '5D'
    }
    
    recovery_save_path = os.path.join(pkls_dir, "gait_simplified_recovery_5d.pkl")
    with open(recovery_save_path, 'wb') as f:
        pickle.dump(recovery_data, f)
        
    print(f"5D Recovery data saved to: {recovery_save_path}")
    print("\\n=== 5D Simplified GMR Trajectory Recovery Complete ===")


if __name__ == "__main__":
    main()