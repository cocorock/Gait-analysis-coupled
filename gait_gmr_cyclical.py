#!/usr/bin/env python3
"""
Simplified Gait Analysis GMR Trajectory Recovery Script - Cyclical Version

This script loads a trained TPGMM model (cyclical version) and performs simplified trajectory recovery
following the logic from gait_example1time.ipynb more directly.

MODIFICATION: Uses cyclical TPGMM model with continuous trajectories.
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
    """Load the trained TPGMM model from pickle file."""
    print(f"Loading trained cyclical model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("Cyclical model loaded successfully!")
    print(f"Number of frames: {model_data['num_frames']}")
    print(f"Number of components: {model_data['n_components']}")
    print(f"Feature dimensions: {model_data['feature_dims']}")
    print(f"Feature names: {model_data['feature_names']}")
    
    # Check if this is indeed a cyclical model
    if 'cyclical_modification' in model_data:
        print(f"Cyclical modification: {model_data['cyclical_modification']}")
    else:
        print("Warning: Model does not have cyclical_modification flag")
    
    return model_data


def extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0):
    """Extract a sample trajectory from the training data."""
    
    # Get the sample trajectory
    sample_trajectory = model_data['all_trajectories'][frame_idx][trajectory_idx]
    
    print(f"Sample cyclical trajectory shape: {sample_trajectory.shape}")
    print(f"Time range: {sample_trajectory[:, 0].min():.3f} to {sample_trajectory[:, 0].max():.3f}")
    print(f"First point time: {sample_trajectory[0, 0]:.3f}, Last point time: {sample_trajectory[-1, 0]:.3f}")
    
    # Verify cyclical property
    first_pos = sample_trajectory[0, 1:]  # All features except time
    last_pos = sample_trajectory[-1, 1:]
    pos_diff = np.linalg.norm(first_pos - last_pos)
    print(f"Cyclical continuity check - position difference between first and last point: {pos_diff:.6f}")
    
    return sample_trajectory


def predict_using_tpgmm_gmr(tpgmm, time_input, feature_idx_to_predict):
    """
    Prediction using TaskPaGMMM GaussianMixtureRegression with 3 frames.
    Uses proper frame transformations extracted from training data.
    """
    
    # Create GMR instance with time as input (index 0)
    gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0])
    
    # Load the extracted frame transformations
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    transform_path = os.path.join(pkls_dir, "frame_transformation_analysis.pkl")
    
    with open(transform_path, 'rb') as f:
        transform_data = pickle.load(f)
    
    translations_dict = transform_data['translations']
    
    # Extract position feature translations only (4 features: right_x, right_y, left_x, left_y)
    position_features = [1, 2, 5, 6]  # indices for position features in original data
    
    # Map feature indices to position-only indices
    pos_translation_map = {}
    for i, orig_idx in enumerate(position_features):
        if orig_idx in feature_idx_to_predict:
            pos_idx = feature_idx_to_predict.index(orig_idx)
            pos_translation_map[pos_idx] = i
    
    num_output_features = len(feature_idx_to_predict)
    
    # Create translation matrices for 3 frames (3, num_output_features)
    translation = np.zeros((3, num_output_features))
    
    # Apply extracted translations for position features only
    # NOTE: We need to invert the translations since we want reference-to-frame, not frame-to-reference
    for frame_idx, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
        frame_translation = translations_dict[frame_name]
        
        # Map position translations to output feature indices (with inversion)
        for out_idx, pos_idx in pos_translation_map.items():
            translation[frame_idx, out_idx] = -frame_translation[pos_idx]  # Invert the translation
    
    # Create rotation matrices for 3 frames (3, num_output_features, num_output_features)
    # Use identity matrices as we're not rotating the coordinate system
    rotation_matrix = np.eye(num_output_features)[None].repeat(3, axis=0)
    
    print(f"GMR setup with proper frame transformations (cyclical):")
    print(f"  Translation shape: {translation.shape}")
    print(f"  Rotation shape: {rotation_matrix.shape}")
    print(f"  Frame translations:")
    for i, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
        print(f"    {frame_name}: {translation[i]}")
    
    # Fit the GMR with the 3-frame transformations
    gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
    
    # Prepare input data for prediction (time points)
    time_input_reshaped = time_input.reshape(-1, 1)
    
    # Predict using GMR
    predicted_output, predicted_covariance = gmr.predict(time_input_reshaped)
    
    return predicted_output, predicted_covariance


def print_trajectory_points(original_trajectory, predicted_trajectory, feature_names):
    """Print first and last 5 points of trajectories for cyclical verification."""
    print("\n=== Trajectory Points Analysis ===")
    
    # Feature indices mapping
    # Original: [time, right_pos_x, right_pos_y, right_vel_x, right_vel_y, left_pos_x, left_pos_y, left_vel_x, left_vel_y]
    # Predicted: [right_pos_x, right_pos_y, right_vel_x, right_vel_y, left_pos_x, left_pos_y, left_vel_x, left_vel_y]
    
    print("Format: right_pos(x,y) left_vel(x,y) left_pos(x,y) right_vel(x,y)")
    print()
    
    print("ORIGINAL TRAJECTORY:")
    print("First 5 points:")
    for i in range(5):
        time_val = original_trajectory[i, 0]
        right_pos_x = original_trajectory[i, 1]
        right_pos_y = original_trajectory[i, 2] 
        right_vel_x = original_trajectory[i, 3]
        right_vel_y = original_trajectory[i, 4]
        left_pos_x = original_trajectory[i, 5]
        left_pos_y = original_trajectory[i, 6]
        left_vel_x = original_trajectory[i, 7]
        left_vel_y = original_trajectory[i, 8]
        
        print(f"  Point {i+1} (t={time_val:.3f}): right_pos({right_pos_x:.6f},{right_pos_y:.6f}) left_vel({left_vel_x:.6f},{left_vel_y:.6f}) left_pos({left_pos_x:.6f},{left_pos_y:.6f}) right_vel({right_vel_x:.6f},{right_vel_y:.6f})")
    
    print()
    print("Last 5 points:")
    for i in range(-5, 0):
        time_val = original_trajectory[i, 0]
        right_pos_x = original_trajectory[i, 1]
        right_pos_y = original_trajectory[i, 2]
        right_vel_x = original_trajectory[i, 3]
        right_vel_y = original_trajectory[i, 4]
        left_pos_x = original_trajectory[i, 5]
        left_pos_y = original_trajectory[i, 6]
        left_vel_x = original_trajectory[i, 7]
        left_vel_y = original_trajectory[i, 8]
        
        print(f"  Point {len(original_trajectory)+i+1} (t={time_val:.3f}): right_pos({right_pos_x:.6f},{right_pos_y:.6f}) left_vel({left_vel_x:.6f},{left_vel_y:.6f}) left_pos({left_pos_x:.6f},{left_pos_y:.6f}) right_vel({right_vel_x:.6f},{right_vel_y:.6f})")
    
    print()
    print("PREDICTED TRAJECTORY:")
    print("First 5 points:")
    for i in range(5):
        right_pos_x = predicted_trajectory[i, 0]
        right_pos_y = predicted_trajectory[i, 1]
        right_vel_x = predicted_trajectory[i, 2]
        right_vel_y = predicted_trajectory[i, 3]
        left_pos_x = predicted_trajectory[i, 4]
        left_pos_y = predicted_trajectory[i, 5]
        left_vel_x = predicted_trajectory[i, 6]
        left_vel_y = predicted_trajectory[i, 7]
        
        print(f"  Point {i+1}: right_pos({right_pos_x:.6f},{right_pos_y:.6f}) left_vel({left_vel_x:.6f},{left_vel_y:.6f}) left_pos({left_pos_x:.6f},{left_pos_y:.6f}) right_vel({right_vel_x:.6f},{right_vel_y:.6f})")
    
    print()
    print("Last 5 points:")
    for i in range(-5, 0):
        right_pos_x = predicted_trajectory[i, 0]
        right_pos_y = predicted_trajectory[i, 1]
        right_vel_x = predicted_trajectory[i, 2]
        right_vel_y = predicted_trajectory[i, 3]
        left_pos_x = predicted_trajectory[i, 4]
        left_pos_y = predicted_trajectory[i, 5]
        left_vel_x = predicted_trajectory[i, 6]
        left_vel_y = predicted_trajectory[i, 7]
        
        print(f"  Point {len(predicted_trajectory)+i+1}: right_pos({right_pos_x:.6f},{right_pos_y:.6f}) left_vel({left_vel_x:.6f},{left_vel_y:.6f}) left_pos({left_pos_x:.6f},{left_pos_y:.6f}) right_vel({right_vel_x:.6f},{right_vel_y:.6f})")
    
    print()
    
    # Compute and display differences
    print("CYCLICAL VERIFICATION:")
    orig_first = original_trajectory[0, 1:]  # Exclude time
    orig_last = original_trajectory[-1, 1:]
    pred_first = predicted_trajectory[0, :]
    pred_last = predicted_trajectory[-1, :]
    
    orig_diff = np.linalg.norm(orig_first - orig_last)
    pred_diff = np.linalg.norm(pred_first - pred_last)
    
    print(f"Original trajectory - First vs Last point difference: {orig_diff:.8f}")
    print(f"Predicted trajectory - First vs Last point difference: {pred_diff:.8f}")
    
    print("=== End Trajectory Points Analysis ===\n")


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


def plot_gaussian_models_with_deviations(model_data, predicted_trajectory=None, predicted_covariance=None, sample_trajectory=None, save_dir="plots"):
    """Plot Gaussian models with trajectory deviations shown as shaded areas."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Use the first frame (FR1)
    frame_idx = 0
    means = tpgmm.means_[frame_idx]  # Shape: (n_components, n_features)
    covariances = tpgmm.covariances_[frame_idx]  # Shape: (n_components, n_features, n_features)
    weights = tpgmm.weights_  # Shape: (n_components,)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for each Gaussian component
    colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Left ankle positions from sample (dims 5, 6 in sample_trajectory)
        sample_x = sample_trajectory[:, 5]
        sample_y = sample_trajectory[:, 6]
        ax1.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax1.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_x) > 1:
            ax1.plot([sample_x[-2], sample_x[-1]], [sample_y[-2], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Left ankle positions trajectory with uncertainty (dims 4, 5 in predicted_trajectory correspond to dims 5, 6)
        x_pos = predicted_trajectory[:, 4]
        y_pos = predicted_trajectory[:, 5]
        x_std = np.sqrt(predicted_covariance[:, 4, 4])
        y_std = np.sqrt(predicted_covariance[:, 4, 4])
        
        ax1.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery (Cyclical)', zorder=10)
        ax1.fill_between(x_pos, y_pos - y_std, y_pos + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax1.plot(x_pos[0], y_pos[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        ax1.plot(x_pos[-1], y_pos[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(x_pos) > 1:
            ax1.plot([x_pos[-2], x_pos[-1]], [y_pos[-2], y_pos[-1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Plot position Gaussians
    # Left ankle positions (dims 5, 6)
    for k in range(tpgmm._n_components):
        mean_pos = means[k, [5, 6]]  # [left_ankle_pos_x, left_ankle_pos_y]
        cov_pos = covariances[k][np.ix_([5, 6], [5, 6])]  # 2x2 covariance matrix
        
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
    
    ax1.set_title('Left Ankle Position Gaussians with Deviations (Cyclical)\n(Dimensions 5, 6)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Left Ankle X Position (m)')
    ax1.set_ylabel('Left Ankle Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Right ankle positions from sample (dims 1, 2 in sample_trajectory)
        sample_x = sample_trajectory[:, 1]
        sample_y = sample_trajectory[:, 2]
        ax2.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax2.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_x) > 1:
            ax2.plot([sample_x[-2], sample_x[-1]], [sample_y[-2], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle positions trajectory with uncertainty (dims 0, 1 in predicted_trajectory correspond to dims 1, 2)
        x_pos = predicted_trajectory[:, 0]
        y_pos = predicted_trajectory[:, 1]
        x_std = np.sqrt(predicted_covariance[:, 0, 0])
        y_std = np.sqrt(predicted_covariance[:, 1, 1])
        
        ax2.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery (Cyclical)', zorder=10)
        ax2.fill_between(x_pos, y_pos - y_std, y_pos + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax2.plot(x_pos[0], y_pos[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        ax2.plot(x_pos[-1], y_pos[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(x_pos) > 1:
            ax2.plot([x_pos[-2], x_pos[-1]], [y_pos[-2], y_pos[-1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Right ankle positions (dims 1, 2)
    for k in range(tpgmm._n_components):
        mean_pos = means[k, [1, 2]]  # [right_ankle_pos_x, right_ankle_pos_y]
        cov_pos = covariances[k][np.ix_([1, 2], [1, 2])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax2.scatter(mean_pos[0], mean_pos[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_pos)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_pos, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax2.add_patch(ellipse)
    
    ax2.set_title('Right Ankle Position Gaussians with Deviations (Cyclical)\n(Dimensions 1, 2)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Right Ankle X Position (m)')
    ax2.set_ylabel('Right Ankle Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Left ankle velocities from sample (dims 7, 8 in sample_trajectory)
        sample_x = sample_trajectory[:, 7]
        sample_y = sample_trajectory[:, 8]
        ax3.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax3.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_x) > 1:
            ax3.plot([sample_x[-2], sample_x[-1]], [sample_y[-2], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Left ankle velocities trajectory with uncertainty (dims 6, 7 in predicted_trajectory correspond to dims 7, 8)
        x_vel = predicted_trajectory[:, 6]
        y_vel = predicted_trajectory[:, 7]
        x_std = np.sqrt(predicted_covariance[:, 6, 6])
        y_std = np.sqrt(predicted_covariance[:, 7, 7])
        
        ax3.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery (Cyclical)', zorder=10)
        ax3.fill_between(x_vel, y_vel - y_std, y_vel + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax3.plot(x_vel[0], y_vel[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        ax3.plot(x_vel[-1], y_vel[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(x_vel) > 1:
            ax3.plot([x_vel[-2], x_vel[-1]], [y_vel[-2], y_vel[-1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Plot velocity Gaussians
    # Left ankle velocities (dims 7, 8)
    for k in range(tpgmm._n_components):
        mean_vel = means[k, [7, 8]]  # [left_ankle_vel_x, left_ankle_vel_y]
        cov_vel = covariances[k][np.ix_([7, 8], [7, 8])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax3.scatter(mean_vel[0], mean_vel[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_vel, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax3.add_patch(ellipse)
    
    ax3.set_title('Left Ankle Velocity Gaussians with Deviations (Cyclical)\n(Dimensions 7, 8)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Left Ankle X Velocity (m/s)')
    ax3.set_ylabel('Left Ankle Y Velocity (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Right ankle velocities from sample (dims 3, 4 in sample_trajectory)
        sample_x = sample_trajectory[:, 3]
        sample_y = sample_trajectory[:, 4]
        ax4.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax4.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_x) > 1:
            ax4.plot([sample_x[-2], sample_x[-1]], [sample_y[-2], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle velocities trajectory with uncertainty (dims 2, 3 in predicted_trajectory correspond to dims 3, 4)
        x_vel = predicted_trajectory[:, 2]
        y_vel = predicted_trajectory[:, 3]
        x_std = np.sqrt(predicted_covariance[:, 2, 2])
        y_std = np.sqrt(predicted_covariance[:, 3, 3])
        
        ax4.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label='GMR Recovery (Cyclical)', zorder=10)
        ax4.fill_between(x_vel, y_vel - y_std, y_vel + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        ax4.plot(x_vel[0], y_vel[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        ax4.plot(x_vel[-1], y_vel[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(x_vel) > 1:
            ax4.plot([x_vel[-2], x_vel[-1]], [y_vel[-2], y_vel[-1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Right ankle velocities (dims 3, 4)
    for k in range(tpgmm._n_components):
        mean_vel = means[k, [3, 4]]  # [right_ankle_vel_x, right_ankle_vel_y]
        cov_vel = covariances[k][np.ix_([3, 4], [3, 4])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax4.scatter(mean_vel[0], mean_vel[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_vel, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax4.add_patch(ellipse)
    
    ax4.set_title('Right Ankle Velocity Gaussians with Deviations (Cyclical)\n(Dimensions 3, 4)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Right Ankle X Velocity (m/s)')
    ax4.set_ylabel('Right Ankle Y Velocity (m/s)')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gaussian_models_with_deviations_cyclical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cyclical Gaussian models with deviations visualization saved to {save_dir}/tpgmm_gaussian_models_with_deviations_cyclical.png")


def plot_gaussian_models(model_data, predicted_trajectory=None, sample_trajectory=None, save_dir="plots"):
    """Plot Gaussian models with means and covariance ellipses for position and velocity dimensions."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Use the first frame (FR1)
    frame_idx = 0
    means = tpgmm.means_[frame_idx]  # Shape: (n_components, n_features)
    covariances = tpgmm.covariances_[frame_idx]  # Shape: (n_components, n_features, n_features)
    weights = tpgmm.weights_  # Shape: (n_components,)
    
    # Position dimensions: [1, 2, 5, 6] -> [right_ankle_pos_x, right_ankle_pos_y, left_ankle_pos_x, left_ankle_pos_y]
    pos_dims = [1, 2, 5, 6]
    # Velocity dimensions: [3, 4, 7, 8] -> [right_ankle_vel_x, right_ankle_vel_y, left_ankle_vel_x, left_ankle_vel_y]
    vel_dims = [3, 4, 7, 8]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for each Gaussian component
    colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Left ankle positions from sample (dims 5, 6 in sample_trajectory)
        ax1.plot(sample_trajectory[:, 5], sample_trajectory[:, 6], 'b-', 
                linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax1.plot(sample_trajectory[0, 5], sample_trajectory[0, 6], 'bo', 
                markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_trajectory) > 1:
            ax1.plot([sample_trajectory[-2, 5], sample_trajectory[-1, 5]], 
                    [sample_trajectory[-2, 6], sample_trajectory[-1, 6]], 
                    'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory if provided
    if predicted_trajectory is not None:
        # Left ankle positions trajectory (dims 4, 5 in predicted_trajectory correspond to dims 5, 6)
        ax1.plot(predicted_trajectory[:, 4], predicted_trajectory[:, 5], 'k-', 
                linewidth=3, alpha=0.7, label='GMR Recovery (Cyclical)', zorder=10)
        ax1.plot(predicted_trajectory[0, 4], predicted_trajectory[0, 5], 'ko', 
                markersize=10, label='Recovery Start', zorder=11)
        ax1.plot(predicted_trajectory[-1, 4], predicted_trajectory[-1, 5], 'ks', 
                markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(predicted_trajectory) > 1:
            ax1.plot([predicted_trajectory[-2, 4], predicted_trajectory[-1, 4]], 
                    [predicted_trajectory[-2, 5], predicted_trajectory[-1, 5]], 
                    'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Plot position Gaussians
    # Left ankle positions (dims 5, 6)
    for k in range(tpgmm._n_components):
        mean_pos = means[k, [5, 6]]  # [left_ankle_pos_x, left_ankle_pos_y]
        cov_pos = covariances[k][np.ix_([5, 6], [5, 6])]  # 2x2 covariance matrix
        
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
    
    ax1.set_title('Left Ankle Position Gaussians (Cyclical)\n(Dimensions 5, 6)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Left Ankle X Position (m)')
    ax1.set_ylabel('Left Ankle Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Right ankle positions from sample (dims 1, 2 in sample_trajectory)
        ax2.plot(sample_trajectory[:, 1], sample_trajectory[:, 2], 'b-', 
                linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax2.plot(sample_trajectory[0, 1], sample_trajectory[0, 2], 'bo', 
                markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_trajectory) > 1:
            ax2.plot([sample_trajectory[-2, 1], sample_trajectory[-1, 1]], 
                    [sample_trajectory[-2, 2], sample_trajectory[-1, 2]], 
                    'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory if provided
    if predicted_trajectory is not None:
        # Right ankle positions trajectory (dims 0, 1 in predicted_trajectory correspond to dims 1, 2)
        ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'k-', 
                linewidth=3, alpha=0.7, label='GMR Recovery (Cyclical)', zorder=10)
        ax2.plot(predicted_trajectory[0, 0], predicted_trajectory[0, 1], 'ko', 
                markersize=10, label='Recovery Start', zorder=11)
        ax2.plot(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], 'ks', 
                markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(predicted_trajectory) > 1:
            ax2.plot([predicted_trajectory[-2, 0], predicted_trajectory[-1, 0]], 
                    [predicted_trajectory[-2, 1], predicted_trajectory[-1, 1]], 
                    'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Right ankle positions (dims 1, 2)
    for k in range(tpgmm._n_components):
        mean_pos = means[k, [1, 2]]  # [right_ankle_pos_x, right_ankle_pos_y]
        cov_pos = covariances[k][np.ix_([1, 2], [1, 2])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax2.scatter(mean_pos[0], mean_pos[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_pos)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_pos, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax2.add_patch(ellipse)
    
    ax2.set_title('Right Ankle Position Gaussians (Cyclical)\n(Dimensions 1, 2)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Right Ankle X Position (m)')
    ax2.set_ylabel('Right Ankle Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Left ankle velocities from sample (dims 7, 8 in sample_trajectory)
        ax3.plot(sample_trajectory[:, 7], sample_trajectory[:, 8], 'b-', 
                linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax3.plot(sample_trajectory[0, 7], sample_trajectory[0, 8], 'bo', 
                markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_trajectory) > 1:
            ax3.plot([sample_trajectory[-2, 7], sample_trajectory[-1, 7]], 
                    [sample_trajectory[-2, 8], sample_trajectory[-1, 8]], 
                    'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory if provided
    if predicted_trajectory is not None:
        # Left ankle velocities trajectory (dims 6, 7 in predicted_trajectory correspond to dims 7, 8)
        ax3.plot(predicted_trajectory[:, 6], predicted_trajectory[:, 7], 'k-', 
                linewidth=3, alpha=0.7, label='GMR Recovery (Cyclical)', zorder=10)
        ax3.plot(predicted_trajectory[0, 6], predicted_trajectory[0, 7], 'ko', 
                markersize=10, label='Recovery Start', zorder=11)
        ax3.plot(predicted_trajectory[-1, 6], predicted_trajectory[-1, 7], 'ks', 
                markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(predicted_trajectory) > 1:
            ax3.plot([predicted_trajectory[-2, 6], predicted_trajectory[-1, 6]], 
                    [predicted_trajectory[-2, 7], predicted_trajectory[-1, 7]], 
                    'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Plot velocity Gaussians
    # Left ankle velocities (dims 7, 8)
    for k in range(tpgmm._n_components):
        mean_vel = means[k, [7, 8]]  # [left_ankle_vel_x, left_ankle_vel_y]
        cov_vel = covariances[k][np.ix_([7, 8], [7, 8])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax3.scatter(mean_vel[0], mean_vel[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_vel, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax3.add_patch(ellipse)
    
    ax3.set_title('Left Ankle Velocity Gaussians (Cyclical)\n(Dimensions 7, 8)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Left Ankle X Velocity (m/s)')
    ax3.set_ylabel('Left Ankle Y Velocity (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Right ankle velocities from sample (dims 3, 4 in sample_trajectory)
        ax4.plot(sample_trajectory[:, 3], sample_trajectory[:, 4], 'b-', 
                linewidth=2, alpha=0.7, label='Sample Trajectory', zorder=8)
        ax4.plot(sample_trajectory[0, 3], sample_trajectory[0, 4], 'bo', 
                markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection
        if len(sample_trajectory) > 1:
            ax4.plot([sample_trajectory[-2, 3], sample_trajectory[-1, 3]], 
                    [sample_trajectory[-2, 4], sample_trajectory[-1, 4]], 
                    'b--', alpha=0.5, linewidth=1, label='Sample Cyclical', zorder=9)
    
    # Plot recovered trajectory if provided
    if predicted_trajectory is not None:
        # Right ankle velocities trajectory (dims 2, 3 in predicted_trajectory correspond to dims 3, 4)
        ax4.plot(predicted_trajectory[:, 2], predicted_trajectory[:, 3], 'k-', 
                linewidth=3, alpha=0.7, label='GMR Recovery (Cyclical)', zorder=10)
        ax4.plot(predicted_trajectory[0, 2], predicted_trajectory[0, 3], 'ko', 
                markersize=10, label='Recovery Start', zorder=11)
        ax4.plot(predicted_trajectory[-1, 2], predicted_trajectory[-1, 3], 'ks', 
                markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity
        if len(predicted_trajectory) > 1:
            ax4.plot([predicted_trajectory[-2, 2], predicted_trajectory[-1, 2]], 
                    [predicted_trajectory[-2, 3], predicted_trajectory[-1, 3]], 
                    'r--', alpha=0.7, linewidth=2, label='Cyclical Connection', zorder=11)
    
    # Right ankle velocities (dims 3, 4)
    for k in range(tpgmm._n_components):
        mean_vel = means[k, [3, 4]]  # [right_ankle_vel_x, right_ankle_vel_y]
        cov_vel = covariances[k][np.ix_([3, 4], [3, 4])]  # 2x2 covariance matrix
        
        # Plot mean as point
        ax4.scatter(mean_vel[0], mean_vel[1], c=[colors[k]], s=100*weights[k]*10, 
                   alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        # Plot covariance as ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov_vel)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
        
        ellipse = Ellipse(mean_vel, width, height, angle=angle, 
                         facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
        ax4.add_patch(ellipse)
    
    ax4.set_title('Right Ankle Velocity Gaussians (Cyclical)\n(Dimensions 3, 4)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Right Ankle X Velocity (m/s)')
    ax4.set_ylabel('Right Ankle Y Velocity (m/s)')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gaussian_models_visualization_cyclical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cyclical Gaussian models visualization saved to {save_dir}/tpgmm_gaussian_models_visualization_cyclical.png")


def plot_recovery_results(original_trajectory, predicted_trajectory, pca_results, feature_names, save_dir="plots"):
    """Plot trajectory recovery results with PCA visualization."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Time series comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    time_data = original_trajectory[:, 0]
    
    # Plot key features over time
    feature_plots = [
        (1, 'Right Ankle X Pos'),
        (2, 'Right Ankle Y Pos'), 
        (3, 'Right Ankle X Vel'),
        (4, 'Right Ankle Y Vel')
    ]
    
    for idx, (feat_idx, title) in enumerate(feature_plots):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].plot(time_data, original_trajectory[:, feat_idx], 'b-', label='Original (Cyclical)', linewidth=2)
        axes[row, col].plot(time_data, predicted_trajectory[:, feat_idx-1], 'r--', label='Predicted (Cyclical)', linewidth=2)
        
        # Highlight cyclical points
        axes[row, col].plot(time_data[0], original_trajectory[0, feat_idx], 'bo', markersize=8, label='Start/End')
        axes[row, col].plot(time_data[-1], original_trajectory[-1, feat_idx], 'bs', markersize=8)
        
        axes[row, col].set_title(f'{title} (Cyclical)')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Value')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # PCA plots
    if 'position' in pca_results:
        pca_pos = pca_results['position']['pca_data']
        axes[0, 2].plot(pca_pos[:, 0], pca_pos[:, 1], 'g-', linewidth=2)
        axes[0, 2].plot(pca_pos[0, 0], pca_pos[0, 1], 'go', markersize=8, label='Start/End')
        axes[0, 2].plot(pca_pos[-1, 0], pca_pos[-1, 1], 'gs', markersize=8)
        # Cyclical connection
        if len(pca_pos) > 1:
            axes[0, 2].plot([pca_pos[-2, 0], pca_pos[-1, 0]], [pca_pos[-2, 1], pca_pos[-1, 1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical')
        axes[0, 2].set_title('Position PCA (Cyclical)')
        axes[0, 2].set_xlabel(f'PC1 ({pca_results["position"]["explained_variance"][0]:.2%} variance)')
        axes[0, 2].set_ylabel(f'PC2 ({pca_results["position"]["explained_variance"][1]:.2%} variance)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        axes[0, 2].axis('equal')
    
    if 'velocity' in pca_results:
        pca_vel = pca_results['velocity']['pca_data']
        axes[0, 3].plot(pca_vel[:, 0], pca_vel[:, 1], 'm-', linewidth=2)
        axes[0, 3].plot(pca_vel[0, 0], pca_vel[0, 1], 'mo', markersize=8, label='Start/End')
        axes[0, 3].plot(pca_vel[-1, 0], pca_vel[-1, 1], 'ms', markersize=8)
        # Cyclical connection
        if len(pca_vel) > 1:
            axes[0, 3].plot([pca_vel[-2, 0], pca_vel[-1, 0]], [pca_vel[-2, 1], pca_vel[-1, 1]], 'r--', alpha=0.7, linewidth=2, label='Cyclical')
        axes[0, 3].set_title('Velocity PCA (Cyclical)')
        axes[0, 3].set_xlabel(f'PC1 ({pca_results["velocity"]["explained_variance"][0]:.2%} variance)')
        axes[0, 3].set_ylabel(f'PC2 ({pca_results["velocity"]["explained_variance"][1]:.2%} variance)')
        axes[0, 3].grid(True, alpha=0.3)
        axes[0, 3].legend()
        axes[0, 3].axis('equal')
    
    # Additional PCA comparison plots
    if 'position' in pca_results:
        # Plot original vs PCA reconstruction for positions
        pos_original = pca_results['position']['original_data']
        axes[1, 2].plot(pos_original[:, 0], pos_original[:, 1], 'b-', label='Right Ankle', linewidth=2)
        axes[1, 2].plot(pos_original[:, 2], pos_original[:, 3], 'r-', label='Left Ankle', linewidth=2)
        
        # Cyclical markers
        axes[1, 2].plot(pos_original[0, 0], pos_original[0, 1], 'bo', markersize=6)
        axes[1, 2].plot(pos_original[-1, 0], pos_original[-1, 1], 'bs', markersize=6)
        axes[1, 2].plot(pos_original[0, 2], pos_original[0, 3], 'ro', markersize=6)
        axes[1, 2].plot(pos_original[-1, 2], pos_original[-1, 3], 'rs', markersize=6)
        
        axes[1, 2].set_title('Original Position Trajectories (Cyclical)')
        axes[1, 2].set_xlabel('X Position (m)')
        axes[1, 2].set_ylabel('Y Position (m)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')
    
    if 'velocity' in pca_results:
        # Plot original vs PCA reconstruction for velocities
        vel_original = pca_results['velocity']['original_data']
        axes[1, 3].plot(vel_original[:, 0], vel_original[:, 1], 'b-', label='Right Ankle', linewidth=2)
        axes[1, 3].plot(vel_original[:, 2], vel_original[:, 3], 'r-', label='Left Ankle', linewidth=2)
        
        # Cyclical markers
        axes[1, 3].plot(vel_original[0, 0], vel_original[0, 1], 'bo', markersize=6)
        axes[1, 3].plot(vel_original[-1, 0], vel_original[-1, 1], 'bs', markersize=6)
        axes[1, 3].plot(vel_original[0, 2], vel_original[0, 3], 'ro', markersize=6)
        axes[1, 3].plot(vel_original[-1, 2], vel_original[-1, 3], 'rs', markersize=6)
        
        axes[1, 3].set_title('Original Velocity Trajectories (Cyclical)')
        axes[1, 3].set_xlabel('X Velocity (m/s)')
        axes[1, 3].set_ylabel('Y Velocity (m/s)')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
        axes[1, 3].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gmr_recovery_results_cyclical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cyclical recovery results plots saved to {save_dir}/ directory")


def main():
    """Main simplified GMR trajectory recovery pipeline."""
    print("=== Simplified Gait Analysis GMR Trajectory Recovery - Cyclical Version ===")
    
    # Configuration
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    model_path = os.path.join(pkls_dir, "gait_tpgmm_model_cyclical.pkl")
    
    # Step 1: Load trained model
    print("\nStep 1: Loading trained cyclical TPGMM model...")
    model_data = load_trained_model(model_path)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Step 2: Extract sample trajectory
    print("\nStep 2: Extracting sample cyclical trajectory...")
    sample_trajectory = extract_sample_trajectory(model_data, frame_idx=0, trajectory_idx=0)
    
    # Step 3: Prepare input data (time vector)
    print("\nStep 3: Preparing time input...")
    time_input = sample_trajectory[:, 0]  # Time column
    print(f"Time input shape: {time_input.shape}")
    
    # Step 4: Define output features to predict (exclude time)
    output_feature_indices = list(range(1, len(feature_names)))  # [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Output feature indices: {output_feature_indices}")
    print(f"Output features: {[feature_names[i] for i in output_feature_indices]}")
    
    # Step 5: Perform GMR prediction using TaskPaGMMM
    print("\nStep 5: Performing cyclical GMR prediction using TaskPaGMMM...")
    predicted_output, predicted_covariance = predict_using_tpgmm_gmr(
        tpgmm, time_input, output_feature_indices
    )
    
    print(f"Predicted output shape: {predicted_output.shape}")
    print(f"Predicted covariance shape: {predicted_covariance.shape}")
    
    # Verify cyclical properties
    first_pred = predicted_output[0, :]
    last_pred = predicted_output[-1, :]
    pred_diff = np.linalg.norm(first_pred - last_pred)
    print(f"Cyclical continuity check - predicted difference between first and last point: {pred_diff:.6f}")
    
    # Print detailed trajectory points for analysis
    print_trajectory_points(sample_trajectory, predicted_output, feature_names)
    
    # Step 6: Apply PCA to extract principal components
    print("\nStep 6: Applying PCA for dimensionality reduction...")
    pca_results = apply_pca_to_features(predicted_output, feature_names[1:])  # Exclude time from PCA
    
    # Step 7: Plot Gaussian models visualization
    print("\nStep 7: Creating cyclical Gaussian models visualization...")
    plot_gaussian_models(model_data, predicted_output, sample_trajectory)
    
    # Step 8: Plot Gaussian models with deviations
    print("\nStep 8: Creating cyclical Gaussian models with deviations visualization...")
    plot_gaussian_models_with_deviations(model_data, predicted_output, predicted_covariance, sample_trajectory)
    
    # Step 9: Plot recovery results
    print("\nStep 9: Creating cyclical recovery results visualizations...")
    plot_recovery_results(sample_trajectory, predicted_output, pca_results, feature_names)
    
    # Step 10: Save recovered trajectory
    print("\nStep 10: Saving recovered cyclical trajectory...")
    recovery_data = {
        'original_trajectory': sample_trajectory,
        'predicted_trajectory': predicted_output,
        'prediction_covariance': predicted_covariance,
        'pca_results': pca_results,
        'feature_names': feature_names,
        'time_input': time_input,
        'output_feature_indices': output_feature_indices,
        'cyclical_modification': True  # Flag to indicate this is cyclical data
    }
    
    recovery_save_path = os.path.join(pkls_dir, "gait_simplified_recovery_cyclical.pkl")
    with open(recovery_save_path, 'wb') as f:
        pickle.dump(recovery_data, f)
        
    print(f"Cyclical recovery data saved to: {recovery_save_path}")
    print("\n=== Simplified GMR Trajectory Recovery (Cyclical) Complete ===")


if __name__ == "__main__":
    main()