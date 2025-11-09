#!/usr/bin/env python3
"""
Simplified Gait Analysis GMR Trajectory Recovery Script - Cyclical Version

This script loads a trained TPGMM model (cyclical version) and performs simplified trajectory recovery
following the logic from gait_example1time.ipynb more directly.

MODIFICATION: Uses cyclical TPGMM model with configurable overlap points.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import os
import sys
import argparse

# Add TaskPaGMMM to Python path
sys.path.append('TaskPaGMMM')
from tpgmm.gmr.gmr import GaussianMixtureRegression


def load_trained_model(model_path, overlap_points):
    """Load the trained cyclical TPGMM model from pickle file."""
    print(f"Loading trained {overlap_points}-point cyclical model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"{overlap_points}-point cyclical model loaded successfully!")
    print(f"Number of frames: {model_data['num_frames']}")
    print(f"Number of components: {model_data['n_components']}")
    print(f"Feature dimensions: {model_data['feature_dims']}")
    print(f"Feature names: {model_data['feature_names']}")
    
    # Check if this is indeed a cyclical model
    if 'cyclical_modification' in model_data:
        print(f"Cyclical modification: {model_data['cyclical_modification']}")
    if 'cyclical_overlap_points' in model_data:
        print(f"Cyclical overlap points: {model_data['cyclical_overlap_points']}")
    else:
        print("Warning: Model does not have cyclical_overlap_points flag")
    
    return model_data


def extract_sample_trajectory(model_data, overlap_points, frame_idx=0, trajectory_idx=0):
    """Extract a sample trajectory from the training data."""
    
    # Get the sample trajectory
    sample_trajectory = model_data['all_trajectories'][frame_idx][trajectory_idx]
    
    print(f"Sample {overlap_points}-point cyclical trajectory shape: {sample_trajectory.shape}")
    print(f"Time range: {sample_trajectory[:, 0].min():.3f} to {sample_trajectory[:, 0].max():.3f}")
    print(f"First point time: {sample_trajectory[0, 0]:.3f}, Last point time: {sample_trajectory[-1, 0]:.3f}")
    
    # Verify cyclical property
    if overlap_points > 0:
        first_n_pos = sample_trajectory[:overlap_points, 1:]  # First n points (all features except time)
        last_n_pos = sample_trajectory[-overlap_points:, 1:]  # Last n points
        pos_diff = np.linalg.norm(first_n_pos - last_n_pos)
        print(f"{overlap_points}-point cyclical continuity check - position difference between first and last {overlap_points} points: {pos_diff:.6f}")
    else:
        print("0-point overlap - no cyclical continuity check performed")
    
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


def print_trajectory_points(original_trajectory, predicted_trajectory, feature_names, overlap_points):
    """Print first and last n points of trajectories for cyclical verification."""
    print(f"\n=== {overlap_points}-Point Trajectory Points Analysis ===")
    
    # Feature indices mapping
    # Original: [time, right_pos_x, right_pos_y, right_vel_x, right_vel_y, left_pos_x, left_pos_y, left_vel_x, left_vel_y]
    # Predicted: [right_pos_x, right_pos_y, right_vel_x, right_vel_y, left_pos_x, left_pos_y, left_vel_x, left_vel_y]
    
    print("Format: right_pos(x,y) left_vel(x,y) left_pos(x,y) right_vel(x,y)")
    print()
    
    if overlap_points == 0:
        print("0-point overlap - only showing first and last points")
        num_points = 1
    else:
        num_points = overlap_points
    
    print("ORIGINAL TRAJECTORY:")
    print(f"First {num_points} points:")
    for i in range(num_points):
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
    print(f"Last {num_points} points:")
    for i in range(-num_points, 0):
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
    print(f"First {num_points} points:")
    for i in range(num_points):
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
    print(f"Last {num_points} points:")
    for i in range(-num_points, 0):
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
    print(f"{overlap_points}-POINT CYCLICAL VERIFICATION:")
    
    if overlap_points > 0:
        orig_first_n = original_trajectory[:overlap_points, 1:]  # Exclude time, first n points
        orig_last_n = original_trajectory[-overlap_points:, 1:]  # Last n points
        pred_first_n = predicted_trajectory[:overlap_points, :]  # First n points
        pred_last_n = predicted_trajectory[-overlap_points:, :]  # Last n points
        
        orig_diff_n = np.linalg.norm(orig_first_n - orig_last_n)
        pred_diff_n = np.linalg.norm(pred_first_n - pred_last_n)
        
        print(f"Original trajectory - First vs Last {overlap_points} points difference: {orig_diff_n:.8f}")
        print(f"Predicted trajectory - First vs Last {overlap_points} points difference: {pred_diff_n:.8f}")
    
    # Also compute single point differences for comparison
    orig_first = original_trajectory[0, 1:]  # Exclude time
    orig_last = original_trajectory[-1, 1:]
    pred_first = predicted_trajectory[0, :]
    pred_last = predicted_trajectory[-1, :]
    
    orig_diff_1 = np.linalg.norm(orig_first - orig_last)
    pred_diff_1 = np.linalg.norm(pred_first - pred_last)
    
    print(f"Original trajectory - First vs Last single point difference: {orig_diff_1:.8f}")
    print(f"Predicted trajectory - First vs Last single point difference: {pred_diff_1:.8f}")
    
    print(f"=== End {overlap_points}-Point Trajectory Points Analysis ===\n")


def plot_gaussian_models_with_deviations(model_data, overlap_points, predicted_trajectory=None, predicted_covariance=None, sample_trajectory=None, save_dir="plots"):
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
        ax1.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label=f'Sample Trajectory ({overlap_points}-Point)', zorder=8)
        ax1.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection (overlap)
        if overlap_points > 0 and len(sample_x) > overlap_points:
            ax1.plot([sample_x[-(overlap_points+1)], sample_x[-1]], [sample_y[-(overlap_points+1)], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label=f'Sample {overlap_points}-Point Overlap', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Left ankle positions trajectory with uncertainty (dims 4, 5 in predicted_trajectory correspond to dims 5, 6)
        x_pos = predicted_trajectory[:, 4]
        y_pos = predicted_trajectory[:, 5]
        x_std = np.sqrt(predicted_covariance[:, 4, 4])
        y_std = np.sqrt(predicted_covariance[:, 4, 4])
        
        ax1.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label=f'GMR Recovery ({overlap_points}-Point)', zorder=10)
        ax1.fill_between(x_pos, y_pos - y_std, y_pos + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        #ax1.plot(x_pos[0], y_pos[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        #ax1.plot(x_pos[-1], y_pos[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity (overlap region)
        if overlap_points > 0 and len(x_pos) > overlap_points:
            ax1.plot([x_pos[-(overlap_points+1)], x_pos[-1]], [y_pos[-(overlap_points+1)], y_pos[-1]], 'r--', alpha=0.7, linewidth=2, label=f'Recovery {overlap_points}-Point Gap', zorder=11)
    
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
    
    ax1.set_title(f'Left Ankle Position Gaussians with Deviations ({overlap_points}-Point)\n(Dimensions 5, 6)', fontsize=14, fontweight='bold')
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
        ax2.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label=f'Sample Trajectory ({overlap_points}-Point)', zorder=8)
        ax2.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection (overlap)
        if overlap_points > 0 and len(sample_x) > overlap_points:
            ax2.plot([sample_x[-(overlap_points+1)], sample_x[-1]], [sample_y[-(overlap_points+1)], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label=f'Sample {overlap_points}-Point Overlap', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle positions trajectory with uncertainty (dims 0, 1 in predicted_trajectory correspond to dims 1, 2)
        x_pos = predicted_trajectory[:, 0]
        y_pos = predicted_trajectory[:, 1]
        x_std = np.sqrt(predicted_covariance[:, 0, 0])
        y_std = np.sqrt(predicted_covariance[:, 1, 1])
        
        ax2.plot(x_pos, y_pos, 'k-', linewidth=3, alpha=0.8, label=f'GMR Recovery ({overlap_points}-Point)', zorder=10)
        ax2.fill_between(x_pos, y_pos - y_std, y_pos + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        #ax2.plot(x_pos[0], y_pos[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        #ax2.plot(x_pos[-1], y_pos[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity (overlap region)
        if overlap_points > 0 and len(x_pos) > overlap_points:
            ax2.plot([x_pos[-(overlap_points+1)], x_pos[-1]], [y_pos[-(overlap_points+1)], y_pos[-1]], 'r--', alpha=0.7, linewidth=2, label=f'Recovery {overlap_points}-Point Gap', zorder=11)
    
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
    
    ax2.set_title(f'Right Ankle Position Gaussians with Deviations ({overlap_points}-Point)\n(Dimensions 1, 2)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Right Ankle X Position (m)')
    ax2.set_ylabel('Right Ankle Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Left ankle velocities from sample (dims 7, 8 in sample_trajectory)
        sample_x = sample_trajectory[:, 7]
        sample_y = sample_trajectory[:, 8]
        ax3.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label=f'Sample Trajectory ({overlap_points}-Point)', zorder=8)
        ax3.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection (overlap)
        if overlap_points > 0 and len(sample_x) > overlap_points:
            ax3.plot([sample_x[-(overlap_points+1)], sample_x[-1]], [sample_y[-(overlap_points+1)], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label=f'Sample {overlap_points}-Point Overlap', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Left ankle velocities trajectory with uncertainty (dims 6, 7 in predicted_trajectory correspond to dims 7, 8)
        x_vel = predicted_trajectory[:, 6]
        y_vel = predicted_trajectory[:, 7]
        x_std = np.sqrt(predicted_covariance[:, 6, 6])
        y_std = np.sqrt(predicted_covariance[:, 7, 7])
        
        ax3.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label=f'GMR Recovery ({overlap_points}-Point)', zorder=10)
        ax3.fill_between(x_vel, y_vel - y_std, y_vel + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        #ax3.plot(x_vel[0], y_vel[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        #ax3.plot(x_vel[-1], y_vel[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity (overlap region)
        if overlap_points > 0 and len(x_vel) > overlap_points:
            ax3.plot([x_vel[-(overlap_points+1)], x_vel[-1]], [y_vel[-(overlap_points+1)], y_vel[-1]], 'r--', alpha=0.7, linewidth=2, label=f'Recovery {overlap_points}-Point Gap', zorder=11)
    
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
    
    ax3.set_title(f'Left Ankle Velocity Gaussians with Deviations ({overlap_points}-Point)\n(Dimensions 7, 8)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Left Ankle X Velocity (m/s)')
    ax3.set_ylabel('Left Ankle Y Velocity (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot sample trajectory if provided
    if sample_trajectory is not None:
        # Right ankle velocities from sample (dims 3, 4 in sample_trajectory)
        sample_x = sample_trajectory[:, 3]
        sample_y = sample_trajectory[:, 4]
        ax4.plot(sample_x, sample_y, 'b-', linewidth=2, alpha=0.7, label=f'Sample Trajectory ({overlap_points}-Point)', zorder=8)
        ax4.plot(sample_x[0], sample_y[0], 'bo', markersize=6, label='Sample Start/End', zorder=9)
        
        # Sample cyclical connection (overlap)
        if overlap_points > 0 and len(sample_x) > overlap_points:
            ax4.plot([sample_x[-(overlap_points+1)], sample_x[-1]], [sample_y[-(overlap_points+1)], sample_y[-1]], 'b--', alpha=0.5, linewidth=1, label=f'Sample {overlap_points}-Point Overlap', zorder=9)
    
    # Plot recovered trajectory with deviations if provided
    if predicted_trajectory is not None and predicted_covariance is not None:
        # Right ankle velocities trajectory with uncertainty (dims 2, 3 in predicted_trajectory correspond to dims 3, 4)
        x_vel = predicted_trajectory[:, 2]
        y_vel = predicted_trajectory[:, 3]
        x_std = np.sqrt(predicted_covariance[:, 2, 2])
        y_std = np.sqrt(predicted_covariance[:, 3, 3])
        
        ax4.plot(x_vel, y_vel, 'k-', linewidth=3, alpha=0.8, label=f'GMR Recovery ({overlap_points}-Point)', zorder=10)
        ax4.fill_between(x_vel, y_vel - y_std, y_vel + y_std, alpha=0.3, color='gray', label='±1σ deviation', zorder=5)
        #ax4.plot(x_vel[0], y_vel[0], 'ko', markersize=10, label='Recovery Start', zorder=11)
        #ax4.plot(x_vel[-1], y_vel[-1], 'ks', markersize=10, label='Recovery End', zorder=11)
        
        # Highlight cyclical continuity (overlap region)
        if overlap_points > 0 and len(x_vel) > overlap_points:
            ax4.plot([x_vel[-(overlap_points+1)], x_vel[-1]], [y_vel[-(overlap_points+1)], y_vel[-1]], 'r--', alpha=0.7, linewidth=2, label=f'Recovery {overlap_points}-Point Gap', zorder=11)
    
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
    
    ax4.set_title(f'Right Ankle Velocity Gaussians with Deviations ({overlap_points}-Point)\n(Dimensions 3, 4)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Right Ankle X Velocity (m/s)')
    ax4.set_ylabel('Right Ankle Y Velocity (m/s)')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tpgmm_gaussian_models_with_deviations_cyclical_{overlap_points}pt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{overlap_points}-point cyclical Gaussian models with deviations visualization saved to {save_dir}/tpgmm_gaussian_models_with_deviations_cyclical_{overlap_points}pt.png")


def main():
    """Main simplified GMR trajectory recovery pipeline."""
    parser = argparse.ArgumentParser(description='Gait Analysis GMR Trajectory Recovery with configurable overlap points')
    parser.add_argument('--overlap_points', type=int, default=10, 
                       help='Number of overlap points for cyclical continuity (default: 10)')
    
    args = parser.parse_args()
    overlap_points = args.overlap_points
    
    print(f"=== Simplified Gait Analysis GMR Trajectory Recovery - {overlap_points}-Point Cyclical Version ===")
    
    # Configuration
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    model_path = os.path.join(pkls_dir, f"gait_tpgmm_model_cyclical_{overlap_points}pt.pkl")
    
    # Step 1: Load trained model
    print(f"\nStep 1: Loading trained {overlap_points}-point cyclical TPGMM model...")
    model_data = load_trained_model(model_path, overlap_points)
    
    tpgmm = model_data['tpgmm']
    feature_names = model_data['feature_names']
    
    # Step 2: Extract sample trajectory
    print(f"\nStep 2: Extracting sample {overlap_points}-point cyclical trajectory...")
    sample_trajectory = extract_sample_trajectory(model_data, overlap_points, frame_idx=0, trajectory_idx=0)
    
    # Step 3: Prepare input data (time vector)
    print("\nStep 3: Preparing time input...")
    time_input = sample_trajectory[:, 0]  # Time column
    print(f"Time input shape: {time_input.shape}")
    
    # Step 4: Define output features to predict (exclude time)
    output_feature_indices = list(range(1, len(feature_names)))  # [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Output feature indices: {output_feature_indices}")
    print(f"Output features: {[feature_names[i] for i in output_feature_indices]}")
    
    # Step 5: Perform GMR prediction using TaskPaGMMM
    print(f"\nStep 5: Performing {overlap_points}-point cyclical GMR prediction using TaskPaGMMM...")
    predicted_output, predicted_covariance = predict_using_tpgmm_gmr(
        tpgmm, time_input, output_feature_indices
    )
    
    print(f"Predicted output shape: {predicted_output.shape}")
    print(f"Predicted covariance shape: {predicted_covariance.shape}")
    
    # Verify cyclical properties
    if overlap_points > 0:
        first_n_pred = predicted_output[:overlap_points, :]
        last_n_pred = predicted_output[-overlap_points:, :]
        pred_diff_n = np.linalg.norm(first_n_pred - last_n_pred)
        print(f"{overlap_points}-point cyclical continuity check - predicted difference between first and last {overlap_points} points: {pred_diff_n:.6f}")
    
    first_pred = predicted_output[0, :]
    last_pred = predicted_output[-1, :]
    pred_diff_1 = np.linalg.norm(first_pred - last_pred)
    print(f"Single-point cyclical continuity check - predicted difference between first and last point: {pred_diff_1:.6f}")
    
    # Print detailed trajectory points for analysis
    print_trajectory_points(sample_trajectory, predicted_output, feature_names, overlap_points)
    
    # Step 6: Create visualization
    print(f"\nStep 6: Creating {overlap_points}-point cyclical Gaussian models with deviations visualization...")
    plot_gaussian_models_with_deviations(model_data, overlap_points, predicted_output, predicted_covariance, sample_trajectory)
    
    # Step 7: Save recovered trajectory
    print(f"\nStep 7: Saving recovered {overlap_points}-point cyclical trajectory...")
    recovery_data = {
        'original_trajectory': sample_trajectory,
        'predicted_trajectory': predicted_output,
        'prediction_covariance': predicted_covariance,
        'feature_names': feature_names,
        'time_input': time_input,
        'output_feature_indices': output_feature_indices,
        'cyclical_modification': True,  # Flag to indicate this is cyclical data
        'cyclical_overlap_points': overlap_points   # Flag to indicate n-point overlap
    }
    
    recovery_save_path = os.path.join(pkls_dir, f"gait_simplified_recovery_cyclical_{overlap_points}pt.pkl")
    with open(recovery_save_path, 'wb') as f:
        pickle.dump(recovery_data, f)
        
    print(f"{overlap_points}-point cyclical recovery data saved to: {recovery_save_path}")
    print(f"\n=== Simplified GMR Trajectory Recovery ({overlap_points}-Point Cyclical) Complete ===")


if __name__ == "__main__":
    main()