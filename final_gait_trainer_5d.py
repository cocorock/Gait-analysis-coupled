#!/usr/bin/env python3
"""
Final Gait Analysis TPGMM Training Script - 5D Version

This script trains a Task Parameterized Gaussian Mixture Model (TPGMM) with 2 frames of reference
for gait analysis using only right ankle position and velocity data (5D feature space).
Based on final_gait_trainer.py but simplified to 5D: time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, right_ankle_vel_y.
"""

import json
import numpy as np
import pickle
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import the TPGMM implementation from TaskPaGMMM
import sys
sys.path.append('TaskPaGMMM')
from tpgmm.tpgmm.tpgmm import TPGMM


def load_gait_data(json_path):
    """Load and process gait data from JSON file - 5D version with only FR1 and FR2."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get kinematics data from only 2 frames (FR1 and FR2)
    frames = ['FR1', 'FR2']
    all_trajectories = []
    
    # Process each frame of reference
    for frame_idx, frame in enumerate(frames):
        kinematics = data['kinematics_data'][frame]['right_leg_kinematics']
        num_cycles = len(kinematics)
        interpolation_points = data['parameters']['interpolation_points']
        
        if frame_idx == 0:  # Print info only once
            print(f"Processing {num_cycles} gait cycles from {len(frames)} frames")
            print(f"Each cycle has {interpolation_points} interpolation points")
        
        frame_trajectories = []
        
        for cycle_idx, cycle_data in enumerate(kinematics):
            # Extract only right ankle data
            right_ankle_pos = np.array(cycle_data['right_ankle_pos'])
            right_ankle_vel = np.array(cycle_data['right_ankle_vel'])
            
            # Create time vector (normalized 0-1)
            time_vector = np.linspace(0, 1, interpolation_points)
            
            # Create 5-dimensional feature space:
            # [time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, right_ankle_vel_y]
            trajectory = np.column_stack([
                time_vector,               # time (0-1)
                right_ankle_pos[0, :],     # right ankle X position
                right_ankle_pos[1, :],     # right ankle Y position
                right_ankle_vel[0, :],     # right ankle X velocity
                right_ankle_vel[1, :]      # right ankle Y velocity
            ])
            
            frame_trajectories.append(trajectory)
        
        # Convert to numpy array and store
        frame_trajectories = np.array(frame_trajectories)
        all_trajectories.append(frame_trajectories)
        print(f"{frame} trajectories shape: {frame_trajectories.shape}")
    
    return all_trajectories


def plot_input_trajectories(all_trajectories):
    """Plot input trajectories for FR1 and FR2 frames - right ankle only."""
    frames = ['FR1', 'FR2']
    frame_colors = {'FR1': 'blue', 'FR2': 'red'}
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot right ankle positions only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for frame_idx, (frame_name, trajectories) in enumerate(zip(frames, all_trajectories)):
        color = frame_colors[frame_name]
        
        # Right ankle positions - all frames in one plot
        for traj_idx, traj in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.8
            linewidth = 1.5 if traj_idx == 0 else 1
            label = frame_name if traj_idx == 0 else ""
            ax.plot(traj[:, 1], traj[:, 2], color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    ax.set_title('Right Ankle Position Trajectories - 5D Version\\nFR1 and FR2 Combined', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('plots/input_trajectories_positions_5d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot right ankle velocities only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for frame_idx, (frame_name, trajectories) in enumerate(zip(frames, all_trajectories)):
        color = frame_colors[frame_name]
        
        # Right ankle velocities - all frames in one plot
        for traj_idx, traj in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.8
            linewidth = 1.5 if traj_idx == 0 else 1
            label = frame_name if traj_idx == 0 else ""
            ax.plot(traj[:, 3], traj[:, 4], color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    ax.set_title('Right Ankle Velocity Trajectories - 5D Version\\nFR1 and FR2 Combined', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Velocity (m/s)')
    ax.set_ylabel('Y Velocity (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('plots/input_trajectories_velocities_5d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Input trajectory plots saved to plots/ directory (5D version)")


def prepare_tpgmm_data(all_trajectories):
    """Prepare data for TPGMM training - 5D version with 2 frames."""
    # Convert list of 2 trajectory arrays to stacked format
    # all_trajectories: list of 2 arrays, each with shape (num_cycles, num_points, num_features)
    
    num_frames = len(all_trajectories)
    num_trajectories, num_samples, num_features = all_trajectories[0].shape
    
    # Stack the data for all frames
    reshaped_trajectories = np.stack(all_trajectories, axis=0)
    
    # Reshape to (num_frames, num_trajectories * num_samples, num_features)
    reshaped_trajectories = reshaped_trajectories.reshape(num_frames, num_trajectories * num_samples, num_features)
    
    print(f"Reshaped trajectories shape: {reshaped_trajectories.shape}")
    print(f"Features: [time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, right_ankle_vel_y]")
    
    return reshaped_trajectories


def find_optimal_components(reshaped_trajectories, component_range=(3, 12)):
    """Find optimal number of components using BIC scores with visualization - 5D version."""
    print("\\nFinding optimal number of components (5D version)...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    n_components_list = []
    bic_scores = []
    
    best_n_components = None
    lowest_bic_score = float('inf')
    
    # Loop through n_components (reduced range for 5D)
    for n_components in range(component_range[0], component_range[1]):
        print(f'Fitting TPGMM with n_components={n_components}...')
        
        # Define the TPGMM model with the current n_components
        tpgmm = TPGMM(n_components=n_components, verbose=False, threshold=1e-5, reg_factor=1e-8)
        
        # Fit the model with the trajectories
        tpgmm.fit(reshaped_trajectories)
        
        # Calculate the BIC score
        bic_score = tpgmm.bic(reshaped_trajectories)
        
        print(f'n_components={n_components}: BIC={bic_score:.2f}')
        
        # Store results
        n_components_list.append(n_components)
        bic_scores.append(bic_score)
        
        # Update the best n_components and lowest BIC score if the current BIC is lower
        if bic_score < lowest_bic_score:
            lowest_bic_score = bic_score
            best_n_components = n_components
    
    # Plot BIC scores
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # BIC plot
    ax1.plot(n_components_list, bic_scores, 'bo-', linewidth=2, markersize=8, label='BIC')
    ax1.axvline(x=best_n_components, color='red', linestyle='--', linewidth=2, 
               label=f'Optimal (n={best_n_components})')
    ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BIC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Bayesian Information Criterion (BIC) - 5D Version', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig('plots/model_selection_criteria_5d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\\n=== Model Selection Results (5D) ===')
    print(f'BIC optimal n_components: {best_n_components} (BIC = {lowest_bic_score:.2f})')
    print(f'Model selection plot saved to plots/model_selection_criteria_5d.png')
    
    # Return results dictionary
    return {
        'best_n_components': best_n_components,
        'n_components_list': n_components_list,
        'bic_scores': bic_scores,
        'lowest_bic_score': lowest_bic_score
    }


def main():
    """Main training pipeline - 5D version."""
    print("=== Final Gait Analysis TPGMM Training - 5D Version ===")
    
    # Configuration
    reg_factor = 1e-2
    threshold = 1e-4
    json_path = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    
    # Create pkls directory if it doesn't exist
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    os.makedirs(pkls_dir, exist_ok=True)
    
    model_save_path = os.path.join(pkls_dir, "gait_tpgmm_model_5d.pkl")
    
    print(f"JSON data path: {json_path}")
    print(f"Model save path: {model_save_path}")
    print()
    
    # Step 1: Load and process data
    print("Step 1: Loading gait data (5D version)...")
    all_trajectories = load_gait_data(json_path)
    
    # Step 2: Plot input trajectories
    print("\\nStep 2: Plotting input trajectories (5D version)...")
    plot_input_trajectories(all_trajectories)
    
    # Step 3: Prepare TPGMM data
    print("\\nStep 3: Preparing TPGMM data (5D version)...")
    reshaped_trajectories = prepare_tpgmm_data(all_trajectories)
    
    # Step 4: Find optimal components using BIC
    print("\\nStep 4: Finding optimal components using BIC (5D version)...")
    model_selection_results = find_optimal_components(reshaped_trajectories, component_range=(3, 17))
    best_n_components = model_selection_results['best_n_components']
    
    # Step 5: Train final TPGMM with optimal components
    print(f"\\nStep 5: Training final TPGMM with n_components={best_n_components} (5D version)...")
    tpgmm = TPGMM(n_components=best_n_components, verbose=True, reg_factor=reg_factor, threshold=threshold)
    tpgmm.fit(reshaped_trajectories)
    
    print("TPGMM training completed (5D version)!")
    
    # Step 6: Save model
    print("\\nStep 6: Saving model (5D version)...")
    model_data = {
        'tpgmm': tpgmm,
        'n_components': best_n_components,
        'weights_': tpgmm.weights_,
        'means_': tpgmm.means_,
        'covariances_': tpgmm.covariances_,
        'all_trajectories': all_trajectories,
        'reshaped_trajectories': reshaped_trajectories,
        'model_selection_results': model_selection_results,
        'feature_names': [
            'time', 'right_ankle_pos_x', 'right_ankle_pos_y', 'right_ankle_vel_x', 'right_ankle_vel_y'
        ],
        'frame_names': ['FR1', 'FR2'],
        'num_frames': 2,
        'feature_dims': 5
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Model saved to: {model_save_path}")
    print("\\n=== Training Complete (5D Version) ===")
    print("The 5D model can now be loaded for Gaussian Mixture Regression (GMR).")


if __name__ == "__main__":
    main()