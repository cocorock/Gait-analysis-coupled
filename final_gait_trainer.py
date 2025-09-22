#!/usr/bin/env python3
"""
Final Gait Analysis TPGMM Training Script

This script trains a Task Parameterized Gaussian Mixture Model (TPGMM) with 3 frames of reference
for gait analysis using ankle position and velocity data from both legs.
"""

import json
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# Import the local TPGMM implementation
from tpgmm_local import TPGMM


def load_gait_data(json_path):
    """Load and process gait data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get kinematics data
    kinematics = data['kinematics_data']['FR1']['right_leg_kinematics']
    num_cycles = len(kinematics)
    interpolation_points = data['parameters']['interpolation_points']
    
    print(f"Processing {num_cycles} gait cycles")
    print(f"Each cycle has {interpolation_points} interpolation points")
    
    # Process each gait cycle
    all_trajectories = []
    
    for cycle_idx, cycle_data in enumerate(kinematics):
        # Extract required data
        right_ankle_pos = np.array(cycle_data['right_ankle_pos'])
        right_ankle_vel = np.array(cycle_data['right_ankle_vel'])
        left_ankle_pos = np.array(cycle_data['left_ankle_pos'])
        left_ankle_vel = np.array(cycle_data['left_ankle_vel'])
        
        # Create time vector
        time_vector = np.linspace(0, 1, interpolation_points)
        
        # Combine features: [time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, 
        #                   right_ankle_vel_y, left_ankle_pos_x, left_ankle_pos_y, 
        #                   left_ankle_vel_x, left_ankle_vel_y]
        trajectory = np.column_stack([
            time_vector,
            right_ankle_pos[0, :],
            right_ankle_pos[1, :],
            right_ankle_vel[0, :],
            right_ankle_vel[1, :],
            left_ankle_pos[0, :],
            left_ankle_pos[1, :],
            left_ankle_vel[0, :],
            left_ankle_vel[1, :]
        ])
        
        all_trajectories.append(trajectory)
        
    return np.array(all_trajectories)


def create_frames_of_reference(trajectories):
    """Create 3 frames of reference for task parameterization."""
    num_cycles, num_points, num_features = trajectories.shape
    
    # Define frame reference points
    start_frame_idx = 0
    mid_frame_idx = num_points // 2
    end_frame_idx = num_points - 1
    
    frames_data = []
    
    for frame_idx, ref_point in enumerate([start_frame_idx, mid_frame_idx, end_frame_idx]):
        print(f"Creating frame {frame_idx + 1} with reference point at index {ref_point}")
        
        frame_trajectories = []
        
        for cycle_idx in range(num_cycles):
            # Get reference point for this cycle
            ref_values = trajectories[cycle_idx, ref_point, :]
            
            # Transform trajectory relative to reference point
            transformed_traj = trajectories[cycle_idx].copy()
            
            # Translate position features relative to reference
            pos_dims = [1, 2, 5, 6]  # position dimensions
            for dim in pos_dims:
                transformed_traj[:, dim] -= ref_values[dim]
            
            frame_trajectories.append(transformed_traj)
        
        # Concatenate all cycles for this frame
        frame_data = np.concatenate(frame_trajectories, axis=0)
        frames_data.append(frame_data)
        
    return np.array(frames_data)


def main():
    """Main training pipeline."""
    print("=== Final Gait Analysis TPGMM Training ===")
    
    # Configuration
    json_path = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    model_save_path = "/home/jemajuinta/ws/Gait-analysis-coupled/gait_tpgmm_model.pkl"
    n_components = 6
    
    print(f"JSON data path: {json_path}")
    print(f"Model save path: {model_save_path}")
    print(f"Number of components: {n_components}")
    print()
    
    # Step 1: Load and process data
    print("Step 1: Loading gait data...")
    trajectories = load_gait_data(json_path)
    print(f"Loaded trajectories shape: {trajectories.shape}")
    
    # Step 2: Create frames of reference
    print("\nStep 2: Creating frames of reference...")
    frames_data = create_frames_of_reference(trajectories)
    print(f"Frames data shape: {frames_data.shape}")
    
    # Step 3: Train TPGMM
    print("\nStep 3: Training TPGMM...")
    tpgmm = TPGMM(n_components=n_components, verbose=True)
    tpgmm.fit(frames_data)
    
    print("TPGMM training completed!")
    print(f"Final log-likelihood: {tpgmm.log_likelihood_}")
    
    # Step 4: Save model
    print("\nStep 4: Saving model...")
    model_data = {
        'tpgmm': tpgmm,
        'n_components': n_components,
        'weights_': tpgmm.weights_,
        'means_': tpgmm.means_,
        'covariances_': tpgmm.covariances_,
        'log_likelihood_': tpgmm.log_likelihood_,
        'feature_names': [
            'time', 'right_ankle_pos_x', 'right_ankle_pos_y', 
            'right_ankle_vel_x', 'right_ankle_vel_y',
            'left_ankle_pos_x', 'left_ankle_pos_y',
            'left_ankle_vel_x', 'left_ankle_vel_y'
        ]
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Model saved to: {model_save_path}")
    print("\n=== Training Complete ===")
    print("The model can now be loaded for Gaussian Mixture Regression (GMR).")


if __name__ == "__main__":
    main()