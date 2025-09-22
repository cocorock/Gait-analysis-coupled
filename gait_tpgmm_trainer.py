#!/usr/bin/env python3
"""
TPGMM Gait Analysis Training Script

This script trains a Task Parameterized Gaussian Mixture Model (TPGMM) with 3 frames of reference
for gait analysis using ankle position and velocity data from both legs.

Features:
- 9 dimensions: time + 8 ankle features (position & velocity for both ankles, x & y components)
- 3 frames of reference for task parameterization
- Saves trained model as .pkl file for later GMR use
"""

import json
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# Add the TPGMM module to path
sys.path.append(str(Path(__file__).parent / "TaskParameterizedGaussianMixtureModels"))

try:
    from tpgmm.tpgmm.tpgmm import TPGMM
    from tpgmm.gmr.gmr import GaussianMixtureRegression
except ImportError:
    # Try importing from the current directory files
    sys.path.insert(0, str(Path(__file__).parent))
    from tpgmm import TPGMM
    from gmr import GaussianMixtureRegression


class GaitDataProcessor:
    """Processes gait analysis JSON data for TPGMM training."""
    
    def __init__(self, json_path: str, interpolation_points: int = 200):
        """
        Initialize the gait data processor.
        
        Args:
            json_path: Path to the gait analysis JSON file
            interpolation_points: Number of interpolation points per trajectory
        """
        self.json_path = json_path
        self.interpolation_points = interpolation_points
        self.data = None
        
    def load_data(self):
        """Load gait analysis data from JSON file."""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded gait data from {self.json_path}")
        
    def extract_trajectories(self):
        """
        Extract and organize trajectory data for TPGMM training.
        
        Returns:
            numpy.ndarray: Trajectory data with shape (num_frames, num_points, num_features)
                          where num_features = 9 (time + 8 ankle features)
        """
        if self.data is None:
            self.load_data()
            
        # Get kinematics data for right_leg_kinematics frame
        kinematics = self.data['kinematics_data']['FR1']['right_leg_kinematics']
        num_cycles = len(kinematics)
        
        print(f"Processing {num_cycles} gait cycles")
        print(f"Each cycle has {self.interpolation_points} interpolation points")
        
        # Initialize arrays to store all trajectory data
        all_trajectories = []
        
        # Process each gait cycle
        for cycle_idx, cycle_data in enumerate(kinematics):
            # Extract required data
            right_ankle_pos = np.array(cycle_data['right_ankle_pos'])  # Shape: (2, 200) for x,y
            right_ankle_vel = np.array(cycle_data['right_ankle_vel'])  # Shape: (2, 200) for x,y
            left_ankle_pos = np.array(cycle_data['left_ankle_pos'])    # Shape: (2, 200) for x,y
            left_ankle_vel = np.array(cycle_data['left_ankle_vel'])    # Shape: (2, 200) for x,y
            
            # Create time vector (normalized 0 to 1 for each cycle)
            time_vector = np.linspace(0, 1, self.interpolation_points)
            
            # Combine features: [time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, 
            #                   right_ankle_vel_y, left_ankle_pos_x, left_ankle_pos_y, 
            #                   left_ankle_vel_x, left_ankle_vel_y]
            trajectory = np.column_stack([
                time_vector,                    # dimension 0: time
                right_ankle_pos[0, :],         # dimension 1: right_ankle_pos_x
                right_ankle_pos[1, :],         # dimension 2: right_ankle_pos_y
                right_ankle_vel[0, :],         # dimension 3: right_ankle_vel_x
                right_ankle_vel[1, :],         # dimension 4: right_ankle_vel_y
                left_ankle_pos[0, :],          # dimension 5: left_ankle_pos_x
                left_ankle_pos[1, :],          # dimension 6: left_ankle_pos_y
                left_ankle_vel[0, :],          # dimension 7: left_ankle_vel_x
                left_ankle_vel[1, :]           # dimension 8: left_ankle_vel_y
            ])
            
            all_trajectories.append(trajectory)
            
        all_trajectories = np.array(all_trajectories)
        print(f"Extracted trajectories shape: {all_trajectories.shape}")
        
        return all_trajectories
    
    def create_frames_of_reference(self, trajectories):
        """
        Create 3 frames of reference for task parameterization:
        1. Initial frame (start of gait cycle)
        2. Mid-cycle frame (middle of gait cycle) 
        3. Final frame (end of gait cycle)
        
        Args:
            trajectories: Array of shape (num_cycles, num_points, num_features)
            
        Returns:
            numpy.ndarray: Transformed data with shape (num_frames, total_points, num_features)
        """
        num_cycles, num_points, num_features = trajectories.shape
        
        # Define frame reference points (indices in the trajectory)
        start_frame_idx = 0
        mid_frame_idx = num_points // 2
        end_frame_idx = num_points - 1
        
        # Create 3 frames of reference
        frames_data = []
        
        for frame_idx, ref_point in enumerate([start_frame_idx, mid_frame_idx, end_frame_idx]):
            print(f"Creating frame {frame_idx + 1} with reference point at index {ref_point}")
            
            # Transform trajectories relative to the reference frame
            frame_trajectories = []
            
            for cycle_idx in range(num_cycles):
                # Get reference point for this cycle
                ref_values = trajectories[cycle_idx, ref_point, :]
                
                # Transform trajectory relative to reference point
                # (keep time as is, translate spatial features)
                transformed_traj = trajectories[cycle_idx].copy()
                
                # Translate position features (dimensions 1,2,5,6) relative to reference
                pos_dims = [1, 2, 5, 6]  # position dimensions
                for dim in pos_dims:
                    transformed_traj[:, dim] -= ref_values[dim]
                
                frame_trajectories.append(transformed_traj)
            
            # Concatenate all cycles for this frame
            frame_data = np.concatenate(frame_trajectories, axis=0)
            frames_data.append(frame_data)
            
        frames_array = np.array(frames_data)
        print(f"Created frames data shape: {frames_array.shape}")
        
        return frames_array


class GaitTPGMMTrainer:
    """Trains TPGMM model for gait analysis."""
    
    def __init__(self, n_components: int = 6, **tpgmm_kwargs):
        """
        Initialize the TPGMM trainer.
        
        Args:
            n_components: Number of Gaussian components
            **tpgmm_kwargs: Additional arguments for TPGMM
        """
        self.n_components = n_components
        
        # Set default parameters
        default_kwargs = {
            'verbose': True
        }
        default_kwargs.update(tpgmm_kwargs)
        
        self.tpgmm = TPGMM(n_components=n_components, **default_kwargs)
        self.is_trained = False
        
    def train(self, frames_data):
        """
        Train the TPGMM model.
        
        Args:
            frames_data: Array of shape (num_frames, total_points, num_features)
        """
        print(f"Training TPGMM with {self.n_components} components...")
        print(f"Input data shape: {frames_data.shape}")
        
        # Train the model
        self.tpgmm.fit(frames_data)
        self.is_trained = True
        
        print("TPGMM training completed!")
        print(f"Final log-likelihood: {self.tpgmm.log_likelihood_}")
        
    def save_model(self, save_path: str):
        """
        Save the trained TPGMM model to a pickle file.
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'tpgmm': self.tpgmm,
            'n_components': self.n_components,
            'weights_': self.tpgmm.weights_,
            'means_': self.tpgmm.means_,
            'covariances_': self.tpgmm.covariances_,
            'log_likelihood_': self.tpgmm.log_likelihood_,
            'feature_names': [
                'time', 'right_ankle_pos_x', 'right_ankle_pos_y', 
                'right_ankle_vel_x', 'right_ankle_vel_y',
                'left_ankle_pos_x', 'left_ankle_pos_y',
                'left_ankle_vel_x', 'left_ankle_vel_y'
            ],
            'feature_stats': None
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to: {save_path}")
        
    @staticmethod
    def load_model(model_path: str):
        """
        Load a trained TPGMM model from a pickle file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            dict: Model data including TPGMM instance and parameters
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"Model loaded from: {model_path}")
        print(f"Number of components: {model_data['n_components']}")
        print(f"Log-likelihood: {model_data['log_likelihood_']}")
        
        return model_data


def main():
    """Main training pipeline."""
    # Configuration
    json_path = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    model_save_path = "/home/jemajuinta/ws/Gait-analysis-coupled/gait_tpgmm_model.pkl"
    n_components = 6
    interpolation_points = 200
    
    print("=== Gait Analysis TPGMM Training ===")
    print(f"JSON data path: {json_path}")
    print(f"Model save path: {model_save_path}")
    print(f"Number of components: {n_components}")
    print(f"Interpolation points: {interpolation_points}")
    print()
    
    # Step 1: Process gait data
    print("Step 1: Processing gait data...")
    processor = GaitDataProcessor(json_path, interpolation_points)
    trajectories = processor.extract_trajectories()
    
    # Step 2: Create frames of reference
    print("\nStep 2: Creating frames of reference...")
    frames_data = processor.create_frames_of_reference(trajectories)
    
    # Step 3: Train TPGMM
    print("\nStep 3: Training TPGMM...")
    trainer = GaitTPGMMTrainer(n_components=n_components, verbose=True)
    trainer.train(frames_data)
    
    # Step 4: Save model
    print("\nStep 4: Saving model...")
    trainer.save_model(model_save_path)
    
    print("\n=== Training Complete ===")
    print(f"Trained model saved to: {model_save_path}")
    print("The model can now be loaded for Gaussian Mixture Regression (GMR).")


if __name__ == "__main__":
    main()