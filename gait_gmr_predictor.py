#!/usr/bin/env python3
"""
Gait Analysis GMR Prediction Script

This script loads a trained TPGMM model and performs Gaussian Mixture Regression (GMR)
for gait trajectory prediction and analysis.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add the TPGMM module to path
sys.path.append(str(Path(__file__).parent / "TaskParameterizedGaussianMixtureModels"))

from tpgmm.gmr.gmr import GaussianMixtureRegression


class GaitGMRPredictor:
    """Performs GMR prediction using trained gait TPGMM model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the GMR predictor.
        
        Args:
            model_path: Path to the saved TPGMM model
        """
        self.model_path = model_path
        self.model_data = None
        self.gmr = None
        self.load_model()
        
    def load_model(self):
        """Load the trained TPGMM model."""
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
            
        print(f"Loaded TPGMM model from: {self.model_path}")
        print(f"Number of components: {self.model_data['n_components']}")
        print(f"Model log-likelihood: {self.model_data['log_likelihood_']}")
        print(f"Feature names: {self.model_data['feature_names']}")
        
        # Create GMR instance from TPGMM
        tpgmm = self.model_data['tpgmm']
        
        # Time is input (dimension 0), spatial features are output (dimensions 1-8)
        input_indices = [0]  # time dimension
        
        self.gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_indices)
        
        print("GMR predictor initialized successfully!")
        
    def predict_trajectory(self, time_points: np.ndarray, 
                          start_frame_translation: np.ndarray = None,
                          mid_frame_translation: np.ndarray = None,
                          end_frame_translation: np.ndarray = None):
        """
        Predict gait trajectory using GMR.
        
        Args:
            time_points: Array of time points for prediction (normalized 0-1)
            start_frame_translation: Translation for start frame (shape: 8,)
            mid_frame_translation: Translation for mid frame (shape: 8,)  
            end_frame_translation: Translation for end frame (shape: 8,)
            
        Returns:
            tuple: (predicted_means, predicted_covariances)
        """
        if self.gmr is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Default translations (can be customized based on specific gait requirements)
        if start_frame_translation is None:
            start_frame_translation = np.zeros(8)  # All spatial features at origin
            
        if mid_frame_translation is None:
            mid_frame_translation = np.zeros(8)  # Mid-cycle reference
            
        if end_frame_translation is None:
            end_frame_translation = np.zeros(8)  # End-cycle reference
            
        # Combine translations for all 3 frames
        translation = np.array([start_frame_translation, 
                               mid_frame_translation, 
                               end_frame_translation])
        
        # Identity rotation matrices (no rotation)
        rotation_matrix = np.eye(8)[None].repeat(3, axis=0)
        
        # Fit GMR with frame transformations
        try:
            self.gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Singular matrix encountered. Using regularization.")
            # Add small regularization to avoid singular matrices
            for i in range(3):
                rotation_matrix[i] += np.eye(8) * 1e-6
            self.gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
        
        # Prepare input data (time points)
        input_data = time_points.reshape(-1, 1)
        
        # Predict using GMR
        predicted_means, predicted_covariances = self.gmr.predict(input_data)
        
        return predicted_means, predicted_covariances
        
    def predict_full_gait_cycle(self, num_points: int = 200):
        """
        Predict a complete gait cycle.
        
        Args:
            num_points: Number of points to predict
            
        Returns:
            dict: Dictionary containing predicted trajectory and metadata
        """
        # Create time vector for full gait cycle
        time_points = np.linspace(0, 1, num_points)
        
        # Predict trajectory
        means, covariances = self.predict_trajectory(time_points)
        
        # Extract individual features
        feature_names = self.model_data['feature_names'][1:]  # Exclude time
        
        trajectory_dict = {
            'time': time_points,
            'predicted_means': means,
            'predicted_covariances': covariances,
            'features': {}
        }
        
        # Organize by feature names
        for i, feature_name in enumerate(feature_names):
            trajectory_dict['features'][feature_name] = means[:, i]
            
        return trajectory_dict
        
    def plot_predictions(self, trajectory_dict: dict, save_path: str = None):
        """
        Plot predicted gait trajectories.
        
        Args:
            trajectory_dict: Dictionary from predict_full_gait_cycle()
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Predicted Gait Trajectories', fontsize=16)
        
        time = trajectory_dict['time']
        features = trajectory_dict['features']
        
        # Plot each feature
        feature_names = list(features.keys())
        
        for i, (ax, feature_name) in enumerate(zip(axes.flat, feature_names)):
            feature_data = features[feature_name]
            
            ax.plot(time, feature_data, 'b-', linewidth=2, label='Predicted')
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel(feature_name.replace('_', ' ').title())
            ax.set_title(f'{feature_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            
        plt.show()
        
    def compare_with_original(self, json_path: str, cycle_index: int = 0):
        """
        Compare predictions with original data from JSON file.
        
        Args:
            json_path: Path to original gait analysis JSON
            cycle_index: Index of cycle to compare with
        """
        import json
        
        # Load original data
        with open(json_path, 'r') as f:
            original_data = json.load(f)
            
        # Extract original trajectory for comparison
        cycle_data = original_data['kinematics_data']['FR1']['right_leg_kinematics'][cycle_index]
        
        # Predict trajectory
        predicted_dict = self.predict_full_gait_cycle()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Predicted vs Original Gait Cycle {cycle_index}', fontsize=16)
        
        time = predicted_dict['time']
        
        # Original data mapping
        original_mapping = {
            'right_ankle_pos_x': np.array(cycle_data['right_ankle_pos'][0]),
            'right_ankle_pos_y': np.array(cycle_data['right_ankle_pos'][1]),
            'right_ankle_vel_x': np.array(cycle_data['right_ankle_vel'][0]),
            'right_ankle_vel_y': np.array(cycle_data['right_ankle_vel'][1]),
            'left_ankle_pos_x': np.array(cycle_data['left_ankle_pos'][0]),
            'left_ankle_pos_y': np.array(cycle_data['left_ankle_pos'][1]),
            'left_ankle_vel_x': np.array(cycle_data['left_ankle_vel'][0]),
            'left_ankle_vel_y': np.array(cycle_data['left_ankle_vel'][1])
        }
        
        for i, (ax, feature_name) in enumerate(zip(axes.flat, original_mapping.keys())):
            # Original data
            original_values = original_mapping[feature_name]
            original_time = np.linspace(0, 1, len(original_values))
            
            # Predicted data
            predicted_values = predicted_dict['features'][feature_name]
            
            ax.plot(original_time, original_values, 'r-', linewidth=2, label='Original', alpha=0.7)
            ax.plot(time, predicted_values, 'b--', linewidth=2, label='Predicted')
            ax.set_xlabel('Normalized Time')
            ax.set_ylabel(feature_name.replace('_', ' ').title())
            ax.set_title(f'{feature_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        plt.show()


def demo_gmr_prediction():
    """Demonstration of GMR prediction functionality."""
    print("=== Gait Analysis GMR Prediction Demo ===")
    
    # Configuration
    model_path = "/home/jemajuinta/ws/Gait-analysis-coupled/gait_tpgmm_model.pkl"
    json_path = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    
    try:
        # Initialize predictor
        print("Loading trained TPGMM model...")
        predictor = GaitGMRPredictor(model_path)
        
        # Predict full gait cycle
        print("\nPredicting full gait cycle...")
        trajectory = predictor.predict_full_gait_cycle()
        
        print(f"Predicted trajectory shape: {trajectory['predicted_means'].shape}")
        print(f"Features: {list(trajectory['features'].keys())}")
        
        # Plot predictions
        print("\nGenerating plots...")
        predictor.plot_predictions(trajectory, save_path="gait_predictions.png")
        
        # Compare with original (if JSON file exists)
        try:
            print("\nComparing with original data...")
            predictor.compare_with_original(json_path, cycle_index=0)
        except FileNotFoundError:
            print("Original JSON file not found, skipping comparison.")
            
        print("\n=== Demo Complete ===")
        
    except FileNotFoundError:
        print(f"Model file not found at: {model_path}")
        print("Please train the model first using gait_tpgmm_trainer.py")


if __name__ == "__main__":
    demo_gmr_prediction()