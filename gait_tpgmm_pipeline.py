#!/usr/bin/env python3
"""
Complete Gait Analysis TPGMM Pipeline

This script combines TPGMM training and GMR trajectory recovery into a single pipeline.
It includes:
1. Data loading and preprocessing
2. TPGMM model training with optimal component selection
3. Frame transformation extraction
4. GMR trajectory recovery with proper coordinate frame handling
5. Comprehensive visualization and analysis

Usage:
    python gait_tpgmm_pipeline.py [--train-only] [--recover-only] [--config CONFIG_FILE]
"""

import json
import numpy as np
import pickle
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# Add TaskPaGMMM to Python path
sys.path.append('TaskPaGMMM')
from tpgmm.tpgmm.tpgmm import TPGMM
from tpgmm.gmr.gmr import GaussianMixtureRegression


class GaitTPGMMPipeline:
    """Complete pipeline for TPGMM-based gait analysis."""
    
    def __init__(self, config=None):
        """Initialize the pipeline with configuration."""
        self.config = config or self._default_config()
        self.model_data = None
        self.frame_transforms = None
        self.original_plots_dir = None
        
    def _default_config(self):
        """Default configuration for the pipeline."""
        return {
            'json_path': "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json",
            'pkls_dir': "/home/jemajuinta/ws/Gait-analysis-coupled/pkls",
            'plots_dir': "testing_plots",
            'model_name': "gait_tpgmm_model_final.pkl",
            'recovery_name': "gait_tpgmm_recovery_final.pkl",
            'transforms_name': "frame_transformation_analysis.pkl",
            
            # TPGMM training parameters
            'reg_factor': 1e-3,
            'threshold': 1e-6,
            'component_range': (3, 16),
            'max_iter': 100,
            'min_iter': 5,
            
            # GMR parameters
            'sample_frame_idx': 0,
            'sample_trajectory_idx': 0,
            
            # Visualization parameters
            'figure_dpi': 300,
            'bbox_inches': 'tight'
        }
    
    def load_gait_data(self):
        """Load and process gait data from JSON file."""
        print("=== Loading Gait Data ===")
        json_path = self.config['json_path']
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get kinematics data from all 3 frames
        frames = ['FR1', 'FR2', 'FR3']
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
                # Extract ankle data for both legs
                right_ankle_pos = np.array(cycle_data['right_ankle_pos'])
                right_ankle_vel = np.array(cycle_data['right_ankle_vel']) 
                left_ankle_pos = np.array(cycle_data['left_ankle_pos'])
                left_ankle_vel = np.array(cycle_data['left_ankle_vel'])
                
                # Create time vector (normalized 0-1)
                time_vector = np.linspace(0, 1, interpolation_points)
                
                # Create 9-dimensional feature space
                trajectory = np.column_stack([
                    time_vector,               # time (0-1)
                    right_ankle_pos[0, :],     # right ankle X position
                    right_ankle_pos[1, :],     # right ankle Y position
                    right_ankle_vel[0, :],     # right ankle X velocity
                    right_ankle_vel[1, :],     # right ankle Y velocity
                    left_ankle_pos[0, :],      # left ankle X position
                    left_ankle_pos[1, :],      # left ankle Y position
                    left_ankle_vel[0, :],      # left ankle X velocity
                    left_ankle_vel[1, :]       # left ankle Y velocity
                ])
                
                frame_trajectories.append(trajectory)
            
            # Convert to numpy array and store
            frame_trajectories = np.array(frame_trajectories)
            all_trajectories.append(frame_trajectories)
            print(f"{frame} trajectories shape: {frame_trajectories.shape}")
        
        return all_trajectories
    
    def plot_input_trajectories(self, all_trajectories):
        """Plot input trajectories for all frames of reference."""
        print("=== Plotting Input Trajectories ===")
        frames = ['FR1', 'FR2', 'FR3']
        frame_colors = {'FR1': 'blue', 'FR2': 'red', 'FR3': 'green'}
        
        os.makedirs(self.config['plots_dir'], exist_ok=True)
        
        # Plot ankle positions
        fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(16, 8))
        
        for frame_idx, (frame_name, trajectories) in enumerate(zip(frames, all_trajectories)):
            color = frame_colors[frame_name]
            
            # Right ankle positions
            for traj_idx, traj in enumerate(trajectories):
                alpha = 0.6 if traj_idx > 0 else 0.8
                linewidth = 1.5 if traj_idx == 0 else 1
                label = frame_name if traj_idx == 0 else ""
                ax_right.plot(traj[:, 1], traj[:, 2], color=color, alpha=alpha, linewidth=linewidth, label=label)
            
            # Left ankle positions
            for traj_idx, traj in enumerate(trajectories):
                alpha = 0.6 if traj_idx > 0 else 0.8
                linewidth = 1.5 if traj_idx == 0 else 1
                label = frame_name if traj_idx == 0 else ""
                ax_left.plot(traj[:, 5], traj[:, 6], color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        ax_right.set_title('Right Ankle Position Trajectories\nAll Frames Combined', fontsize=14, fontweight='bold')
        ax_right.set_xlabel('X Position (m)')
        ax_right.set_ylabel('Y Position (m)')
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
        ax_right.axis('equal')
        
        ax_left.set_title('Left Ankle Position Trajectories\nAll Frames Combined', fontsize=14, fontweight='bold')
        ax_left.set_xlabel('X Position (m)')
        ax_left.set_ylabel('Y Position (m)')
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()
        ax_left.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.config["plots_dir"]}/input_trajectories_positions.png', 
                   dpi=self.config['figure_dpi'], bbox_inches=self.config['bbox_inches'])
        plt.close()
        
        print("Input trajectory plots saved")
    
    def prepare_tpgmm_data(self, all_trajectories):
        """Prepare data for TPGMM training."""
        print("=== Preparing TPGMM Data ===")
        
        num_frames = len(all_trajectories)
        num_trajectories, num_samples, num_features = all_trajectories[0].shape
        
        # Stack the data for all frames
        reshaped_trajectories = np.stack(all_trajectories, axis=0)
        
        # Reshape to (num_frames, num_trajectories * num_samples, num_features)
        reshaped_trajectories = reshaped_trajectories.reshape(num_frames, num_trajectories * num_samples, num_features)
        
        print(f"Reshaped trajectories shape: {reshaped_trajectories.shape}")
        print(f"Features: [time, right_ankle_pos_x, right_ankle_pos_y, right_ankle_vel_x, right_ankle_vel_y,")
        print(f"          left_ankle_pos_x, left_ankle_pos_y, left_ankle_vel_x, left_ankle_vel_y]")
        
        return reshaped_trajectories
    
    def find_optimal_components(self, reshaped_trajectories):
        """Find optimal number of components using BIC scores."""
        print("=== Finding Optimal Components ===")
        
        os.makedirs(self.config['plots_dir'], exist_ok=True)
        
        component_range = self.config['component_range']
        n_components_list = []
        bic_scores = []
        
        best_n_components = None
        lowest_bic_score = float('inf')
        
        # Loop through n_components
        for n_components in range(component_range[0], component_range[1]):
            print(f'Fitting TPGMM with n_components={n_components}...')
            
            # Define the TPGMM model
            tpgmm = TPGMM(
                n_components=n_components, 
                verbose=False, 
                threshold=self.config['threshold'], 
                reg_factor=1e-8,  # Use smaller reg_factor for model selection
                max_iter=self.config['max_iter'],
                min_iter=self.config['min_iter']
            )
            
            # Fit the model
            tpgmm.fit(reshaped_trajectories)
            
            # Calculate BIC score
            bic_score = tpgmm.bic(reshaped_trajectories)
            print(f'n_components={n_components}: BIC={bic_score:.2f}')
            
            # Store results
            n_components_list.append(n_components)
            bic_scores.append(bic_score)
            
            # Update best components
            if bic_score < lowest_bic_score:
                lowest_bic_score = bic_score
                best_n_components = n_components
        
        # Plot BIC scores
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax1.plot(n_components_list, bic_scores, 'bo-', linewidth=2, markersize=8, label='BIC')
        ax1.axvline(x=best_n_components, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal (n={best_n_components})')
        ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax1.set_ylabel('BIC Score', fontsize=12, fontweight='bold')
        ax1.set_title('Bayesian Information Criterion (BIC)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.config["plots_dir"]}/model_selection_criteria.png', 
                   dpi=self.config['figure_dpi'], bbox_inches=self.config['bbox_inches'])
        plt.close()
        
        print(f'BIC optimal n_components: {best_n_components} (BIC = {lowest_bic_score:.2f})')
        
        return {
            'best_n_components': best_n_components,
            'n_components_list': n_components_list,
            'bic_scores': bic_scores,
            'lowest_bic_score': lowest_bic_score
        }
    
    def train_tpgmm(self, reshaped_trajectories, all_trajectories):
        """Train the final TPGMM model."""
        print("=== Training TPGMM ===")
        
        # Find optimal components
        model_selection_results = self.find_optimal_components(reshaped_trajectories)
        best_n_components = model_selection_results['best_n_components']
        
        # Train final model with optimal components
        print(f"Training final TPGMM with n_components={best_n_components}...")
        tpgmm = TPGMM(
            n_components=best_n_components, 
            verbose=True, 
            reg_factor=self.config['reg_factor'], 
            threshold=self.config['threshold'],
            max_iter=self.config['max_iter'],
            min_iter=self.config['min_iter']
        )
        tpgmm.fit(reshaped_trajectories)
        
        print("TPGMM training completed!")
        
        # Prepare model data
        feature_names = [
            'time', 'right_ankle_pos_x', 'right_ankle_pos_y', 'right_ankle_vel_x', 'right_ankle_vel_y',
            'left_ankle_pos_x', 'left_ankle_pos_y', 'left_ankle_vel_x', 'left_ankle_vel_y'
        ]
        
        self.model_data = {
            'tpgmm': tpgmm,
            'n_components': best_n_components,
            'weights_': tpgmm.weights_,
            'means_': tpgmm.means_,
            'covariances_': tpgmm.covariances_,
            'all_trajectories': all_trajectories,
            'reshaped_trajectories': reshaped_trajectories,
            'model_selection_results': model_selection_results,
            'feature_names': feature_names,
            'frame_names': ['FR1', 'FR2', 'FR3'],
            'num_frames': 3,
            'feature_dims': 9,
            'config': self.config
        }
        
        # Save model
        os.makedirs(self.config['pkls_dir'], exist_ok=True)
        model_path = os.path.join(self.config['pkls_dir'], self.config['model_name'])
        with open(model_path, 'wb') as f:
            pickle.dump(self.model_data, f)
        
        print(f"Model saved to: {model_path}")
        
        return self.model_data
    
    def extract_frame_transformations(self):
        """Extract frame transformations from training data."""
        print("=== Extracting Frame Transformations ===")
        
        if self.model_data is None:
            raise ValueError("Model data not available. Train model first or load existing model.")
        
        all_trajectories = self.model_data['all_trajectories']
        feature_names = self.model_data['feature_names']
        
        # Get position feature indices (exclude time)
        pos_indices = [i for i, name in enumerate(feature_names) if 'pos' in name]
        print(f"Position feature indices: {pos_indices}")
        
        # Extract position data for each frame
        frame_positions = {}
        frame_means = {}
        
        for frame_idx, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
            trajectories = all_trajectories[frame_idx]
            
            # Extract position data (exclude time column)
            position_data = trajectories[:, :, pos_indices]
            
            # Flatten to get all position points
            flattened_positions = position_data.reshape(-1, len(pos_indices))
            
            frame_positions[frame_name] = flattened_positions
            frame_means[frame_name] = np.mean(flattened_positions, axis=0)
            
            print(f"{frame_name}: {flattened_positions.shape[0]} position points")
            print(f"  Mean positions: {frame_means[frame_name]}")
        
        # Calculate translation vectors (using FR1 as reference)
        translations = {}
        translations['FR1'] = np.zeros(len(pos_indices))  # Reference frame
        translations['FR2'] = frame_means['FR2'] - frame_means['FR1']
        translations['FR3'] = frame_means['FR3'] - frame_means['FR1']
        
        print("\nTranslation vectors (relative to FR1):")
        for frame_name, translation in translations.items():
            print(f"{frame_name}: {translation}")
        
        self.frame_transforms = {
            'translations': translations,
            'frame_positions': frame_positions,
            'frame_means': frame_means,
            'position_indices': pos_indices,
            'position_features': [feature_names[i] for i in pos_indices]
        }
        
        # Save transformations
        transform_path = os.path.join(self.config['pkls_dir'], self.config['transforms_name'])
        with open(transform_path, 'wb') as f:
            pickle.dump(self.frame_transforms, f)
        
        print(f"Frame transformations saved to: {transform_path}")
        
        return self.frame_transforms
    
    def predict_using_gmr(self, time_input, feature_idx_to_predict):
        """Perform GMR prediction with proper frame transformations."""
        print("=== Performing GMR Prediction ===")
        
        if self.model_data is None or self.frame_transforms is None:
            raise ValueError("Model data or frame transformations not available.")
        
        tpgmm = self.model_data['tpgmm']
        
        # Create GMR instance with time as input
        gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0])
        
        # Get frame transformations
        translations_dict = self.frame_transforms['translations']
        
        # Extract position feature translations
        position_features = [1, 2, 5, 6]  # indices for position features in original data
        
        # Map feature indices to position-only indices
        pos_translation_map = {}
        for i, orig_idx in enumerate(position_features):
            if orig_idx in feature_idx_to_predict:
                pos_idx = feature_idx_to_predict.index(orig_idx)
                pos_translation_map[pos_idx] = i
        
        num_output_features = len(feature_idx_to_predict)
        
        # Create translation matrices for 3 frames
        translation = np.zeros((3, num_output_features))
        
        # Apply extracted translations for position features only (inverted)
        for frame_idx, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
            frame_translation = translations_dict[frame_name]
            
            # Map position translations to output feature indices (with inversion)
            for out_idx, pos_idx in pos_translation_map.items():
                translation[frame_idx, out_idx] = -frame_translation[pos_idx]  # Invert
        
        # Create identity rotation matrices
        rotation_matrix = np.eye(num_output_features)[None].repeat(3, axis=0)
        
        print(f"GMR setup with proper frame transformations:")
        print(f"  Translation shape: {translation.shape}")
        print(f"  Rotation shape: {rotation_matrix.shape}")
        for i, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
            print(f"    {frame_name}: {translation[i]}")
        
        # Fit and predict
        gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
        time_input_reshaped = time_input.reshape(-1, 1)
        predicted_output, predicted_covariance = gmr.predict(time_input_reshaped)
        
        return predicted_output, predicted_covariance
    
    def plot_gaussian_models_with_recovery(self, predicted_trajectory=None, predicted_covariance=None):
        """Plot Gaussian models with recovered trajectory overlay."""
        print("=== Creating Gaussian Models Visualization ===")
        
        if self.model_data is None:
            raise ValueError("Model data not available.")
        
        os.makedirs(self.config['plots_dir'], exist_ok=True)
        
        tpgmm = self.model_data['tpgmm']
        
        # Use the first frame (FR1)
        frame_idx = 0
        means = tpgmm.means_[frame_idx]
        covariances = tpgmm.covariances_[frame_idx]
        weights = tpgmm.weights_
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define colors for each Gaussian component
        colors = plt.cm.tab10(np.linspace(0, 1, tpgmm._n_components))
        
        # Plot recovered trajectory if provided
        if predicted_trajectory is not None:
            # Left ankle positions (dims 4, 5 in predicted_trajectory)
            ax1.plot(predicted_trajectory[:, 4], predicted_trajectory[:, 5], 'k-', 
                    linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
            ax1.plot(predicted_trajectory[0, 4], predicted_trajectory[0, 5], 'ko', 
                    markersize=8, label='Start', zorder=11)
            ax1.plot(predicted_trajectory[-1, 4], predicted_trajectory[-1, 5], 'ks', 
                    markersize=8, label='End', zorder=11)
            
            # Right ankle positions (dims 0, 1 in predicted_trajectory)
            ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'k-', 
                    linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
            ax2.plot(predicted_trajectory[0, 0], predicted_trajectory[0, 1], 'ko', 
                    markersize=8, label='Start', zorder=11)
            ax2.plot(predicted_trajectory[-1, 0], predicted_trajectory[-1, 1], 'ks', 
                    markersize=8, label='End', zorder=11)
            
            # Left ankle velocities (dims 6, 7 in predicted_trajectory)
            ax3.plot(predicted_trajectory[:, 6], predicted_trajectory[:, 7], 'k-', 
                    linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
            ax3.plot(predicted_trajectory[0, 6], predicted_trajectory[0, 7], 'ko', 
                    markersize=8, label='Start', zorder=11)
            ax3.plot(predicted_trajectory[-1, 6], predicted_trajectory[-1, 7], 'ks', 
                    markersize=8, label='End', zorder=11)
            
            # Right ankle velocities (dims 2, 3 in predicted_trajectory)
            ax4.plot(predicted_trajectory[:, 2], predicted_trajectory[:, 3], 'k-', 
                    linewidth=3, alpha=0.8, label='GMR Recovery', zorder=10)
            ax4.plot(predicted_trajectory[0, 2], predicted_trajectory[0, 3], 'ko', 
                    markersize=8, label='Start', zorder=11)
            ax4.plot(predicted_trajectory[-1, 2], predicted_trajectory[-1, 3], 'ks', 
                    markersize=8, label='End', zorder=11)
        
        # Plot Gaussians for each subplot
        plot_configs = [
            (ax1, [5, 6], 'Left Ankle Position Gaussians\n(Dimensions 5, 6)', 'Left Ankle X Position (m)', 'Left Ankle Y Position (m)'),
            (ax2, [1, 2], 'Right Ankle Position Gaussians\n(Dimensions 1, 2)', 'Right Ankle X Position (m)', 'Right Ankle Y Position (m)'),
            (ax3, [7, 8], 'Left Ankle Velocity Gaussians\n(Dimensions 7, 8)', 'Left Ankle X Velocity (m/s)', 'Left Ankle Y Velocity (m/s)'),
            (ax4, [3, 4], 'Right Ankle Velocity Gaussians\n(Dimensions 3, 4)', 'Right Ankle X Velocity (m/s)', 'Right Ankle Y Velocity (m/s)')
        ]
        
        for ax, dims, title, xlabel, ylabel in plot_configs:
            for k in range(tpgmm._n_components):
                mean_vals = means[k, dims]
                cov_vals = covariances[k][np.ix_(dims, dims)]
                
                # Plot mean as point
                ax.scatter(mean_vals[0], mean_vals[1], c=[colors[k]], s=100*weights[k]*10, 
                          alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
                
                # Plot covariance as ellipse
                eigenvals, eigenvecs = np.linalg.eigh(cov_vals)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
                
                ellipse = Ellipse(mean_vals, width, height, angle=angle, 
                                facecolor=colors[k], alpha=0.3, edgecolor=colors[k], linewidth=2, zorder=1)
                ax.add_patch(ellipse)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            if predicted_trajectory is not None:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.config["plots_dir"]}/tpgmm_complete_analysis.png', 
                   dpi=self.config['figure_dpi'], bbox_inches=self.config['bbox_inches'])
        plt.close()
        
        print("Gaussian models visualization saved")
    
    def apply_pca_to_features(self, trajectories, feature_names, n_components=2):
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
    
    def plot_recovery_results(self, original_trajectory, predicted_trajectory, pca_results, feature_names):
        """Plot trajectory recovery results with PCA visualization."""
        print("=== Creating Recovery Results Visualization ===")
        
        os.makedirs(self.config['plots_dir'], exist_ok=True)
        
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
            
            axes[row, col].plot(time_data, original_trajectory[:, feat_idx], 'b-', label='Original', linewidth=2)
            axes[row, col].plot(time_data, predicted_trajectory[:, feat_idx-1], 'r--', label='Predicted', linewidth=2)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Value')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # PCA plots
        if 'position' in pca_results:
            pca_pos = pca_results['position']['pca_data']
            axes[0, 2].plot(pca_pos[:, 0], pca_pos[:, 1], 'g-', linewidth=2)
            axes[0, 2].set_title('Position PCA (First 2 Components)')
            axes[0, 2].set_xlabel(f'PC1 ({pca_results["position"]["explained_variance"][0]:.2%} variance)')
            axes[0, 2].set_ylabel(f'PC2 ({pca_results["position"]["explained_variance"][1]:.2%} variance)')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].axis('equal')
        
        if 'velocity' in pca_results:
            pca_vel = pca_results['velocity']['pca_data']
            axes[0, 3].plot(pca_vel[:, 0], pca_vel[:, 1], 'm-', linewidth=2)
            axes[0, 3].set_title('Velocity PCA (First 2 Components)')
            axes[0, 3].set_xlabel(f'PC1 ({pca_results["velocity"]["explained_variance"][0]:.2%} variance)')
            axes[0, 3].set_ylabel(f'PC2 ({pca_results["velocity"]["explained_variance"][1]:.2%} variance)')
            axes[0, 3].grid(True, alpha=0.3)
            axes[0, 3].axis('equal')
        
        # Additional PCA comparison plots
        if 'position' in pca_results:
            # Plot original vs PCA reconstruction for positions
            pos_original = pca_results['position']['original_data']
            axes[1, 2].plot(pos_original[:, 0], pos_original[:, 1], 'b-', label='Right Ankle', linewidth=2)
            axes[1, 2].plot(pos_original[:, 2], pos_original[:, 3], 'r-', label='Left Ankle', linewidth=2)
            axes[1, 2].set_title('Original Position Trajectories')
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
            axes[1, 3].set_title('Original Velocity Trajectories')
            axes[1, 3].set_xlabel('X Velocity (m/s)')
            axes[1, 3].set_ylabel('Y Velocity (m/s)')
            axes[1, 3].legend()
            axes[1, 3].grid(True, alpha=0.3)
            axes[1, 3].axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.config["plots_dir"]}/tpgmm_gmr_recovery_results.png', 
                   dpi=self.config['figure_dpi'], bbox_inches=self.config['bbox_inches'])
        plt.close()
        
        print("Recovery results plots saved")
    
    def run_complete_pipeline(self):
        """Run the complete TPGMM pipeline from training to recovery."""
        print("=== Running Complete TPGMM Pipeline ===")
        
        # Step 1: Load data
        all_trajectories = self.load_gait_data()
        
        # Step 2: Plot input trajectories
        self.plot_input_trajectories(all_trajectories)
        
        # Step 3: Prepare TPGMM data
        reshaped_trajectories = self.prepare_tpgmm_data(all_trajectories)
        
        # Step 4: Train TPGMM
        self.train_tpgmm(reshaped_trajectories, all_trajectories)
        
        # Step 5: Extract frame transformations
        self.extract_frame_transformations()
        
        # Step 6: Perform GMR recovery
        sample_trajectory = self.model_data['all_trajectories'][
            self.config['sample_frame_idx']
        ][self.config['sample_trajectory_idx']]
        
        time_input = sample_trajectory[:, 0]
        output_feature_indices = list(range(1, len(self.model_data['feature_names'])))
        
        predicted_output, predicted_covariance = self.predict_using_gmr(
            time_input, output_feature_indices
        )
        
        print(f"Predicted output shape: {predicted_output.shape}")
        print(f"Predicted covariance shape: {predicted_covariance.shape}")
        
        # Step 7: Apply PCA analysis
        pca_results = self.apply_pca_to_features(predicted_output, self.model_data['feature_names'][1:])
        
        # Step 8: Create comprehensive visualization
        self.plot_gaussian_models_with_recovery(predicted_output, predicted_covariance)
        
        # Step 9: Create recovery results visualization
        self.plot_recovery_results(sample_trajectory, predicted_output, pca_results, self.model_data['feature_names'])
        
        # Step 10: Save recovery results
        recovery_data = {
            'original_trajectory': sample_trajectory,
            'predicted_trajectory': predicted_output,
            'prediction_covariance': predicted_covariance,
            'pca_results': pca_results,
            'feature_names': self.model_data['feature_names'],
            'time_input': time_input,
            'output_feature_indices': output_feature_indices,
            'config': self.config
        }
        
        recovery_path = os.path.join(self.config['pkls_dir'], self.config['recovery_name'])
        with open(recovery_path, 'wb') as f:
            pickle.dump(recovery_data, f)
        
        print(f"Recovery data saved to: {recovery_path}")
        print("=== Complete Pipeline Finished Successfully ===")
        
        return recovery_data
    
    def load_existing_model(self):
        """Load existing trained model."""
        model_path = os.path.join(self.config['pkls_dir'], self.config['model_name'])
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        transform_path = os.path.join(self.config['pkls_dir'], self.config['transforms_name'])
        if os.path.exists(transform_path):
            with open(transform_path, 'rb') as f:
                self.frame_transforms = pickle.load(f)
        
        print(f"Loaded existing model from: {model_path}")
        return self.model_data
    
    def update_config_for_sweep(self, reg_factor, threshold):
        """Update configuration for parameter sweep."""
        if self.original_plots_dir is None:
            self.original_plots_dir = self.config['plots_dir']
        
        # Update parameters
        self.config['reg_factor'] = reg_factor
        self.config['threshold'] = threshold
        
        # Update plots directory with parameter values
        param_dir = f"reg_factor={reg_factor:.0e}_threshold={threshold:.0e}"
        self.config['plots_dir'] = os.path.join(self.original_plots_dir, param_dir)
        
        # Update model and recovery filenames
        self.config['model_name'] = f"gait_tpgmm_model_reg{reg_factor:.0e}_thresh{threshold:.0e}.pkl"
        self.config['recovery_name'] = f"gait_tpgmm_recovery_reg{reg_factor:.0e}_thresh{threshold:.0e}.pkl"
    
    def run_parameter_sweep(self, reg_factors, thresholds):
        """Run parameter sweep with preloaded trajectories."""
        print("=== Running Parameter Sweep ===" )
        
        # Load and plot trajectories only once
        print("Loading trajectories (once)...")
        all_trajectories = self.load_gait_data()
        self.plot_input_trajectories(all_trajectories)
        reshaped_trajectories = self.prepare_tpgmm_data(all_trajectories)
        
        results = {}
        total_combinations = len(reg_factors) * len(thresholds)
        current_combination = 0
        
        for reg_factor in reg_factors:
            for threshold in thresholds:
                current_combination += 1
                print(f"\n=== Parameter Combination {current_combination}/{total_combinations} ===")
                print(f"reg_factor: {reg_factor:.0e}, threshold: {threshold:.0e}")
                
                # Update configuration for this parameter combination
                self.update_config_for_sweep(reg_factor, threshold)
                
                try:
                    # Train model with current parameters
                    model_data = self.train_tpgmm(reshaped_trajectories, all_trajectories)
                    
                    # Extract frame transformations
                    self.extract_frame_transformations()
                    
                    # Perform recovery
                    sample_trajectory = model_data['all_trajectories'][
                        self.config['sample_frame_idx']
                    ][self.config['sample_trajectory_idx']]
                    
                    time_input = sample_trajectory[:, 0]
                    output_feature_indices = list(range(1, len(model_data['feature_names'])))
                    
                    predicted_output, predicted_covariance = self.predict_using_gmr(
                        time_input, output_feature_indices
                    )
                    
                    # Apply PCA analysis
                    pca_results = self.apply_pca_to_features(predicted_output, model_data['feature_names'][1:])
                    
                    # Create visualizations
                    self.plot_gaussian_models_with_recovery(predicted_output, predicted_covariance)
                    self.plot_recovery_results(sample_trajectory, predicted_output, pca_results, model_data['feature_names'])
                    
                    # Save recovery results
                    recovery_data = {
                        'original_trajectory': sample_trajectory,
                        'predicted_trajectory': predicted_output,
                        'prediction_covariance': predicted_covariance,
                        'pca_results': pca_results,
                        'feature_names': model_data['feature_names'],
                        'time_input': time_input,
                        'output_feature_indices': output_feature_indices,
                        'config': self.config.copy()
                    }
                    
                    recovery_path = os.path.join(self.config['pkls_dir'], self.config['recovery_name'])
                    with open(recovery_path, 'wb') as f:
                        pickle.dump(recovery_data, f)
                    
                    # Store results
                    results[(reg_factor, threshold)] = {
                        'model_data': model_data,
                        'recovery_data': recovery_data,
                        'plots_dir': self.config['plots_dir']
                    }
                    
                    print(f"✓ Completed: reg_factor={reg_factor:.0e}, threshold={threshold:.0e}")
                    
                except Exception as e:
                    print(f"✗ Failed: reg_factor={reg_factor:.0e}, threshold={threshold:.0e}")
                    print(f"  Error: {str(e)}")
                    results[(reg_factor, threshold)] = {'error': str(e)}
        
        print(f"\n=== Parameter Sweep Complete ===")
        print(f"Processed {len(results)} parameter combinations")
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"Successful: {successful}/{len(results)}")
        
        return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Complete Gait Analysis TPGMM Pipeline')
    parser.add_argument('--train-only', action='store_true', help='Only perform training')
    parser.add_argument('--recover-only', action='store_true', help='Only perform recovery (requires existing model)')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--param-sweep', action='store_true', help='Perform parameter sweep with powers of 10')
    parser.add_argument('--reg-min', type=float, default=1e-4, help='Minimum reg_factor value as power of 10 (default: 1e-4)')
    parser.add_argument('--reg-max', type=float, default=1e-1, help='Maximum reg_factor value as power of 10 (default: 1e-1)')
    parser.add_argument('--thresh-min', type=float, default=1e-6, help='Minimum threshold value as power of 10 (default: 1e-6)')
    parser.add_argument('--thresh-max', type=float, default=1e-2, help='Maximum threshold value as power of 10 (default: 1e-2)')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize pipeline
    pipeline = GaitTPGMMPipeline(config)
    
    if args.param_sweep:
        # Generate parameter ranges as powers of 10
        import numpy as np
        
        # Calculate the power range for reg_factor
        reg_min_exp = int(np.log10(args.reg_min))
        reg_max_exp = int(np.log10(args.reg_max))
        reg_factors = [10**i for i in range(reg_min_exp, reg_max_exp + 1)]
        
        # Calculate the power range for threshold
        thresh_min_exp = int(np.log10(args.thresh_min))
        thresh_max_exp = int(np.log10(args.thresh_max))
        thresholds = [10**i for i in range(thresh_min_exp, thresh_max_exp + 1)]
        
        print(f"Parameter sweep configuration:")
        print(f"  reg_factor range: {args.reg_min:.0e} to {args.reg_max:.0e}")
        print(f"  threshold range: {args.thresh_min:.0e} to {args.thresh_max:.0e}")
        print(f"  reg_factor values: {[f'{rf:.0e}' for rf in reg_factors]}")
        print(f"  threshold values: {[f'{th:.0e}' for th in thresholds]}")
        print(f"  Total combinations: {len(reg_factors) * len(thresholds)}")
        
        # Run parameter sweep
        results = pipeline.run_parameter_sweep(reg_factors, thresholds)
        
    elif args.recover_only:
        # Load existing model and perform recovery only
        pipeline.load_existing_model()
        if pipeline.frame_transforms is None:
            pipeline.extract_frame_transformations()
        
        sample_trajectory = pipeline.model_data['all_trajectories'][
            pipeline.config['sample_frame_idx']
        ][pipeline.config['sample_trajectory_idx']]
        
        time_input = sample_trajectory[:, 0]
        output_feature_indices = list(range(1, len(pipeline.model_data['feature_names'])))
        
        predicted_output, predicted_covariance = pipeline.predict_using_gmr(
            time_input, output_feature_indices
        )
        
        pipeline.plot_gaussian_models_with_recovery(predicted_output, predicted_covariance)
        
    elif args.train_only:
        # Perform training only
        all_trajectories = pipeline.load_gait_data()
        pipeline.plot_input_trajectories(all_trajectories)
        reshaped_trajectories = pipeline.prepare_tpgmm_data(all_trajectories)
        pipeline.train_tpgmm(reshaped_trajectories, all_trajectories)
        pipeline.extract_frame_transformations()
        
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()