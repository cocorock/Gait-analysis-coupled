"""
FIXED: Gait Analysis using Task-Parameterized Gaussian Mixture Models (TPGMM)
This version properly handles trajectory data instead of individual points.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our implementations
from tpgmm_implementation import TPGMM
from gmr_implementation import TaskParameterizedGMR

# Set style for better plots
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    print("Seaborn not available, using default matplotlib style")
    plt.style.use('default')

class GaitAnalysisTTPGMM:
    """
    FIXED: Complete gait analysis pipeline using TPGMM
    Now properly handles trajectory-level modeling instead of point-level
    """
    
    def __init__(self, data_path):
        """
        Initialize with path to gait data JSON file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.tpgmm_model = None
        self.gmr_model = None
        self.scaler = StandardScaler()
        
        # Analysis parameters
        self.feature_names = [
            'x_pos', 'y_pos', 'x_vel', 'y_vel', 
            'ankle_right_rad', 'ankle_left_rad', 'time'
        ]
        
        print("=" * 60)
        print("FIXED GAIT ANALYSIS WITH TASK-PARAMETERIZED GMM")
        print("=" * 60)
        print(f"Data source: {data_path}")
        print("Implementation: Custom TPGMM with TRAJECTORY-LEVEL modeling")
        print()
    
    def load_and_process_data(self):
        """
        Load and process the gait analysis JSON data - FIXED VERSION
        """
        print("STEP 1: Loading and Processing Data")
        print("-" * 40)
        
        # Load JSON data
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        print(f"✓ Loaded data from: {os.path.basename(self.data_path)}")
        print(f"  Creation date: {self.raw_data['export_info']['creation_date']}")
        print(f"  Number of source files: {len(self.raw_data['export_info']['source_files'])}")
        print()
        
        # Process kinematics data
        fr1_data = self.raw_data['kinematics_data']['FR1']
        fr2_data = self.raw_data['kinematics_data']['FR2']
        
        # Extract trajectories for FR1 and FR2
        trajectories_fr1, trajectories_fr2 = self._extract_trajectories(fr1_data, fr2_data)
        
        print(f"✓ Extracted trajectories:")
        print(f"  FR1 shape: {trajectories_fr1.shape}")
        print(f"  FR2 shape: {trajectories_fr2.shape}")
        print(f"  Features: {self.feature_names}")
        print(f"  Number of demonstrations: {len(trajectories_fr1)}")
        print()
        
        # FIXED: Prepare data for TPGMM with trajectory features instead of all points
        self.processed_data = self._prepare_tpgmm_data_fixed(trajectories_fr1, trajectories_fr2)
        
        return self.processed_data
    
    def _extract_trajectories(self, fr1_data, fr2_data):
        """
        Extract and organize trajectory data from JSON structure
        """
        trajectories_fr1 = []
        trajectories_fr2 = []
        
        # Process FR1 data
        for demo in fr1_data['right_leg_kinematics']:
            # Extract positions, velocities, and angles
            right_ankle_pos = np.array(demo['right_ankle_pos'])
            right_ankle_vel = np.array(demo['right_ankle_vel'])
            
            # Create time vector
            n_points = right_ankle_pos.shape[1]
            time_vector = np.linspace(0, 1, n_points)
            
            # Stack features: [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]
            trajectory = np.vstack([
                right_ankle_pos[0, :],  # x_pos
                right_ankle_pos[1, :],  # y_pos  
                right_ankle_vel[0, :],  # x_vel
                right_ankle_vel[1, :],  # y_vel
                np.zeros(n_points),     # ankle_right_rad (placeholder)
                np.zeros(n_points),     # ankle_left_rad (placeholder)
                time_vector             # time
            ]).T
            
            trajectories_fr1.append(trajectory)
        
        # Process FR2 data similarly
        for demo in fr2_data['right_leg_kinematics']:
            right_ankle_pos = np.array(demo['right_ankle_pos'])
            right_ankle_vel = np.array(demo['right_ankle_vel'])
            
            n_points = right_ankle_pos.shape[1]
            time_vector = np.linspace(0, 1, n_points)
            
            trajectory = np.vstack([
                right_ankle_pos[0, :],
                right_ankle_pos[1, :],
                right_ankle_vel[0, :],
                right_ankle_vel[1, :],
                np.zeros(n_points),
                np.zeros(n_points),
                time_vector
            ]).T
            
            trajectories_fr2.append(trajectory)
        
        return np.array(trajectories_fr1), np.array(trajectories_fr2)
    
    def _prepare_tpgmm_data_fixed(self, trajectories_fr1, trajectories_fr2):
        """
        FIXED: Prepare data for TPGMM using trajectory-level features instead of all points
        This creates a much smaller dataset suitable for TPGMM
        """
        print("STEP 2: Preparing TPGMM Data Structure (TRAJECTORY-LEVEL)")
        print("-" * 50)
        
        n_demos_fr1, n_points, n_features = trajectories_fr1.shape
        n_demos_fr2, _, _ = trajectories_fr2.shape
        
        # Instead of using all points, extract trajectory-level features
        trajectory_features = []
        
        print("Extracting trajectory-level features...")
        
        # Process FR1 trajectories
        for i in range(n_demos_fr1):
            traj = trajectories_fr1[i]
            features = self._extract_trajectory_features(traj, frame='FR1')
            trajectory_features.append(features)
        
        # Process FR2 trajectories  
        for i in range(n_demos_fr2):
            traj = trajectories_fr2[i]
            features = self._extract_trajectory_features(traj, frame='FR2')
            trajectory_features.append(features)
        
        # Convert to numpy array
        X_trajectories = np.array(trajectory_features)
        
        print(f"✓ Trajectory features extracted:")
        print(f"  Shape: {X_trajectories.shape}")
        print(f"  Features per trajectory: {X_trajectories.shape[1]}")
        
        # Normalize trajectory features
        X_normalized = self.scaler.fit_transform(X_trajectories)
        
        # Create task parameters for trajectory-level data
        n_total_demos = n_demos_fr1 + n_demos_fr2
        n_frames = 2
        n_traj_features = X_normalized.shape[1]
        
        # Initialize task parameter matrices
        A = np.zeros((n_total_demos, n_frames, n_traj_features, n_traj_features))
        b = np.zeros((n_total_demos, n_frames, n_traj_features))
        
        # Set up reference frames
        for demo_idx in range(n_total_demos):
            for frame_idx in range(n_frames):
                # Identity transformation as baseline
                A[demo_idx, frame_idx] = np.eye(n_traj_features)
                
                # Frame-specific transformations
                if frame_idx == 0:  # FR1
                    b[demo_idx, frame_idx] = np.zeros(n_traj_features)
                else:  # FR2 - different reference frame
                    # Small translation to represent different frame origin
                    b[demo_idx, frame_idx] = np.random.randn(n_traj_features) * 0.1
        
        print(f"✓ FIXED: Prepared TPGMM data:")
        print(f"  Total trajectory samples: {X_normalized.shape[0]} (was 12000 points!)")
        print(f"  Trajectory features: {X_normalized.shape[1]}")
        print(f"  Demonstrations: {n_total_demos}")
        print(f"  Reference frames: {n_frames}")
        print(f"  Task parameter shapes: A{A.shape}, b{b.shape}")
        print()
        
        return {
            'X': X_normalized,
            'X_original': X_trajectories,
            'A': A,
            'b': b,
            'trajectories_fr1': trajectories_fr1,
            'trajectories_fr2': trajectories_fr2,
            'feature_names': self._get_trajectory_feature_names(),
            'n_demos_fr1': n_demos_fr1,
            'n_demos_fr2': n_demos_fr2
        }
    
    def _extract_trajectory_features(self, trajectory, frame='FR1'):
        """
        Extract meaningful features from a single trajectory
        Instead of using all 200 points, extract summary statistics
        """
        # Trajectory shape: (n_points, n_features)
        # Features: [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]
        
        features = []
        
        # Position features
        x_pos = trajectory[:, 0]
        y_pos = trajectory[:, 1]
        x_vel = trajectory[:, 2] 
        y_vel = trajectory[:, 3]
        
        # Statistical features for positions
        features.extend([
            np.mean(x_pos), np.std(x_pos), np.min(x_pos), np.max(x_pos),  # x position stats
            np.mean(y_pos), np.std(y_pos), np.min(y_pos), np.max(y_pos),  # y position stats
        ])
        
        # Statistical features for velocities
        features.extend([
            np.mean(x_vel), np.std(x_vel), np.min(x_vel), np.max(x_vel),  # x velocity stats
            np.mean(y_vel), np.std(y_vel), np.min(y_vel), np.max(y_vel),  # y velocity stats
        ])
        
        # Kinematic features
        speed = np.sqrt(x_vel**2 + y_vel**2)
        features.extend([
            np.mean(speed), np.std(speed), np.max(speed),  # speed statistics
        ])
        
        # Trajectory shape features
        total_distance = np.sum(np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2))
        displacement = np.sqrt((x_pos[-1] - x_pos[0])**2 + (y_pos[-1] - y_pos[0])**2)
        
        features.extend([
            total_distance,    # total path length
            displacement,      # start-to-end displacement
            total_distance / displacement if displacement > 1e-6 else 1,  # path efficiency
        ])
        
        # Temporal features
        duration = 1.0  # normalized time
        features.append(duration)
        
        # Frame identifier (helps distinguish FR1 vs FR2)
        frame_id = 0.0 if frame == 'FR1' else 1.0
        features.append(frame_id)
        
        return np.array(features)
    
    def _get_trajectory_feature_names(self):
        """
        Get names for trajectory-level features
        """
        names = [
            'x_pos_mean', 'x_pos_std', 'x_pos_min', 'x_pos_max',
            'y_pos_mean', 'y_pos_std', 'y_pos_min', 'y_pos_max',
            'x_vel_mean', 'x_vel_std', 'x_vel_min', 'x_vel_max',
            'y_vel_mean', 'y_vel_std', 'y_vel_min', 'y_vel_max',
            'speed_mean', 'speed_std', 'speed_max',
            'total_distance', 'displacement', 'path_efficiency',
            'duration', 'frame_id'
        ]
        return names
    
    def fit_tpgmm_model(self, n_components_range=[2, 3, 4, 5]):
        """
        Fit TPGMM model with model selection - MUCH FASTER NOW!
        """
        print("STEP 3: TPGMM Model Fitting and Selection (TRAJECTORY-LEVEL)")
        print("-" * 60)
        
        X = self.processed_data['X']
        A = self.processed_data['A']
        b = self.processed_data['b']
        
        print(f"Fitting TPGMM to {X.shape[0]} TRAJECTORY samples (not individual points!)")
        print(f"Each trajectory represented by {X.shape[1]} features")
        print()
        
        best_model = None
        best_score = -np.inf
        model_scores = {}
        
        print("Testing different numbers of components...")
        print()
        
        for n_components in n_components_range:
            print(f"Fitting TPGMM with {n_components} components...")
            
            # Fit model with fewer iterations since we have much less data
            model = TPGMM(
                n_components=n_components,
                n_frames=2,
                reg_covar=1e-6,
                max_iter=20,  # Reduced since we have fewer samples
                random_state=42
            )
            
            try:
                model.fit(X, A, b)
                
                # Use final log-likelihood as score
                score = model.log_likelihood_history_[-1] if model.log_likelihood_history_ else -np.inf
                model_scores[n_components] = score
                
                print(f"  Final log-likelihood: {score:.4f}")
                print(f"  Converged: {model.converged_}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"  Failed to fit model with {n_components} components: {e}")
                model_scores[n_components] = -np.inf
            
            print()
        
        self.tpgmm_model = best_model
        
        print(f"✓ Best model selected:")
        print(f"  Components: {best_model.n_components}")
        print(f"  Log-likelihood: {best_score:.4f}")
        print(f"  Model info: {best_model.get_model_info()}")
        print()
        
        # Plot model selection results
        self._plot_model_selection(model_scores)
        
        return best_model
    
    def perform_regression_analysis(self):
        """
        Perform GMR analysis for trajectory prediction
        """
        print("STEP 4: Gaussian Mixture Regression Analysis")
        print("-" * 40)
        
        if self.tpgmm_model is None:
            raise ValueError("TPGMM model must be fitted first")
        
        # Initialize GMR
        self.gmr_model = TaskParameterizedGMR(self.tpgmm_model)
        
        # Define regression task: predict position features from velocity features
        input_dims = [8, 9, 10, 11, 23]  # velocity stats + frame_id
        output_dims = [0, 1, 2, 3]  # position stats
        
        feature_names = self.processed_data['feature_names']
        print(f"Regression setup:")
        print(f"  Input dimensions: {[feature_names[i] for i in input_dims]}")
        print(f"  Output dimensions: {[feature_names[i] for i in output_dims]}")
        print()
        
        # Prepare test data
        X = self.processed_data['X']
        A = self.processed_data['A']
        b = self.processed_data['b']
        task_params = (A, b)
        
        # Split data for evaluation
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        
        # Perform regression on test data
        X_input_test = X_test[:, input_dims]
        Y_true_test = X_test[:, output_dims]
        
        print("Performing regression on test data...")
        Y_pred, Y_std = self.gmr_model.predict(
            X_input_test, input_dims, output_dims, task_params, return_std=True
        )
        
        # Evaluate performance
        metrics = self.gmr_model.evaluate_prediction_quality(
            X_input_test, Y_true_test, input_dims, output_dims, task_params
        )
        
        print(f"✓ Regression results:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Mean R²: {metrics['mean_r2']:.4f}")
        print(f"  R² per dimension: {[f'{r2:.3f}' for r2 in metrics['r2_scores']]}")
        print()
        
        return metrics
    
    def _plot_model_selection(self, model_scores):
        """
        Plot model selection results
        """
        plt.figure(figsize=(12, 5))
        
        components = list(model_scores.keys())
        scores = list(model_scores.values())
        
        plt.subplot(1, 2, 1)
        plt.plot(components, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Log-likelihood')
        plt.title('TPGMM Model Selection (Trajectory-Level)')
        plt.grid(True, alpha=0.3)
        
        # Plot convergence of best model
        plt.subplot(1, 2, 2)
        if self.tpgmm_model and self.tpgmm_model.log_likelihood_history_:
            plt.plot(self.tpgmm_model.log_likelihood_history_, 'r-', linewidth=2)
            plt.xlabel('EM Iteration')
            plt.ylabel('Log-likelihood')
            plt.title(f'Convergence (K={self.tpgmm_model.n_components})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/model_selection.png', dpi=150)
        plt.show()
        print("Model selection plot saved as model_selection.png")
    
    def visualize_original_data(self):
        """
        Visualize the original gait data
        """
        print("STEP 5: Visualizing Results")
        print("-" * 40)
        
        trajectories_fr1 = self.processed_data['trajectories_fr1']
        trajectories_fr2 = self.processed_data['trajectories_fr2']
        
        # Plot original trajectories
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Position trajectories
        axes[0, 0].set_title('X Position Trajectories')
        for i, traj in enumerate(trajectories_fr1):
            axes[0, 0].plot(traj[:, 6], traj[:, 0], 'b-', alpha=0.7, label='FR1' if i == 0 else '')
        for i, traj in enumerate(trajectories_fr2):
            axes[0, 0].plot(traj[:, 6], traj[:, 0], 'r-', alpha=0.7, label='FR2' if i == 0 else '')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Y Position Trajectories')
        for traj in trajectories_fr1:
            axes[0, 1].plot(traj[:, 6], traj[:, 1], 'b-', alpha=0.7)
        for traj in trajectories_fr2:
            axes[0, 1].plot(traj[:, 6], traj[:, 1], 'r-', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Velocity trajectories  
        axes[0, 2].set_title('X Velocity Trajectories')
        for traj in trajectories_fr1:
            axes[0, 2].plot(traj[:, 6], traj[:, 2], 'b-', alpha=0.7)
        for traj in trajectories_fr2:
            axes[0, 2].plot(traj[:, 6], traj[:, 2], 'r-', alpha=0.7)
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('X Velocity')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Y Velocity Trajectories')
        for traj in trajectories_fr1:
            axes[1, 0].plot(traj[:, 6], traj[:, 3], 'b-', alpha=0.7)
        for traj in trajectories_fr2:
            axes[1, 0].plot(traj[:, 6], traj[:, 3], 'r-', alpha=0.7)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Y Velocity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D position plot
        axes[1, 1].set_title('2D Position Trajectories')
        for traj in trajectories_fr1:
            axes[1, 1].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7)
        for traj in trajectories_fr2:
            axes[1, 1].plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7)
        axes[1, 1].set_xlabel('X Position')
        axes[1, 1].set_ylabel('Y Position')
        axes[1, 1].axis('equal')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Trajectory-level features
        X_traj = self.processed_data['X_original']
        n_fr1 = self.processed_data['n_demos_fr1']
        
        axes[1, 2].set_title('Trajectory Feature Comparison')
        # Plot first few trajectory features
        axes[1, 2].scatter(X_traj[:n_fr1, 0], X_traj[:n_fr1, 4], 
                          c='blue', alpha=0.7, label='FR1', s=50)
        axes[1, 2].scatter(X_traj[n_fr1:, 0], X_traj[n_fr1:, 4], 
                          c='red', alpha=0.7, label='FR2', s=50)
        axes[1, 2].set_xlabel('X Position Mean')
        axes[1, 2].set_ylabel('Y Position Mean')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Gait Analysis Data Visualization (TRAJECTORY-LEVEL)', fontsize=16)
        plt.tight_layout()
        plt.savefig('/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/gait_visualization.png', dpi=150)
        plt.show()
        
        print("✓ Visualization complete - saved as gait_visualization.png")
    
    def run_complete_analysis(self):
        """
        Run the complete gait analysis pipeline - FIXED VERSION
        """
        try:
            # Load and process data
            self.load_and_process_data()
            
            # Visualize original data
            self.visualize_original_data()
            
            # Fit TPGMM model (now much faster!)
            self.fit_tpgmm_model()
            
            # Perform regression analysis
            metrics = self.perform_regression_analysis()
            
            print("=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"✓ TPGMM successfully fitted with {self.tpgmm_model.n_components} components")
            print(f"✓ GMR achieved RMSE: {metrics['rmse']:.4f}")
            print(f"✓ Mean R² score: {metrics['mean_r2']:.4f}")
            print()
            print("✅ FIXED: Now using trajectory-level features instead of individual points!")
            print("This makes the algorithm much faster and more appropriate for gait analysis.")
            
            return self.tpgmm_model, self.gmr_model, metrics
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    """
    Main function to run the FIXED gait analysis
    """
    # Path to the gait data
    data_path = "/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/examples/7days1/gait_analysis_export_subject35v4.json"
    
    # Create analyzer
    analyzer = GaitAnalysisTTPGMM(data_path)
    
    # Run complete analysis
    tpgmm_model, gmr_model, metrics = analyzer.run_complete_analysis()
    
    return analyzer, tpgmm_model, gmr_model, metrics

if __name__ == "__main__":
    analyzer, tpgmm_model, gmr_model, metrics = main()