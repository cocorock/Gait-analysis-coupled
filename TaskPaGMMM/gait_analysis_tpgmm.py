"""
Gait Analysis using Task-Parameterized Gaussian Mixture Models (TPGMM)
Based on: Calinon et al., "Robot Learning with Task-parameterized Generative Models", ISRR 2015

This script demonstrates the new TPGMM implementation on gait analysis data.
It processes the JSON data, fits the TPGMM model, and performs regression.
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
    Complete gait analysis pipeline using TPGMM
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
        print("GAIT ANALYSIS WITH TASK-PARAMETERIZED GMM")
        print("=" * 60)
        print(f"Data source: {data_path}")
        print("Implementation: Custom TPGMM based on Calinon et al. (2015)")
        print()
    
    def load_and_process_data(self):
        """
        Load and process the gait analysis JSON data
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
        
        # Prepare data for TPGMM
        self.processed_data = self._prepare_tpgmm_data(trajectories_fr1, trajectories_fr2)
        
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
            
            # Get corresponding left ankle data
            if fr1_data['left_leg_kinematics']:
                left_demo = fr1_data['left_leg_kinematics'][0]  # Assuming same indexing
                left_ankle_pos = np.array(left_demo['left_ankle_pos'])
            
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
    
    def _prepare_tpgmm_data(self, trajectories_fr1, trajectories_fr2):
        """
        Prepare data for TPGMM fitting
        """
        print("STEP 2: Preparing TPGMM Data Structure")
        print("-" * 40)
        
        n_demos_fr1, n_points_fr1, n_features = trajectories_fr1.shape
        n_demos_fr2, n_points_fr2, _ = trajectories_fr2.shape
        
        # Ensure consistent number of points
        min_points = min(n_points_fr1, n_points_fr2)
        trajectories_fr1 = trajectories_fr1[:, :min_points, :]
        trajectories_fr2 = trajectories_fr2[:, :min_points, :]
        
        # Concatenate all trajectory data
        all_data = []
        for demo_idx in range(min(n_demos_fr1, n_demos_fr2)):
            all_data.append(trajectories_fr1[demo_idx])
            all_data.append(trajectories_fr2[demo_idx])
        
        X = np.vstack(all_data)
        
        # Normalize data
        X_normalized = self.scaler.fit_transform(X)
        
        # Create task parameters (reference frames)
        n_demos = len(all_data)
        n_frames = 2  # FR1 and FR2
        
        # Initialize task parameter matrices
        A = np.zeros((n_demos, n_frames, n_features, n_features))
        b = np.zeros((n_demos, n_frames, n_features))
        
        # Set up reference frames
        for demo_idx in range(n_demos):
            for frame_idx in range(n_frames):
                # Identity transformation as baseline (can be customized)
                A[demo_idx, frame_idx] = np.eye(n_features)
                
                # Frame-specific translations
                if frame_idx == 0:  # FR1
                    b[demo_idx, frame_idx] = np.zeros(n_features)
                else:  # FR2 - could have different origin
                    b[demo_idx, frame_idx] = np.array([0.1, 0.1, 0, 0, 0, 0, 0])
        
        print(f"✓ Prepared TPGMM data:")
        print(f"  Total samples: {X_normalized.shape[0]}")
        print(f"  Features: {X_normalized.shape[1]}")
        print(f"  Demonstrations: {n_demos}")
        print(f"  Reference frames: {n_frames}")
        print(f"  Task parameter shapes: A{A.shape}, b{b.shape}")
        print()
        
        return {
            'X': X_normalized,
            'X_original': X,
            'A': A,
            'b': b,
            'trajectories_fr1': trajectories_fr1,
            'trajectories_fr2': trajectories_fr2,
            'feature_names': self.feature_names
        }
    
    def fit_tpgmm_model(self, n_components_range=[3, 4, 5, 6, 7, 8]):
        """
        Fit TPGMM model with model selection
        """
        print("STEP 3: TPGMM Model Fitting and Selection")
        print("-" * 40)
        
        X = self.processed_data['X']
        A = self.processed_data['A']
        b = self.processed_data['b']
        
        best_model = None
        best_score = -np.inf
        model_scores = {}
        
        print("Testing different numbers of components...")
        print()
        
        for n_components in n_components_range:
            print(f"Fitting TPGMM with {n_components} components...")
            
            # Fit model
            model = TPGMM(
                n_components=n_components,
                n_frames=2,
                reg_covar=1e-6,
                max_iter=50,
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
        
        # Define regression task: predict positions and velocities given time
        input_dims = [6]  # time dimension
        output_dims = [0, 1, 2, 3]  # x_pos, y_pos, x_vel, y_vel
        
        print(f"Regression setup:")
        print(f"  Input dimensions: {[self.feature_names[i] for i in input_dims]}")
        print(f"  Output dimensions: {[self.feature_names[i] for i in output_dims]}")
        print()
        
        # Prepare test data
        X = self.processed_data['X']
        A = self.processed_data['A']
        b = self.processed_data['b']
        task_params = (A, b)
        
        # Split data for evaluation
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
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
        
        # Generate trajectory predictions
        self._generate_trajectory_predictions(task_params)
        
        return metrics
    
    def _generate_trajectory_predictions(self, task_params):
        """
        Generate and visualize trajectory predictions
        """
        print("Generating trajectory predictions...")
        
        # Create time points for prediction
        time_points = np.linspace(0, 1, 100)
        input_dims = [6]  # time
        output_dims = [0, 1, 2, 3]  # positions and velocities
        
        # Predict trajectory
        t_pred, Y_pred, Y_std = self.gmr_model.predict_trajectory(
            time_points, input_dims, output_dims, task_params, return_std=True
        )
        
        # Plot results
        self._plot_trajectory_predictions(t_pred, Y_pred, Y_std, output_dims)
        
    def _plot_model_selection(self, model_scores):
        """
        Plot model selection results
        """
        plt.figure(figsize=(10, 6))
        
        components = list(model_scores.keys())
        scores = list(model_scores.values())
        
        plt.subplot(1, 2, 1)
        plt.plot(components, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Log-likelihood')
        plt.title('TPGMM Model Selection')
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
        plt.show()
    
    def _plot_trajectory_predictions(self, time_points, Y_pred, Y_std, output_dims):
        """
        Plot trajectory prediction results
        """
        feature_labels = [self.feature_names[i] for i in output_dims]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, label) in enumerate(zip(axes, feature_labels)):
            # Plot prediction with uncertainty
            ax.plot(time_points, Y_pred[:, i], 'b-', linewidth=2, label='Prediction')
            ax.fill_between(
                time_points, 
                Y_pred[:, i] - 2*Y_std[:, i],
                Y_pred[:, i] + 2*Y_std[:, i],
                alpha=0.3, color='blue', label='±2σ Uncertainty'
            )
            
            ax.set_xlabel('Time (normalized)')
            ax.set_ylabel(label)
            ax.set_title(f'TPGMM Prediction: {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Task-Parameterized GMR Trajectory Predictions', fontsize=16)
        plt.tight_layout()
        plt.show()
    
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
        
        # Phase portrait (x vs x_dot)
        axes[1, 2].set_title('Phase Portrait (X)')
        for traj in trajectories_fr1:
            axes[1, 2].plot(traj[:, 0], traj[:, 2], 'b-', alpha=0.7)
        for traj in trajectories_fr2:
            axes[1, 2].plot(traj[:, 0], traj[:, 2], 'r-', alpha=0.7)
        axes[1, 2].set_xlabel('X Position')
        axes[1, 2].set_ylabel('X Velocity')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Gait Analysis Data Visualization', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualization complete")
    
    def run_complete_analysis(self):
        """
        Run the complete gait analysis pipeline
        """
        try:
            # Load and process data
            self.load_and_process_data()
            
            # Visualize original data
            self.visualize_original_data()
            
            # Fit TPGMM model
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
            print("The new TPGMM implementation based on Calinon et al. (2015)")
            print("has been successfully applied to your gait analysis data!")
            
            return self.tpgmm_model, self.gmr_model, metrics
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    """
    Main function to run the gait analysis
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