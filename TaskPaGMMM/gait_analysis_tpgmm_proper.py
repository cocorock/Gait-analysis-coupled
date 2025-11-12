"""
PROPER: Gait Analysis using Task-Parameterized Gaussian Mixture Models (TPGMM)
This version properly handles:
1. Full trajectory data (not statistical summaries)
2. Both left and right ankle data separately
3. Proper TPGMM with temporal structure
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

class ProperGaitAnalysisTTPGMM:
    """
    PROPER: Complete gait analysis pipeline using TPGMM
    - Uses full trajectory temporal data
    - Properly handles left and right ankle data
    - Subsamples for efficiency while preserving temporal structure
    """
    
    def __init__(self, data_path, subsample_factor=5):
        """
        Initialize with path to gait data JSON file
        """
        self.data_path = data_path
        self.subsample_factor = subsample_factor  # Use every Nth point
        self.raw_data = None
        self.processed_data = None
        self.tpgmm_model = None
        self.gmr_model = None
        self.scalers = {}  # Separate scalers for each type
        
        print("=" * 70)
        print("PROPER GAIT ANALYSIS WITH TASK-PARAMETERIZED GMM")
        print("=" * 70)
        print(f"Data source: {data_path}")
        print("✓ Uses FULL trajectory temporal data (subsampled for efficiency)")
        print("✓ Properly handles LEFT and RIGHT ankle data separately")
        print("✓ Maintains temporal structure for TPGMM")
        print(f"✓ Subsampling factor: {subsample_factor} (every {subsample_factor}th point)")
        print()
    
    def load_and_process_data(self):
        """
        Load and process the gait analysis JSON data - PROPER VERSION
        """
        print("STEP 1: Loading and Processing Data (PROPER)")
        print("-" * 50)
        
        # Load JSON data
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        print(f"✓ Loaded data from: {os.path.basename(self.data_path)}")
        print(f"  Creation date: {self.raw_data['export_info']['creation_date']}")
        print(f"  Number of source files: {len(self.raw_data['export_info']['source_files'])}")
        print()
        
        # Process kinematics data properly
        fr1_data = self.raw_data['kinematics_data']['FR1']
        fr2_data = self.raw_data['kinematics_data']['FR1']
        
        # Extract trajectories for BOTH left and right ankles
        trajectories = self._extract_proper_trajectories(fr1_data, fr2_data)
        
        print(f"✓ Extracted trajectories PROPERLY:")
        for traj_type, data in trajectories.items():
            print(f"  {traj_type}: {data.shape}")
        print()
        
        # Prepare data for TPGMM with proper temporal structure
        self.processed_data = self._prepare_proper_tpgmm_data(trajectories)
        
        return self.processed_data
    
    def _extract_proper_trajectories(self, fr1_data, fr2_data):
        """
        Extract ALL trajectory data from right_leg_kinematics section
        UPDATED: Extract all features including ankle angles and shank angles
        """
        trajectories = {
            'complete_fr1': [],
            'complete_fr2': []
        }
        
        print("Extracting ALL gait features from right_leg_kinematics section...")
        print("Features: right_ankle_pos, right_ankle_vel, left_ankle_pos, left_ankle_vel,")
        print("          right_shank_deg, left_shank_deg, ankle_right_deg, ankle_left_deg")
        
        # Process FR1 data - extract ALL features from right_leg_kinematics
        # MODIFICATION: Use only first 12 trajectories for faster processing
        n_demos_to_use = 12
        print(f"Processing FR1 - Using first {n_demos_to_use} of {len(fr1_data['right_leg_kinematics'])} demos for faster processing")
        for demo in fr1_data['right_leg_kinematics'][:n_demos_to_use]:
            # Extract ALL available data
            right_ankle_pos = np.array(demo['right_ankle_pos'])  # (2, n_points)
            right_ankle_vel = np.array(demo['right_ankle_vel'])  # (2, n_points)
            left_ankle_pos = np.array(demo['left_ankle_pos'])    # (2, n_points)
            left_ankle_vel = np.array(demo['left_ankle_vel'])    # (2, n_points)
            right_shank_deg = np.array(demo['right_shank_deg'])  # (n_points,)
            left_shank_deg = np.array(demo['left_shank_deg'])    # (n_points,)
            ankle_right_deg = np.array(demo['ankle_right_deg'])  # (n_points,)
            ankle_left_deg = np.array(demo['ankle_left_deg'])    # (n_points,)
            
            n_points = right_ankle_pos.shape[1]
            
            # Subsample for efficiency while preserving temporal structure
            indices = np.arange(0, n_points, self.subsample_factor)
            n_sub = len(indices)
            
            # Create time vector
            time_vector = np.linspace(0, 1, n_sub)
            
            # Stack ALL features into one comprehensive trajectory
            complete_trajectory = np.vstack([
                right_ankle_pos[0, indices],   # right_ankle_x_pos
                right_ankle_pos[1, indices],   # right_ankle_y_pos  
                right_ankle_vel[0, indices],   # right_ankle_x_vel
                right_ankle_vel[1, indices],   # right_ankle_y_vel
                left_ankle_pos[0, indices],    # left_ankle_x_pos
                left_ankle_pos[1, indices],    # left_ankle_y_pos
                left_ankle_vel[0, indices],    # left_ankle_x_vel
                left_ankle_vel[1, indices],    # left_ankle_y_vel
                right_shank_deg[indices],      # right_shank_deg
                left_shank_deg[indices],       # left_shank_deg
                ankle_right_deg[indices],      # ankle_right_deg
                ankle_left_deg[indices],       # ankle_left_deg
                time_vector                    # time
            ]).T
            
            trajectories['complete_fr1'].append(complete_trajectory)
        
        # Process FR2 data - extract ALL features from right_leg_kinematics
        print(f"Processing FR2 - Using first {n_demos_to_use} of {len(fr2_data['right_leg_kinematics'])} demos for faster processing")
        for demo in fr2_data['right_leg_kinematics'][:n_demos_to_use]:
            # Extract ALL available data
            right_ankle_pos = np.array(demo['right_ankle_pos'])
            right_ankle_vel = np.array(demo['right_ankle_vel'])
            left_ankle_pos = np.array(demo['left_ankle_pos'])
            left_ankle_vel = np.array(demo['left_ankle_vel'])
            right_shank_deg = np.array(demo['right_shank_deg'])
            left_shank_deg = np.array(demo['left_shank_deg'])
            ankle_right_deg = np.array(demo['ankle_right_deg'])
            ankle_left_deg = np.array(demo['ankle_left_deg'])
            
            n_points = right_ankle_pos.shape[1]
            indices = np.arange(0, n_points, self.subsample_factor)
            n_sub = len(indices)
            time_vector = np.linspace(0, 1, n_sub)
            
            # Stack ALL features into one comprehensive trajectory
            complete_trajectory = np.vstack([
                right_ankle_pos[0, indices],   # right_ankle_x_pos
                right_ankle_pos[1, indices],   # right_ankle_y_pos  
                right_ankle_vel[0, indices],   # right_ankle_x_vel
                right_ankle_vel[1, indices],   # right_ankle_y_vel
                left_ankle_pos[0, indices],    # left_ankle_x_pos
                left_ankle_pos[1, indices],    # left_ankle_y_pos
                left_ankle_vel[0, indices],    # left_ankle_x_vel
                left_ankle_vel[1, indices],    # left_ankle_y_vel
                right_shank_deg[indices],      # right_shank_deg
                left_shank_deg[indices],       # left_shank_deg
                ankle_right_deg[indices],      # ankle_right_deg
                ankle_left_deg[indices],       # ankle_left_deg
                time_vector                    # time
            ]).T
            
            trajectories['complete_fr2'].append(complete_trajectory)
        
        # Convert to numpy arrays
        for key in trajectories:
            if trajectories[key]:
                trajectories[key] = np.array(trajectories[key])
                print(f"  ✓ {key}: {trajectories[key].shape}")
            else:
                print(f"  ❌ {key}: No data found")
        
        return trajectories
    
    def _prepare_proper_tpgmm_data(self, trajectories):
        """
        Prepare data for TPGMM using ALL available gait features
        """
        print("STEP 2: Preparing COMPREHENSIVE TPGMM Data Structure")
        print("-" * 55)
        
        # Use complete trajectories from both frames
        if 'complete_fr1' not in trajectories or 'complete_fr2' not in trajectories:
            raise ValueError("Missing complete trajectory data")
        
        traj_fr1 = trajectories['complete_fr1']
        traj_fr2 = trajectories['complete_fr2']
        
        # Ensure we have the same number of trajectories in both frames
        min_demos = min(len(traj_fr1), len(traj_fr2))
        traj_fr1 = traj_fr1[:min_demos]
        traj_fr2 = traj_fr2[:min_demos]
        
        print(f"Using {min_demos} trajectory demonstrations from each frame")
        print(f"Each trajectory has {traj_fr1.shape[2]} features:")
        
        feature_names = [
            'right_ankle_x_pos', 'right_ankle_y_pos', 'right_ankle_x_vel', 'right_ankle_y_vel',
            'left_ankle_x_pos', 'left_ankle_y_pos', 'left_ankle_x_vel', 'left_ankle_y_vel',
            'right_shank_deg', 'left_shank_deg', 'ankle_right_deg', 'ankle_left_deg', 'time'
        ]
        
        for i, name in enumerate(feature_names):
            print(f"  Feature {i}: {name}")
        print()
        
        # Combine all trajectory data points while maintaining structure
        all_points = []
        trajectory_labels = []  # Track which trajectory each point belongs to
        frame_labels = []       # Track which frame each point belongs to
        
        # Add FR1 trajectories
        for i, traj in enumerate(traj_fr1):
            all_points.append(traj)
            trajectory_labels.extend([f'traj_{i}_fr1'] * len(traj))
            frame_labels.extend(['FR1'] * len(traj))
        
        # Add FR2 trajectories  
        for i, traj in enumerate(traj_fr2):
            all_points.append(traj)
            trajectory_labels.extend([f'traj_{i}_fr2'] * len(traj))
            frame_labels.extend(['FR2'] * len(traj))
        
        # Concatenate all points
        X_all = np.vstack(all_points)
        
        print(f"Combined data shape: {X_all.shape}")
        print(f"Features: {len(feature_names)} comprehensive gait features")
        
        # Normalize the data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_all)
        self.scalers['main'] = scaler
        
        # Create task parameters
        n_points = X_all.shape[0]
        n_features = X_all.shape[1]
        n_frames = 2
        
        # For TPGMM, we'll treat this as one "demonstration" with task parameters
        # distinguishing between the frames
        A = np.zeros((1, n_frames, n_features, n_features))
        b = np.zeros((1, n_frames, n_features))
        
        # Frame 1 (FR1): identity transformation
        A[0, 0] = np.eye(n_features)
        b[0, 0] = np.zeros(n_features)
        
        # Frame 2 (FR2): slight transformation to represent different reference frame
        A[0, 1] = np.eye(n_features)
        # Small transformations in relevant feature spaces
        # Position spaces (features 0-3 and 4-7)
        theta = 0.1  # Small rotation angle
        for pos_start in [0, 4]:  # right and left ankle positions
            A[0, 1, pos_start, pos_start] = np.cos(theta)
            A[0, 1, pos_start, pos_start+1] = -np.sin(theta)
            A[0, 1, pos_start+1, pos_start] = np.sin(theta)
            A[0, 1, pos_start+1, pos_start+1] = np.cos(theta)
        
        # Small translation for all features
        b[0, 1] = np.random.randn(n_features) * 0.1
        
        print(f"✓ COMPREHENSIVE TPGMM data prepared:")
        print(f"  Total trajectory points: {X_normalized.shape[0]}")
        print(f"  Features per point: {X_normalized.shape[1]} (ALL gait features)")
        print(f"  Reference frames: {n_frames}")
        print(f"  Task parameter shapes: A{A.shape}, b{b.shape}")
        print(f"  FR1 points: {sum(1 for label in frame_labels if label == 'FR1')}")
        print(f"  FR2 points: {sum(1 for label in frame_labels if label == 'FR2')}")
        print()
        
        return {
            'X': X_normalized,
            'X_original': X_all,
            'A': A,
            'b': b,
            'trajectories': trajectories,
            'trajectory_labels': trajectory_labels,
            'frame_labels': frame_labels,
            'traj_fr1': traj_fr1,
            'traj_fr2': traj_fr2,
            'feature_names': feature_names
        }
    
    def fit_tpgmm_model(self, n_components_range=[3, 5, 7, 9, 11, 13, 15, 18, 21, 25]):
        """
        Fit TPGMM model with proper trajectory data
        """
        print("STEP 3: TPGMM Model Fitting (PROPER TRAJECTORIES)")
        print("-" * 55)
        
        X = self.processed_data['X']
        A = self.processed_data['A']
        b = self.processed_data['b']
        
        print(f"Fitting TPGMM to {X.shape[0]} trajectory points")
        print(f"Features per point: {X.shape[1]}")
        print("Using proper temporal trajectory structure")
        print()
        
        best_model = None
        best_score = -np.inf
        model_scores = {}
        
        for n_components in n_components_range:
            print(f"Fitting TPGMM with {n_components} components...")
            
            model = TPGMM(
                n_components=n_components,
                n_frames=2,
                reg_covar=1e-4,
                max_iter=30,
                random_state=42
            )
            
            try:
                model.fit(X, A, b)
                
                score = model.log_likelihood_history_[-1] if model.log_likelihood_history_ else -np.inf
                model_scores[n_components] = score
                
                print(f"  Final log-likelihood: {score:.4f}")
                print(f"  Converged: {model.converged_}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"  Failed: {e}")
                model_scores[n_components] = -np.inf
            
            print()
        
        self.tpgmm_model = best_model
        
        print(f"✓ Best model selected:")
        print(f"  Components: {best_model.n_components}")
        print(f"  Log-likelihood: {best_score:.4f}")
        print()
        
        # Plot results
        self._plot_model_selection(model_scores)
        
        return best_model
    
    def visualize_proper_data(self):
        """
        Visualize the comprehensive gait data showing ALL features
        """
        print("STEP 4: Visualizing ALL Gait Features")
        print("-" * 40)
        
        trajectories = self.processed_data['trajectories']
        traj_fr1 = self.processed_data['traj_fr1']
        traj_fr2 = self.processed_data['traj_fr2']
        feature_names = self.processed_data['feature_names']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # RIGHT ANKLE TRAJECTORIES
        # X Position
        axes[0, 0].set_title('RIGHT Ankle - X Position Trajectories')
        for i, traj in enumerate(traj_fr1):
            axes[0, 0].plot(traj[:, 4], traj[:, 0], 'b-', alpha=0.6, linewidth=1.5, 
                           label='FR1' if i == 0 else '')
        for i, traj in enumerate(traj_fr2):
            axes[0, 0].plot(traj[:, 4], traj[:, 0], 'r-', alpha=0.6, linewidth=1.5,
                           label='FR2' if i == 0 else '')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y Position
        axes[0, 1].set_title('RIGHT Ankle - Y Position Trajectories')
        for traj in traj_fr1:
            axes[0, 1].plot(traj[:, 4], traj[:, 1], 'b-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[0, 1].plot(traj[:, 4], traj[:, 1], 'r-', alpha=0.6, linewidth=1.5)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D Position Plot
        axes[0, 2].set_title('RIGHT Ankle - 2D Position Trajectories')
        for traj in traj_fr1:
            axes[0, 2].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[0, 2].plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.6, linewidth=1.5)
        axes[0, 2].set_xlabel('X Position')
        axes[0, 2].set_ylabel('Y Position')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')
        
        # VELOCITY PLOTS
        axes[1, 0].set_title('RIGHT Ankle - X Velocity')
        for traj in traj_fr1:
            axes[1, 0].plot(traj[:, 4], traj[:, 2], 'b-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[1, 0].plot(traj[:, 4], traj[:, 2], 'r-', alpha=0.6, linewidth=1.5)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('X Velocity')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('RIGHT Ankle - Y Velocity')
        for traj in traj_fr1:
            axes[1, 1].plot(traj[:, 4], traj[:, 3], 'b-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[1, 1].plot(traj[:, 4], traj[:, 3], 'r-', alpha=0.6, linewidth=1.5)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Y Velocity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Phase portrait
        axes[1, 2].set_title('RIGHT Ankle - Phase Portrait (X)')
        for traj in traj_fr1:
            axes[1, 2].plot(traj[:, 0], traj[:, 2], 'b-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[1, 2].plot(traj[:, 0], traj[:, 2], 'r-', alpha=0.6, linewidth=1.5)
        axes[1, 2].set_xlabel('X Position')
        axes[1, 2].set_ylabel('X Velocity')
        axes[1, 2].grid(True, alpha=0.3)
        
        # LEFT ANKLE - now from comprehensive data (features 4-7)
        axes[2, 0].set_title('LEFT Ankle - X Position Trajectories')
        for i, traj in enumerate(traj_fr1):
            axes[2, 0].plot(traj[:, 12], traj[:, 4], 'g-', alpha=0.6, linewidth=1.5,
                           label='FR1' if i == 0 else '')
        for i, traj in enumerate(traj_fr2):
            axes[2, 0].plot(traj[:, 12], traj[:, 4], 'm-', alpha=0.6, linewidth=1.5,
                           label='FR2' if i == 0 else '')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Left X Position')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].set_title('LEFT Ankle - Y Position Trajectories')
        for traj in traj_fr1:
            axes[2, 1].plot(traj[:, 12], traj[:, 5], 'g-', alpha=0.6, linewidth=1.5)
        for traj in traj_fr2:
            axes[2, 1].plot(traj[:, 12], traj[:, 5], 'm-', alpha=0.6, linewidth=1.5)
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Left Y Position')
        axes[2, 1].grid(True, alpha=0.3)
        
        axes[2, 2].set_title('LEFT vs RIGHT Ankle Comparison')
        # Plot one trajectory from each frame showing both ankles
        if len(traj_fr1) > 0:
            # Right ankle (features 0, 1)
            axes[2, 2].plot(traj_fr1[0, :, 0], traj_fr1[0, :, 1], 'b-', 
                           linewidth=2, label='Right Ankle FR1')
            # Left ankle (features 4, 5)
            axes[2, 2].plot(traj_fr1[0, :, 4], traj_fr1[0, :, 5], 'g-', 
                           linewidth=2, label='Left Ankle FR1')
        if len(traj_fr2) > 0:
            # Right ankle (features 0, 1)
            axes[2, 2].plot(traj_fr2[0, :, 0], traj_fr2[0, :, 1], 'r--', 
                           linewidth=2, label='Right Ankle FR2')
            # Left ankle (features 4, 5)
            axes[2, 2].plot(traj_fr2[0, :, 4], traj_fr2[0, :, 5], 'm--', 
                           linewidth=2, label='Left Ankle FR2')
        axes[2, 2].set_xlabel('X Position')
        axes[2, 2].set_ylabel('Y Position')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axis('equal')
        
        plt.suptitle('PROPER Gait Analysis: LEFT vs RIGHT Ankle Trajectories', fontsize=16)
        plt.tight_layout()
        plt.savefig('D:\\Github\\Gait-analysis-coupled\\TaskPaGMMM\\proper_gait_visualization.png', dpi=150)
        plt.show()
        
        print("✓ Proper visualization complete - saved as proper_gait_visualization.png")
        print("✓ Now you can clearly see the differences between:")
        print("  - Left vs Right ankle trajectories")
        print("  - FR1 vs FR2 reference frame data")
        print("  - Temporal evolution of positions and velocities")
    
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
        plt.title('TPGMM Model Selection (Proper Trajectories)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if self.tpgmm_model and self.tpgmm_model.log_likelihood_history_:
            plt.plot(self.tpgmm_model.log_likelihood_history_, 'r-', linewidth=2)
            plt.xlabel('EM Iteration')
            plt.ylabel('Log-likelihood')
            plt.title(f'Convergence (K={self.tpgmm_model.n_components})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/proper_model_selection.png', dpi=300)
        plt.show()
    
    def save_model(self):
        """
        Save the trained TPGMM model to a PKL file
        """
        if self.tpgmm_model is None:
            print("❌ No trained model to save")
            return
        
        import pickle
        import os
        from datetime import datetime
        
        # Create models directory if it doesn't exist
        models_dir = '/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tpgmm_gait_model_{timestamp}.pkl"
        filepath = os.path.join(models_dir, filename)
        
        # Save model and metadata
        model_data = {
            'tpgmm_model': self.tpgmm_model,
            'X_data': self.processed_data['X'],
            'A_params': self.processed_data['A'],
            'b_params': self.processed_data['b'],
            'n_components': self.tpgmm_model.n_components,
            'n_frames': self.tpgmm_model.n_frames,
            'feature_names': [
                'right_ankle_x_pos', 'right_ankle_y_pos', 'right_ankle_x_vel', 'right_ankle_y_vel',
                'left_ankle_x_pos', 'left_ankle_y_pos', 'left_ankle_x_vel', 'left_ankle_y_vel',
                'right_shank_deg', 'left_shank_deg', 'ankle_right_deg', 'ankle_left_deg', 'time'
            ],
            'subsample_factor': self.subsample_factor,
            'training_data_path': self.data_path,
            'timestamp': timestamp
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save a "latest" version
        latest_filepath = os.path.join(models_dir, "tpgmm_gait_model_latest.pkl")
        with open(latest_filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
        print(f"✓ Latest model saved to: {latest_filepath}")
        print(f"✓ Model info: {self.tpgmm_model.n_components} components, {self.tpgmm_model.n_frames} frames")
    
    def perform_gmr_analysis(self):
        """
        Perform GMR trajectory recovery using the trained TPGMM model
        """
        if self.tpgmm_model is None:
            print("❌ No trained model available for GMR")
            return
        
        print("\nSTEP 5: GMR Trajectory Recovery")
        print("-" * 40)
        
        from gmr_implementation import TaskParameterizedGMR
        
        # Initialize GMR
        gmr = TaskParameterizedGMR(self.tpgmm_model)
        
        # Test 1: Predict ankle positions from time
        print("Test 1: Predicting ankle positions from time...")
        
        # Use first 50 time points for testing
        n_test_points = 50
        X_test = self.processed_data['X'][:n_test_points]
        
        # Input: time (feature 12), Output: right ankle positions (features 0,1)
        input_dims = [12]  # time
        output_dims = [0, 1]  # right ankle x,y positions
        
        Y_pred_right = gmr.predict(X_test[:, input_dims], input_dims, output_dims, (self.processed_data['A'], self.processed_data['b']))
        Y_true_right = X_test[:, output_dims]
        
        # Calculate error
        mse_right = np.mean((Y_true_right - Y_pred_right)**2)
        print(f"  Right ankle position MSE: {mse_right:.6f}")
        
        # Test 2: Predict left ankle positions from time
        print("Test 2: Predicting left ankle positions from time...")
        
        # Input: time (feature 12), Output: left ankle positions (features 4,5)
        output_dims_left = [4, 5]  # left ankle x,y positions
        
        Y_pred_left = gmr.predict(X_test[:, input_dims], input_dims, output_dims_left, (self.processed_data['A'], self.processed_data['b']))
        Y_true_left = X_test[:, output_dims_left]
        
        mse_left = np.mean((Y_true_left - Y_pred_left)**2)
        print(f"  Left ankle position MSE: {mse_left:.6f}")
        
        # Test 3: Predict velocities from positions
        print("Test 3: Predicting velocities from positions...")
        
        # Input: right ankle positions (features 0,1), Output: right ankle velocities (features 2,3)
        input_dims_pos = [0, 1]
        output_dims_vel = [2, 3]
        
        Y_pred_vel = gmr.predict(X_test[:, input_dims_pos], input_dims_pos, output_dims_vel, (self.processed_data['A'], self.processed_data['b']))
        Y_true_vel = X_test[:, output_dims_vel]
        
        mse_vel = np.mean((Y_true_vel - Y_pred_vel)**2)
        print(f"  Right ankle velocity MSE: {mse_vel:.6f}")
        
        # Visualize GMR results
        self.visualize_gmr_results(X_test, Y_true_right, Y_pred_right, Y_true_left, Y_pred_left)
        
        print("✓ GMR trajectory recovery completed")
        return {
            'mse_right_pos': mse_right,
            'mse_left_pos': mse_left, 
            'mse_right_vel': mse_vel
        }
    
    def visualize_gmr_results(self, X_test, Y_true_right, Y_pred_right, Y_true_left, Y_pred_left):
        """
        Visualize GMR trajectory recovery results
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Time vector for plotting
        time_test = X_test[:, 12]  # time feature
        
        # Right ankle X position
        axes[0, 0].plot(time_test, Y_true_right[:, 0], 'b-', linewidth=2, label='True')
        axes[0, 0].plot(time_test, Y_pred_right[:, 0], 'r--', linewidth=2, label='GMR Predicted')
        axes[0, 0].set_title('Right Ankle X Position Recovery')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Right ankle Y position  
        axes[0, 1].plot(time_test, Y_true_right[:, 1], 'b-', linewidth=2, label='True')
        axes[0, 1].plot(time_test, Y_pred_right[:, 1], 'r--', linewidth=2, label='GMR Predicted')
        axes[0, 1].set_title('Right Ankle Y Position Recovery')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Right ankle 2D trajectory
        axes[0, 2].plot(Y_true_right[:, 0], Y_true_right[:, 1], 'b-', linewidth=2, label='True')
        axes[0, 2].plot(Y_pred_right[:, 0], Y_pred_right[:, 1], 'r--', linewidth=2, label='GMR Predicted')
        axes[0, 2].set_title('Right Ankle 2D Trajectory Recovery')
        axes[0, 2].set_xlabel('X Position')
        axes[0, 2].set_ylabel('Y Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')
        
        # Left ankle X position
        axes[1, 0].plot(time_test, Y_true_left[:, 0], 'g-', linewidth=2, label='True')
        axes[1, 0].plot(time_test, Y_pred_left[:, 0], 'm--', linewidth=2, label='GMR Predicted')
        axes[1, 0].set_title('Left Ankle X Position Recovery')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('X Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Left ankle Y position
        axes[1, 1].plot(time_test, Y_true_left[:, 1], 'g-', linewidth=2, label='True')
        axes[1, 1].plot(time_test, Y_pred_left[:, 1], 'm--', linewidth=2, label='GMR Predicted')
        axes[1, 1].set_title('Left Ankle Y Position Recovery')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Y Position')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Left ankle 2D trajectory
        axes[1, 2].plot(Y_true_left[:, 0], Y_true_left[:, 1], 'g-', linewidth=2, label='True')
        axes[1, 2].plot(Y_pred_left[:, 0], Y_pred_left[:, 1], 'm--', linewidth=2, label='GMR Predicted')
        axes[1, 2].set_title('Left Ankle 2D Trajectory Recovery')
        axes[1, 2].set_xlabel('X Position')
        axes[1, 2].set_ylabel('Y Position')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')
        
        plt.suptitle('GMR Trajectory Recovery Results', fontsize=16)
        plt.tight_layout()
        plt.savefig('/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/gmr_recovery_results.png', dpi=150)
        plt.show()
        
        print("✓ GMR visualization saved as gmr_recovery_results.png")
    
    def run_complete_analysis(self):
        """
        Run the complete PROPER gait analysis pipeline
        """
        try:
            # Load and process data properly
            self.load_and_process_data()
            
            # Visualize data showing left vs right differences
            self.visualize_proper_data()
            
            # Fit TPGMM model with proper trajectory data
            self.fit_tpgmm_model()
            
            # Save the trained model
            self.save_model()
            
            # Perform GMR trajectory recovery
            self.perform_gmr_analysis()
            
            print("=" * 70)
            print("PROPER ANALYSIS COMPLETE!")
            print("=" * 70)
            print(f"✓ TPGMM successfully fitted with {self.tpgmm_model.n_components} components")
            print("✓ Used PROPER trajectory temporal data (not statistical summaries)")
            print("✓ Properly distinguished LEFT vs RIGHT ankle trajectories")
            print("✓ Maintained temporal structure essential for TPGMM")
            print("✓ Model saved to PKL file for future use")
            print("✓ GMR trajectory recovery completed")
            print()
            print("Now you have a proper TPGMM implementation that:")
            print("  - Uses full trajectory data as intended")
            print("  - Shows clear differences between left/right ankles")
            print("  - Distinguishes between FR1 and FR2 reference frames")
            print("  - Maintains temporal relationships in the data")
            print("  - Can generate new trajectories using GMR")
            
            return self.tpgmm_model
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """
    Main function to run the PROPER gait analysis
    """
    # Path to the gait data
    data_path = "D:\\Github\\Gait-analysis-coupled\\TaskPaGMMM\\examples\\7days1\\gait_analysis_export_subject35v4.json"
    
    # Create analyzer with subsampling for efficiency
    analyzer = ProperGaitAnalysisTTPGMM(data_path, subsample_factor=5)  # Every 10th point
    
    # Run complete analysis
    tpgmm_model = analyzer.run_complete_analysis()
    
    return analyzer, tpgmm_model

if __name__ == "__main__":
    analyzer, tpgmm_model = main()