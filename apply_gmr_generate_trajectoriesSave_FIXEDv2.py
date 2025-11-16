"""
GMR TRAJECTORY GENERATION SCRIPT - FIXED VERSION
Load trained TPGMM model and generate trajectories using Gaussian Mixture Regression

FIXES:
- Changed to generate trajectories ONLY from FR1 (Hip frame)
- Removed global averaging across all 3 frames
- Added transformation alignment to match FR1 trajectories

This script:
1. Loads trained TPGMM model from PKL file (auto-selects folder based on reg_factor)
2. Applies Gaussian Mixture Regression (GMR) using ONLY FR1 parameters
3. Generates smooth trajectory predictions in FR1 coordinate system
4. Calculates statistics and uncertainties
5. Generates and saves trajectory-related figures:
   - Time series trajectory plots
   - X-Y position trajectories
   - Vx-Vy velocity trajectories

Input:
- models/train_tpgmm_model_reg{reg_factor}/trained_model.pkl
  (folder automatically selected based on reg_factor parameter)

Output:
- models/train_tpgmm_model_reg{reg_factor}/gmr_trajectory_11dims.png
- models/train_tpgmm_model_reg{reg_factor}/ankle_position_trajectories_xy.png
- models/train_tpgmm_model_reg{reg_factor}/ankle_velocity_trajectories_vxvy.png

Note: Model training is done separately in train_tpgmm_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import time
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from scipy.linalg import cholesky
from sklearn.cluster import KMeans

# Set numpy print options for cleaner output
np.set_printoptions(precision=2, suppress=True)

# Global variable for output folder
OUTPUT_FOLDER = "."  # Default to current directory

# Configure matplotlib to use Palatino Linotype font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'Times New Roman']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 16


class OptimizedTPGMM:
    """
    OPTIMIZED Task-Parameterized Gaussian Mixture Model
    
    This class is required for unpickling the trained model.
    """
    
    def __init__(self, n_components, n_frames, n_features, reg_factor=1e-5, 
                 max_iter=100, tol=1e-4, verbose=True):
        """Initialize Optimized TPGMM model"""
        self.K = n_components
        self.P = n_frames
        self.D = n_features
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Model parameters
        self.priors = None
        self.means = None
        self.covars = None
        
        # Cached values for speed
        self._inv_covars = None
        self._log_dets = None
        self._chol_covars = None
        
        self.log_likelihood_history = []
        self.converged = False
        self.n_iter = 0
    
    def _regularize_covar(self, covar):
        """Add regularization to covariance matrix"""
        return covar + self.reg_factor * np.eye(self.D)
    
    def _compute_cached_values(self):
        """Pre-compute inverse covariances and log-determinants for speed"""
        self._inv_covars = np.zeros((self.P, self.K, self.D, self.D))
        self._log_dets = np.zeros((self.P, self.K))
        self._chol_covars = np.zeros((self.P, self.K, self.D, self.D))
        
        for p in range(self.P):
            for k in range(self.K):
                try:
                    # Cholesky decomposition for numerical stability
                    L = cholesky(self.covars[p, k], lower=True)
                    self._chol_covars[p, k] = L
                    self._inv_covars[p, k] = np.linalg.inv(self.covars[p, k])
                    self._log_dets[p, k] = 2 * np.sum(np.log(np.diag(L)))
                except:
                    # Fallback to standard computation
                    self._inv_covars[p, k] = np.linalg.inv(self.covars[p, k])
                    self._log_dets[p, k] = np.linalg.slogdet(self.covars[p, k])[1]


class OptimizedGMR_FR1Only:
    """
    Optimized Gaussian Mixture Regression for TPGMM - FR1 ONLY VERSION
    
    This version uses ONLY Frame 1 (FR1) parameters for trajectory generation,
    avoiding the global averaging across all frames.
    """
    def __init__(self, tpgmm_model, frame_index=0):
        self.model = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
        self.frame_index = frame_index  # Which frame to use (0 = FR1)
    
    def predict(self, X_query, input_dims, output_dims, A_frame, b_frame):
        """
        GMR prediction using ONLY the specified frame (optimized)
        
        Parameters:
        -----------
        X_query : array, shape (n_query, n_input)
            Query points for input dimensions
        input_dims : list
            Indices of input dimensions
        output_dims : list
            Indices of output dimensions
        A_frame : array, shape (D, D)
            Transformation matrix for the frame
        b_frame : array, shape (D,)
            Translation vector for the frame
        """
        n_query = X_query.shape[0]
        n_out = len(output_dims)
        
        # Use ONLY the specified frame (FR1)
        p = self.frame_index
        
        # Pre-compute transformed Gaussians for the selected frame
        mu_global = np.zeros((self.K, self.D))
        Sigma_global = np.zeros((self.K, self.D, self.D))
        
        for k in range(self.K):
            # Transform mean
            mu_global[k] = A_frame @ self.model.means[p, k] + b_frame
            
            # Transform covariance
            Sigma_global[k] = A_frame @ self.model.covars[p, k] @ A_frame.T
        
        # GMR for each query point
        mu_out = np.zeros((n_query, n_out))
        sigma_out = np.zeros((n_query, n_out, n_out))
        
        for t in range(n_query):
            x_in = X_query[t]
            
            # Compute responsibilities
            h = np.zeros(self.K)
            for k in range(self.K):
                mu_in = mu_global[k][input_dims]
                Sigma_in = Sigma_global[k][np.ix_(input_dims, input_dims)]
                Sigma_in_inv = np.linalg.inv(Sigma_in)
                
                diff = x_in - mu_in
                h[k] = self.model.priors[k] * np.exp(-0.5 * diff @ Sigma_in_inv @ diff)
            
            h = h / (np.sum(h) + 1e-10)
            
            # Condition on input
            for k in range(self.K):
                mu_k = mu_global[k]
                Sigma_k = Sigma_global[k]
                
                Sigma_in = Sigma_k[np.ix_(input_dims, input_dims)]
                Sigma_out_in = Sigma_k[np.ix_(output_dims, input_dims)]
                Sigma_out_out = Sigma_k[np.ix_(output_dims, output_dims)]
                
                Sigma_in_inv = np.linalg.inv(Sigma_in)
                
                mu_out_k = mu_k[output_dims] + Sigma_out_in @ Sigma_in_inv @ (x_in - mu_k[input_dims])
                Sigma_out_k = Sigma_out_out - Sigma_out_in @ Sigma_in_inv @ Sigma_out_in.T
                
                mu_out[t] += h[k] * mu_out_k
                sigma_out[t] += h[k] * (Sigma_out_k + np.outer(mu_out_k, mu_out_k))
            
            sigma_out[t] -= np.outer(mu_out[t], mu_out[t])
        
        return mu_out, sigma_out


def load_model(model_path):
    """Load trained TPGMM model from PKL file"""
    print("\n" + "="*70)
    print("LOADING TRAINED TPGMM MODEL")
    print("="*70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Loaded model from: {model_path}")
    print(f"\nModel Information:")
    print(f"  K = {model_data['metadata']['K']} components")
    print(f"  Frames: {model_data['metadata']['n_frames']}")
    print(f"  Dimensions: {model_data['metadata']['n_features']}")
    print(f"  Demonstrations: {model_data['metadata']['n_demonstrations']}")
    print(f"  Timesteps: {model_data['metadata']['n_timesteps']}")
    print(f"  Best BIC: {model_data['metadata']['best_bic']:.2f}")
    print(f"\nFrame Descriptions:")
    for frame, desc in model_data['metadata']['frame_descriptions'].items():
        print(f"  {frame}: {desc}")
    
    return model_data


def compute_transformation_to_fr1(mu_generated, trajectories_fr1):
    """
    Compute transformation parameters to align generated trajectory with FR1
    
    Uses the first point of the generated trajectory and the average first point
    of FR1 trajectories to compute translation.
    
    For rotation, we use the initial direction of motion.
    
    Parameters:
    -----------
    mu_generated : array, shape (n_timesteps, 10)
        Generated trajectory from GMR (excludes time dimension)
    trajectories_fr1 : list of arrays
        Original FR1 trajectories with shape (n_timesteps, 11) including time
        
    Returns:
    --------
    mu_aligned : array, shape (n_timesteps, 10)
        Aligned trajectory in FR1 coordinate system (excludes time)
    translation : array, shape (10,)
        Translation vector applied (excludes time)
    rotation_angle : float
        Rotation angle applied (radians)
    """
    print("\n" + "="*70)
    print("COMPUTING TRANSFORMATION TO ALIGN WITH FR1")
    print("="*70)
    
    # Compute average first point from FR1 trajectories (exclude time dimension 10)
    first_points_fr1 = np.array([traj[0, :10] for traj in trajectories_fr1])  # Only dims 0-9
    avg_first_point = np.mean(first_points_fr1, axis=0)
    
    # Get first point of generated trajectory (already has 10 dims)
    first_point_gen = mu_generated[0, :]
    
    # Compute translation
    translation = avg_first_point - first_point_gen
    
    print(f"✓ FR1 average first point: {avg_first_point[:6]}")  # Show first 6 dims
    print(f"✓ Generated first point: {first_point_gen[:6]}")
    print(f"✓ Translation vector: {translation[:6]}")
    
    # Apply translation
    mu_aligned = mu_generated + translation
    
    # For rotation alignment, we'll use the initial motion direction
    # Get direction from first few points
    n_init = min(10, len(mu_generated))
    
    # Right ankle initial direction (dims 0-1)
    gen_dir_right = mu_generated[n_init, 0:2] - mu_generated[0, 0:2]
    # Get average direction from FR1 trajectories (dims 0-1, excluding time dim 10)
    avg_trajectory = np.mean([traj[:, :10] for traj in trajectories_fr1], axis=0)
    fr1_dir_right = avg_trajectory[n_init, 0:2] - avg_trajectory[0, 0:2]
    
    # Compute rotation angle
    gen_angle = np.arctan2(gen_dir_right[1], gen_dir_right[0])
    fr1_angle = np.arctan2(fr1_dir_right[1], fr1_dir_right[0])
    rotation_angle = fr1_angle - gen_angle
    
    print(f"✓ Rotation angle: {np.degrees(rotation_angle):.2f} degrees")
    
    # Apply rotation to position and velocity components
    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    # Rotate around the FR1 average first point
    # Right ankle positions (dims 0-1)
    mu_aligned[:, 0:2] = (rotation_matrix @ (mu_aligned[:, 0:2] - avg_first_point[0:2]).T).T + avg_first_point[0:2]
    # Right ankle velocities (dims 2-3)
    mu_aligned[:, 2:4] = (rotation_matrix @ mu_aligned[:, 2:4].T).T
    
    # Left ankle positions (dims 5-6)
    mu_aligned[:, 5:7] = (rotation_matrix @ (mu_aligned[:, 5:7] - avg_first_point[5:7]).T).T + avg_first_point[5:7]
    # Left ankle velocities (dims 7-8)
    mu_aligned[:, 7:9] = (rotation_matrix @ mu_aligned[:, 7:9].T).T
    
    print(f"✓ Applied 2D rotation and translation alignment")
    print(f"✓ Aligned first point: {mu_aligned[0, :6]}")
    
    return mu_aligned, translation, rotation_angle


def generate_trajectory_with_gmr_fr1_only(model, trajectories_fr1):
    """
    Generate smooth trajectory using optimized GMR (11 dimensions, FR1 ONLY)
    
    This version uses ONLY Frame 1 (FR1 - Hip frame) for trajectory generation,
    avoiding the global averaging that was causing misalignment.
    """
    print("\n" + "="*70)
    print("GENERATING TRAJECTORIES WITH GMR (11 Dimensions, FR1 ONLY)")
    print("="*70)
    print("NOTE: Using ONLY FR1 (Hip frame) parameters - no global averaging")
    
    n_features = 11
    
    # Use ONLY FR1 transformation (identity for now, can be customized)
    A_frame = np.eye(n_features)
    b_frame = np.zeros(n_features)
    
    n_query = 200
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    # Use FR1-only GMR
    gmr = OptimizedGMR_FR1Only(model, frame_index=0)  # frame_index=0 means FR1
    
    input_dims = [10]  # Time is dimension 10
    output_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All other dimensions
    
    print("Generating trajectory from FR1...")
    start_time = time.time()
    
    mu_generated, sigma_generated = gmr.predict(
        time_query, input_dims, output_dims, A_frame, b_frame
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Generated trajectory in {elapsed:.3f} seconds")
    print(f"✓ Trajectory shape: {mu_generated.shape}")
    
    # Align generated trajectory with FR1
    mu_aligned, translation, rotation = compute_transformation_to_fr1(mu_generated, trajectories_fr1)
    
    return time_query, mu_aligned, sigma_generated, translation, rotation


def visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1):
    """Visualize GMR results with all 11 dimensions"""
    print("\n" + "="*70)
    print("VISUALIZING GMR RESULTS (11 Dimensions)")
    print("="*70)
    
    # Define feature names for left and right ankles
    right_ankle_features = [
        'Right Ankle X Position',
        'Right Ankle Y Position',
        'Right Ankle X Velocity',
        'Right Ankle Y Velocity',
        'Right Ankle Angle'
    ]
    
    left_ankle_features = [
        'Left Ankle X Position',
        'Left Ankle Y Position',
        'Left Ankle X Velocity',
        'Left Ankle Y Velocity',
        'Left Ankle Angle'
    ]
    
    # Map dimensions: right ankle (0-4) and left ankle (5-9)
    right_ankle_dims = [0, 1, 2, 3, 4]
    left_ankle_dims = [5, 6, 7, 8, 9]
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 18))
    
    time_flat = time_query.flatten()
    
    # Plot right ankle on left column (column 0) and left ankle on right column (column 1)
    for row in range(5):
        # Right ankle (left column)
        right_dim = right_ankle_dims[row]
        for traj in trajectories_fr1:
            axes[row, 0].plot(traj[:, 10], traj[:, right_dim], 'gray', alpha=0.15, linewidth=0.8)
        
        mu_right = mu_generated[:, right_dim]
        std_right = np.sqrt(sigma_generated[:, right_dim, right_dim])
        
        axes[row, 0].plot(time_flat, mu_right, 'b-', linewidth=2.5, label='GMR Prediction (FR1)', zorder=10)
        axes[row, 0].fill_between(time_flat, mu_right - 2*std_right, mu_right + 2*std_right, 
                              color='lightblue', alpha=0.4, label='±2σ ', zorder=5)
        
        axes[row, 0].set_xlabel('Normalized Time', fontsize=11)
        axes[row, 0].set_ylabel(right_ankle_features[row], fontsize=11)
        axes[row, 0].set_title(f'{right_ankle_features[row]} vs Time', fontsize=12, fontweight='bold')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].legend(fontsize=9)
        
        print(f"\n{right_ankle_features[row]}:")
        print(f"  Mean: {np.mean(mu_right):.4f}")
        print(f"  Std Dev: {np.std(mu_right):.4f}")
        print(f"  Avg Uncertainty: {np.mean(std_right):.4f}")
        
        # Left ankle (right column)
        left_dim = left_ankle_dims[row]
        for traj in trajectories_fr1:
            axes[row, 1].plot(traj[:, 10], traj[:, left_dim], 'gray', alpha=0.15, linewidth=0.8)
        
        mu_left = mu_generated[:, left_dim]
        std_left = np.sqrt(sigma_generated[:, left_dim, left_dim])
        
        axes[row, 1].plot(time_flat, mu_left, 'r-', linewidth=2.5, label='GMR Prediction (FR1)', zorder=10)
        axes[row, 1].fill_between(time_flat, mu_left - 2*std_left, mu_left + 2*std_left, 
                              color='lightcoral', alpha=0.4, label='±2σ ', zorder=5)
        
        axes[row, 1].set_xlabel('Normalized Time', fontsize=11)
        axes[row, 1].set_ylabel(left_ankle_features[row], fontsize=11)
        axes[row, 1].set_title(f'{left_ankle_features[row]} vs Time', fontsize=12, fontweight='bold')
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].legend(fontsize=9)
        
        print(f"\n{left_ankle_features[row]}:")
        print(f"  Mean: {np.mean(mu_left):.4f}")
        print(f"  Std Dev: {np.std(mu_left):.4f}")
        print(f"  Avg Uncertainty: {np.mean(std_left):.4f}")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, 'gmr_trajectory_11dims.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved trajectory plot: {save_path}")
    plt.show()


def plot_position_trajectories_xy(trajectories_fr1, mu_generated, sigma_generated):
    """Plot X-Y position trajectories for both ankles"""
    print("\n" + "="*70)
    print("X-Y POSITION TRAJECTORY VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Right ankle
    for traj in trajectories_fr1:
        axes[0].plot(traj[:, 0], traj[:, 1], 'gray', alpha=0.15, linewidth=0.8)
    
    # Plot generated trajectory with closed loop
    x_right = np.append(mu_generated[:, 0], mu_generated[0, 0])
    y_right = np.append(mu_generated[:, 1], mu_generated[0, 1])
    axes[0].plot(x_right, y_right, 'b-', linewidth=2.5, 
                label='GMR Prediction (FR1)', zorder=10)
    
    std_x = np.sqrt(sigma_generated[:, 0, 0])
    std_y = np.sqrt(sigma_generated[:, 1, 1])
    
    for i in range(0, len(mu_generated), 1):
        ellipse = Ellipse((mu_generated[i, 0], mu_generated[i, 1]),
                         width=2*std_x[i], height=2*std_y[i],
                         facecolor='lightblue',
                         alpha=0.3, linewidth=1)
        axes[0].add_patch(ellipse)
    
    axes[0].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Right Ankle Position Trajectory (X-Y)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_aspect('equal', adjustable='box')
    
    # Left ankle
    for traj in trajectories_fr1:
        axes[1].plot(traj[:, 5], traj[:, 6], 'gray', alpha=0.15, linewidth=0.8)
    
    # Plot generated trajectory with closed loop
    x_left = np.append(mu_generated[:, 5], mu_generated[0, 5])
    y_left = np.append(mu_generated[:, 6], mu_generated[0, 6])
    axes[1].plot(x_left, y_left, 'r-', linewidth=2.5,
                label='GMR Prediction (FR1)', zorder=10)
    
    std_x = np.sqrt(sigma_generated[:, 5, 5])
    std_y = np.sqrt(sigma_generated[:, 6, 6])
    
    for i in range(0, len(mu_generated), 1):
        ellipse = Ellipse((mu_generated[i, 5], mu_generated[i, 6]),
                         width=2*std_x[i], height=2*std_y[i],
                         facecolor='lightcoral',
                         alpha=0.3, linewidth=1)
        axes[1].add_patch(ellipse)
    
    axes[1].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Left Ankle Position Trajectory (X-Y)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, 'ankle_position_trajectories_xy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()


def plot_velocity_trajectories_vxvy(trajectories_fr1, mu_generated, sigma_generated):
    """Plot Vx-Vy velocity trajectories for both ankles"""
    print("\n" + "="*70)
    print("Vx-Vy VELOCITY TRAJECTORY VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Right ankle
    for traj in trajectories_fr1:
        axes[0].plot(traj[:, 2], traj[:, 3], 'gray', alpha=0.15, linewidth=0.8)
    
    # Plot generated trajectory with closed loop
    vx_right = np.append(mu_generated[:, 2], mu_generated[0, 2])
    vy_right = np.append(mu_generated[:, 3], mu_generated[0, 3])
    axes[0].plot(vx_right, vy_right, 'b-', linewidth=2.5,
                label='GMR Prediction (FR1)', zorder=10)
    
    std_vx = np.sqrt(sigma_generated[:, 2, 2])
    std_vy = np.sqrt(sigma_generated[:, 3, 3])
    
    for i in range(0, len(mu_generated), 1):
        ellipse = Ellipse((mu_generated[i, 2], mu_generated[i, 3]),
                         width=2*std_vx[i], height=2*std_vy[i],
                         facecolor='lightblue',
                         alpha=0.3, linewidth=1)
        axes[0].add_patch(ellipse)
    
    axes[0].set_xlabel('X Velocity (m/s)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Y Velocity (m/s)', fontsize=12, fontweight='bold')
    axes[0].set_title('Right Ankle Velocity Trajectory (Vx-Vy)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_aspect('equal', adjustable='box')
    
    # Left ankle
    for traj in trajectories_fr1:
        axes[1].plot(traj[:, 7], traj[:, 8], 'gray', alpha=0.15, linewidth=0.8)
    
    # Plot generated trajectory with closed loop
    vx_left = np.append(mu_generated[:, 7], mu_generated[0, 7])
    vy_left = np.append(mu_generated[:, 8], mu_generated[0, 8])
    axes[1].plot(vx_left, vy_left, 'r-', linewidth=2.5,
                label='GMR Prediction (FR1)', zorder=10)
    
    std_vx = np.sqrt(sigma_generated[:, 7, 7])
    std_vy = np.sqrt(sigma_generated[:, 8, 8])
    
    for i in range(0, len(mu_generated), 1):
        ellipse = Ellipse((mu_generated[i, 7], mu_generated[i, 8]),
                         width=2*std_vx[i], height=2*std_vy[i],
                         facecolor='lightcoral',
                         alpha=0.3, linewidth=1)
        axes[1].add_patch(ellipse)
    
    axes[1].set_xlabel('X Velocity (m/s)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Y Velocity (m/s)', fontsize=12, fontweight='bold')
    axes[1].set_title('Left Ankle Velocity Trajectory (Vx-Vy)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, 'ankle_velocity_trajectories_vxvy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()


def main(model_folder=None, reg_factor=1e-5):
    """
    Main GMR application pipeline - FR1 ONLY VERSION
    
    Parameters:
    -----------
    model_folder : str, optional
        Custom path to model folder. If None, automatically constructs path based on reg_factor
    reg_factor : float
        Regularization factor used during training (default: 1e-5)
        Used to construct folder name: train_tpgmm_model_reg{reg_factor:.0e}
    """
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION - FR1 ONLY VERSION")
    print("="*70)
    print("FIXED: Generates trajectories using ONLY FR1 (Hip frame)")
    print("       No global averaging across frames")
    
    # Construct model path based on reg_factor if not provided
    if model_folder is None:
        model_folder = f"train_tpgmm_model_reg{reg_factor:.0e}"
        print(f"✓ Using auto-constructed folder: {model_folder}")
    else:
        print(f"✓ Using custom folder: {model_folder}")
    
    model_path = os.path.join(model_folder, "trained_model.pkl")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        print(f"\nExpected location: {model_path}")
        print("\nPlease either:")
        print(f"  1. Train a model with reg_factor={reg_factor} using train_tpgmm_model.py")
        print(f"  2. Provide a custom model_folder path")
        print(f"  3. Adjust the reg_factor parameter to match your trained model")
        sys.exit(1)
    
    # Extract output folder from model path
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = os.path.dirname(model_path)
    
    overall_start = time.time()
    
    # Load trained model
    model_data = load_model(model_path)
    
    best_model = model_data['model']
    trajectories_fr1 = model_data['trajectories_fr1']
    
    # Generate trajectories with GMR (FR1 ONLY)
    time_query, mu_generated, sigma_generated, translation, rotation = generate_trajectory_with_gmr_fr1_only(
        best_model, trajectories_fr1
    )
    
    # Visualize results
    visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1)
    
    # Visualize X-Y position trajectories
    plot_position_trajectories_xy(trajectories_fr1, mu_generated, sigma_generated)
    
    # Visualize Vx-Vy velocity trajectories
    plot_velocity_trajectories_vxvy(trajectories_fr1, mu_generated, sigma_generated)
    
    overall_time = time.time() - overall_start
    
    # Save trajectory data for post-processing
    trajectory_file = os.path.join(OUTPUT_FOLDER, 'gmr_trajectory_fr1.npy')
    time_file = os.path.join(OUTPUT_FOLDER, 'time_query.npy')
    sigma_file = os.path.join(OUTPUT_FOLDER, 'gmr_sigma_fr1.npy')
    transform_file = os.path.join(OUTPUT_FOLDER, 'transformation_params.npz')
    
    np.save(trajectory_file, mu_generated)
    np.save(time_file, time_query)
    np.save(sigma_file, sigma_generated)
    np.savez(transform_file, translation=translation, rotation_angle=rotation)
    
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION COMPLETE! (FR1 ONLY)")
    print("="*70)
    print(f"✓ Model: K = {best_model.K} components")
    print(f"✓ Frame: FR1 (Hip frame) ONLY - no global averaging")
    print(f"✓ Alignment: Translation + Rotation applied")
    print(f"✓ Total execution time: {overall_time:.2f} seconds")
    print(f"\n✓ All figures saved in folder: {OUTPUT_FOLDER}/")
    print("\n✓ Generated GMR visualizations:")
    print(f"  1. {OUTPUT_FOLDER}/gmr_trajectory_11dims.png")
    print(f"  2. {OUTPUT_FOLDER}/ankle_position_trajectories_xy.png")
    print(f"  3. {OUTPUT_FOLDER}/ankle_velocity_trajectories_vxvy.png")
    print(f"\n✓ Saved trajectory data:")
    print(f"  - {trajectory_file}")
    print(f"  - {time_file}")
    print(f"  - {sigma_file}")
    print(f"  - {transform_file}")
    print("="*70)
    
    return mu_generated, sigma_generated, translation, rotation


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION SCRIPT - FR1 ONLY VERSION")
    print("="*70)
    print("This script loads a trained TPGMM model and generates trajectories")
    print("using ONLY FR1 (Hip frame) - avoiding global averaging.\n")
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    # Option 1: Auto-construct path based on reg_factor
    reg_factor = 3e-4  # Must match the reg_factor used during training
    model_folder = None  # Will auto-construct: train_tpgmm_model_reg5e-04
    
    # Option 2: Manually specify model folder path (uncomment to use)
    # model_folder = "train_tpgmm_model_reg5e-04"  # Custom path
    # reg_factor = 5e-4  # For reference only when using manual path
    # ============================================================
    
    print(f"Configuration:")
    if model_folder is None:
        print(f"  Regularization factor: {reg_factor}")
        print(f"  Model folder: auto-constructed from reg_factor")
    else:
        print(f"  Model folder: {model_folder} (manual)")
    print()
    
    # Run GMR trajectory generation (FR1 ONLY)
    mu_gen, sigma_gen, trans, rot = main(model_folder=model_folder, reg_factor=reg_factor)
    
    print(f"\n✅ Trajectory generation complete! (FR1 frame only)")
    print(f"   Translation: {trans[:6]}")
    print(f"   Rotation: {np.degrees(rot):.2f} degrees")