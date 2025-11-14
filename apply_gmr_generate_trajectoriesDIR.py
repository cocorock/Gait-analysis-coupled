"""
GMR TRAJECTORY GENERATION SCRIPT
Load trained TPGMM model and generate trajectories using Gaussian Mixture Regression

This script:
1. Loads trained TPGMM model from PKL file (auto-selects folder based on reg_factor)
2. Applies Gaussian Mixture Regression (GMR)
3. Generates smooth trajectory predictions
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
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


class OptimizedGMR:
    """
    Optimized Gaussian Mixture Regression for TPGMM
    """
    def __init__(self, tpgmm_model):
        self.model = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
    
    def predict(self, X_query, input_dims, output_dims, A_frames, b_frames):
        """
        GMR prediction with task parameters (optimized)
        """
        n_query = X_query.shape[0]
        n_out = len(output_dims)
        
        # Pre-compute products of Gaussians for all frames
        mu_prod = np.zeros((self.K, self.D))
        Sigma_prod_inv = np.zeros((self.K, self.D, self.D))
        
        for k in range(self.K):
            Sigma_inv_sum = np.zeros((self.D, self.D))
            weighted_mu_sum = np.zeros(self.D)
            
            for p in range(self.P):
                A = A_frames[p]
                b = b_frames[p]
                
                # Transform mean
                mu_global = A @ self.model.means[p, k] + b
                
                # Transform covariance
                Sigma_global = A @ self.model.covars[p, k] @ A.T
                
                # Inverse and accumulate
                Sigma_inv = np.linalg.inv(Sigma_global)
                Sigma_inv_sum += Sigma_inv
                weighted_mu_sum += Sigma_inv @ mu_global
            
            Sigma_prod_inv[k] = Sigma_inv_sum
            mu_prod[k] = np.linalg.solve(Sigma_inv_sum, weighted_mu_sum)
        
        # GMR for each query point
        mu_out = np.zeros((n_query, n_out))
        sigma_out = np.zeros((n_query, n_out, n_out))
        
        for t in range(n_query):
            x_in = X_query[t]
            
            # Compute responsibilities
            h = np.zeros(self.K)
            for k in range(self.K):
                mu_in = mu_prod[k][input_dims]
                Sigma_in_inv = Sigma_prod_inv[k][np.ix_(input_dims, input_dims)]
                
                diff = x_in - mu_in
                h[k] = self.model.priors[k] * np.exp(-0.5 * diff @ Sigma_in_inv @ diff)
            
            h = h / (np.sum(h) + 1e-10)
            
            # Condition on input
            for k in range(self.K):
                mu_k = mu_prod[k]
                Sigma_k_inv = Sigma_prod_inv[k]
                
                # Compute conditional mean and covariance
                Sigma_k = np.linalg.inv(Sigma_k_inv)
                
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


def generate_trajectory_with_gmr(model, trajectories_fr1):
    """Generate smooth trajectory using optimized GMR (11 dimensions, 3 frames)"""
    print("\n" + "="*70)
    print("GENERATING TRAJECTORIES WITH GMR (11 Dimensions, 3 Frames)")
    print("="*70)
    
    n_features = 11
    A_frames = [np.eye(n_features), np.eye(n_features), np.eye(n_features)]
    b_frames = [np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)]
    
    n_query = 200
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    gmr = OptimizedGMR(model)
    
    input_dims = [10]  # Time is dimension 10
    output_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All other dimensions
    
    print("Generating trajectory...")
    start_time = time.time()
    
    mu_generated, sigma_generated = gmr.predict(
        time_query, input_dims, output_dims, A_frames, b_frames
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Generated trajectory in {elapsed:.3f} seconds")
    print(f"✓ Trajectory shape: {mu_generated.shape}")
    
    return time_query, mu_generated, sigma_generated


def visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1):
    """Visualize GMR results with all 11 dimensions"""
    print("\n" + "="*70)
    print("VISUALIZING GMR RESULTS (11 Dimensions)")
    print("="*70)
    
    feature_names = [
        'Right Ankle X Position',
        'Right Ankle Y Position',
        'Right Ankle X Velocity',
        'Right Ankle Y Velocity',
        'Right Ankle Angle',
        'Left Ankle X Position',
        'Left Ankle Y Position',
        'Left Ankle X Velocity',
        'Left Ankle Y Velocity',
        'Left Ankle Angle'
    ]
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    time_flat = time_query.flatten()
    
    for dim in range(10):
        # Plot demonstrations
        for traj in trajectories_fr1:
            axes[dim].plot(traj[:, 10], traj[:, dim], 'gray', alpha=0.15, linewidth=0.8)
        
        # Plot GMR prediction
        mu = mu_generated[:, dim]
        std = np.sqrt(sigma_generated[:, dim, dim])
        
        axes[dim].plot(time_flat, mu, 'b-', linewidth=2.5, label='GMR Prediction', zorder=10)
        axes[dim].fill_between(time_flat, mu - 2*std, mu + 2*std, 
                              color='lightblue', alpha=0.4, label='±2σ Uncertainty', zorder=5)
        
        axes[dim].set_xlabel('Normalized Time', fontsize=11)
        axes[dim].set_ylabel(feature_names[dim], fontsize=11)
        axes[dim].set_title(f'{feature_names[dim]} vs Time', fontsize=12, fontweight='bold')
        axes[dim].grid(True, alpha=0.3)
        axes[dim].legend(fontsize=9)
        
        print(f"\n{feature_names[dim]}:")
        print(f"  Mean: {np.mean(mu):.4f}")
        print(f"  Std Dev: {np.std(mu):.4f}")
        print(f"  Avg Uncertainty: {np.mean(std):.4f}")
    
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
    
    axes[0].plot(mu_generated[:, 0], mu_generated[:, 1], 'b-', linewidth=2.5, 
                label='GMR Prediction', zorder=10)
    
    std_x = np.sqrt(sigma_generated[:, 0, 0])
    std_y = np.sqrt(sigma_generated[:, 1, 1])
    
    for i in range(0, len(mu_generated), 20):
        ellipse = Ellipse((mu_generated[i, 0], mu_generated[i, 1]),
                         width=4*std_x[i], height=4*std_y[i],
                         facecolor='lightblue', edgecolor='blue',
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
    
    axes[1].plot(mu_generated[:, 5], mu_generated[:, 6], 'r-', linewidth=2.5,
                label='GMR Prediction', zorder=10)
    
    std_x = np.sqrt(sigma_generated[:, 5, 5])
    std_y = np.sqrt(sigma_generated[:, 6, 6])
    
    for i in range(0, len(mu_generated), 20):
        ellipse = Ellipse((mu_generated[i, 5], mu_generated[i, 6]),
                         width=4*std_x[i], height=4*std_y[i],
                         facecolor='lightcoral', edgecolor='red',
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
    
    axes[0].plot(mu_generated[:, 2], mu_generated[:, 3], 'b-', linewidth=2.5,
                label='GMR Prediction', zorder=10)
    
    std_vx = np.sqrt(sigma_generated[:, 2, 2])
    std_vy = np.sqrt(sigma_generated[:, 3, 3])
    
    for i in range(0, len(mu_generated), 20):
        ellipse = Ellipse((mu_generated[i, 2], mu_generated[i, 3]),
                         width=4*std_vx[i], height=4*std_vy[i],
                         facecolor='lightblue', edgecolor='blue',
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
    
    axes[1].plot(mu_generated[:, 7], mu_generated[:, 8], 'r-', linewidth=2.5,
                label='GMR Prediction', zorder=10)
    
    std_vx = np.sqrt(sigma_generated[:, 7, 7])
    std_vy = np.sqrt(sigma_generated[:, 8, 8])
    
    for i in range(0, len(mu_generated), 20):
        ellipse = Ellipse((mu_generated[i, 7], mu_generated[i, 8]),
                         width=4*std_vx[i], height=4*std_vy[i],
                         facecolor='lightcoral', edgecolor='red',
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
    Main GMR application pipeline
    
    Parameters:
    -----------
    model_folder : str, optional
        Custom path to model folder. If None, automatically constructs path based on reg_factor
    reg_factor : float
        Regularization factor used during training (default: 1e-5)
        Used to construct folder name: train_tpgmm_model_reg{reg_factor:.0e}
    """
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION")
    print("="*70)
    
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
    
    # Generate trajectories with GMR
    time_query, mu_generated, sigma_generated = generate_trajectory_with_gmr(
        best_model, trajectories_fr1
    )
    
    # Visualize results
    visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1)
    
    # Visualize X-Y position trajectories
    plot_position_trajectories_xy(trajectories_fr1, mu_generated, sigma_generated)
    
    # Visualize Vx-Vy velocity trajectories
    plot_velocity_trajectories_vxvy(trajectories_fr1, mu_generated, sigma_generated)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION COMPLETE!")
    print("="*70)
    print(f"✓ Model: K = {best_model.K} components")
    print(f"✓ Frames: 3 (FR1: Hip, FR2: Right foot, FR3: Left foot)")
    print(f"✓ Total execution time: {overall_time:.2f} seconds")
    print(f"\n✓ All figures saved in folder: {OUTPUT_FOLDER}/")
    print("\n✓ Generated GMR visualizations:")
    print(f"  1. {OUTPUT_FOLDER}/gmr_trajectory_11dims.png")
    print(f"  2. {OUTPUT_FOLDER}/ankle_position_trajectories_xy.png")
    print(f"  3. {OUTPUT_FOLDER}/ankle_velocity_trajectories_vxvy.png")
    print("="*70)
    
    return mu_generated, sigma_generated


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GMR TRAJECTORY GENERATION SCRIPT")
    print("="*70)
    print("This script loads a trained TPGMM model and generates trajectories.\n")
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    # Option 1: Auto-construct path based on reg_factor
    reg_factor = 5e-4  # Must match the reg_factor used during training
    model_folder = None  # Will auto-construct: train_tpgmm_model_reg1e-05
    
    # Option 2: Manually specify model folder path (uncomment to use)
    # model_folder = "train_tpgmm_model_reg1e-05"  # Custom path
    # reg_factor = 1e-5  # For reference only when using manual path
    # ============================================================
    
    print(f"Configuration:")
    if model_folder is None:
        print(f"  Regularization factor: {reg_factor}")
        print(f"  Model folder: auto-constructed from reg_factor")
    else:
        print(f"  Model folder: {model_folder} (manual)")
    print()
    
    # Run GMR trajectory generation
    mu_gen, sigma_gen = main(model_folder=model_folder, reg_factor=reg_factor)
    
    print(f"\n✅ Trajectory generation complete!")
