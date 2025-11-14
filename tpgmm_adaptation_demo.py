"""
TPGMM ADAPTATION DEMONSTRATION SCRIPT
=====================================
Demonstrates TPGMM's ability to adapt trajectories to task parameter changes

This script systematically varies the FR2 (Right Foot) frame parameters:
1. Horizontal position (X-axis): ±0.2m in 5 steps
2. Vertical position (Y-axis): ±0.1m in 5 steps  
3. Orientation (rotation): ±15° in 5 steps
4. Combined variations: Position + Orientation grids

Outputs:
- Individual adaptation plots for each parameter
- Overlay comparison plots
- Grid/heatmap visualizations
- Quantitative analysis plots
- Animation frames (optional)

Author: Victor
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from matplotlib.patches import Ellipse, FancyArrow
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

# Configure matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

np.set_printoptions(precision=3, suppress=True)


class OptimizedTPGMM:
    """TPGMM class for unpickling"""
    def __init__(self, n_components, n_frames, n_features, reg_factor=1e-5, 
                 max_iter=100, tol=1e-4, verbose=True):
        self.K = n_components
        self.P = n_frames
        self.D = n_features
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.priors = None
        self.means = None
        self.covars = None
        self._inv_covars = None
        self._log_dets = None
        self._chol_covars = None
        self.log_likelihood_history = []
        self.converged = False
        self.n_iter = 0


class OptimizedGMR:
    """Optimized GMR for TPGMM with task parameters"""
    def __init__(self, tpgmm_model):
        self.model = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
    
    def predict(self, X_query, input_dims, output_dims, A_frames, b_frames):
        """GMR prediction with task parameters"""
        n_query = X_query.shape[0]
        n_out = len(output_dims)
        
        # Product of Gaussians across frames
        mu_prod = np.zeros((self.K, self.D))
        Sigma_prod_inv = np.zeros((self.K, self.D, self.D))
        
        for k in range(self.K):
            Sigma_inv_sum = np.zeros((self.D, self.D))
            weighted_mu_sum = np.zeros(self.D)
            
            for p in range(self.P):
                A = A_frames[p]
                b = b_frames[p]
                
                # Transform to global frame
                mu_global = A @ self.model.means[p, k] + b
                Sigma_global = A @ self.model.covars[p, k] @ A.T
                
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
            
            # Conditional distribution
            for k in range(self.K):
                mu_k = mu_prod[k]
                Sigma_k = np.linalg.inv(Sigma_prod_inv[k])
                
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
    """Load trained TPGMM model"""
    print("\n" + "="*70)
    print("LOADING TRAINED TPGMM MODEL")
    print("="*70)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Loaded model from: {model_path}")
    print(f"  K = {model_data['metadata']['K']} components")
    print(f"  Frames: {model_data['metadata']['n_frames']}")
    print(f"  Dimensions: {model_data['metadata']['n_features']}")
    
    return model_data


def create_transformation_matrix_2d(tx, ty, theta):
    """
    Create 11D transformation matrix for 2D translation and rotation
    
    Parameters:
    -----------
    tx, ty : float
        Translation in x and y
    theta : float
        Rotation angle in radians
    
    Returns:
    --------
    A : array (11, 11)
        Transformation matrix
    b : array (11,)
        Translation vector
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # 2D rotation matrix
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    
    # 11D transformation matrix
    A = np.eye(11)
    
    # Apply rotation to positions and velocities
    # Right ankle position (dims 0-1)
    A[0:2, 0:2] = R
    # Right ankle velocity (dims 2-3)
    A[2:4, 2:4] = R
    # Left ankle position (dims 5-6)
    A[5:7, 5:7] = R
    # Left ankle velocity (dims 7-8)
    A[7:9, 7:9] = R
    
    # Translation vector
    b = np.zeros(11)
    b[0] = tx  # Right ankle X
    b[1] = ty  # Right ankle Y
    b[5] = tx  # Left ankle X
    b[6] = ty  # Left ankle Y
    
    return A, b


def generate_adapted_trajectory(model, A_frames, b_frames, n_query=200):
    """
    Generate trajectory with given task parameters
    
    Parameters:
    -----------
    model : OptimizedTPGMM
        Trained TPGMM model
    A_frames : list of arrays
        Transformation matrices for each frame [A_FR1, A_FR2, A_FR3]
    b_frames : list of arrays  
        Translation vectors for each frame [b_FR1, b_FR2, b_FR3]
    n_query : int
        Number of query points
        
    Returns:
    --------
    mu_generated : array (n_query, 10)
        Generated trajectory
    sigma_generated : array (n_query, 10, 10)
        Covariance matrices
    """
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    gmr = OptimizedGMR(model)
    
    input_dims = [10]  # Time
    output_dims = list(range(10))  # All other dimensions
    
    mu_generated, sigma_generated = gmr.predict(
        time_query, input_dims, output_dims, A_frames, b_frames
    )
    
    return mu_generated, sigma_generated


def compute_trajectory_metrics(trajectory):
    """
    Compute quantitative metrics for a trajectory
    
    Parameters:
    -----------
    trajectory : array (n_timesteps, 10)
        Trajectory data
        
    Returns:
    --------
    metrics : dict
        Dictionary with computed metrics
    """
    # Right ankle metrics (dims 0-4)
    right_x = trajectory[:, 0]
    right_y = trajectory[:, 1]
    right_vx = trajectory[:, 2]
    right_vy = trajectory[:, 3]
    
    # Left ankle metrics (dims 5-9)
    left_x = trajectory[:, 5]
    left_y = trajectory[:, 6]
    left_vx = trajectory[:, 7]
    left_vy = trajectory[:, 8]
    
    metrics = {
        'right_step_length': np.max(right_x) - np.min(right_x),
        'right_step_height': np.max(right_y) - np.min(right_y),
        'right_peak_velocity': np.max(np.sqrt(right_vx**2 + right_vy**2)),
        'left_step_length': np.max(left_x) - np.min(left_x),
        'left_step_height': np.max(left_y) - np.min(left_y),
        'left_peak_velocity': np.max(np.sqrt(left_vx**2 + left_vy**2)),
        'stride_length': np.max(right_x) - np.min(left_x),
    }
    
    return metrics


def test_horizontal_variation(model, output_folder, baseline_frames):
    """Test horizontal position variations of FR2"""
    print("\n" + "="*70)
    print("TEST 1: HORIZONTAL POSITION VARIATION (FR2)")
    print("="*70)
    
    # Define horizontal shifts
    x_shifts = np.linspace(-0.2, 0.2, 5)
    
    trajectories = []
    metrics_list = []
    
    for dx in x_shifts:
        # Modify FR2 transformation
        A_FR2, b_FR2 = create_transformation_matrix_2d(dx, 0, 0)
        A_frames = [baseline_frames[0][0], A_FR2, baseline_frames[2][0]]
        b_frames = [baseline_frames[0][1], b_FR2, baseline_frames[2][1]]
        
        mu, sigma = generate_adapted_trajectory(model, A_frames, b_frames)
        trajectories.append((mu, sigma))
        metrics_list.append(compute_trajectory_metrics(mu))
        
        print(f"  dx = {dx:+.3f}m: Step length = {metrics_list[-1]['right_step_length']:.3f}m")
    
    # Visualize
    plot_horizontal_adaptation(trajectories, x_shifts, output_folder)
    plot_metrics_vs_horizontal(metrics_list, x_shifts, output_folder)
    
    return trajectories, metrics_list, x_shifts


def test_vertical_variation(model, output_folder, baseline_frames):
    """Test vertical position variations of FR2"""
    print("\n" + "="*70)
    print("TEST 2: VERTICAL POSITION VARIATION (FR2)")
    print("="*70)
    
    # Define vertical shifts
    y_shifts = np.linspace(-0.1, 0.1, 5)
    
    trajectories = []
    metrics_list = []
    
    for dy in y_shifts:
        # Modify FR2 transformation
        A_FR2, b_FR2 = create_transformation_matrix_2d(0, dy, 0)
        A_frames = [baseline_frames[0][0], A_FR2, baseline_frames[2][0]]
        b_frames = [baseline_frames[0][1], b_FR2, baseline_frames[2][1]]
        
        mu, sigma = generate_adapted_trajectory(model, A_frames, b_frames)
        trajectories.append((mu, sigma))
        metrics_list.append(compute_trajectory_metrics(mu))
        
        print(f"  dy = {dy:+.3f}m: Step height = {metrics_list[-1]['right_step_height']:.3f}m")
    
    # Visualize
    plot_vertical_adaptation(trajectories, y_shifts, output_folder)
    plot_metrics_vs_vertical(metrics_list, y_shifts, output_folder)
    
    return trajectories, metrics_list, y_shifts


def test_orientation_variation(model, output_folder, baseline_frames):
    """Test orientation variations of FR2"""
    print("\n" + "="*70)
    print("TEST 3: ORIENTATION VARIATION (FR2)")
    print("="*70)
    
    # Define rotation angles (±15 degrees)
    angles_deg = np.linspace(-15, 15, 5)
    angles_rad = np.deg2rad(angles_deg)
    
    trajectories = []
    metrics_list = []
    
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        # Modify FR2 transformation
        A_FR2, b_FR2 = create_transformation_matrix_2d(0, 0, angle_rad)
        A_frames = [baseline_frames[0][0], A_FR2, baseline_frames[2][0]]
        b_frames = [baseline_frames[0][1], b_FR2, baseline_frames[2][1]]
        
        mu, sigma = generate_adapted_trajectory(model, A_frames, b_frames)
        trajectories.append((mu, sigma))
        metrics_list.append(compute_trajectory_metrics(mu))
        
        print(f"  θ = {angle_deg:+.1f}°: Peak velocity = {metrics_list[-1]['right_peak_velocity']:.3f}m/s")
    
    # Visualize
    plot_orientation_adaptation(trajectories, angles_deg, output_folder)
    plot_metrics_vs_orientation(metrics_list, angles_deg, output_folder)
    
    return trajectories, metrics_list, angles_deg


def test_combined_variation(model, output_folder, baseline_frames):
    """Test combined position and orientation variations"""
    print("\n" + "="*70)
    print("TEST 4: COMBINED POSITION + ORIENTATION VARIATION (FR2)")
    print("="*70)
    
    # Define grid
    x_values = np.linspace(-0.15, 0.15, 3)
    angle_values = np.linspace(-10, 10, 3)
    
    trajectories_grid = []
    
    for dx in x_values:
        row = []
        for angle_deg in angle_values:
            angle_rad = np.deg2rad(angle_deg)
            A_FR2, b_FR2 = create_transformation_matrix_2d(dx, 0, angle_rad)
            A_frames = [baseline_frames[0][0], A_FR2, baseline_frames[2][0]]
            b_frames = [baseline_frames[0][1], b_FR2, baseline_frames[2][1]]
            
            mu, sigma = generate_adapted_trajectory(model, A_frames, b_frames)
            row.append((mu, sigma))
            print(f"  dx = {dx:+.2f}m, θ = {angle_deg:+.1f}°")
        
        trajectories_grid.append(row)
    
    # Visualize
    plot_combined_grid(trajectories_grid, x_values, angle_values, output_folder)
    
    return trajectories_grid


def plot_horizontal_adaptation(trajectories, x_shifts, output_folder):
    """Plot trajectories for horizontal variations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color gradient
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(x_shifts)))
    
    # Right ankle trajectories
    for i, ((mu, sigma), dx) in enumerate(zip(trajectories, x_shifts)):
        axes[0].plot(mu[:, 0], mu[:, 1], color=colors[i], linewidth=2.5,
                    label=f'dx = {dx:+.2f}m', alpha=0.8)
        # Mark start and end
        axes[0].plot(mu[0, 0], mu[0, 1], 'o', color=colors[i], markersize=8)
        axes[0].plot(mu[-1, 0], mu[-1, 1], 's', color=colors[i], markersize=8)
    
    axes[0].set_xlabel('X Position (m)', fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontweight='bold')
    axes[0].set_title('Right Ankle Adaptation (Horizontal Shift)', fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Left ankle trajectories
    for i, ((mu, sigma), dx) in enumerate(zip(trajectories, x_shifts)):
        axes[1].plot(mu[:, 5], mu[:, 6], color=colors[i], linewidth=2.5,
                    label=f'dx = {dx:+.2f}m', alpha=0.8)
        axes[1].plot(mu[0, 5], mu[0, 6], 'o', color=colors[i], markersize=8)
        axes[1].plot(mu[-1, 5], mu[-1, 6], 's', color=colors[i], markersize=8)
    
    axes[1].set_xlabel('X Position (m)', fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontweight='bold')
    axes[1].set_title('Left Ankle Adaptation (Horizontal Shift)', fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'adaptation_horizontal.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: adaptation_horizontal.png")
    plt.close()


def plot_vertical_adaptation(trajectories, y_shifts, output_folder):
    """Plot trajectories for vertical variations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(y_shifts)))
    
    # Right ankle
    for i, ((mu, sigma), dy) in enumerate(zip(trajectories, y_shifts)):
        axes[0].plot(mu[:, 0], mu[:, 1], color=colors[i], linewidth=2.5,
                    label=f'dy = {dy:+.2f}m', alpha=0.8)
        axes[0].plot(mu[0, 0], mu[0, 1], 'o', color=colors[i], markersize=8)
        axes[0].plot(mu[-1, 0], mu[-1, 1], 's', color=colors[i], markersize=8)
    
    axes[0].set_xlabel('X Position (m)', fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontweight='bold')
    axes[0].set_title('Right Ankle Adaptation (Vertical Shift)', fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Left ankle
    for i, ((mu, sigma), dy) in enumerate(zip(trajectories, y_shifts)):
        axes[1].plot(mu[:, 5], mu[:, 6], color=colors[i], linewidth=2.5,
                    label=f'dy = {dy:+.2f}m', alpha=0.8)
        axes[1].plot(mu[0, 5], mu[0, 6], 'o', color=colors[i], markersize=8)
        axes[1].plot(mu[-1, 5], mu[-1, 6], 's', color=colors[i], markersize=8)
    
    axes[1].set_xlabel('X Position (m)', fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontweight='bold')
    axes[1].set_title('Left Ankle Adaptation (Vertical Shift)', fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'adaptation_vertical.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: adaptation_vertical.png")
    plt.close()


def plot_orientation_adaptation(trajectories, angles, output_folder):
    """Plot trajectories for orientation variations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.twilight(np.linspace(0.2, 0.8, len(angles)))
    
    # Right ankle
    for i, ((mu, sigma), angle) in enumerate(zip(trajectories, angles)):
        axes[0].plot(mu[:, 0], mu[:, 1], color=colors[i], linewidth=2.5,
                    label=f'θ = {angle:+.1f}°', alpha=0.8)
        axes[0].plot(mu[0, 0], mu[0, 1], 'o', color=colors[i], markersize=8)
        axes[0].plot(mu[-1, 0], mu[-1, 1], 's', color=colors[i], markersize=8)
        
        # Draw orientation arrow at end point
        arrow_length = 0.05
        dx = arrow_length * np.cos(np.deg2rad(angle))
        dy = arrow_length * np.sin(np.deg2rad(angle))
        axes[0].arrow(mu[-1, 0], mu[-1, 1], dx, dy, 
                     head_width=0.02, head_length=0.015, fc=colors[i], ec=colors[i], alpha=0.6)
    
    axes[0].set_xlabel('X Position (m)', fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontweight='bold')
    axes[0].set_title('Right Ankle Adaptation (Orientation)', fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Left ankle
    for i, ((mu, sigma), angle) in enumerate(zip(trajectories, angles)):
        axes[1].plot(mu[:, 5], mu[:, 6], color=colors[i], linewidth=2.5,
                    label=f'θ = {angle:+.1f}°', alpha=0.8)
        axes[1].plot(mu[0, 5], mu[0, 6], 'o', color=colors[i], markersize=8)
        axes[1].plot(mu[-1, 5], mu[-1, 6], 's', color=colors[i], markersize=8)
        
        arrow_length = 0.05
        dx = arrow_length * np.cos(np.deg2rad(angle))
        dy = arrow_length * np.sin(np.deg2rad(angle))
        axes[1].arrow(mu[-1, 5], mu[-1, 6], dx, dy,
                     head_width=0.02, head_length=0.015, fc=colors[i], ec=colors[i], alpha=0.6)
    
    axes[1].set_xlabel('X Position (m)', fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontweight='bold')
    axes[1].set_title('Left Ankle Adaptation (Orientation)', fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'adaptation_orientation.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: adaptation_orientation.png")
    plt.close()


def plot_combined_grid(trajectories_grid, x_values, angle_values, output_folder):
    """Plot grid of combined variations"""
    n_rows = len(x_values)
    n_cols = len(angle_values)
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, dx in enumerate(x_values):
        for j, angle in enumerate(angle_values):
            ax = fig.add_subplot(gs[i, j])
            mu, sigma = trajectories_grid[i][j]
            
            # Plot both ankles
            ax.plot(mu[:, 0], mu[:, 1], 'b-', linewidth=2, label='Right', alpha=0.8)
            ax.plot(mu[:, 5], mu[:, 6], 'r-', linewidth=2, label='Left', alpha=0.8)
            
            # Mark start points
            ax.plot(mu[0, 0], mu[0, 1], 'bo', markersize=6)
            ax.plot(mu[0, 5], mu[0, 6], 'ro', markersize=6)
            
            ax.set_title(f'dx={dx:+.2f}m, θ={angle:+.1f}°', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if i == n_rows - 1:
                ax.set_xlabel('X (m)', fontsize=9)
            if j == 0:
                ax.set_ylabel('Y (m)', fontsize=9)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    
    fig.suptitle('TPGMM Adaptation: Combined Position + Orientation Variations', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(output_folder, 'adaptation_combined_grid.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: adaptation_combined_grid.png")
    plt.close()


def plot_metrics_vs_horizontal(metrics_list, x_shifts, output_folder):
    """Plot metrics vs horizontal shift"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    step_lengths_right = [m['right_step_length'] for m in metrics_list]
    step_heights_right = [m['right_step_height'] for m in metrics_list]
    peak_vels_right = [m['right_peak_velocity'] for m in metrics_list]
    stride_lengths = [m['stride_length'] for m in metrics_list]
    
    axes[0, 0].plot(x_shifts, step_lengths_right, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Horizontal Shift (m)', fontweight='bold')
    axes[0, 0].set_ylabel('Step Length (m)', fontweight='bold')
    axes[0, 0].set_title('Step Length vs Horizontal Shift', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x_shifts, step_heights_right, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Horizontal Shift (m)', fontweight='bold')
    axes[0, 1].set_ylabel('Step Height (m)', fontweight='bold')
    axes[0, 1].set_title('Step Height vs Horizontal Shift', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(x_shifts, peak_vels_right, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Horizontal Shift (m)', fontweight='bold')
    axes[1, 0].set_ylabel('Peak Velocity (m/s)', fontweight='bold')
    axes[1, 0].set_title('Peak Velocity vs Horizontal Shift', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x_shifts, stride_lengths, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Horizontal Shift (m)', fontweight='bold')
    axes[1, 1].set_ylabel('Stride Length (m)', fontweight='bold')
    axes[1, 1].set_title('Stride Length vs Horizontal Shift', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'metrics_vs_horizontal.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: metrics_vs_horizontal.png")
    plt.close()


def plot_metrics_vs_vertical(metrics_list, y_shifts, output_folder):
    """Plot metrics vs vertical shift"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    step_lengths_right = [m['right_step_length'] for m in metrics_list]
    step_heights_right = [m['right_step_height'] for m in metrics_list]
    peak_vels_right = [m['right_peak_velocity'] for m in metrics_list]
    stride_lengths = [m['stride_length'] for m in metrics_list]
    
    axes[0, 0].plot(y_shifts, step_lengths_right, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Vertical Shift (m)', fontweight='bold')
    axes[0, 0].set_ylabel('Step Length (m)', fontweight='bold')
    axes[0, 0].set_title('Step Length vs Vertical Shift', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(y_shifts, step_heights_right, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Vertical Shift (m)', fontweight='bold')
    axes[0, 1].set_ylabel('Step Height (m)', fontweight='bold')
    axes[0, 1].set_title('Step Height vs Vertical Shift', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(y_shifts, peak_vels_right, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Vertical Shift (m)', fontweight='bold')
    axes[1, 0].set_ylabel('Peak Velocity (m/s)', fontweight='bold')
    axes[1, 0].set_title('Peak Velocity vs Vertical Shift', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(y_shifts, stride_lengths, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Vertical Shift (m)', fontweight='bold')
    axes[1, 1].set_ylabel('Stride Length (m)', fontweight='bold')
    axes[1, 1].set_title('Stride Length vs Vertical Shift', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'metrics_vs_vertical.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: metrics_vs_vertical.png")
    plt.close()


def plot_metrics_vs_orientation(metrics_list, angles, output_folder):
    """Plot metrics vs orientation"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    step_lengths_right = [m['right_step_length'] for m in metrics_list]
    step_heights_right = [m['right_step_height'] for m in metrics_list]
    peak_vels_right = [m['right_peak_velocity'] for m in metrics_list]
    stride_lengths = [m['stride_length'] for m in metrics_list]
    
    axes[0, 0].plot(angles, step_lengths_right, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Orientation (degrees)', fontweight='bold')
    axes[0, 0].set_ylabel('Step Length (m)', fontweight='bold')
    axes[0, 0].set_title('Step Length vs Orientation', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(angles, step_heights_right, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Orientation (degrees)', fontweight='bold')
    axes[0, 1].set_ylabel('Step Height (m)', fontweight='bold')
    axes[0, 1].set_title('Step Height vs Orientation', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(angles, peak_vels_right, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Orientation (degrees)', fontweight='bold')
    axes[1, 0].set_ylabel('Peak Velocity (m/s)', fontweight='bold')
    axes[1, 0].set_title('Peak Velocity vs Orientation', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(angles, stride_lengths, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Orientation (degrees)', fontweight='bold')
    axes[1, 1].set_ylabel('Stride Length (m)', fontweight='bold')
    axes[1, 1].set_title('Stride Length vs Orientation', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'metrics_vs_orientation.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: metrics_vs_orientation.png")
    plt.close()


def main(model_folder=None, reg_factor=5e-4):
    """
    Main adaptation demonstration pipeline
    
    Parameters:
    -----------
    model_folder : str, optional
        Path to model folder
    reg_factor : float
        Regularization factor for auto-constructing path
    """
    print("\n" + "="*70)
    print("TPGMM ADAPTATION DEMONSTRATION")
    print("="*70)
    print("Testing trajectory adaptation to task parameter variations")
    print("Target: FR2 (Right Foot Frame)")
    print("  - Horizontal position: ±0.2m")
    print("  - Vertical position: ±0.1m")
    print("  - Orientation: ±15°")
    print("="*70)
    
    # Construct model path
    if model_folder is None:
        model_folder = f"train_tpgmm_model_reg{reg_factor:.0e}"
    
    model_path = os.path.join(model_folder, "trained_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Create output folder
    output_folder = os.path.join(model_folder, "adaptation_results")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder}")
    
    # Load model
    model_data = load_model(model_path)
    model = model_data['model']
    
    # Define baseline frame transformations (identity)
    baseline_frames = [
        (np.eye(11), np.zeros(11)),  # FR1
        (np.eye(11), np.zeros(11)),  # FR2
        (np.eye(11), np.zeros(11)),  # FR3
    ]
    
    # Run tests
    print("\n" + "="*70)
    print("RUNNING ADAPTATION TESTS")
    print("="*70)
    
    # Test 1: Horizontal variation
    h_traj, h_metrics, h_shifts = test_horizontal_variation(model, output_folder, baseline_frames)
    
    # Test 2: Vertical variation
    v_traj, v_metrics, v_shifts = test_vertical_variation(model, output_folder, baseline_frames)
    
    # Test 3: Orientation variation
    o_traj, o_metrics, o_angles = test_orientation_variation(model, output_folder, baseline_frames)
    
    # Test 4: Combined variation
    combined_traj = test_combined_variation(model, output_folder, baseline_frames)
    
    print("\n" + "="*70)
    print("ADAPTATION DEMONSTRATION COMPLETE!")
    print("="*70)
    print(f"✓ All results saved in: {output_folder}/")
    print("\n✓ Generated figures:")
    print(f"  1. adaptation_horizontal.png")
    print(f"  2. adaptation_vertical.png")
    print(f"  3. adaptation_orientation.png")
    print(f"  4. adaptation_combined_grid.png")
    print(f"  5. metrics_vs_horizontal.png")
    print(f"  6. metrics_vs_vertical.png")
    print(f"  7. metrics_vs_orientation.png")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TPGMM ADAPTATION DEMONSTRATION SCRIPT")
    print("="*70)
    
    # Configuration
    reg_factor = 5e-4
    model_folder = None  # Auto-construct from reg_factor
    
    # Run demonstration
    main(model_folder=model_folder, reg_factor=reg_factor)
    
    print("\n✅ Demonstration complete!")
