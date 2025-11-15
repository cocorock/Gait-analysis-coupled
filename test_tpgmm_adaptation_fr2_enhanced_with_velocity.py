"""
TPGMM ADAPTATION TESTING - FR2 PARAMETER VARIATION (Enhanced with Velocities)
==============================================================================
Tests TPGMM's adaptation capability by varying the FR2 (Right Foot) task parameter.

ENHANCEMENTS:
- Added color bars to show parameter variation ranges
- Overlay FR1 trajectory (from baseline) on all adapted trajectory plots
- FR1 trajectory is shifted to match adapted trajectories' mean position
- NEW: Added velocity trajectory plots for each test

This script:
1. Loads trained TPGMM model
2. Plots trajectories from each frame (FR1, FR2, FR3) - POSITIONS & VELOCITIES
3. Generates baseline GMR trajectory (no adaptation)
4. Varies FR2 x-position from -1.0 to 1.0 and overlays adapted trajectories
5. Varies FR2 y-position from -0.3 to 0.3 and overlays adapted trajectories
6. Varies FR2 orientation from -15 to +15 degrees and overlays adapted trajectories

Output:
- POSITION FIGURES (3 total):
  * Figure 1: X-position adaptation (two plots: Right Ankle, Left Ankle)
  * Figure 2: Y-position adaptation (two plots: Right Ankle, Left Ankle)
  * Figure 3: Orientation adaptation (two plots: Right Ankle, Left Ankle)
- VELOCITY FIGURES (3 total):
  * Figure 4: X-position adaptation velocities
  * Figure 5: Y-position adaptation velocities
  * Figure 6: Orientation adaptation velocities
- Each shows:
  * Individual frame trajectories (FR1, FR2, FR3) - thin lines
  * Baseline GMR trajectory (no adaptation) - thick black line
  * FR1 trajectory (from baseline) - thick red dashed line
  * Multiple adapted trajectories with varying FR2 parameters - colored gradient
  * Color bar showing parameter variation range

Author: Victor
Date: November 2024
Enhanced with Velocities: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import pickle
import os
import sys

# Configure matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 6
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
    """Optimized GMR using product of frames"""
    def __init__(self, tpgmm_model):
        self.model = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
    
    def predict(self, X_query, input_dims, output_dims, A_frames, b_frames):
        """GMR prediction using product of all frames"""
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
    
    def get_frame_trajectories(self, X_query, input_dims, output_dims, A_frames, b_frames):
        """
        Get individual trajectories from each frame's perspective
        
        Returns:
        --------
        frame_trajectories : list of arrays
            List of 3 trajectories, one from each frame [FR1, FR2, FR3]
        """
        n_query = X_query.shape[0]
        n_out = len(output_dims)
        
        frame_trajectories = []
        
        for p in range(self.P):
            A = A_frames[p]
            b = b_frames[p]
            
            # Transform Gaussians for this frame
            mu_frame = np.zeros((self.K, self.D))
            Sigma_frame = np.zeros((self.K, self.D, self.D))
            
            for k in range(self.K):
                mu_frame[k] = A @ self.model.means[p, k] + b
                Sigma_frame[k] = A @ self.model.covars[p, k] @ A.T
            
            # GMR for this frame only
            mu_out = np.zeros((n_query, n_out))
            
            for t in range(n_query):
                x_in = X_query[t]
                
                # Compute responsibilities
                h = np.zeros(self.K)
                for k in range(self.K):
                    mu_in = mu_frame[k][input_dims]
                    Sigma_in = Sigma_frame[k][np.ix_(input_dims, input_dims)]
                    Sigma_in_inv = np.linalg.inv(Sigma_in)
                    diff = x_in - mu_in
                    h[k] = self.model.priors[k] * np.exp(-0.5 * diff @ Sigma_in_inv @ diff)
                
                h = h / (np.sum(h) + 1e-10)
                
                # Conditional mean
                for k in range(self.K):
                    mu_k = mu_frame[k]
                    Sigma_k = Sigma_frame[k]
                    
                    Sigma_in = Sigma_k[np.ix_(input_dims, input_dims)]
                    Sigma_out_in = Sigma_k[np.ix_(output_dims, input_dims)]
                    Sigma_in_inv = np.linalg.inv(Sigma_in)
                    
                    mu_out_k = mu_k[output_dims] + Sigma_out_in @ Sigma_in_inv @ (x_in - mu_k[input_dims])
                    mu_out[t] += h[k] * mu_out_k
            
            frame_trajectories.append(mu_out)
        
        return frame_trajectories


def create_transformation_matrix_2d(x, y, theta):
    """
    Create 2D transformation matrix for 11 dimensions
    
    Parameters:
    -----------
    x : float
        Translation in x (meters)
    y : float
        Translation in y (meters)
    theta : float
        Rotation angle (degrees)
    
    Returns:
    --------
    A : ndarray (11, 11)
        Rotation matrix (block diagonal)
    b : ndarray (11,)
        Translation vector
    """
    # Convert to radians
    theta_rad = np.deg2rad(theta)
    
    # 2D rotation matrix
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    R2D = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # Block diagonal matrix (11x11)
    A = np.eye(11)
    
    # Apply rotation to position pairs
    A[0:2, 0:2] = R2D  # Right ankle (X, Y)
    A[2:4, 2:4] = R2D  # Right ankle (Xd, Yd)
    A[5:7, 5:7] = R2D  # Left ankle (X, Y)
    A[7:9, 7:9] = R2D  # Left ankle (Xd, Yd)
    # Dimension 4 (Right Yd) and 9 (Left Yd) are scalars - no rotation
    # Dimension 10 is time - no rotation
    
    # Translation vector (11,)
    b = np.zeros(11)
    b[0] = x  # Right ankle X
    b[1] = y  # Right ankle Y
    b[5] = x  # Left ankle X
    b[6] = y  # Left ankle Y
    
    return A, b


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
    print(f"  Frames = {model_data['metadata']['n_frames']}")
    print(f"  Dimensions = {model_data['metadata']['n_features']}")
    print(f"  Demonstrations = {model_data['metadata']['n_demonstrations']}")
    
    return model_data

def generate_trajectories_with_adaptation(model, fr2_x_values, fr2_y_values=None, fr2_theta_values=None):
    """
    Generate trajectories with varying FR2 x-position, y-position, and/or orientation
    
    Parameters:
    -----------
    model : OptimizedTPGMM
        Trained TPGMM model
    fr2_x_values : array
        Array of x-position offsets for FR2 frame
    fr2_y_values : array, optional
        Array of y-position offsets for FR2 frame
    fr2_theta_values : array, optional
        Array of theta offsets for FR2 frame (in degrees)
    
    Returns:
    --------
    baseline_frame_trajs : list
        Frame trajectories with no adaptation [FR1, FR2, FR3]
    baseline_gmr : array
        Baseline GMR trajectory (no adaptation)
    fr1_trajectory : array
        FR1 frame trajectory (from baseline)
    adapted_trajectories_x : list of arrays
        List of adapted GMR trajectories for each FR2 x-value
    adapted_trajectories_y : list of arrays or None
        List of adapted GMR trajectories for each FR2 y-value
    adapted_trajectories_theta : list of arrays or None
        List of adapted GMR trajectories for each FR2 theta-value
    """
    print("\n" + "="*70)
    print("GENERATING TRAJECTORIES WITH ADAPTATION")
    print("="*70)
    
    # Time query points
    n_query = 200
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    # Input and output dimensions
    input_dims = [10]  # Time
    output_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All other dimensions
    
    gmr = OptimizedGMR(model)
    
    # =========================================
    # 1. BASELINE (No Adaptation)
    # =========================================
    print("\n1. Generating baseline trajectories (no adaptation)...")
    
    A_FR1, b_FR1 = create_transformation_matrix_2d(0, 0, 0)
    A_FR2, b_FR2 = create_transformation_matrix_2d(0, 0, 0)
    A_FR3, b_FR3 = create_transformation_matrix_2d(0, 0, 0)
    
    A_frames_baseline = [A_FR1, A_FR2, A_FR3]
    b_frames_baseline = [b_FR1, b_FR2, b_FR3]
    
    # Get individual frame trajectories
    baseline_frame_trajs = gmr.get_frame_trajectories(
        time_query, input_dims, output_dims, A_frames_baseline, b_frames_baseline
    )
    
    # Get GMR trajectory (product of frames)
    baseline_gmr, _ = gmr.predict(
        time_query, input_dims, output_dims, A_frames_baseline, b_frames_baseline
    )
    
    # Extract FR1 trajectory
    fr1_trajectory = baseline_frame_trajs[0]
    
    print(f"✓ Baseline FR1 trajectory: {baseline_frame_trajs[0].shape}")
    print(f"✓ Baseline FR2 trajectory: {baseline_frame_trajs[1].shape}")
    print(f"✓ Baseline FR3 trajectory: {baseline_frame_trajs[2].shape}")
    print(f"✓ Baseline GMR trajectory: {baseline_gmr.shape}")
    print(f"✓ FR1 trajectory extracted for overlay")
    
    # =========================================
    # 2. ADAPTED TRAJECTORIES - X-POSITION
    # =========================================
    print(f"\n2. Generating {len(fr2_x_values)} adapted trajectories (X-position)...")
    print(f"   FR2 x-position range: [{fr2_x_values[0]:.3f}, {fr2_x_values[-1]:.3f}]")
    
    adapted_trajectories_x = []
    
    for i, fr2_x in enumerate(fr2_x_values):
        # Keep FR1 and FR3 unchanged
        A_FR1_adapt, b_FR1_adapt = create_transformation_matrix_2d(0, 0, 0)
        A_FR3_adapt, b_FR3_adapt = create_transformation_matrix_2d(0, 0, 0)
        
        # Vary FR2 x-position
        A_FR2_adapt, b_FR2_adapt = create_transformation_matrix_2d(fr2_x, 0, 0)
        
        A_frames_adapt = [A_FR1_adapt, A_FR2_adapt, A_FR3_adapt]
        b_frames_adapt = [b_FR1_adapt, b_FR2_adapt, b_FR3_adapt]
        
        # Generate adapted GMR trajectory
        adapted_gmr, _ = gmr.predict(
            time_query, input_dims, output_dims, A_frames_adapt, b_frames_adapt
        )
        
        adapted_trajectories_x.append(adapted_gmr)
        
        if (i + 1) % 3 == 0 or i == 0 or i == len(fr2_x_values) - 1:
            print(f"   [{i+1}/{len(fr2_x_values)}] FR2_x = {fr2_x:+.3f}m → Generated")
    
    print(f"✓ Generated {len(adapted_trajectories_x)} adapted trajectories (X-position)")
    
    # =========================================
    # 3. ADAPTED TRAJECTORIES - Y-POSITION (Optional)
    # =========================================
    adapted_trajectories_y = None
    
    if fr2_y_values is not None:
        print(f"\n3. Generating {len(fr2_y_values)} adapted trajectories (Y-position)...")
        print(f"   FR2 y-position range: [{fr2_y_values[0]:.3f}, {fr2_y_values[-1]:.3f}]")
        
        adapted_trajectories_y = []
        
        for i, fr2_y in enumerate(fr2_y_values):
            # Keep FR1 and FR3 unchanged
            A_FR1_adapt, b_FR1_adapt = create_transformation_matrix_2d(0, 0, 0)
            A_FR3_adapt, b_FR3_adapt = create_transformation_matrix_2d(0, 0, 0)
            
            # Vary FR2 y-position
            A_FR2_adapt, b_FR2_adapt = create_transformation_matrix_2d(0, fr2_y, 0)
            
            A_frames_adapt = [A_FR1_adapt, A_FR2_adapt, A_FR3_adapt]
            b_frames_adapt = [b_FR1_adapt, b_FR2_adapt, b_FR3_adapt]
            
            # Generate adapted GMR trajectory
            adapted_gmr, _ = gmr.predict(
                time_query, input_dims, output_dims, A_frames_adapt, b_frames_adapt
            )
            
            adapted_trajectories_y.append(adapted_gmr)
            
            if (i + 1) % 3 == 0 or i == 0 or i == len(fr2_y_values) - 1:
                print(f"   [{i+1}/{len(fr2_y_values)}] FR2_y = {fr2_y:+.3f}m → Generated")
        
        print(f"✓ Generated {len(adapted_trajectories_y)} adapted trajectories (Y-position)")
    
    # =========================================
    # 4. ADAPTED TRAJECTORIES - ORIENTATION (Optional)
    # =========================================
    adapted_trajectories_theta = None
    
    if fr2_theta_values is not None:
        print(f"\n4. Generating {len(fr2_theta_values)} adapted trajectories (Orientation)...")
        print(f"   FR2 theta range: [{fr2_theta_values[0]:.3f}, {fr2_theta_values[-1]:.3f}]")
        
        adapted_trajectories_theta = []
        
        for i, fr2_theta in enumerate(fr2_theta_values):
            # Keep FR1 and FR3 unchanged
            A_FR1_adapt, b_FR1_adapt = create_transformation_matrix_2d(0, 0, 0)
            A_FR3_adapt, b_FR3_adapt = create_transformation_matrix_2d(0, 0, 0)
            
            # Vary FR2 orientation
            A_FR2_adapt, b_FR2_adapt = create_transformation_matrix_2d(0, 0, fr2_theta)
            
            A_frames_adapt = [A_FR1_adapt, A_FR2_adapt, A_FR3_adapt]
            b_frames_adapt = [b_FR1_adapt, b_FR2_adapt, b_FR3_adapt]
            
            # Generate adapted GMR trajectory
            adapted_gmr, _ = gmr.predict(
                time_query, input_dims, output_dims, A_frames_adapt, b_frames_adapt
            )
            
            adapted_trajectories_theta.append(adapted_gmr)
            
            if (i + 1) % 3 == 0 or i == 0 or i == len(fr2_theta_values) - 1:
                print(f"   [{i+1}/{len(fr2_theta_values)}] FR2_θ = {fr2_theta:+.3f}° → Generated")
        
        print(f"✓ Generated {len(adapted_trajectories_theta)} adapted trajectories (Orientation)")
    
    return baseline_frame_trajs, baseline_gmr, fr1_trajectory, \
           adapted_trajectories_x, adapted_trajectories_y, adapted_trajectories_theta


def plot_adaptation_test_x(baseline_frame_trajs, baseline_gmr, fr1_trajectory,
                           adapted_trajectories, fr2_x_values, output_folder):
    """Plot adaptation test for FR2 X-position variation with color bar and FR1 overlay"""
    print("\n" + "="*70)
    print("PLOTTING X-POSITION ADAPTATION TEST")
    print("="*70)
    
    # Create color map
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=fr2_x_values.min(), vmax=fr2_x_values.max())
    adapt_colors = [cmap(norm(val)) for val in fr2_x_values]
    
    # Create figure with 3 subplots: 2 for data, 1 for colorbar
    fig = plt.figure(figsize=(16, 6))
    
    # GridSpec for layout: 2 main plots + 1 colorbar
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Calculate FR1 displacement for each ankle using FIRST POINT
    # Use middle adapted trajectory (no transformation) as reference
    middle_idx = len(adapted_trajectories) // 2
    
    # Right ankle (dims 0-1)
    fr1_first_right = fr1_trajectory[0, 0:2]
    adapted_first_right = adapted_trajectories[middle_idx][0, 0:2]
    fr1_displacement_right = 0#adapted_first_right - fr1_first_right
    
    # Left ankle (dims 5-6)
    fr1_first_left = fr1_trajectory[0, 5:7]
    adapted_first_left = adapted_trajectories[middle_idx][0, 5:7]
    fr1_displacement_left = 0#adapted_first_left - fr1_first_left
    
    # =========================================
    # RIGHT ANKLE (dimensions 0-1)
    # =========================================
    
    # 1. Plot baseline GMR trajectory (thick black line) - CLOSED
    x_closed = np.append(baseline_gmr[:, 0], baseline_gmr[0, 0])
    y_closed = np.append(baseline_gmr[:, 1], baseline_gmr[0, 1])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end of baseline
    axes[0].plot(baseline_gmr[0, 0], baseline_gmr[0, 1], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 0], baseline_gmr[-1, 1], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # 2. Plot FR1 trajectory (shifted to match adapted mean)
    # fr1_shifted_right = fr1_trajectory[:, 0:2] + fr1_displacement_right
    # axes[0].plot(fr1_shifted_right[:, 0], fr1_shifted_right[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    # 3. Plot adapted trajectories (colored gradient)
    for i, (adapted_traj, fr2_x, color) in enumerate(
        zip(adapted_trajectories, fr2_x_values, adapt_colors)
    ):
        # Only label first and last
        if i == 0:
            label = f'FR2_x = {fr2_x:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_x = {fr2_x:+.2f}m (max)'
        else:
            label = None
        
        # Calculate displacement from baseline's mean position and anti-transform
        baseline_mean = np.mean(baseline_gmr[:, 0:2], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 0:2], axis=0)
        displacement = [0 , 0]#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 0] - displacement[0]
        y_adapted = adapted_traj[:, 1] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[0].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure right ankle plot
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Position\nAdaptation via FR2 X-Position', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_ylim(-0.35, 0.1)
    
    # =========================================
    # LEFT ANKLE (dimensions 5-6)
    # =========================================
    
    # 1. Plot baseline GMR trajectory - CLOSED
    x_closed = np.append(baseline_gmr[:, 5], baseline_gmr[0, 5])
    y_closed = np.append(baseline_gmr[:, 6], baseline_gmr[0, 6])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end
    axes[1].plot(baseline_gmr[0, 5], baseline_gmr[0, 6], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 5], baseline_gmr[-1, 6], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # # 2. Plot FR1 trajectory (shifted to match adapted mean)
    # fr1_shifted_left = fr1_trajectory[:, 5:7] + fr1_displacement_left
    # axes[1].plot(fr1_shifted_left[:, 0], fr1_shifted_left[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    # 3. Plot adapted trajectories
    for i, (adapted_traj, fr2_x, color) in enumerate(
        zip(adapted_trajectories, fr2_x_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_x = {fr2_x:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_x = {fr2_x:+.2f}m (max)'
        else:
            label = None
        
        # Calculate displacement from baseline's mean position and anti-transform
        baseline_mean = np.mean(baseline_gmr[:, 5:7], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 5:7], axis=0)
        displacement = [0 , 0]#0#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 5] - displacement[0]
        y_adapted = adapted_traj[:, 6] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[1].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure left ankle plot
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Position\nAdaptation via FR2 X-Position', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_ylim(-0.35, 0.1)
    
    # =========================================
    # COLOR BAR
    # =========================================
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 X-Position (m)', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('TPGMM Adaptation Test: Varying FR2 (Right Foot) X-Position', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_x_position.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()

def plot_adaptation_test_y(baseline_frame_trajs, baseline_gmr, fr1_trajectory,
                           adapted_trajectories, fr2_y_values, output_folder):
    """Plot adaptation test for FR2 Y-position variation with color bar and FR1 overlay"""
    print("\n" + "="*70)
    print("PLOTTING Y-POSITION ADAPTATION TEST")
    print("="*70)
    
    # Create color map
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=fr2_y_values.min(), vmax=fr2_y_values.max())
    adapt_colors = [cmap(norm(val)) for val in fr2_y_values]
    
    # Create figure with 3 subplots: 2 for data, 1 for colorbar
    fig = plt.figure(figsize=(16, 6))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Calculate FR1 displacement for each ankle using FIRST POINT
    # Use middle adapted trajectory (no transformation) as reference
    middle_idx = len(adapted_trajectories) // 2
    
    fr1_first_right = fr1_trajectory[0, 0:2]
    adapted_first_right = adapted_trajectories[middle_idx][0, 0:2]
    fr1_displacement_right = 0#adapted_first_right - fr1_first_right
    
    fr1_first_left = fr1_trajectory[0, 5:7]
    adapted_first_left = adapted_trajectories[middle_idx][0, 5:7]
    fr1_displacement_left = 0#adapted_first_left - fr1_first_left
    
    # =========================================
    # RIGHT ANKLE (dimensions 0-1)
    # =========================================
    
    x_closed = np.append(baseline_gmr[:, 0], baseline_gmr[0, 0])
    y_closed = np.append(baseline_gmr[:, 1], baseline_gmr[0, 1])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    axes[0].plot(baseline_gmr[0, 0], baseline_gmr[0, 1], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 0], baseline_gmr[-1, 1], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # # Plot FR1 trajectory
    # fr1_shifted_right = fr1_trajectory[:, 0:2] + fr1_displacement_right
    # axes[0].plot(fr1_shifted_right[:, 0], fr1_shifted_right[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    for i, (adapted_traj, fr2_y, color) in enumerate(
        zip(adapted_trajectories, fr2_y_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_y = {fr2_y:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_y = {fr2_y:+.2f}m (max)'
        else:
            label = None
        
        baseline_mean = np.mean(baseline_gmr[:, 0:2], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 0:2], axis=0)
        displacement = [0 , 0]#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 0] - displacement[0]
        y_adapted = adapted_traj[:, 1] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[0].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Position\nAdaptation via FR2 Y-Position', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # =========================================
    # LEFT ANKLE (dimensions 5-6)
    # =========================================
    
    x_closed = np.append(baseline_gmr[:, 5], baseline_gmr[0, 5])
    y_closed = np.append(baseline_gmr[:, 6], baseline_gmr[0, 6])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    axes[1].plot(baseline_gmr[0, 5], baseline_gmr[0, 6], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 5], baseline_gmr[-1, 6], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # # Plot FR1 trajectory
    # fr1_shifted_left = fr1_trajectory[:, 5:7] + fr1_displacement_left
    # axes[1].plot(fr1_shifted_left[:, 0], fr1_shifted_left[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    for i, (adapted_traj, fr2_y, color) in enumerate(
        zip(adapted_trajectories, fr2_y_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_y = {fr2_y:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_y = {fr2_y:+.2f}m (max)'
        else:
            label = None
        
        baseline_mean = np.mean(baseline_gmr[:, 5:7], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 5:7], axis=0)
        displacement = [0 , 0]#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 5] - displacement[0]
        y_adapted = adapted_traj[:, 6] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[1].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Position\nAdaptation via FR2 Y-Position', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # =========================================
    # COLOR BAR
    # =========================================
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 Y-Position (m)', fontsize=12, fontweight='bold')
    
    fig.suptitle('TPGMM Adaptation Test: Varying FR2 (Right Foot) Y-Position', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_y_position.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()

def plot_adaptation_test_theta(baseline_frame_trajs, baseline_gmr, fr1_trajectory,
                               adapted_trajectories, fr2_theta_values, output_folder):
    """Plot adaptation test for FR2 Orientation variation with color bar and FR1 overlay"""
    print("\n" + "="*70)
    print("PLOTTING ORIENTATION ADAPTATION TEST")
    print("="*70)
    
    # Create color map
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=fr2_theta_values.min(), vmax=fr2_theta_values.max())
    adapt_colors = [cmap(norm(val)) for val in fr2_theta_values]
    
    # Create figure with 3 subplots: 2 for data, 1 for colorbar
    fig = plt.figure(figsize=(16, 6))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Calculate FR1 displacement for each ankle using FIRST POINT
    # Use middle adapted trajectory (no transformation) as reference
    middle_idx = len(adapted_trajectories) // 2
    
    fr1_first_right = fr1_trajectory[-1, 0:2]
    adapted_first_right = adapted_trajectories[middle_idx][-1, 0:2]
    fr1_displacement_right = 0#adapted_first_right - fr1_first_right
    
    fr1_first_left = fr1_trajectory[-1, 5:7]
    adapted_first_left = adapted_trajectories[middle_idx][-1, 5:7]
    fr1_displacement_left = 0#adapted_first_left - fr1_first_left
    
    # =========================================
    # RIGHT ANKLE (dimensions 0-1)
    # =========================================
    
    x_closed = np.append(baseline_gmr[:, 0], baseline_gmr[0, 0])
    y_closed = np.append(baseline_gmr[:, 1], baseline_gmr[0, 1])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    axes[0].plot(baseline_gmr[0, 0], baseline_gmr[0, 1], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 0], baseline_gmr[-1, 1], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # # Plot FR1 trajectory
    # fr1_shifted_right = fr1_trajectory[:, 0:2] + fr1_displacement_right
    # axes[0].plot(fr1_shifted_right[:, 0], fr1_shifted_right[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    for i, (adapted_traj, fr2_theta, color) in enumerate(
        zip(adapted_trajectories, fr2_theta_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_θ = {fr2_theta:+.2f}° (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_θ = {fr2_theta:+.2f}° (max)'
        else:
            label = None
        
        baseline_mean = np.mean(baseline_gmr[:, 0:2], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 0:2], axis=0)
        displacement = [0 , 0]#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 0] - displacement[0]
        y_adapted = adapted_traj[:, 1] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[0].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Position\nAdaptation via FR2 Orientation', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # =========================================
    # LEFT ANKLE (dimensions 5-6)
    # =========================================
    
    x_closed = np.append(baseline_gmr[:, 5], baseline_gmr[0, 5])
    y_closed = np.append(baseline_gmr[:, 6], baseline_gmr[0, 6])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    axes[1].plot(baseline_gmr[0, 5], baseline_gmr[0, 6], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 5], baseline_gmr[-1, 6], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # # Plot FR1 trajectory
    # fr1_shifted_left = fr1_trajectory[:, 5:7] + fr1_displacement_left
    # axes[1].plot(fr1_shifted_left[:, 0], fr1_shifted_left[:, 1],
    #             'r--', linewidth=3.0, alpha=0.8, label='FR1 trajectory',
    #             zorder=5)
    
    for i, (adapted_traj, fr2_theta, color) in enumerate(
        zip(adapted_trajectories, fr2_theta_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_θ = {fr2_theta:+.2f}° (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_θ = {fr2_theta:+.2f}° (max)'
        else:
            label = None
        
        baseline_mean = np.mean(baseline_gmr[:, 5:7], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 5:7], axis=0)
        displacement = [0 , 0]#adapted_mean - baseline_mean
        
        # Close the trajectory
        x_adapted = adapted_traj[:, 5] - displacement[0]
        y_adapted = adapted_traj[:, 6] - displacement[1]
        x_closed = np.append(x_adapted, x_adapted[0])
        y_closed = np.append(y_adapted, y_adapted[0])
        
        axes[1].plot(x_closed, y_closed,
                   color=color, linewidth=1.0, alpha=0.7,
                   label=label, zorder=4)
    
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Position\nAdaptation via FR2 Orientation', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # =========================================
    # COLOR BAR
    # =========================================
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 Orientation (degrees)', fontsize=12, fontweight='bold')
    
    fig.suptitle('TPGMM Adaptation Test: Varying FR2 (Right Foot) Orientation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_orientation.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()


def plot_adaptation_test_x_velocity(baseline_frame_trajs, baseline_gmr, fr1_trajectory, 
                                    adapted_trajectories, fr2_x_values, output_folder):
    """Plot velocity adaptation test results for FR2 x-position variation"""
    
    print(f"\n{'='*70}")
    print(f"PLOTTING VELOCITY ADAPTATION: FR2 X-POSITION")
    print(f"{'='*70}")
    
    # Create colormap
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=fr2_x_values[0], vmax=fr2_x_values[-1])
    adapt_colors = [cmap(norm(fr2_x)) for fr2_x in fr2_x_values]
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # RIGHT ANKLE VELOCITY (dimensions 2-3)
    x_closed = np.append(baseline_gmr[:, 2], baseline_gmr[0, 2])
    y_closed = np.append(baseline_gmr[:, 3], baseline_gmr[0, 3])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[0].plot(baseline_gmr[0, 2], baseline_gmr[0, 3], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 2], baseline_gmr[-1, 3], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_x, color) in enumerate(zip(adapted_trajectories, fr2_x_values, adapt_colors)):
        label = f'FR2_x = {fr2_x:+.3f}m (min)' if i == 0 else (f'FR2_x = {fr2_x:+.3f}m (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 2], adapted_traj[0, 2])
        y_closed = np.append(adapted_traj[:, 3], adapted_traj[0, 3])
        axes[0].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[0].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Velocity\nAdaptation via FR2 X-Position', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # LEFT ANKLE VELOCITY (dimensions 7-8)
    x_closed = np.append(baseline_gmr[:, 7], baseline_gmr[0, 7])
    y_closed = np.append(baseline_gmr[:, 8], baseline_gmr[0, 8])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[1].plot(baseline_gmr[0, 7], baseline_gmr[0, 8], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 7], baseline_gmr[-1, 8], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_x, color) in enumerate(zip(adapted_trajectories, fr2_x_values, adapt_colors)):
        label = f'FR2_x = {fr2_x:+.3f}m (min)' if i == 0 else (f'FR2_x = {fr2_x:+.3f}m (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 7], adapted_traj[0, 7])
        y_closed = np.append(adapted_traj[:, 8], adapted_traj[0, 8])
        axes[1].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[1].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Velocity\nAdaptation via FR2 X-Position', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # COLOR BAR
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 X-Position (m)', fontsize=12, fontweight='bold')
    
    fig.suptitle('TPGMM Velocity Adaptation: Varying FR2 X-Position', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_x_velocity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()


def plot_adaptation_test_y_velocity(baseline_frame_trajs, baseline_gmr, fr1_trajectory, 
                                    adapted_trajectories, fr2_y_values, output_folder):
    """Plot velocity adaptation test results for FR2 y-position variation"""
    
    print(f"\n{'='*70}")
    print(f"PLOTTING VELOCITY ADAPTATION: FR2 Y-POSITION")
    print(f"{'='*70}")
    
    # Create colormap
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=fr2_y_values[0], vmax=fr2_y_values[-1])
    adapt_colors = [cmap(norm(fr2_y)) for fr2_y in fr2_y_values]
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # RIGHT ANKLE VELOCITY (dimensions 2-3)
    x_closed = np.append(baseline_gmr[:, 2], baseline_gmr[0, 2])
    y_closed = np.append(baseline_gmr[:, 3], baseline_gmr[0, 3])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[0].plot(baseline_gmr[0, 2], baseline_gmr[0, 3], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 2], baseline_gmr[-1, 3], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_y, color) in enumerate(zip(adapted_trajectories, fr2_y_values, adapt_colors)):
        label = f'FR2_y = {fr2_y:+.3f}m (min)' if i == 0 else (f'FR2_y = {fr2_y:+.3f}m (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 2], adapted_traj[0, 2])
        y_closed = np.append(adapted_traj[:, 3], adapted_traj[0, 3])
        axes[0].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[0].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Velocity\nAdaptation via FR2 Y-Position', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # LEFT ANKLE VELOCITY (dimensions 7-8)
    x_closed = np.append(baseline_gmr[:, 7], baseline_gmr[0, 7])
    y_closed = np.append(baseline_gmr[:, 8], baseline_gmr[0, 8])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[1].plot(baseline_gmr[0, 7], baseline_gmr[0, 8], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 7], baseline_gmr[-1, 8], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_y, color) in enumerate(zip(adapted_trajectories, fr2_y_values, adapt_colors)):
        label = f'FR2_y = {fr2_y:+.3f}m (min)' if i == 0 else (f'FR2_y = {fr2_y:+.3f}m (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 7], adapted_traj[0, 7])
        y_closed = np.append(adapted_traj[:, 8], adapted_traj[0, 8])
        axes[1].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[1].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Velocity\nAdaptation via FR2 Y-Position', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # COLOR BAR
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 Y-Position (m)', fontsize=12, fontweight='bold')
    
    fig.suptitle('TPGMM Velocity Adaptation: Varying FR2 Y-Position', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_y_velocity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()


def plot_adaptation_test_theta_velocity(baseline_frame_trajs, baseline_gmr, fr1_trajectory, 
                                       adapted_trajectories, fr2_theta_values, output_folder):
    """Plot velocity adaptation test results for FR2 orientation variation"""
    
    print(f"\n{'='*70}")
    print(f"PLOTTING VELOCITY ADAPTATION: FR2 ORIENTATION")
    print(f"{'='*70}")
    
    # Create colormap
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=fr2_theta_values[0], vmax=fr2_theta_values[-1])
    adapt_colors = [cmap(norm(fr2_theta)) for fr2_theta in fr2_theta_values]
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # RIGHT ANKLE VELOCITY (dimensions 2-3)
    x_closed = np.append(baseline_gmr[:, 2], baseline_gmr[0, 2])
    y_closed = np.append(baseline_gmr[:, 3], baseline_gmr[0, 3])
    axes[0].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[0].plot(baseline_gmr[0, 2], baseline_gmr[0, 3], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[0].plot(baseline_gmr[-1, 2], baseline_gmr[-1, 3], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_theta, color) in enumerate(zip(adapted_trajectories, fr2_theta_values, adapt_colors)):
        label = f'FR2_θ = {fr2_theta:+.2f}° (min)' if i == 0 else (f'FR2_θ = {fr2_theta:+.2f}° (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 2], adapted_traj[0, 2])
        y_closed = np.append(adapted_traj[:, 3], adapted_traj[0, 3])
        axes[0].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[0].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Velocity\nAdaptation via FR2 Orientation', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # LEFT ANKLE VELOCITY (dimensions 7-8)
    x_closed = np.append(baseline_gmr[:, 7], baseline_gmr[0, 7])
    y_closed = np.append(baseline_gmr[:, 8], baseline_gmr[0, 8])
    axes[1].plot(x_closed, y_closed, 
                'k--', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    axes[1].plot(baseline_gmr[0, 7], baseline_gmr[0, 8], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5,
                label='Start/End points')
    axes[1].plot(baseline_gmr[-1, 7], baseline_gmr[-1, 8], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    for i, (adapted_traj, fr2_theta, color) in enumerate(zip(adapted_trajectories, fr2_theta_values, adapt_colors)):
        label = f'FR2_θ = {fr2_theta:+.2f}° (min)' if i == 0 else (f'FR2_θ = {fr2_theta:+.2f}° (max)' if i == len(adapted_trajectories) - 1 else None)
        # Close the trajectory
        x_closed = np.append(adapted_traj[:, 7], adapted_traj[0, 7])
        y_closed = np.append(adapted_traj[:, 8], adapted_traj[0, 8])
        axes[1].plot(x_closed, y_closed, color=color, linewidth=1.0, alpha=0.7, label=label, zorder=4)
    
    axes[1].set_xlabel('X Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Velocity (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Velocity\nAdaptation via FR2 Orientation', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # COLOR BAR
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('FR2 Orientation (degrees)', fontsize=12, fontweight='bold')
    
    fig.suptitle('TPGMM Velocity Adaptation: Varying FR2 Orientation', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_orientation_velocity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()

def main(model_folder=None, reg_factor=1e-4):
    """
    Main adaptation testing pipeline
    
    Parameters:
    -----------
    model_folder : str, optional
        Path to model folder
    reg_factor : float
        Regularization factor used in training
    """
    print("\n" + "="*70)
    print("TPGMM ADAPTATION TEST - FR2 PARAMETER VARIATION (ENHANCED)")
    print("="*70)
    print("Testing adaptation capability by varying FR2 (Right Foot) parameters")
    print("  X-Position: -1.0m to +1.0m")
    print("  Y-Position: -0.3m to +0.3m")
    print("  Orientation: -15deg to +15deg")
    print("\nENHANCEMENTS:")
    print("  ✓ Color bars showing parameter variation ranges")
    print("  ✓ FR1 trajectory overlay (red dashed line)")
    print("  ✓ FR1 shifted to match adapted trajectories' mean position")
    print("="*70)
    
    # Construct model path
    if model_folder is None:
        model_folder = f"train_tpgmm_model_reg{reg_factor:.0e}"
    
    model_path = os.path.join(model_folder, "trained_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        print(f"   Please train a model first using train_tpgmm_model.py")
        sys.exit(1)
    
    # Create output folder
    output_folder = os.path.join(model_folder, "adaptation_test")
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder: {output_folder}")
    
    # Load model
    model_data = load_model(model_path)
    model = model_data['model']
    
    # Define FR2 x-position values to test
    fr2_x_values = np.linspace(-0.4, 0.4, 7)
    
    # Define FR2 y-position values to test
    fr2_y_values = np.linspace(-0.3, 0.3,7)
    
    # Define FR2 orientation values to test
    fr2_theta_values = np.linspace(-25.0, 25.0, 7)
    
    print(f"\n✓ Testing {len(fr2_x_values)} FR2 x-positions:")
    print(f"   Range: [{fr2_x_values[0]:.3f}, {fr2_x_values[-1]:.3f}] meters")
    print(f"   Step: {fr2_x_values[1] - fr2_x_values[0]:.3f} meters")
    
    print(f"\n✓ Testing {len(fr2_y_values)} FR2 y-positions:")
    print(f"   Range: [{fr2_y_values[0]:.3f}, {fr2_y_values[-1]:.3f}] meters")
    print(f"   Step: {fr2_y_values[1] - fr2_y_values[0]:.3f} meters")
    
    print(f"\n✓ Testing {len(fr2_theta_values)} FR2 orientations:")
    print(f"   Range: [{fr2_theta_values[0]:.3f}, {fr2_theta_values[-1]:.3f}] degrees")
    print(f"   Step: {fr2_theta_values[1] - fr2_theta_values[0]:.3f} degrees")
    
    # Generate trajectories
    baseline_frame_trajs, baseline_gmr, fr1_trajectory, \
    adapted_trajectories_x, adapted_trajectories_y, adapted_trajectories_theta = \
        generate_trajectories_with_adaptation(model, fr2_x_values, fr2_y_values, fr2_theta_values)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Plot X-position adaptation
    plot_adaptation_test_x(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_x,
        fr2_x_values, output_folder
    )
    
    # Plot Y-position adaptation
    plot_adaptation_test_y(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_y,
        fr2_y_values, output_folder
    )
    
    # Plot Orientation adaptation
    plot_adaptation_test_theta(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_theta,
        fr2_theta_values, output_folder
    )
    
    # Plot VELOCITY adaptations
    plot_adaptation_test_x_velocity(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_x,
        fr2_x_values, output_folder
    )
    
    plot_adaptation_test_y_velocity(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_y,
        fr2_y_values, output_folder
    )
    
    plot_adaptation_test_theta_velocity(
        baseline_frame_trajs, baseline_gmr, fr1_trajectory, adapted_trajectories_theta,
        fr2_theta_values, output_folder
    )
    
    print("\n" + "="*70)
    print("ADAPTATION TEST COMPLETE!")
    print("="*70)
    print(f"✓ Results saved in: {output_folder}/")
    print(f"✓ POSITION Figures:")
    print(f"  1. tpgmm_adaptation_fr2_x_position.png")
    print(f"  2. tpgmm_adaptation_fr2_y_position.png")
    print(f"  3. tpgmm_adaptation_fr2_orientation.png")
    print(f"\n✓ VELOCITY Figures:")
    print(f"  4. tpgmm_adaptation_fr2_x_velocity.png")
    print(f"  5. tpgmm_adaptation_fr2_y_velocity.png")
    print(f"  6. tpgmm_adaptation_fr2_orientation_velocity.png")
    print("\nThe plots show:")
    print("  - Thick black line: Baseline GMR (no adaptation)")
    print("  - Thick red dashed line: FR1 trajectory (shifted to match adapted mean)")
    print("  - Colored gradient lines: Adapted trajectories with varying FR2 parameters")
    print("    * Coolwarm colormap (Position): Blue (min) → Red (max)")
    print("    * Viridis colormap (Orientation): Purple (min) → Yellow (max)")
    print("  - Color bar: Shows parameter variation range")
    print("  - Circle: Start point")
    print("  - Square: End point")
    print("\n✓ This demonstrates TPGMM's ability to adapt BOTH position and velocity")
    print("  trajectories based on task parameters (frame transformations)!")
    print("  The FR1 trajectory overlay shows the reference frame contribution.")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TPGMM ADAPTATION TESTING (ENHANCED)")
    print("="*70)
    
    # Configuration
    reg_factor = 3e-4  # Should match your training script
    model_folder = None  # Will use default: train_tpgmm_model_regXXX
    
    # Run adaptation test
    main(model_folder=model_folder, reg_factor=reg_factor)
    
    print("\n✅ Adaptation test complete!")
