"""
TPGMM ADAPTATION TESTING - FR2 POSITION VARIATION
==================================================
Tests TPGMM's adaptation capability by varying the FR2 (Right Foot) task parameter.

This script:
1. Loads trained TPGMM model
2. Plots trajectories from each frame (FR1, FR2, FR3) - POSITIONS ONLY
3. Generates baseline GMR trajectory (no adaptation)
4. Varies FR2 x-position from -1.0 to 1.0 and overlays adapted trajectories
5. Varies FR2 y-position from -0.5 to 0.5 and overlays adapted trajectories

Output:
- Figure 1: X-position adaptation (two plots: Right Ankle, Left Ankle)
- Figure 2: Y-position adaptation (two plots: Right Ankle, Left Ankle)
- Each shows:
  * Individual frame trajectories (FR1, FR2, FR3) - thin lines
  * Baseline GMR trajectory (no adaptation) - thick black line
  * Multiple adapted trajectories with varying FR2 position - colored gradient

Author: Victor
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
plt.rcParams['legend.fontsize'] = 9
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
    x, y : float
        Translation in x and y
    theta : float
        Rotation angle in radians
    
    Returns:
    --------
    A : (11, 11) array
        Transformation matrix (rotation + identity for non-spatial dims)
    b : (11,) array
        Translation vector
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rotation matrix for 2D positions and velocities
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    
    # Full transformation matrix (11x11)
    A = np.eye(11)
    A[0:2, 0:2] = R  # Right ankle position
    A[2:4, 2:4] = R  # Right ankle velocity
    # A[4, 4] = 1    # Right ankle angle (no change)
    A[5:7, 5:7] = R  # Left ankle position
    A[7:9, 7:9] = R  # Left ankle velocity
    # A[9, 9] = 1    # Left ankle angle (no change)
    # A[10, 10] = 1  # Time (no change)
    
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


def generate_trajectories_with_adaptation(model, fr2_x_values, fr2_y_values=None):
    """
    Generate trajectories with varying FR2 x-position and/or y-position
    
    Parameters:
    -----------
    model : OptimizedTPGMM
        Trained TPGMM model
    fr2_x_values : array
        Array of x-position offsets for FR2 frame
    fr2_y_values : array, optional
        Array of y-position offsets for FR2 frame
    
    Returns:
    --------
    baseline_frame_trajs : list
        Frame trajectories with no adaptation [FR1, FR2, FR3]
    baseline_gmr : array
        Baseline GMR trajectory (no adaptation)
    adapted_trajectories_x : list of arrays
        List of adapted GMR trajectories for each FR2 x-value
    adapted_trajectories_y : list of arrays or None
        List of adapted GMR trajectories for each FR2 y-value (if fr2_y_values provided)
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
    
    print(f"✓ Baseline FR1 trajectory: {baseline_frame_trajs[0].shape}")
    print(f"✓ Baseline FR2 trajectory: {baseline_frame_trajs[1].shape}")
    print(f"✓ Baseline FR3 trajectory: {baseline_frame_trajs[2].shape}")
    print(f"✓ Baseline GMR trajectory: {baseline_gmr.shape}")
    
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
    
    return baseline_frame_trajs, baseline_gmr, adapted_trajectories_x, adapted_trajectories_y


def plot_adaptation_test_x(baseline_frame_trajs, baseline_gmr, adapted_trajectories, 
                         fr2_x_values, output_folder):
    """
    Plot adaptation test results - X-POSITION VARIATION
    
    Parameters:
    -----------
    baseline_frame_trajs : list
        Frame trajectories [FR1, FR2, FR3]
    baseline_gmr : array
        Baseline GMR trajectory
    adapted_trajectories : list of arrays
        Adapted GMR trajectories
    fr2_x_values : array
        FR2 x-position values used
    output_folder : str
        Where to save figures
    """
    print("\n" + "="*70)
    print("CREATING ADAPTATION VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define colors for frames
    frame_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    frame_labels = ['FR1 (Hip)', 'FR2 (Right Foot)', 'FR3 (Left Foot)']
    frame_styles = ['-', '--', '-.']
    
    # Color map for adapted trajectories (from blue to red)
    n_adapt = len(adapted_trajectories)
    cmap = cm.get_cmap('coolwarm')
    adapt_colors = [cmap(i / (n_adapt - 1)) for i in range(n_adapt)]
    
    # =========================================
    # RIGHT ANKLE (dimensions 0-1)
    # =========================================
    
    # 1. Plot individual frame trajectories (thin, semi-transparent)
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(baseline_frame_trajs, frame_colors, frame_labels, frame_styles)
    ):
        axes[0].plot(frame_traj[:, 0], frame_traj[:, 1], 
                   color=frame_color, linestyle=frame_style, linewidth=1.5,
                   alpha=0.4, label=frame_label, zorder=3)
    
    # 2. Plot baseline GMR trajectory (thick black line)
    axes[0].plot(baseline_gmr[:, 0], baseline_gmr[:, 1], 
                'k-', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end of baseline
    axes[0].plot(baseline_gmr[0, 0], baseline_gmr[0, 1], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[0].plot(baseline_gmr[-1, 0], baseline_gmr[-1, 1], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
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
        displacement = adapted_mean - baseline_mean
        
        axes[0].plot(adapted_traj[:, 0] - displacement[0], adapted_traj[:, 1] - displacement[1],
                   color=color, linewidth=2.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure right ankle plot
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Position\nAdaptation via FR2 X-Position', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # =========================================
    # LEFT ANKLE (dimensions 5-6)
    # =========================================
    
    # 1. Plot individual frame trajectories
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(baseline_frame_trajs, frame_colors, frame_labels, frame_styles)
    ):
        axes[1].plot(frame_traj[:, 5], frame_traj[:, 6], 
                   color=frame_color, linestyle=frame_style, linewidth=1.5,
                   alpha=0.4, label=frame_label, zorder=3)
    
    # 2. Plot baseline GMR trajectory
    axes[1].plot(baseline_gmr[:, 5], baseline_gmr[:, 6], 
                'k-', linewidth=2.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end
    axes[1].plot(baseline_gmr[0, 5], baseline_gmr[0, 6], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[1].plot(baseline_gmr[-1, 5], baseline_gmr[-1, 6], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
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
        displacement = adapted_mean - baseline_mean
        
        axes[1].plot(adapted_traj[:, 5] - displacement[0], adapted_traj[:, 6] - displacement[1],
                   color=color, linewidth=2.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure left ankle plot
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Position\nAdaptation via FR2 X-Position', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # Main title
    fig.suptitle('TPGMM Adaptation Test: Varying FR2 (Right Foot) X-Position', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_x_position.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()


def plot_adaptation_test_y(baseline_frame_trajs, baseline_gmr, adapted_trajectories, 
                         fr2_y_values, output_folder):
    """
    Plot adaptation test results - Y-POSITION VARIATION
    
    Parameters:
    -----------
    baseline_frame_trajs : list
        Frame trajectories [FR1, FR2, FR3]
    baseline_gmr : array
        Baseline GMR trajectory
    adapted_trajectories : list of arrays
        Adapted GMR trajectories
    fr2_y_values : array
        FR2 y-position values used
    output_folder : str
        Where to save figures
    """
    print("\n" + "="*70)
    print("CREATING Y-POSITION ADAPTATION VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define colors for frames
    frame_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    frame_labels = ['FR1 (Hip)', 'FR2 (Right Foot)', 'FR3 (Left Foot)']
    frame_styles = ['-', '--', '-.']
    
    # Color map for adapted trajectories (from blue to red)
    n_adapt = len(adapted_trajectories)
    cmap = cm.get_cmap('coolwarm')
    adapt_colors = [cmap(i / (n_adapt - 1)) for i in range(n_adapt)]
    
    # =========================================
    # RIGHT ANKLE (dimensions 0-1)
    # =========================================
    
    # 1. Plot individual frame trajectories (thin, semi-transparent)
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(baseline_frame_trajs, frame_colors, frame_labels, frame_styles)
    ):
        axes[0].plot(frame_traj[:, 0], frame_traj[:, 1], 
                   color=frame_color, linestyle=frame_style, linewidth=1.5,
                   alpha=0.4, label=frame_label, zorder=3)
    
    # 2. Plot baseline GMR trajectory (thick black line)
    axes[0].plot(baseline_gmr[:, 0], baseline_gmr[:, 1], 
                'k-', linewidth=3.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end of baseline
    axes[0].plot(baseline_gmr[0, 0], baseline_gmr[0, 1], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[0].plot(baseline_gmr[-1, 0], baseline_gmr[-1, 1], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # 3. Plot adapted trajectories (colored gradient)
    for i, (adapted_traj, fr2_y, color) in enumerate(
        zip(adapted_trajectories, fr2_y_values, adapt_colors)
    ):
        # Only label first and last
        if i == 0:
            label = f'FR2_y = {fr2_y:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_y = {fr2_y:+.2f}m (max)'
        else:
            label = None
        
        # Calculate displacement from baseline's mean position and anti-transform
        baseline_mean = np.mean(baseline_gmr[:, 0:2], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 0:2], axis=0)
        displacement = adapted_mean - baseline_mean
        
        axes[0].plot(adapted_traj[:, 0] - displacement[0], adapted_traj[:, 1] - displacement[1],
                   color=color, linewidth=2.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure right ankle plot
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Position\nAdaptation via FR2 Y-Position', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9, framealpha=0.9)
    axes[0].set_aspect('equal', adjustable='box')
    
    # =========================================
    # LEFT ANKLE (dimensions 5-6)
    # =========================================
    
    # 1. Plot individual frame trajectories
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(baseline_frame_trajs, frame_colors, frame_labels, frame_styles)
    ):
        axes[1].plot(frame_traj[:, 5], frame_traj[:, 6], 
                   color=frame_color, linestyle=frame_style, linewidth=1.5,
                   alpha=0.4, label=frame_label, zorder=3)
    
    # 2. Plot baseline GMR trajectory
    axes[1].plot(baseline_gmr[:, 5], baseline_gmr[:, 6], 
                'k-', linewidth=3.0, label='Baseline GMR (no adaptation)',
                zorder=5, alpha=0.8)
    
    # Mark start and end
    axes[1].plot(baseline_gmr[0, 5], baseline_gmr[0, 6], 'ko', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[1].plot(baseline_gmr[-1, 5], baseline_gmr[-1, 6], 'ks', 
                markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    
    # 3. Plot adapted trajectories
    for i, (adapted_traj, fr2_y, color) in enumerate(
        zip(adapted_trajectories, fr2_y_values, adapt_colors)
    ):
        if i == 0:
            label = f'FR2_y = {fr2_y:+.2f}m (min)'
        elif i == len(adapted_trajectories) - 1:
            label = f'FR2_y = {fr2_y:+.2f}m (max)'
        else:
            label = None
        
        # Calculate displacement from baseline's mean position and anti-transform
        baseline_mean = np.mean(baseline_gmr[:, 5:7], axis=0)
        adapted_mean = np.mean(adapted_traj[:, 5:7], axis=0)
        displacement = adapted_mean - baseline_mean
        
        axes[1].plot(adapted_traj[:, 5] - displacement[0], adapted_traj[:, 6] - displacement[1],
                   color=color, linewidth=2.0, alpha=0.7,
                   label=label, zorder=4)
    
    # Configure left ankle plot
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Position\nAdaptation via FR2 Y-Position', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9, framealpha=0.9)
    axes[1].set_aspect('equal', adjustable='box')
    
    # Main title
    fig.suptitle('TPGMM Adaptation Test: Varying FR2 (Right Foot) Y-Position', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_adaptation_fr2_y_position.png')
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
    print("TPGMM ADAPTATION TEST - FR2 POSITION VARIATION")
    print("="*70)
    print("Testing adaptation capability by varying FR2 (Right Foot) position")
    print("  X-Position: -1.0m to +1.0m")
    print("  Y-Position: -0.5m to +0.5m")
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
    # From -1.0 to +1.0 with 11 points
    fr2_x_values = np.linspace(-1.0, 1.0, 11)
    
    # Define FR2 y-position values to test
    # From -0.5 to +0.5 with 11 points
    fr2_y_values = np.linspace(-0.3, 0.3, 11)
    
    print(f"\n✓ Testing {len(fr2_x_values)} FR2 x-positions:")
    print(f"   Range: [{fr2_x_values[0]:.3f}, {fr2_x_values[-1]:.3f}] meters")
    print(f"   Step: {fr2_x_values[1] - fr2_x_values[0]:.3f} meters")
    
    print(f"\n✓ Testing {len(fr2_y_values)} FR2 y-positions:")
    print(f"   Range: [{fr2_y_values[0]:.3f}, {fr2_y_values[-1]:.3f}] meters")
    print(f"   Step: {fr2_y_values[1] - fr2_y_values[0]:.3f} meters")
    
    # Generate trajectories
    baseline_frame_trajs, baseline_gmr, adapted_trajectories_x, adapted_trajectories_y = \
        generate_trajectories_with_adaptation(model, fr2_x_values, fr2_y_values)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Plot X-position adaptation
    plot_adaptation_test_x(
        baseline_frame_trajs, baseline_gmr, adapted_trajectories_x,
        fr2_x_values, output_folder
    )
    
    # Plot Y-position adaptation
    plot_adaptation_test_y(
        baseline_frame_trajs, baseline_gmr, adapted_trajectories_y,
        fr2_y_values, output_folder
    )
    
    print("\n" + "="*70)
    print("ADAPTATION TEST COMPLETE!")
    print("="*70)
    print(f"✓ Results saved in: {output_folder}/")
    print(f"✓ Figures:")
    print(f"  1. tpgmm_adaptation_fr2_x_position.png")
    print(f"  2. tpgmm_adaptation_fr2_y_position.png")
    print("\nThe plots show:")
    print("  - Thin colored lines: Individual frame trajectories (FR1, FR2, FR3)")
    print("  - Thick black line: Baseline GMR (no adaptation)")
    print("  - Colored gradient lines: Adapted trajectories with varying FR2 position")
    print("    * Blue → Red:")
    print("      - X-position: -1.0m to +1.0m")
    print("      - Y-position: -0.5m to +0.5m")
    print("  - Circle: Start point")
    print("  - Square: End point")
    print("\n✓ This demonstrates TPGMM's ability to adapt trajectories based on")
    print("  task parameters (frame transformations)!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TPGMM ADAPTATION TESTING")
    print("="*70)
    
    # Configuration
    reg_factor = 2e-4  # Should match your training script
    model_folder = None  # Will use default: train_tpgmm_model_reg1e-04
    
    # Run adaptation test
    main(model_folder=model_folder, reg_factor=reg_factor)
    
    print("\n✅ Adaptation test complete!")
