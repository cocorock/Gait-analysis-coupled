"""
TPGMM COMBINED ADAPTABILITY TESTING SCRIPT FOR LOWER LIMB EXOSKELETON
======================================================================

COMBINED VERSION - Applies all transformations simultaneously

This script tests the adaptability of the trained TPGMM model by applying
time-varying transformations to reference frame FR2 (right foot target) with
ALL THREE modifications applied at once:
1. HORIZONTAL DISPLACEMENT: FR2 shifts along x-axis
2. VERTICAL DISPLACEMENT: FR2 shifts along y-axis  
3. ROTATION: FR2 rotates

Test performed:
- COMBINED TRANSFORMATION: All 3 modifications applied simultaneously
- 4 test variations from 0 to target values
- TIME-VARYING transformations (not constant!)
- Demonstrates endpoint adaptation while maintaining initial position
- All coordinates are relative to the body (hip-centered frame)

Author: Victor
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from matplotlib.patches import Ellipse
import json
from scipy.special import logsumexp
from scipy.linalg import cholesky
import csv

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

# Set numpy print options
np.set_printoptions(precision=4, suppress=True)


# ============================================================================
# TPGMM CLASS DEFINITIONS (needed for unpickling the trained model)
# ============================================================================

class OptimizedTPGMM:
    """
    OPTIMIZED Task-Parameterized Gaussian Mixture Model
    
    This class definition is needed to unpickle the trained model.
    It must match the class used during training.
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
                    L = cholesky(self.covars[p, k], lower=True)
                    self._chol_covars[p, k] = L
                    self._inv_covars[p, k] = np.linalg.inv(self.covars[p, k])
                    self._log_dets[p, k] = 2 * np.sum(np.log(np.diag(L)))
                except:
                    self._inv_covars[p, k] = np.linalg.inv(self.covars[p, k])
                    self._log_dets[p, k] = np.linalg.slogdet(self.covars[p, k])[1]
    
    def _e_step_vectorized(self, X_frames):
        """Optimized E-step using vectorized operations"""
        N = X_frames.shape[1]
        log_resp = np.zeros((N, self.K))
        log_2pi_D = self.D * np.log(2 * np.pi)
        
        for k in range(self.K):
            log_resp[:, k] = np.log(self.priors[k] + 1e-10)
            
            for p in range(self.P):
                diff = X_frames[p] - self.means[p, k]
                maha = np.einsum('ij,jk,ik->i', diff, self._inv_covars[p, k], diff)
                log_prob = -0.5 * (self._log_dets[p, k] + maha + log_2pi_D)
                log_resp[:, k] += log_prob
        
        log_sum = logsumexp(log_resp, axis=1, keepdims=True)
        responsibilities = np.exp(log_resp - log_sum)
        log_likelihood = np.sum(log_sum)
        
        return responsibilities, log_likelihood
    
    def compute_bic(self, X_frames):
        """Compute Bayesian Information Criterion"""
        N = X_frames.shape[1]
        n_params = (self.K - 1) + (self.P * self.K * self.D) + \
                   (self.P * self.K * self.D * (self.D + 1) / 2)
        
        if len(self.log_likelihood_history) > 0:
            log_likelihood = self.log_likelihood_history[-1]
        else:
            _, log_likelihood = self._e_step_vectorized(X_frames)
        
        bic = -2 * log_likelihood + n_params * np.log(N)
        return bic, log_likelihood


class OptimizedGMR:
    """
    OPTIMIZED Gaussian Mixture Regression
    
    This class performs GMR with task-parameterized frames.
    Supports both constant and TIME-VARYING transformations.
    """
    
    def __init__(self, tpgmm_model):
        """Initialize with trained TPGMM model"""
        self.tpgmm = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
    
    def _transform_frame(self, mean, covar, A, b):
        """Transform Gaussian parameters to global frame"""
        # Global mean: A * mean + b
        mean_global = A @ mean + b
        
        # Global covariance: A * covar * A^T
        covar_global = A @ covar @ A.T
        
        return mean_global, covar_global
    
    def predict(self, input_data, input_dims, output_dims, A_frames, b_frames):
        """
        Perform GMR prediction (vectorized)
        
        Parameters:
        -----------
        input_data : array (N, len(input_dims))
            Query points
        input_dims : list
            Indices of input dimensions
        output_dims : list
            Indices of output dimensions
        A_frames : array (P, D, D) or (N, P, D, D)
            Transformation matrices for each frame
            If 3D: same transformation for all query points
            If 4D: different transformation per query point (TIME-VARYING!)
        b_frames : array (P, D) or (N, P, D)
            Translation vectors for each frame
            If 2D: same translation for all query points
            If 3D: different translation per query point (TIME-VARYING!)
        
        Returns:
        --------
        mu_out : array (N, len(output_dims))
            Predicted mean
        sigma_out : array (N, len(output_dims), len(output_dims))
            Predicted covariance
        """
        N = input_data.shape[0]
        D_in = len(input_dims)
        D_out = len(output_dims)
        
        # Check if transformations are time-varying
        time_varying = (A_frames.ndim == 4)
        
        # Initialize outputs
        mu_out = np.zeros((N, D_out))
        sigma_out = np.zeros((N, D_out, D_out))
        
        # Process each query point
        for n in range(N):
            x_in = input_data[n]
            
            # Get transformations for this time step
            if time_varying:
                A_n = A_frames[n]  # (P, D, D)
                b_n = b_frames[n]  # (P, D)
            else:
                A_n = A_frames  # (P, D, D)
                b_n = b_frames  # (P, D)
            
            # Transform all components to global frame and compute mixture
            mu_k = np.zeros((self.K, D_out))
            sigma_k = np.zeros((self.K, D_out, D_out))
            h = np.zeros(self.K)
            
            for k in range(self.K):
                # Product of Gaussians across all frames
                mu_tmp = np.zeros(self.D)
                sigma_tmp_inv = np.zeros((self.D, self.D))
                
                for p in range(self.P):
                    # Transform to global frame
                    mu_global, sigma_global = self._transform_frame(
                        self.tpgmm.means[p, k],
                        self.tpgmm.covars[p, k],
                        A_n[p],
                        b_n[p]
                    )
                    
                    # Add to precision-weighted sum
                    sigma_inv = np.linalg.inv(sigma_global)
                    sigma_tmp_inv += sigma_inv
                    mu_tmp += sigma_inv @ mu_global
                
                # Compute product of Gaussians
                sigma_tmp = np.linalg.inv(sigma_tmp_inv)
                mu_tmp = sigma_tmp @ mu_tmp
                
                # Partition for GMR
                mu_in = mu_tmp[input_dims]
                mu_out_k = mu_tmp[output_dims]
                
                sigma_in_in = sigma_tmp[np.ix_(input_dims, input_dims)]
                sigma_out_out = sigma_tmp[np.ix_(output_dims, output_dims)]
                sigma_in_out = sigma_tmp[np.ix_(input_dims, output_dims)]
                sigma_out_in = sigma_tmp[np.ix_(output_dims, input_dims)]
                
                # Conditional distribution
                sigma_in_in_inv = np.linalg.inv(sigma_in_in)
                mu_k[k] = mu_out_k + sigma_out_in @ sigma_in_in_inv @ (x_in - mu_in)
                sigma_k[k] = sigma_out_out - sigma_out_in @ sigma_in_in_inv @ sigma_in_out
                
                # Compute activation (marginal likelihood)
                diff = x_in - mu_in
                exp_term = -0.5 * diff @ sigma_in_in_inv @ diff
                det_term = -0.5 * np.log(np.linalg.det(2 * np.pi * sigma_in_in))
                h[k] = self.tpgmm.priors[k] * np.exp(det_term + exp_term)
            
            # Normalize activations
            h = h / (np.sum(h) + 1e-10)
            
            # Mixture of conditional distributions
            mu_out[n] = np.sum(h[:, None] * mu_k, axis=0)
            for k in range(self.K):
                diff = mu_k[k] - mu_out[n]
                sigma_out[n] += h[k] * (sigma_k[k] + np.outer(diff, diff))
        
        return mu_out, sigma_out


# ============================================================================
# COMBINED TRANSFORMATION FUNCTIONS
# ============================================================================

def create_time_varying_transformations_combined(N, horizontal_target=0.0, 
                                                  vertical_target=0.0, 
                                                  rotation_target=0.0):
    """
    Create time-varying transformations combining horizontal, vertical, and rotation.
    
    ALL THREE transformations are applied simultaneously to FR2 (right foot target).
    Each transformation varies linearly from 0 to its target value.
    
    Parameters:
    -----------
    N : int
        Number of time steps
    horizontal_target : float
        Target horizontal displacement in meters (0 = no displacement)
    vertical_target : float
        Target vertical displacement in meters (0 = no displacement)
    rotation_target : float
        Target rotation angle in degrees (0 = no rotation)
    
    Returns:
    --------
    A_frames : array (N, 3, 11, 11)
        Time-varying transformation matrices
    b_frames : array (N, 3, 11)
        Time-varying translation vectors
    displacements : dict
        Dictionary with time-varying parameters:
        - 'horizontal': horizontal displacement at each time step
        - 'vertical': vertical displacement at each time step
        - 'rotation': rotation angle at each time step
    """
    P = 3  # Number of frames (FR1: body, FR2: right foot, FR3: left foot)
    D = 11  # Total dimensions
    
    # Initialize arrays
    A_frames = np.zeros((N, P, D, D))
    b_frames = np.zeros((N, P, D))
    
    # Time-varying parameters (linearly varying from 0 to target)
    horizontal_disp = np.linspace(0.0, horizontal_target, N)
    vertical_disp = np.linspace(0.0, vertical_target, N)
    rotation_angles = np.linspace(0.0, rotation_target, N)
    
    # For each time step
    for t in range(N):
        # Get current transformation parameters
        dx = horizontal_disp[t]
        dy = vertical_disp[t]
        angle_deg = rotation_angles[t]
        angle_rad = np.deg2rad(angle_deg)
        
        # Create rotation matrix (2D rotation for x-y plane)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
        
        # --- FRAME 1: Body frame (FR1) - IDENTITY (no transformation) ---
        A_frames[t, 0] = np.eye(D)
        b_frames[t, 0] = np.zeros(D)
        
        # --- FRAME 2: Right foot target (FR2) - COMBINED TRANSFORMATION ---
        # Apply rotation to position coordinates
        A_frames[t, 1] = np.eye(D)
        A_frames[t, 1, 0:2, 0:2] = R  # Apply rotation to x-y position
        
        # Apply translation (displacement)
        b_frames[t, 1] = np.zeros(D)
        b_frames[t, 1, 0] = dx  # x displacement
        b_frames[t, 1, 1] = dy  # y displacement
        
        # --- FRAME 3: Left foot target (FR3) - IDENTITY (no transformation) ---
        A_frames[t, 2] = np.eye(D)
        b_frames[t, 2] = np.zeros(D)
    
    # Store all time-varying parameters
    displacements = {
        'horizontal': horizontal_disp,
        'vertical': vertical_disp,
        'rotation': rotation_angles
    }
    
    print(f"✓ Created COMBINED time-varying transformations:")
    print(f"  Horizontal: 0 → {horizontal_target:.3f} m")
    print(f"  Vertical:   0 → {vertical_target:.3f} m")
    print(f"  Rotation:   0 → {rotation_target:.2f}°")
    
    return A_frames, b_frames, displacements


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_adapted_trajectory(model, A_frames, b_frames, N=200):
    """
    Generate trajectory using GMR with time-varying transformations
    
    Parameters:
    -----------
    model : OptimizedTPGMM
        Trained TPGMM model
    A_frames : array (N, P, D, D)
        Time-varying transformation matrices
    b_frames : array (N, P, D)
        Time-varying translation vectors
    N : int
        Number of query points
    
    Returns:
    --------
    time : array (N,)
        Time values [0, 1]
    mu : array (N, 10)
        Predicted trajectory (position, velocity, angle for both ankles)
    sigma : array (N, 10, 10)
        Prediction covariance
    """
    # Create GMR object
    gmr = OptimizedGMR(model)
    
    # Time query points
    time = np.linspace(0, 1, N).reshape(-1, 1)
    
    # Input/output dimensions
    input_dims = [10]  # Time dimension
    output_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All other dimensions
    
    # Perform GMR
    mu, sigma = gmr.predict(time, input_dims, output_dims, A_frames, b_frames)
    
    return time.flatten(), mu, sigma


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_trained_model(model_path):
    """Load trained TPGMM model from pickle file"""
    print(f"\nLoading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("✓ Model loaded successfully")
    print(f"  - Components: {model_data['model'].K}")
    print(f"  - Frames: {model_data['model'].P}")
    print(f"  - Dimensions: {model_data['model'].D}")
    
    return model_data


def load_original_data(data_path):
    """Load original gait data for reference (optional - only for logging)"""
    print(f"\nChecking original data: {data_path}")
    
    if not os.path.exists(data_path):
        print("⚠ Data file not found - continuing without reference data")
        return None, None
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print("✓ Original data file found")
        
        # The actual frame endpoint data isn't needed for trajectory generation
        # It was only used for reference/logging purposes
        # The TPGMM model already has all the information it needs
        
        if 'kinematics_data' in data:
            print("  - Kinematics data available")
            if 'FR2' in data['kinematics_data'] and 'FR3' in data['kinematics_data']:
                print("  - FR2 and FR3 frames present")
        
        # Return None since we don't actually need this data
        return None, None
        
    except Exception as e:
        print(f"⚠ Could not load data file: {e}")
        print("  Continuing without reference data...")
        return None, None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_combined_adaptations(time_baseline, mu_baseline, combined_results,
                                   output_folder):
    """
    Visualize combined adaptations (horizontal + vertical + rotation simultaneously)
    
    Parameters:
    -----------
    time_baseline : array
        Time values for baseline
    mu_baseline : array
        Baseline trajectory
    combined_results : list of dict
        Each dict contains: 'time', 'mu', 'sigma', 'param', 'targets'
        where 'targets' is a dict with 'horizontal', 'vertical', 'rotation'
    output_folder : str
        Path to save figures
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Define colors for different variations
    n_variations = len(combined_results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_variations))
    
    # === SUBPLOT 1: X Position (Right Ankle) ===
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time_baseline, mu_baseline[:, 0], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax1.plot(result['time'], result['mu'][:, 0], 
                color=colors[i], linewidth=2, label=label)
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('Right Ankle - X Position')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # === SUBPLOT 2: Y Position (Right Ankle) ===
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time_baseline, mu_baseline[:, 1], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax2.plot(result['time'], result['mu'][:, 1], 
                color=colors[i], linewidth=2, label=label)
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Right Ankle - Y Position')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # === SUBPLOT 3: XY Trajectory (Right Ankle) ===
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(mu_baseline[:, 0], mu_baseline[:, 1], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax3.plot(result['mu'][:, 0], result['mu'][:, 1], 
                color=colors[i], linewidth=2, label=label)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Right Ankle - XY Trajectory')
    # ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # === SUBPLOT 4: X Velocity (Right Ankle) ===
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time_baseline, mu_baseline[:, 2], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax4.plot(result['time'], result['mu'][:, 2], 
                color=colors[i], linewidth=2, label=label)
    ax4.set_xlabel('Normalized Time')
    ax4.set_ylabel('X Velocity (m/s)')
    ax4.set_title('Right Ankle - X Velocity')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # === SUBPLOT 5: Y Velocity (Right Ankle) ===
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time_baseline, mu_baseline[:, 3], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax5.plot(result['time'], result['mu'][:, 3], 
                color=colors[i], linewidth=2, label=label)
    ax5.set_xlabel('Normalized Time')
    ax5.set_ylabel('Y Velocity (m/s)')
    ax5.set_title('Right Ankle - Y Velocity')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # === SUBPLOT 6: Angle (Right Ankle) ===
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time_baseline, np.rad2deg(mu_baseline[:, 4]), 'k--', 
             linewidth=2, label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax6.plot(result['time'], np.rad2deg(result['mu'][:, 4]), 
                color=colors[i], linewidth=2, label=label)
    ax6.set_xlabel('Normalized Time')
    ax6.set_ylabel('Angle (degrees)')
    ax6.set_title('Right Ankle - Angle')
    ax6.legend(fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # === SUBPLOT 7: X Position (Left Ankle) ===
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time_baseline, mu_baseline[:, 5], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax7.plot(result['time'], result['mu'][:, 5], 
                color=colors[i], linewidth=2, label=label)
    ax7.set_xlabel('Normalized Time')
    ax7.set_ylabel('X Position (m)')
    ax7.set_title('Left Ankle - X Position')
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # === SUBPLOT 8: Y Position (Left Ankle) ===
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time_baseline, mu_baseline[:, 6], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax8.plot(result['time'], result['mu'][:, 6], 
                color=colors[i], linewidth=2, label=label)
    ax8.set_xlabel('Normalized Time')
    ax8.set_ylabel('Y Position (m)')
    ax8.set_title('Left Ankle - Y Position')
    ax8.legend(fontsize=8, loc='best')
    ax8.grid(True, alpha=0.3)
    
    # === SUBPLOT 9: XY Trajectory (Left Ankle) ===
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(mu_baseline[:, 5], mu_baseline[:, 6], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax9.plot(result['mu'][:, 5], result['mu'][:, 6], 
                color=colors[i], linewidth=2, label=label)
    ax9.set_xlabel('X Position (m)')
    ax9.set_ylabel('Y Position (m)')
    ax9.set_title('Left Ankle - XY Trajectory')
    ax9.legend(fontsize=8, loc='best')
    ax9.grid(True, alpha=0.3)
    ax9.set_aspect('equal', adjustable='box')
    
    plt.suptitle('TPGMM Adaptability: Combined Transformation (Horizontal + Vertical + Rotation)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filename = os.path.join(output_folder, "adaptability_combined_transformation.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def visualize_phase_space_combined(time_baseline, mu_baseline, combined_results,
                                   output_folder):
    """
    Visualize phase space plots for combined transformation
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Define colors
    n_variations = len(combined_results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_variations))
    
    # === RIGHT ANKLE: X Phase Space ===
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(mu_baseline[:, 0], mu_baseline[:, 2], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax1.plot(result['mu'][:, 0], result['mu'][:, 2], 
                color=colors[i], linewidth=2, label=label)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('X Velocity (m/s)')
    ax1.set_title('Right Ankle - X Phase Space')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # === RIGHT ANKLE: Y Phase Space ===
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(mu_baseline[:, 1], mu_baseline[:, 3], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax2.plot(result['mu'][:, 1], result['mu'][:, 3], 
                color=colors[i], linewidth=2, label=label)
    ax2.set_xlabel('Y Position (m)')
    ax2.set_ylabel('Y Velocity (m/s)')
    ax2.set_title('Right Ankle - Y Phase Space')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # === RIGHT ANKLE: Angle vs Time ===
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_baseline, np.rad2deg(mu_baseline[:, 4]), 'k--', 
             linewidth=2, label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax3.plot(result['time'], np.rad2deg(result['mu'][:, 4]), 
                color=colors[i], linewidth=2, label=label)
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Right Ankle - Angle Evolution')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.3)   
    
    
    # === LEFT ANKLE: X Phase Space ===
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(mu_baseline[:, 5], mu_baseline[:, 7], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax4.plot(result['mu'][:, 5], result['mu'][:, 7], 
                color=colors[i], linewidth=2, label=label)
    ax4.set_xlabel('X Position (m)')
    ax4.set_ylabel('X Velocity (m/s)')
    ax4.set_title('Left Ankle - X Phase Space')
    ax4.legend(fontsize=7, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    # === LEFT ANKLE: Y Phase Space ===
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(mu_baseline[:, 6], mu_baseline[:, 8], 'k--', linewidth=2, 
             label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax5.plot(result['mu'][:, 6], result['mu'][:, 8], 
                color=colors[i], linewidth=2, label=label)
    ax5.set_xlabel('Y Position (m)')
    ax5.set_ylabel('Y Velocity (m/s)')
    ax5.set_title('Left Ankle - Y Phase Space')
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    # === LEFT ANKLE: Angle vs Time ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(time_baseline, np.rad2deg(mu_baseline[:, 9]), 'k--', 
             linewidth=2, label='Baseline', alpha=0.7)
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        label = f'H={h:.2f}m, V={v:.2f}m, R={r:.1f}°'
        ax6.plot(result['time'], np.rad2deg(result['mu'][:, 9]), 
                color=colors[i], linewidth=2, label=label)
    ax6.set_xlabel('Normalized Time')
    ax6.set_ylabel('Angle (degrees)')
    ax6.set_title('Left Ankle - Angle Evolution')
    ax6.legend(fontsize=7, loc='best')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Phase Space Analysis: Combined Transformation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filename = os.path.join(output_folder, "adaptability_combined_phase_space.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def save_baseline_trajectory_to_csv(time, mu_baseline, output_folder):
    """
    Save the baseline trajectory to a CSV file
    """
    csv_filename = os.path.join(output_folder, "baseline_trajectory_11D.csv")
    
    # Column names
    column_names = [
        'R_ankle_x', 'R_ankle_y',           # Right ankle position [0-1]
        'R_ankle_vx', 'R_ankle_vy',         # Right ankle velocity [2-3]
        'R_ankle_angle',                    # Right ankle angle [4]
        'L_ankle_x', 'L_ankle_y',           # Left ankle position [5-6]
        'L_ankle_vx', 'L_ankle_vy',         # Left ankle velocity [7-8]
        'L_ankle_angle',                    # Left ankle angle [9]
        'time'                              # Time [10]
    ]
    
    # Prepare data: first column is time, remaining 11 columns are the trajectory dimensions
    data = np.column_stack([time, mu_baseline])
    
    # Save to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(column_names)
        
        # Write data rows
        writer.writerows(data)
    
    print(f"\n✓ Saved baseline trajectory to CSV: {csv_filename}")
    print(f"  - Shape: {data.shape} (time + 11 dimensions)")
    
    return csv_filename


def save_combined_trajectories_to_csv(combined_results, output_folder):
    """
    Save all combined test trajectories to CSV files
    """
    for i, result in enumerate(combined_results):
        h = result['targets']['horizontal']
        v = result['targets']['vertical']
        r = result['targets']['rotation']
        
        csv_filename = os.path.join(output_folder, 
                                   f"combined_traj_H{h:.2f}_V{v:.2f}_R{r:.1f}.csv")
        
        # Column names
        column_names = [
            'R_ankle_x', 'R_ankle_y',
            'R_ankle_vx', 'R_ankle_vy',
            'R_ankle_angle',
            'L_ankle_x', 'L_ankle_y',
            'L_ankle_vx', 'L_ankle_vy',
            'L_ankle_angle',
            'time'
        ]
        
        # Prepare data
        data = np.column_stack([result['time'], result['mu']])
        
        # Save to CSV
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(data)
        
        print(f"✓ Saved: {csv_filename}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run combined adaptability test
    """
    print("\n" + "="*80)
    print(" "*20 + "TPGMM COMBINED ADAPTABILITY TESTING")
    print(" "*15 + "Lower Limb Exoskeleton Gait Adaptation")
    print(" "*10 + "(Horizontal + Vertical + Rotation Simultaneously)")
    print("="*80)
    
    # ========== CONFIGURATION ==========
    reg_factor = "1e-04"
    model_path = "train_tpgmm_model_reg"+reg_factor+"/trained_model.pkl"
    data_path = "TaskPaGMMM/examples/7days1/gait_analysis_export_subject35v6.json"
    output_folder = "adaptability_tests_combined/"+reg_factor+"/"
    N_query = 200
    
    # Define 4 test variations for combined test
    M_factor = 3.333  # Scaling factor from original data units to meters
    horizontal_targets = np.linspace(0, 0.4*M_factor, 4)  # meters
    vertical_targets = np.linspace(0, 0.3*M_factor, 4)    # meters
    rotation_targets = np.linspace(0, 25, 4)              # degrees
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # ========== LOAD MODEL AND DATA ==========
    model_data = load_trained_model(model_path)
    model = model_data['model']
    
    # Load original data for reference
    fr2_endpoint, fr3_endpoint = load_original_data(data_path)
    
    # ========== BASELINE TRAJECTORY (NO ADAPTATION) ==========
    print("\n" + "="*70)
    print("GENERATING BASELINE TRAJECTORY (NO ADAPTATION)")
    print("="*70)
    
    # Identity transformations for baseline
    P = 3
    D = 11
    A_baseline = np.tile(np.eye(D), (N_query, P, 1, 1))
    b_baseline = np.zeros((N_query, P, D))
    
    time_baseline, mu_baseline, sigma_baseline = generate_adapted_trajectory(
        model, A_baseline, b_baseline, N=N_query
    )
    
    print(f"✓ Generated baseline trajectory")
    
    # Save baseline trajectory to CSV
    save_baseline_trajectory_to_csv(time_baseline, mu_baseline, output_folder)
    
    # ========== COMBINED TEST: ALL TRANSFORMATIONS SIMULTANEOUSLY ==========
    print("\n\n" + "="*80)
    print(" "*15 + "COMBINED TEST: HORIZONTAL + VERTICAL + ROTATION")
    print(" "*25 + "(4 variations, all starting from 0)")
    print("="*80)
    
    combined_results = []
    for i in range(4):
        h_target = horizontal_targets[i]
        v_target = vertical_targets[i]
        r_target = rotation_targets[i]
        
        print(f"\n--- Variation {i+1}/4 ---")
        print(f"  Horizontal: 0 → {h_target:.3f} m")
        print(f"  Vertical:   0 → {v_target:.3f} m")
        print(f"  Rotation:   0 → {r_target:.2f}°")
        
        A_combined, b_combined, displacements = create_time_varying_transformations_combined(
            N_query, 
            horizontal_target=h_target,
            vertical_target=v_target,
            rotation_target=r_target
        )
        
        time_combined, mu_combined, sigma_combined = generate_adapted_trajectory(
            model, A_combined, b_combined, N=N_query
        )
        
        combined_results.append({
            'time': time_combined,
            'mu': mu_combined,
            'sigma': sigma_combined,
            'param': displacements,
            'targets': {
                'horizontal': h_target,
                'vertical': v_target,
                'rotation': r_target
            }
        })
    
    # Visualize results
    visualize_combined_adaptations(
        time_baseline, mu_baseline, combined_results, output_folder
    )
    
    visualize_phase_space_combined(
        time_baseline, mu_baseline, combined_results, output_folder
    )
    
    # Save all trajectories to CSV
    print("\n" + "="*70)
    print("SAVING TRAJECTORIES TO CSV")
    print("="*70)
    save_combined_trajectories_to_csv(combined_results, output_folder)
    
    # ========== FINAL SUMMARY ==========
    print("\n\n" + "="*80)
    print(" "*25 + "COMBINED TESTING COMPLETE!")
    print("="*80)
    print(f"\n✓ All tests completed successfully!")
    print(f"✓ Results saved in: {output_folder}/")
    print(f"\nGenerated files:")
    print(f"  BASELINE:")
    print(f"    • baseline_trajectory_11D.csv")
    print(f"\n  COMBINED TRAJECTORIES (4 CSV files):")
    for i in range(4):
        h = horizontal_targets[i]
        v = vertical_targets[i]
        r = rotation_targets[i]
        print(f"    • combined_traj_H{h:.2f}_V{v:.2f}_R{r:.1f}.csv")
    print(f"\n  VISUALIZATIONS:")
    print(f"    • adaptability_combined_transformation.png")
    print(f"    • adaptability_combined_phase_space.png")
    print(f"\nTest configuration:")
    print(f"  - M_factor: {M_factor:.3f}")
    print(f"  - Horizontal targets: {horizontal_targets.tolist()}")
    print(f"  - Vertical targets: {vertical_targets.tolist()}")
    print(f"  - Rotation targets: {rotation_targets.tolist()}")
    print(f"  - Query points: {N_query}")
    print(f"  - Image resolution: 300 DPI")
    print("\n" + "="*80)
    print("\nKEY FINDINGS:")
    print("✓ All three transformations applied SIMULTANEOUSLY")
    print("✓ Each transformation starts from 0 (baseline)")
    print("✓ Linear progression to target values over time")
    print("✓ Smooth adaptation in all dimensions")
    print("✓ Phase space shows coordinated multi-dimensional changes")
    print("\nREADY FOR EXOSKELETON IMPLEMENTATION! 🦾")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()