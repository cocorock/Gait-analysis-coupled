"""
TPGMM DEMONSTRATION - PRODUCT OF FRAMES
========================================
Demonstrates TPGMM's trajectory recovery using product of Gaussians
across all 3 frames (FR1: Hip, FR2: Right Foot, FR3: Left Foot)

This script:
1. Loads trained TPGMM model and demonstration trajectories
2. Shows trajectories from each frame's perspective (FR1, FR2, FR3)
3. Generates recovered trajectory using GMR with product of all frames
4. Plots all trajectories together for comparison

Output:
- Single figure with 2 plots (Right Ankle, Left Ankle)
- Each plot shows:
  * Gray thin lines: Original 30 demonstration trajectories
  * Blue solid: FR1 (Hip) trajectory
  * Orange dashed: FR2 (Right Foot) trajectory
  * Green dash-dot: FR3 (Left Foot) trajectory
  * Red thick: GMR recovered trajectory (product of frames)

Author: Victor
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from matplotlib.patches import Ellipse

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


def generate_trajectories_all_frames(model, A_frames, b_frames, n_query=200):
    """
    Generate trajectories from all frames + GMR product
    
    Parameters:
    -----------
    model : OptimizedTPGMM
        Trained model
    A_frames : list of arrays
        Transformation matrices [A_FR1, A_FR2, A_FR3]
    b_frames : list of arrays
        Translation vectors [b_FR1, b_FR2, b_FR3]
    n_query : int
        Number of query points
        
    Returns:
    --------
    frame_trajs : list of arrays
        Trajectories from each frame [FR1, FR2, FR3]
    gmr_traj : array
        GMR trajectory using product of frames
    gmr_sigma : array
        GMR covariance
    """
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    gmr = OptimizedGMR(model)
    
    input_dims = [10]  # Time
    output_dims = list(range(10))  # All other dimensions
    
    # Get individual frame trajectories
    frame_trajs = gmr.get_frame_trajectories(
        time_query, input_dims, output_dims, A_frames, b_frames
    )
    
    # Get GMR trajectory (product of frames)
    gmr_traj, gmr_sigma = gmr.predict(
        time_query, input_dims, output_dims, A_frames, b_frames
    )
    
    return frame_trajs, gmr_traj, gmr_sigma


def plot_adaptation_with_all_frames(frame_trajs_list, gmr_trajs_list, demo_trajectories, output_folder):
    """
    Plot trajectories showing all 3 frames + GMR + demonstration trajectories
    
    Parameters:
    -----------
    frame_trajs_list : list
        Frame trajectories [FR1_traj, FR2_traj, FR3_traj]
    gmr_trajs_list : array
        GMR trajectory
    demo_trajectories : list of arrays
        Original demonstration trajectories from training data
    output_folder : str
        Where to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define frame colors
    frame_colors = ['#1f77b4', "#da6b0a", "#17b117"]  # Blue, Orange, Green
    frame_labels = ['FR1 (Hip)', 'FR2 (Right Foot)', 'FR3 (Left Foot)']
    frame_styles = ['-', '--', '-.']
    
    # Plot demonstration trajectories (gray, thin, transparent)
    for demo_traj in demo_trajectories:
        # Right ankle
        axes[0].plot(demo_traj[:, 0], demo_traj[:, 1], 
                    'gray', linewidth=0.8, alpha=0.15, zorder=1)
        # Left ankle
        axes[1].plot(demo_traj[:, 5], demo_traj[:, 6], 
                    'gray', linewidth=0.8, alpha=0.15, zorder=1)
    
    # Add label for demonstrations (only once)
    axes[0].plot([], [], 'gray', linewidth=0.8, alpha=0.3, label='Demonstrations (n=30)')
    axes[1].plot([], [], 'gray', linewidth=0.8, alpha=0.3, label='Demonstrations (n=30)')
    
    # RIGHT ANKLE (dims 0-1)
    # Plot each frame's trajectory
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(frame_trajs_list, frame_colors, frame_labels, frame_styles)
    ):
        axes[0].plot(frame_traj[:, 0], frame_traj[:, 1], 
                   color=frame_color, linestyle=frame_style, linewidth=2.0,
                   alpha=0.6, label=frame_label, zorder=5)
    
    # Plot GMR trajectory (product of frames)
    axes[0].plot(gmr_trajs_list[:, 0], gmr_trajs_list[:, 1], 
                color='red', linewidth=3.5,
                label='GMR (Product of Frames)', zorder=10, alpha=0.95)
    
    # Mark start and end
    axes[0].plot(gmr_trajs_list[0, 0], gmr_trajs_list[0, 1], 'o', 
                color='red', markersize=12, zorder=11, markeredgecolor='darkred', 
                markeredgewidth=2)
    axes[0].plot(gmr_trajs_list[-1, 0], gmr_trajs_list[-1, 1], 's', 
                color='red', markersize=12, zorder=11, markeredgecolor='darkred',
                markeredgewidth=2)
    
    # LEFT ANKLE (dims 5-6)
    # Plot each frame's trajectory
    for frame_idx, (frame_traj, frame_color, frame_label, frame_style) in enumerate(
        zip(frame_trajs_list, frame_colors, frame_labels, frame_styles)
    ):
        axes[1].plot(frame_traj[:, 5], frame_traj[:, 6], 
                   color=frame_color, linestyle=frame_style, linewidth=2.0,
                   alpha=0.6, label=frame_label, zorder=5)
    
    # Plot GMR trajectory
    axes[1].plot(gmr_trajs_list[:, 5], gmr_trajs_list[:, 6], 
                color='red', linewidth=3.5,
                label='GMR (Product of Frames)', zorder=10, alpha=0.95)
    
    # Mark start and end
    axes[1].plot(gmr_trajs_list[0, 5], gmr_trajs_list[0, 6], 'o', 
                color='red', markersize=12, zorder=11, markeredgecolor='darkred',
                markeredgewidth=2)
    axes[1].plot(gmr_trajs_list[-1, 5], gmr_trajs_list[-1, 6], 's', 
                color='red', markersize=12, zorder=11, markeredgecolor='darkred',
                markeredgewidth=2)
    
    # Configure right ankle plot
    axes[0].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('Right Ankle Trajectory\n(Product of 3 Frames)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_aspect('equal', adjustable='box')
    
    # Configure left ankle plot
    axes[1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    axes[1].set_title('Left Ankle Trajectory\n(Product of 3 Frames)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].set_aspect('equal', adjustable='box')
    
    # Main title
    fig.suptitle('TPGMM: Trajectory Recovery using Product of Frames', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_folder, 'tpgmm_product_frames_with_demonstrations.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()
    plt.close()


def main(model_folder=None, reg_factor=5e-4):
    """
    Main demonstration pipeline
    
    Parameters:
    -----------
    model_folder : str, optional
        Path to model folder
    reg_factor : float
        Regularization factor
    """
    print("\n" + "="*70)
    print("TPGMM - PRODUCT OF FRAMES DEMONSTRATION")
    print("="*70)
    print("Showing trajectories from all 3 frames + GMR recovered trajectory")
    print("With original demonstration trajectories")
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
    
    # Load demonstration trajectories
    demo_trajectories = np.concatenate([
        model_data['trajectories_fr1'],
        model_data['trajectories_fr2'],
        model_data['trajectories_fr3']
    ])
    print(f"\n✓ Loaded {len(demo_trajectories)} demonstration trajectories")
    
    print("\n" + "="*70)
    print("GENERATING TRAJECTORIES")
    print("="*70)
    
    # Define baseline frames (identity - no transformation)
    A_FR1, b_FR1 = create_transformation_matrix_2d(0, 0, 0)
    A_FR2, b_FR2 = create_transformation_matrix_2d(0, 0, 0)
    A_FR3, b_FR3 = create_transformation_matrix_2d(0, 0, 0)
    
    A_frames = [A_FR1, A_FR2, A_FR3]
    b_frames = [b_FR1, b_FR2, b_FR3]
    
    # Generate trajectories
    frame_trajs, gmr_traj, gmr_sigma = generate_trajectories_all_frames(
        model, A_frames, b_frames
    )
    
    print(f"\n✓ Generated FR1 trajectory: shape {frame_trajs[0].shape}")
    print(f"✓ Generated FR2 trajectory: shape {frame_trajs[1].shape}")
    print(f"✓ Generated FR3 trajectory: shape {frame_trajs[2].shape}")
    print(f"✓ Generated GMR trajectory (product): shape {gmr_traj.shape}")
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    plot_adaptation_with_all_frames(
        frame_trajs, gmr_traj, demo_trajectories, output_folder
    )
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print(f"✓ Results saved in: {output_folder}/")
    print(f"✓ Figure: tpgmm_product_frames_with_demonstrations.png")
    print("\nThe plot shows:")
    print("  - Gray thin lines: Original demonstration trajectories (n=30)")
    print("  - Blue solid: FR1 (Hip) frame trajectory")
    print("  - Orange dashed: FR2 (Right Foot) frame trajectory")
    print("  - Green dash-dot: FR3 (Left Foot) frame trajectory")
    print("  - Red thick line: GMR recovered trajectory (product of frames)")
    print("  - Circle: Start point")
    print("  - Square: End point")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TPGMM PRODUCT OF FRAMES DEMONSTRATION")
    print("="*70)
    
    # Configuration
    reg_factor = 1e-4
    model_folder = None
    
    # Run demonstration
    main(model_folder=model_folder, reg_factor=reg_factor)
    
    print("\n✅ Demonstration complete!")