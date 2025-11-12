"""
Gait Analysis using Task-Parameterized Gaussian Mixture Models (TPGMM)
Following the structure of gait_example_modifiedV2.ipynb

This implementation:
1. Loads gait data from JSON with two reference frames (FR1, FR2)
2. Extracts 7-dimensional features: [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]
3. Tests multiple TPGMM models (n_components from 3 to 23) using K-Means initialization and EM
4. Selects optimal model using Bayesian Information Criterion (BIC)
5. Uses Gaussian Mixture Regression (GMR) to generate smooth gait trajectories
6. Visualizes results with uncertainty quantification

Based on:
- Calinon, S. (2016). A Tutorial on Task-Parameterized Movement Learning and Retrieval.
  Intelligent Service Robotics, 9:1, 1-29.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Set numpy print options for cleaner output
np.set_printoptions(precision=2, suppress=True)


class TPGMM:
    """
    Task-Parameterized Gaussian Mixture Model
    
    Implementation based on Calinon's TP-GMM formulation with:
    - Multiple reference frames
    - K-Means initialization
    - EM algorithm for parameter estimation
    - Product of Gaussians for frame combination
    """
    
    def __init__(self, n_components, n_frames, n_features, reg_factor=1e-5, 
                 max_iter=100, tol=1e-4, verbose=True):
        """
        Initialize TPGMM model
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components (K)
        n_frames : int
            Number of reference frames (P)
        n_features : int
            Dimensionality of data (D)
        reg_factor : float
            Regularization factor for covariance matrices
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        verbose : bool
            Print training progress
        """
        self.K = n_components
        self.P = n_frames
        self.D = n_features
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Model parameters: priors, means, covariances for each frame and component
        self.priors = None  # Shape: (K,)
        self.means = None   # Shape: (P, K, D)
        self.covars = None  # Shape: (P, K, D, D)
        
        self.log_likelihood_history = []
        self.converged = False
        self.n_iter = 0
    
    def _regularize_covar(self, covar):
        """Add regularization to covariance matrix"""
        return covar + self.reg_factor * np.eye(self.D)
    
    def _kmeans_init(self, X_frames):
        """
        Initialize parameters using K-Means clustering
        
        Parameters:
        -----------
        X_frames : ndarray, shape (P, N, D)
            Data in all reference frames
        """
        # Average data across frames for initialization
        X_avg = np.mean(X_frames, axis=0)  # (N, D)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_avg)
        
        # Initialize parameters
        self.priors = np.zeros(self.K)
        self.means = np.zeros((self.P, self.K, self.D))
        self.covars = np.zeros((self.P, self.K, self.D, self.D))
        
        for k in range(self.K):
            mask = (labels == k)
            self.priors[k] = np.sum(mask) / len(labels)
            
            for p in range(self.P):
                if np.sum(mask) > 0:
                    self.means[p, k] = np.mean(X_frames[p, mask], axis=0)
                    
                    X_centered = X_frames[p, mask] - self.means[p, k]
                    self.covars[p, k] = (X_centered.T @ X_centered) / np.sum(mask)
                    self.covars[p, k] = self._regularize_covar(self.covars[p, k])
                else:
                    # Handle empty clusters
                    self.means[p, k] = X_frames[p].mean(axis=0)
                    self.covars[p, k] = self._regularize_covar(np.cov(X_frames[p].T))
        
        # Ensure priors sum to 1
        self.priors /= self.priors.sum()
    
    def _e_step(self, X_frames):
        """
        E-step: Compute responsibilities
        
        Parameters:
        -----------
        X_frames : ndarray, shape (P, N, D)
            Data in all reference frames
            
        Returns:
        --------
        responsibilities : ndarray, shape (N, K)
            Posterior probabilities
        log_likelihood : float
            Current log-likelihood
        """
        N = X_frames.shape[1]
        responsibilities = np.zeros((N, self.K))
        
        # Compute weighted likelihood for each point
        for n in range(N):
            for k in range(self.K):
                # Product of Gaussians across frames (Equation 49 in Calinon 2016)
                log_prob = np.log(self.priors[k] + 1e-10)
                
                for p in range(self.P):
                    try:
                        rv = multivariate_normal(
                            mean=self.means[p, k],
                            cov=self.covars[p, k],
                            allow_singular=True
                        )
                        log_prob += rv.logpdf(X_frames[p, n])
                    except:
                        # Handle numerical issues
                        log_prob += -1e10
                
                responsibilities[n, k] = log_prob
        
        # Normalize in log space for numerical stability
        log_sum = np.logaddexp.reduce(responsibilities, axis=1, keepdims=True)
        responsibilities = np.exp(responsibilities - log_sum)
        
        # Compute log-likelihood
        log_likelihood = np.sum(log_sum)
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X_frames, responsibilities):
        """
        M-step: Update parameters
        
        Parameters:
        -----------
        X_frames : ndarray, shape (P, N, D)
            Data in all reference frames
        responsibilities : ndarray, shape (N, K)
            Posterior probabilities from E-step
        """
        N = X_frames.shape[1]
        
        # Update priors (Equation 50 in Calinon 2016)
        self.priors = responsibilities.sum(axis=0) / N
        
        # Update means and covariances for each frame and component
        for p in range(self.P):
            for k in range(self.K):
                resp_k = responsibilities[:, k]
                resp_sum = resp_k.sum()
                
                if resp_sum > 1e-10:
                    # Update means (Equation 51 in Calinon 2016)
                    self.means[p, k] = (resp_k @ X_frames[p]) / resp_sum
                    
                    # Update covariances (Equation 52 in Calinon 2016)
                    X_centered = X_frames[p] - self.means[p, k]
                    self.covars[p, k] = (X_centered.T @ (resp_k[:, None] * X_centered)) / resp_sum
                    self.covars[p, k] = self._regularize_covar(self.covars[p, k])
    
    def fit(self, X_frames):
        """
        Fit TPGMM model using EM algorithm
        
        Parameters:
        -----------
        X_frames : ndarray, shape (P, N, D)
            Data in all reference frames
            P = number of frames
            N = number of data points
            D = dimensionality
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training TPGMM with K={self.K} components")
            print(f"{'='*60}")
        
        # Initialize parameters with K-Means
        self._kmeans_init(X_frames)
        
        prev_log_likelihood = -np.inf
        self.log_likelihood_history = []
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step
            responsibilities, log_likelihood = self._e_step(X_frames)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            improvement = log_likelihood - prev_log_likelihood
            
            if self.verbose:
                print(f"Iteration {iteration+1}: LL = {log_likelihood:.2f}, "
                      f"Improvement = {improvement:.6f}")
            
            if iteration > 0 and improvement < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration+1} iterations")
                self.converged = True
                self.n_iter = iteration + 1
                break
            
            # M-step
            self._m_step(X_frames, responsibilities)
            prev_log_likelihood = log_likelihood
        
        if not self.converged:
            if self.verbose:
                print(f"Reached maximum iterations ({self.max_iter})")
            self.n_iter = self.max_iter
        
        return self
    
    def compute_bic(self, X_frames):
        """
        Compute Bayesian Information Criterion
        
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        Lower BIC indicates better model
        
        Parameters:
        -----------
        X_frames : ndarray, shape (P, N, D)
            Data in all reference frames
            
        Returns:
        --------
        bic : float
            Bayesian Information Criterion
        """
        N = X_frames.shape[1]
        
        # Compute log-likelihood
        _, log_likelihood = self._e_step(X_frames)
        
        # Count parameters
        # Priors: K-1 (sum to 1 constraint)
        # Means: P * K * D
        # Covariances: P * K * D * (D + 1) / 2 (symmetric matrices)
        n_params = (self.K - 1) + (self.P * self.K * self.D) + \
                   (self.P * self.K * self.D * (self.D + 1) // 2)
        
        bic = -2 * log_likelihood + n_params * np.log(N)
        
        return bic
    
    def get_adapted_gmm(self, A_frames, b_frames):
        """
        Get adapted GMM parameters for new frame configurations
        
        Uses product of linearly transformed Gaussians (Equations 5-6 in Calinon 2016)
        
        Parameters:
        -----------
        A_frames : list of ndarray, length P
            Rotation/scaling matrices for each frame, shape (D, D)
        b_frames : list of ndarray, length P
            Translation vectors for each frame, shape (D,)
            
        Returns:
        --------
        adapted_means : ndarray, shape (K, D)
            Adapted means
        adapted_covars : ndarray, shape (K, D, D)
            Adapted covariances
        """
        adapted_means = np.zeros((self.K, self.D))
        adapted_covars = np.zeros((self.K, self.D, self.D))
        
        for k in range(self.K):
            # Transform parameters to global frame
            transformed_means = []
            transformed_covars = []
            transformed_precisions = []
            
            for p in range(self.P):
                # Linear transformation (Equation 5 in Calinon 2016)
                mu_hat = A_frames[p] @ self.means[p, k] + b_frames[p]
                Sigma_hat = A_frames[p] @ self.covars[p, k] @ A_frames[p].T
                
                transformed_means.append(mu_hat)
                transformed_covars.append(Sigma_hat)
                transformed_precisions.append(np.linalg.inv(Sigma_hat))
            
            # Product of Gaussians (Equation 6 in Calinon 2016)
            precision_sum = sum(transformed_precisions)
            adapted_covars[k] = np.linalg.inv(precision_sum)
            
            weighted_means = sum(prec @ mean for prec, mean 
                               in zip(transformed_precisions, transformed_means))
            adapted_means[k] = adapted_covars[k] @ weighted_means
        
        return adapted_means, adapted_covars


class GaussianMixtureRegression:
    """
    Gaussian Mixture Regression for trajectory generation
    
    Implementation based on Calinon's GMR formulation (Section 5.1)
    """
    
    def __init__(self, tpgmm_model):
        """
        Initialize GMR with trained TPGMM
        
        Parameters:
        -----------
        tpgmm_model : TPGMM
            Trained TPGMM model
        """
        self.tpgmm = tpgmm_model
    
    def predict(self, input_data, input_dims, output_dims, A_frames, b_frames):
        """
        Perform Gaussian Mixture Regression
        
        Parameters:
        -----------
        input_data : ndarray, shape (N, len(input_dims))
            Input values (e.g., time points)
        input_dims : list of int
            Indices of input dimensions
        output_dims : list of int
            Indices of output dimensions
        A_frames : list of ndarray
            Frame transformation matrices
        b_frames : list of ndarray
            Frame translation vectors
            
        Returns:
        --------
        mu_output : ndarray, shape (N, len(output_dims))
            Predicted output means
        sigma_output : ndarray, shape (N, len(output_dims), len(output_dims))
            Predicted output covariances
        """
        N = input_data.shape[0]
        n_out = len(output_dims)
        
        mu_output = np.zeros((N, n_out))
        sigma_output = np.zeros((N, n_out, n_out))
        
        # Get adapted GMM for current frame configuration
        adapted_means, adapted_covars = self.tpgmm.get_adapted_gmm(A_frames, b_frames)
        
        for t in range(N):
            xi_input = input_data[t]
            
            # Compute responsibilities (Equation 16 in Calinon 2016)
            responsibilities = np.zeros(self.tpgmm.K)
            for k in range(self.tpgmm.K):
                # Extract input part
                mu_I = adapted_means[k, input_dims]
                Sigma_I = adapted_covars[k][:, input_dims][input_dims, :]
                
                # Compute responsibility
                try:
                    rv = multivariate_normal(mean=mu_I, cov=Sigma_I, allow_singular=True)
                    responsibilities[k] = self.tpgmm.priors[k] * rv.pdf(xi_input)
                except:
                    responsibilities[k] = 1e-10
            
            # Normalize
            resp_sum = responsibilities.sum()
            if resp_sum > 1e-10:
                responsibilities /= resp_sum
            else:
                responsibilities = np.ones(self.tpgmm.K) / self.tpgmm.K
            
            # Compute conditional expectation (Equations 13-19 in Calinon 2016)
            mu_out_t = np.zeros(n_out)
            mu_out_components = []
            
            for k in range(self.tpgmm.K):
                # Block decomposition
                mu_I = adapted_means[k, input_dims]
                mu_O = adapted_means[k, output_dims]
                
                Sigma_II = adapted_covars[k][:, input_dims][input_dims, :]
                Sigma_OO = adapted_covars[k][:, output_dims][output_dims, :]
                Sigma_IO = adapted_covars[k][:, input_dims][output_dims, :]
                Sigma_OI = Sigma_IO.T
                
                # Conditional mean (Equation 14)
                try:
                    Sigma_II_inv = np.linalg.inv(Sigma_II)
                    mu_O_cond = mu_O + Sigma_OI @ Sigma_II_inv @ (xi_input - mu_I)
                except:
                    mu_O_cond = mu_O
                
                mu_out_components.append(mu_O_cond)
                mu_out_t += responsibilities[k] * mu_O_cond
            
            mu_output[t] = mu_out_t
            
            # Compute conditional covariance (Equation 15 and 19)
            sigma_out_t = np.zeros((n_out, n_out))
            for k in range(self.tpgmm.K):
                Sigma_II = adapted_covars[k][:, input_dims][input_dims, :]
                Sigma_OO = adapted_covars[k][:, output_dims][output_dims, :]
                Sigma_IO = adapted_covars[k][:, input_dims][output_dims, :]
                Sigma_OI = Sigma_IO.T
                
                # Conditional covariance
                try:
                    Sigma_II_inv = np.linalg.inv(Sigma_II)
                    Sigma_O_cond = Sigma_OO - Sigma_OI @ Sigma_II_inv @ Sigma_IO
                except:
                    Sigma_O_cond = Sigma_OO
                
                # Equation 19 in Calinon 2016
                mu_diff = mu_out_components[k] - mu_out_t
                sigma_out_t += responsibilities[k] * (
                    Sigma_O_cond + np.outer(mu_diff, mu_diff)
                )
            
            sigma_output[t] = sigma_out_t
        
        return mu_output, sigma_output


def load_gait_data(data_path):
    """Load gait analysis data from JSON file"""
    print("="*70)
    print("STEP 1: Loading Gait Data")
    print("="*70)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded data from: {os.path.basename(data_path)}")
    print(f"  Creation date: {data['export_info']['creation_date']}")
    print(f"  Source files: {len(data['export_info']['source_files'])}")
    
    return data


def extract_trajectories(data):
    """
    Extract trajectories from both reference frames
    
    Features (7D): [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]
    """
    print("\n" + "="*70)
    print("STEP 2: Extracting Trajectories")
    print("="*70)
    
    fr1_data = data['kinematics_data']['FR1']
    fr2_data = data['kinematics_data']['FR2']
    
    trajectories_fr1 = []
    trajectories_fr2 = []
    
    n_demos = len(fr1_data['right_leg_kinematics'])
    print(f"Processing {n_demos} demonstrations...")
    
    for demo_fr1, demo_fr2 in zip(fr1_data['right_leg_kinematics'], 
                                   fr2_data['right_leg_kinematics']):
        # FR1 trajectory
        right_ankle_pos = np.array(demo_fr1['right_ankle_pos'])  # (2, n_points)
        right_ankle_vel = np.array(demo_fr1['right_ankle_vel'])
        ankle_right_deg = np.array(demo_fr1['ankle_right_deg'])
        ankle_left_deg = np.array(demo_fr1['ankle_left_deg'])
        
        n_points = right_ankle_pos.shape[1]
        time_vector = np.linspace(0, 1, n_points)
        
        # Stack features: [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]
        traj_fr1 = np.vstack([
            right_ankle_pos[0, :],           # x position
            right_ankle_pos[1, :],           # y position
            right_ankle_vel[0, :],           # x velocity
            right_ankle_vel[1, :],           # y velocity
            np.deg2rad(ankle_right_deg),     # right ankle angle (rad)
            np.deg2rad(ankle_left_deg),      # left ankle angle (rad)
            time_vector                       # normalized time
        ]).T  # Shape: (n_points, 7)
        
        trajectories_fr1.append(traj_fr1)
        
        # FR2 trajectory (same structure)
        right_ankle_pos = np.array(demo_fr2['right_ankle_pos'])
        right_ankle_vel = np.array(demo_fr2['right_ankle_vel'])
        ankle_right_deg = np.array(demo_fr2['ankle_right_deg'])
        ankle_left_deg = np.array(demo_fr2['ankle_left_deg'])
        
        traj_fr2 = np.vstack([
            right_ankle_pos[0, :],
            right_ankle_pos[1, :],
            right_ankle_vel[0, :],
            right_ankle_vel[1, :],
            np.deg2rad(ankle_right_deg),
            np.deg2rad(ankle_left_deg),
            time_vector
        ]).T
        
        trajectories_fr2.append(traj_fr2)
    
    # Convert to arrays
    trajectories_fr1 = np.array(trajectories_fr1)  # (n_demos, n_points, 7)
    trajectories_fr2 = np.array(trajectories_fr2)
    
    print(f"✓ FR1 trajectories shape: {trajectories_fr1.shape}")
    print(f"✓ FR2 trajectories shape: {trajectories_fr2.shape}")
    print(f"✓ Features: [x_pos, y_pos, x_vel, y_vel, ankle_right_rad, ankle_left_rad, time]")
    
    return trajectories_fr1, trajectories_fr2


def prepare_tpgmm_data(trajectories_fr1, trajectories_fr2):
    """
    Prepare data for TPGMM training
    
    Returns:
    --------
    X_frames : ndarray, shape (2, N, 7)
        Data stacked for two frames
    """
    print("\n" + "="*70)
    print("STEP 3: Preparing Data for TPGMM")
    print("="*70)
    
    # Stack all demonstrations
    X_fr1 = np.vstack(trajectories_fr1)  # (N_total, 7)
    X_fr2 = np.vstack(trajectories_fr2)  # (N_total, 7)
    
    # Stack frames: shape (2, N_total, 7)
    X_frames = np.stack([X_fr1, X_fr2], axis=0)
    
    print(f"✓ Data shape for TPGMM: {X_frames.shape}")
    print(f"  - 2 reference frames")
    print(f"  - {X_frames.shape[1]} total data points")
    print(f"  - 7 features per point")
    
    return X_frames


def train_tpgmm_with_bic_selection(X_frames, component_range=range(3, 24)):
    """
    Train multiple TPGMM models and select best using BIC
    
    Parameters:
    -----------
    X_frames : ndarray, shape (P, N, D)
        Training data
    component_range : range or list
        Numbers of components to test
        
    Returns:
    --------
    best_model : TPGMM
        Model with lowest BIC
    results : dict
        Training results for all models
    """
    print("\n" + "="*70)
    print("STEP 4: TPGMM Training with BIC Model Selection")
    print("="*70)
    
    results = {
        'n_components': [],
        'bic': [],
        'log_likelihood': [],
        'n_iterations': [],
        'models': []
    }
    
    best_bic = np.inf
    best_model = None
    
    for n_comp in component_range:
        print(f"\n{'='*60}")
        print(f"Testing K = {n_comp} components")
        print(f"{'='*60}")
        
        # Initialize and train model
        model = TPGMM(
            n_components=n_comp,
            n_frames=X_frames.shape[0],
            n_features=X_frames.shape[2],
            reg_factor=1e-4,
            max_iter=100,
            tol=1e-3,
            verbose=True
        )
        
        model.fit(X_frames)
        
        # Compute BIC
        bic = model.compute_bic(X_frames)
        final_ll = model.log_likelihood_history[-1]
        
        print(f"\n{'='*60}")
        print(f"Results for K = {n_comp}:")
        print(f"  Final Log-Likelihood: {final_ll:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Iterations: {model.n_iter}")
        print(f"  Converged: {model.converged}")
        print(f"{'='*60}")
        
        # Store results
        results['n_components'].append(n_comp)
        results['bic'].append(bic)
        results['log_likelihood'].append(final_ll)
        results['n_iterations'].append(model.n_iter)
        results['models'].append(model)
        
        # Update best model
        if bic < best_bic:
            best_bic = bic
            best_model = model
            print(f"  *** NEW BEST MODEL (BIC: {bic:.2f}) ***")
    
    print("\n" + "="*70)
    print("BIC Model Selection Complete")
    print("="*70)
    print(f"Best number of components: K = {best_model.K}")
    print(f"Best BIC: {best_bic:.2f}")
    print("="*70)
    
    return best_model, results


def visualize_bic_results(results):
    """Plot BIC scores vs number of components"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # BIC scores
    ax1.plot(results['n_components'], results['bic'], 'bo-', linewidth=2, markersize=8)
    best_idx = np.argmin(results['bic'])
    ax1.plot(results['n_components'][best_idx], results['bic'][best_idx], 
             'r*', markersize=20, label='Best Model')
    ax1.set_xlabel('Number of Components (K)', fontsize=12)
    ax1.set_ylabel('BIC Score', fontsize=12)
    ax1.set_title('Model Selection: Bayesian Information Criterion', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Log-likelihood
    ax2.plot(results['n_components'], results['log_likelihood'], 'go-', linewidth=2, markersize=8)
    ax2.plot(results['n_components'][best_idx], results['log_likelihood'][best_idx], 
             'r*', markersize=20, label='Best Model')
    ax2.set_xlabel('Number of Components (K)', fontsize=12)
    ax2.set_ylabel('Log-Likelihood', fontsize=12)
    ax2.set_title('Final Log-Likelihood vs Number of Components', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('bic_model_selection.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved BIC comparison plot: bic_model_selection.png")
    plt.show()


def generate_trajectory_with_gmr(best_model, trajectories_fr1):
    """
    Generate smooth trajectory using GMR
    
    Parameters:
    -----------
    best_model : TPGMM
        Trained TPGMM model
    trajectories_fr1 : ndarray
        Original trajectories from FR1 for reference
        
    Returns:
    --------
    time_query : ndarray
        Query time points
    mu_generated : ndarray
        Generated trajectory means
    sigma_generated : ndarray
        Generated trajectory covariances
    """
    print("\n" + "="*70)
    print("STEP 5: Trajectory Generation with GMR")
    print("="*70)
    
    # Define reference frames (identity for simple case)
    n_features = 7
    A_frames = [np.eye(n_features), np.eye(n_features)]
    b_frames = [np.zeros(n_features), np.zeros(n_features)]
    
    # Query points (time)
    n_query = 200
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    # Initialize GMR
    gmr = GaussianMixtureRegression(best_model)
    
    # Input: time (dimension 6), Output: all other dimensions (0-5)
    input_dims = [6]  # time
    output_dims = [0, 1, 2, 3, 4, 5]  # positions, velocities, angles
    
    print("Generating trajectory...")
    print(f"  Input dimension: {input_dims} (time)")
    print(f"  Output dimensions: {output_dims} (positions, velocities, angles)")
    
    mu_generated, sigma_generated = gmr.predict(
        time_query, input_dims, output_dims, A_frames, b_frames
    )
    
    print(f"✓ Generated trajectory shape: {mu_generated.shape}")
    print(f"✓ Covariance shape: {sigma_generated.shape}")
    
    return time_query, mu_generated, sigma_generated


def visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1):
    """
    Visualize generated trajectories with uncertainty bands
    """
    print("\n" + "="*70)
    print("STEP 6: Visualizing Results")
    print("="*70)
    
    feature_names = ['X Position', 'Y Position', 'X Velocity', 'Y Velocity', 
                     'Right Ankle Angle', 'Left Ankle Angle']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot original demonstrations
    for i, traj in enumerate(trajectories_fr1):
        time_orig = traj[:, 6]  # time is last dimension
        for dim in range(6):
            axes[dim].plot(time_orig, traj[:, dim], 'b-', alpha=0.2, linewidth=1)
    
    # Plot generated trajectory with uncertainty
    for dim in range(6):
        time_flat = time_query.flatten()
        mu = mu_generated[:, dim]
        std = np.sqrt(sigma_generated[:, dim, dim])
        
        # Plot mean
        axes[dim].plot(time_flat, mu, 'r-', linewidth=2, label='GMR Generated')
        
        # Plot ±1σ uncertainty band
        axes[dim].fill_between(time_flat, mu - std, mu + std, 
                               color='red', alpha=0.2, label='±1σ Uncertainty')
        
        axes[dim].set_xlabel('Normalized Time', fontsize=11)
        axes[dim].set_ylabel(feature_names[dim], fontsize=11)
        axes[dim].set_title(f'{feature_names[dim]} vs Time', fontsize=12, fontweight='bold')
        axes[dim].grid(True, alpha=0.3)
        axes[dim].legend(fontsize=9)
        
        # Print statistics
        print(f"\n{feature_names[dim]}:")
        print(f"  Mean: {np.mean(mu):.4f}")
        print(f"  Std Dev: {np.std(mu):.4f}")
        print(f"  Avg Uncertainty (±1σ): {np.mean(std):.4f}")
    
    plt.tight_layout()
    plt.savefig('gmr_trajectory_generation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved trajectory comparison plot: gmr_trajectory_generation.png")
    plt.show()


def plot_2d_trajectories(trajectories_fr1, mu_generated):
    """Plot 2D ankle trajectories in XY space"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot original demonstrations
    for traj in trajectories_fr1:
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1.5)
    
    # Plot generated trajectory
    ax.plot(mu_generated[:, 0], mu_generated[:, 1], 'r-', linewidth=3, label='GMR Generated')
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Right Ankle 2D Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('2d_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved 2D trajectory plot: 2d_trajectory_comparison.png")
    plt.show()


def main(data_path):
    """
    Main analysis pipeline following Jupyter notebook structure
    """
    print("\n" + "="*70)
    print("GAIT ANALYSIS USING TASK-PARAMETERIZED GMM")
    print("="*70)
    print("Following structure of gait_example_modifiedV2.ipynb")
    print("="*70)
    
    # Load data
    data = load_gait_data(data_path)
    
    # Extract trajectories
    trajectories_fr1, trajectories_fr2 = extract_trajectories(data)
    
    # Visualize original 2D trajectories
    print("\nVisualizing original 2D trajectories...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for traj in trajectories_fr1:
        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=2)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('FR1: Right Ankle Trajectories', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    for traj in trajectories_fr2:
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=2)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_title('FR2: Right Ankle Trajectories', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('original_trajectories_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved original trajectories plot: original_trajectories_2d.png")
    plt.show()
    
    # Prepare TPGMM data
    X_frames = prepare_tpgmm_data(trajectories_fr1, trajectories_fr2)
    
    # Train with BIC selection
    best_model, results = train_tpgmm_with_bic_selection(
        X_frames, 
        component_range=range(3, 24)  # Test K = 3 to 23
    )
    
    # Visualize BIC results
    visualize_bic_results(results)
    
    # Generate trajectory with GMR
    time_query, mu_generated, sigma_generated = generate_trajectory_with_gmr(
        best_model, trajectories_fr1
    )
    
    # Visualize results
    visualize_results(time_query, mu_generated, sigma_generated, trajectories_fr1)
    
    # Plot 2D comparison
    plot_2d_trajectories(trajectories_fr1, mu_generated)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"✓ Best model: K = {best_model.K} components")
    print(f"✓ BIC: {results['bic'][np.argmin(results['bic'])]:.2f}")
    print("✓ All visualizations saved")
    print("="*70)
    
    return best_model, results, mu_generated, sigma_generated


if __name__ == "__main__":
    # Path to gait data
    # Update this path to your actual data file
    data_path = "TaskPaGMMM\\examples\\7days1\\gait_analysis_export_subject35v4.json"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please update the data_path variable with the correct path to your JSON file.")
        sys.exit(1)
    
    # Run analysis
    model, results, mu_gen, sigma_gen = main(data_path)
