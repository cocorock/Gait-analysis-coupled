"""
Task-Parameterized Gaussian Mixture Model (TPGMM) Implementation
Based on: Calinon et al., "Robot Learning with Task-parameterized Generative Models", ISRR 2015

This implementation follows the mathematical formulation from the paper:
- Task parameters define reference frames with positions and orientations
- Gaussian components are defined in each reference frame
- Products of Gaussians for combining information across frames
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm, inv, det
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class TPGMM:
    """
    Task-Parameterized Gaussian Mixture Model
    
    Key concepts from Calinon paper:
    1. Task parameters define coordinate frames with position A_j and orientation b_j
    2. Each Gaussian component k has parameters in each frame j: μ^{(j)}_k, Σ^{(j)}_k  
    3. Product of Gaussians combines information across frames
    4. EM algorithm estimates parameters with task-parameterized structure
    """
    
    def __init__(self, n_components=5, n_frames=2, reg_covar=1e-6, max_iter=100, random_state=42):
        """
        Initialize TPGMM
        
        Args:
            n_components: Number of Gaussian components
            n_frames: Number of task parameter frames
            reg_covar: Regularization for covariance matrices
            max_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_frames = n_frames
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Model parameters
        self.priors_ = None  # π_k: component weights
        self.mu_ = None      # μ^{(j)}_k: means in each frame [n_frames, n_components, n_features]
        self.sigma_ = None   # Σ^{(j)}_k: covariances in each frame [n_frames, n_components, n_features, n_features]
        self.A_ = None       # Task parameter transformations [n_demos, n_frames, n_features, n_features]
        self.b_ = None       # Task parameter translations [n_demos, n_frames, n_features]
        
        # Fitted data info
        self.n_features_ = None
        self.n_demos_ = None
        self.converged_ = False
        self.log_likelihood_history_ = []
        
    def _initialize_parameters(self, X, A, b):
        """Initialize TPGMM parameters using K-means clustering"""
        self.n_demos_, self.n_features_ = X.shape
        
        # Store task parameters
        self.A_ = A.copy()
        self.b_ = b.copy()
        
        # Initialize with K-means clustering on concatenated data
        print(f"Initializing TPGMM with {self.n_components} components...")
        
        # Transform data to global coordinate system for initialization
        X_global = self._transform_to_global(X, A, b, frame_idx=0)
        
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X_global)
        
        # Initialize priors uniformly
        self.priors_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means and covariances for each frame and component
        self.mu_ = np.zeros((self.n_frames, self.n_components, self.n_features_))
        self.sigma_ = np.zeros((self.n_frames, self.n_components, self.n_features_, self.n_features_))
        
        for j in range(self.n_frames):
            for k in range(self.n_components):
                # Get data points assigned to component k
                mask = labels == k
                if np.sum(mask) == 0:
                    # If no points assigned, use random initialization
                    self.mu_[j, k] = np.random.randn(self.n_features_) * 0.1
                    self.sigma_[j, k] = np.eye(self.n_features_) * 0.1
                else:
                    # Transform relevant data to frame j
                    X_frame = self._transform_to_frame(X_global[mask], A, b, j)
                    
                    # Estimate mean and covariance in frame j
                    self.mu_[j, k] = np.mean(X_frame, axis=0)
                    
                    if len(X_frame) > 1:
                        cov = np.cov(X_frame.T)
                        # Add regularization
                        cov += self.reg_covar * np.eye(self.n_features_)
                        self.sigma_[j, k] = cov
                    else:
                        self.sigma_[j, k] = np.eye(self.n_features_) * 0.1
        
        print(f"Initialization complete. Starting EM algorithm...")
    
    def _transform_to_global(self, X, A, b, frame_idx):
        """Transform data from frame coordinates to global coordinates"""
        # For demo n and frame j: x_global = A[n,j] @ x_frame + b[n,j]
        # Here we assume all demos use the same transformation for simplicity
        return X @ A[0, frame_idx].T + b[0, frame_idx]
    
    def _transform_to_frame(self, X_global, A, b, frame_idx):
        """Transform global coordinates to frame coordinates"""
        # x_frame = A[n,j]^{-1} @ (x_global - b[n,j])
        # For simplicity, using transformation from demo 0
        A_inv = inv(A[0, frame_idx])
        return (X_global - b[0, frame_idx]) @ A_inv.T
    
    def _compute_responsibilities(self, X):
        """
        Compute responsibilities using product of Gaussians across frames
        
        From Calinon paper: Product of Gaussians in different frames
        p(ξ|k) ∝ ∏_j N(ξ; μ^{(j)}_k, Σ^{(j)}_k)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for n in range(n_samples):
            for k in range(self.n_components):
                # Compute product of Gaussians across frames
                log_prob = 0.0
                
                for j in range(self.n_frames):
                    # Transform data point to frame j
                    x_frame = self._transform_to_frame(X[n:n+1], self.A_, self.b_, j)[0]
                    
                    # Compute log probability in frame j
                    try:
                        log_prob += multivariate_normal.logpdf(
                            x_frame, self.mu_[j, k], self.sigma_[j, k]
                        )
                    except:
                        # Handle numerical issues
                        log_prob += -1e10
                
                # Add prior
                responsibilities[n, k] = np.log(self.priors_[k]) + log_prob
        
        # Convert to probabilities and normalize
        max_log_resp = np.max(responsibilities, axis=1, keepdims=True)
        responsibilities = np.exp(responsibilities - max_log_resp)
        
        # Normalize responsibilities
        row_sums = np.sum(responsibilities, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10  # Avoid division by zero
        responsibilities /= row_sums
        
        return responsibilities
    
    def _update_parameters(self, X, responsibilities):
        """Update TPGMM parameters using EM algorithm"""
        n_samples = X.shape[0]
        
        # Update priors
        self.priors_ = np.mean(responsibilities, axis=0)
        self.priors_[self.priors_ < 1e-10] = 1e-10  # Regularization
        self.priors_ /= np.sum(self.priors_)  # Normalize
        
        # Update means and covariances for each frame and component
        for j in range(self.n_frames):
            for k in range(self.n_components):
                resp_k = responsibilities[:, k]
                resp_sum = np.sum(resp_k)
                
                if resp_sum < 1e-10:
                    continue
                
                # Transform data to frame j
                X_frame = np.zeros_like(X)
                for n in range(n_samples):
                    X_frame[n] = self._transform_to_frame(X[n:n+1], self.A_, self.b_, j)[0]
                
                # Update mean
                self.mu_[j, k] = np.sum(resp_k.reshape(-1, 1) * X_frame, axis=0) / resp_sum
                
                # Update covariance
                diff = X_frame - self.mu_[j, k]
                weighted_diff = resp_k.reshape(-1, 1) * diff
                cov = (weighted_diff.T @ diff) / resp_sum
                
                # Add regularization
                cov += self.reg_covar * np.eye(self.n_features_)
                self.sigma_[j, k] = cov
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood of the data"""
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for n in range(n_samples):
            sample_likelihood = 0.0
            
            for k in range(self.n_components):
                component_likelihood = self.priors_[k]
                
                # Product of Gaussians across frames
                for j in range(self.n_frames):
                    x_frame = self._transform_to_frame(X[n:n+1], self.A_, self.b_, j)[0]
                    
                    try:
                        pdf_val = multivariate_normal.pdf(x_frame, self.mu_[j, k], self.sigma_[j, k])
                        component_likelihood *= pdf_val
                    except:
                        component_likelihood *= 1e-10
                
                sample_likelihood += component_likelihood
            
            if sample_likelihood > 0:
                log_likelihood += np.log(sample_likelihood)
            else:
                log_likelihood += -1e10
        
        return log_likelihood
    
    def fit(self, X, A, b):
        """
        Fit TPGMM to data with task parameters
        
        Args:
            X: Data matrix [n_samples, n_features]
            A: Task parameter transformations [n_demos, n_frames, n_features, n_features]
            b: Task parameter translations [n_demos, n_frames, n_features]
        """
        print(f"Fitting TPGMM with {self.n_components} components to {X.shape[0]} samples...")
        
        # Initialize parameters
        self._initialize_parameters(X, A, b)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []
        
        for iteration in range(self.max_iter):
            print(f"EM iteration {iteration + 1}/{self.max_iter}")
            
            # E-step: compute responsibilities
            responsibilities = self._compute_responsibilities(X)
            
            # M-step: update parameters
            self._update_parameters(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(log_likelihood)
            
            # Check convergence
            improvement = log_likelihood - prev_log_likelihood
            print(f"  Log-likelihood: {log_likelihood:.4f}, Improvement: {improvement:.6f}")
            
            if improvement < 1e-6:
                print(f"Converged after {iteration + 1} iterations!")
                self.converged_ = True
                break
            
            prev_log_likelihood = log_likelihood
        
        if not self.converged_:
            print(f"Did not converge after {self.max_iter} iterations")
        
        return self
    
    def predict_proba(self, X):
        """Predict component probabilities for new data"""
        return self._compute_responsibilities(X)
    
    def sample(self, n_samples, task_params):
        """Generate samples from the learned TPGMM"""
        A, b = task_params
        samples = np.zeros((n_samples, self.n_features_))
        
        for i in range(n_samples):
            # Sample component
            k = np.random.choice(self.n_components, p=self.priors_)
            
            # Sample from product of Gaussians (approximation)
            # Use first frame for simplicity
            x_frame = np.random.multivariate_normal(self.mu_[0, k], self.sigma_[0, k])
            
            # Transform to global coordinates
            samples[i] = self._transform_to_global(x_frame.reshape(1, -1), A, b, 0)[0]
        
        return samples
    
    def get_model_info(self):
        """Get information about the fitted model"""
        if self.priors_ is None:
            return "Model not fitted yet"
        
        info = {
            'n_components': self.n_components,
            'n_frames': self.n_frames,
            'n_features': self.n_features_,
            'n_demos': self.n_demos_,
            'converged': self.converged_,
            'final_log_likelihood': self.log_likelihood_history_[-1] if self.log_likelihood_history_ else None,
            'priors': self.priors_
        }
        return info