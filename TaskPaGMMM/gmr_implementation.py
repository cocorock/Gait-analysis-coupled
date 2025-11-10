"""
Gaussian Mixture Regression (GMR) Implementation for TPGMM
Based on: Calinon et al., "Robot Learning with Task-parameterized Generative Models", ISRR 2015

This implementation provides regression capabilities for the TPGMM model:
- Conditional probability distributions p(y|x) given input features x
- Task-parameterized regression using multiple reference frames
- Uncertainty estimation through conditional covariances
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

class TaskParameterizedGMR:
    """
    Task-Parameterized Gaussian Mixture Regression
    
    Performs regression using a fitted TPGMM model to predict output variables
    given input variables and task parameters.
    """
    
    def __init__(self, tpgmm_model):
        """
        Initialize GMR with a fitted TPGMM model
        
        Args:
            tpgmm_model: Fitted TPGMM instance
        """
        self.tpgmm = tpgmm_model
        if self.tpgmm.priors_ is None:
            raise ValueError("TPGMM model must be fitted before using for regression")
    
    def predict(self, X_input, input_dims, output_dims, task_params, return_std=False):
        """
        Perform Task-Parameterized GMR
        
        Args:
            X_input: Input data [n_samples, len(input_dims)]
            input_dims: Indices of input dimensions
            output_dims: Indices of output dimensions  
            task_params: Tuple (A, b) of task parameter transformations and translations
            return_std: Whether to return prediction uncertainties
            
        Returns:
            Y_pred: Predicted outputs [n_samples, len(output_dims)]
            Y_std: Prediction uncertainties (if return_std=True)
        """
        A, b = task_params
        n_samples = X_input.shape[0]
        n_output = len(output_dims)
        n_input = len(input_dims)
        
        print(f"Performing TPGMR for {n_samples} samples...")
        print(f"Input dims: {input_dims}, Output dims: {output_dims}")
        
        # Initialize outputs
        Y_pred = np.zeros((n_samples, n_output))
        if return_std:
            Y_std = np.zeros((n_samples, n_output))
        
        for i in range(n_samples):
            x_input = X_input[i]
            
            # Compute activation weights for each component using product of Gaussians
            activations = self._compute_activations(x_input, input_dims, task_params)
            
            # Weighted conditional means and covariances
            weighted_means = []
            weighted_covs = []
            
            for k in range(self.tpgmm.n_components):
                if activations[k] < 1e-10:
                    continue
                
                # Compute conditional distribution for component k across frames
                mu_cond, sigma_cond = self._compute_conditional_distribution(
                    k, x_input, input_dims, output_dims, task_params
                )
                
                weighted_means.append(activations[k] * mu_cond)
                weighted_covs.append(activations[k]**2 * sigma_cond)
            
            if weighted_means:
                # Combine weighted predictions
                Y_pred[i] = np.sum(weighted_means, axis=0)
                
                if return_std:
                    # Combine weighted covariances
                    combined_cov = np.sum(weighted_covs, axis=0)
                    Y_std[i] = np.sqrt(np.diag(combined_cov))
            else:
                # Fallback to zero prediction
                Y_pred[i] = np.zeros(n_output)
                if return_std:
                    Y_std[i] = np.ones(n_output)
        
        print(f"TPGMR prediction complete.")
        
        if return_std:
            return Y_pred, Y_std
        return Y_pred
    
    def _compute_activations(self, x_input, input_dims, task_params):
        """
        Compute activation weights for each component using product of Gaussians
        
        Following Calinon paper: h_k(ξ_I) = p(k|ξ_I) using product of Gaussians
        """
        A, b = task_params
        activations = np.zeros(self.tpgmm.n_components)
        
        for k in range(self.tpgmm.n_components):
            log_prob = np.log(self.tpgmm.priors_[k])
            
            # Product of Gaussians across frames (marginalized over input dimensions)
            for j in range(self.tpgmm.n_frames):
                # Transform input to frame j
                x_full_frame = self._transform_input_to_frame(x_input, input_dims, j, task_params)
                
                # Extract input part of mean and covariance for frame j
                mu_input = self.tpgmm.mu_[j, k, input_dims]
                sigma_input = self.tpgmm.sigma_[j, k][:, :]
                sigma_input = sigma_input[np.ix_(input_dims, input_dims)]
                
                # Add regularization to avoid numerical issues
                sigma_input += 1e-6 * np.eye(len(input_dims))
                
                try:
                    # Compute log probability of input in frame j
                    diff = x_full_frame[input_dims] - mu_input
                    log_prob += -0.5 * (diff.T @ inv(sigma_input) @ diff + 
                                      len(input_dims) * np.log(2 * np.pi) + 
                                      np.log(np.linalg.det(sigma_input)))
                except:
                    log_prob += -1e10  # Handle numerical issues
            
            activations[k] = log_prob
        
        # Convert to probabilities and normalize
        max_log_prob = np.max(activations)
        activations = np.exp(activations - max_log_prob)
        activation_sum = np.sum(activations)
        
        if activation_sum > 0:
            activations /= activation_sum
        else:
            activations = np.ones(self.tpgmm.n_components) / self.tpgmm.n_components
        
        return activations
    
    def _transform_input_to_frame(self, x_input, input_dims, frame_idx, task_params):
        """Transform input data to specified frame coordinates"""
        A, b = task_params
        n_features = self.tpgmm.n_features_
        
        # Create full feature vector (fill non-input dims with zeros for transformation)
        x_full = np.zeros(n_features)
        x_full[input_dims] = x_input
        
        # Transform to frame coordinates
        # x_frame = A^{-1} @ (x_global - b)
        A_inv = inv(A[0, frame_idx])  # Use first demo's transformation
        x_frame = A_inv @ (x_full - b[0, frame_idx])
        
        return x_frame
    
    def _compute_conditional_distribution(self, component_k, x_input, input_dims, output_dims, task_params):
        """
        Compute conditional distribution p(y|x, k) for component k across frames
        
        Uses the conditional Gaussian formula with product of Gaussians
        """
        A, b = task_params
        n_output = len(output_dims)
        
        # Initialize accumulators for precision-weighted combination
        precision_sum = np.zeros((n_output, n_output))
        precision_mean_sum = np.zeros(n_output)
        
        for j in range(self.tpgmm.n_frames):
            # Transform input to frame j
            x_frame = self._transform_input_to_frame(x_input, input_dims, j, task_params)
            
            # Extract frame j parameters for component k
            mu = self.tpgmm.mu_[j, component_k]
            sigma = self.tpgmm.sigma_[j, component_k]
            
            # Partition mean and covariance
            mu_I = mu[input_dims]   # Input mean
            mu_O = mu[output_dims]  # Output mean
            
            sigma_II = sigma[np.ix_(input_dims, input_dims)]
            sigma_IO = sigma[np.ix_(input_dims, output_dims)]
            sigma_OI = sigma[np.ix_(output_dims, input_dims)]
            sigma_OO = sigma[np.ix_(output_dims, output_dims)]
            
            # Add regularization
            sigma_II += 1e-6 * np.eye(len(input_dims))
            sigma_OO += 1e-6 * np.eye(n_output)
            
            try:
                # Conditional mean and covariance for frame j
                sigma_II_inv = inv(sigma_II)
                
                # Conditional mean: μ_O + Σ_OI @ Σ_II^{-1} @ (x_I - μ_I)
                mu_cond_j = mu_O + sigma_OI @ sigma_II_inv @ (x_frame[input_dims] - mu_I)
                
                # Conditional covariance: Σ_OO - Σ_OI @ Σ_II^{-1} @ Σ_IO
                sigma_cond_j = sigma_OO - sigma_OI @ sigma_II_inv @ sigma_IO
                sigma_cond_j += 1e-6 * np.eye(n_output)  # Regularization
                
                # Add to precision-weighted combination
                precision_j = inv(sigma_cond_j)
                precision_sum += precision_j
                precision_mean_sum += precision_j @ mu_cond_j
                
            except:
                # Handle numerical issues - use uninformative prior
                continue
        
        # Combine information from all frames
        if np.linalg.det(precision_sum) > 1e-10:
            try:
                sigma_combined = inv(precision_sum)
                mu_combined = sigma_combined @ precision_mean_sum
            except:
                # Fallback
                sigma_combined = np.eye(n_output) * 0.1
                mu_combined = np.zeros(n_output)
        else:
            # Fallback for numerical issues
            sigma_combined = np.eye(n_output) * 0.1
            mu_combined = np.zeros(n_output)
        
        return mu_combined, sigma_combined
    
    def predict_trajectory(self, time_points, input_dims, output_dims, task_params, return_std=False):
        """
        Predict a complete trajectory given time points
        
        Args:
            time_points: Array of time values [n_timesteps]
            input_dims: Dimensions to use as input (typically time dimension)
            output_dims: Dimensions to predict
            task_params: Task parameter transformations
            return_std: Whether to return uncertainties
            
        Returns:
            Trajectory predictions and optionally uncertainties
        """
        # Prepare input data
        X_input = time_points.reshape(-1, 1)
        
        # Perform prediction
        if return_std:
            Y_pred, Y_std = self.predict(X_input, input_dims, output_dims, task_params, return_std=True)
            return time_points, Y_pred, Y_std
        else:
            Y_pred = self.predict(X_input, input_dims, output_dims, task_params, return_std=False)
            return time_points, Y_pred
    
    def evaluate_prediction_quality(self, X_test, Y_test, input_dims, output_dims, task_params):
        """
        Evaluate prediction quality on test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        Y_pred = self.predict(X_test, input_dims, output_dims, task_params)
        
        # Compute metrics
        mse = np.mean((Y_test - Y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(Y_test - Y_pred))
        
        # R-squared for each output dimension
        r2_scores = []
        for i in range(Y_test.shape[1]):
            ss_res = np.sum((Y_test[:, i] - Y_pred[:, i])**2)
            ss_tot = np.sum((Y_test[:, i] - np.mean(Y_test[:, i]))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_scores.append(r2)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores)
        }