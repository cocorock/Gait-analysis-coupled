"""
Fixed TPGMM Adaptation with Time-Varying Frame Transformations
==============================================================
This corrected version implements time-varying influence of reference frames.

Key changes:
1. FR2 and FR3 transformations are now time-dependent
2. At t=0, frames have minimal influence (trajectory starts at body)
3. At t=1, frames fully define the endpoint (foot reaches target)
"""

import numpy as np

class OptimizedGMR_TimeVarying:
    """GMR with time-varying frame transformations"""
    def __init__(self, tpgmm_model):
        self.model = tpgmm_model
        self.K = tpgmm_model.K
        self.P = tpgmm_model.P
        self.D = tpgmm_model.D
    
    def predict(self, X_query, input_dims, output_dims, A_frames_base, b_frames_base, 
                frame_weights=None, time_dim=10):
        """
        GMR prediction with time-varying frame transformations
        
        Parameters:
        -----------
        X_query : array (n_query, n_input)
            Query points (typically time)
        input_dims : list
            Input dimensions (typically [10] for time)
        output_dims : list
            Output dimensions (typically [0-9] for positions/velocities)
        A_frames_base : list of arrays
            Base rotation matrices for each frame
        b_frames_base : list of arrays
            Base translation vectors for each frame
        frame_weights : array, optional
            Time-varying weights for each frame (P x n_query)
            If None, uses default time-based weighting
        time_dim : int
            Dimension index for time (default: 10)
        """
        n_query = X_query.shape[0]
        n_out = len(output_dims)
        
        # Extract time values (normalized 0-1)
        time_values = X_query[:, 0] if X_query.shape[1] == 1 else X_query[:, time_dim]
        
        # Define time-varying weights if not provided
        if frame_weights is None:
            frame_weights = self._default_frame_weights(time_values)
        
        # GMR for each query point
        mu_out = np.zeros((n_query, n_out))
        sigma_out = np.zeros((n_query, n_out, n_out))
        
        for t in range(n_query):
            x_in = X_query[t]
            current_time = time_values[t]
            
            # Get time-varying transformations for this time step
            A_frames_t, b_frames_t = self._get_time_varying_transforms(
                A_frames_base, b_frames_base, frame_weights[:, t], current_time
            )
            
            # Product of Gaussians across frames (with time-varying transforms)
            mu_prod = np.zeros((self.K, self.D))
            Sigma_prod_inv = np.zeros((self.K, self.D, self.D))
            
            for k in range(self.K):
                Sigma_inv_sum = np.zeros((self.D, self.D))
                weighted_mu_sum = np.zeros(self.D)
                
                for p in range(self.P):
                    A = A_frames_t[p]
                    b = b_frames_t[p]
                    
                    # Transform to global frame
                    mu_global = A @ self.model.means[p, k] + b
                    Sigma_global = A @ self.model.covars[p, k] @ A.T
                    
                    # Add regularization for numerical stability
                    Sigma_global = Sigma_global + 1e-6 * np.eye(self.D)
                    
                    Sigma_inv = np.linalg.inv(Sigma_global)
                    Sigma_inv_sum += Sigma_inv
                    weighted_mu_sum += Sigma_inv @ mu_global
                
                Sigma_prod_inv[k] = Sigma_inv_sum
                mu_prod[k] = np.linalg.solve(Sigma_inv_sum + 1e-6 * np.eye(self.D), 
                                            weighted_mu_sum)
            
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
                Sigma_k = np.linalg.inv(Sigma_prod_inv[k] + 1e-6 * np.eye(self.D))
                
                Sigma_in = Sigma_k[np.ix_(input_dims, input_dims)]
                Sigma_out_in = Sigma_k[np.ix_(output_dims, input_dims)]
                Sigma_out_out = Sigma_k[np.ix_(output_dims, output_dims)]
                Sigma_in_inv = np.linalg.inv(Sigma_in + 1e-6 * np.eye(len(input_dims)))
                
                mu_out_k = mu_k[output_dims] + Sigma_out_in @ Sigma_in_inv @ (x_in - mu_k[input_dims])
                Sigma_out_k = Sigma_out_out - Sigma_out_in @ Sigma_in_inv @ Sigma_out_in.T
                
                mu_out[t] += h[k] * mu_out_k
                sigma_out[t] += h[k] * (Sigma_out_k + np.outer(mu_out_k, mu_out_k))
            
            sigma_out[t] -= np.outer(mu_out[t], mu_out[t])
        
        return mu_out, sigma_out
    
    def _default_frame_weights(self, time_values):
        """
        Default time-varying weights for frames
        
        For gait:
        - FR1 (body): Constant weight throughout
        - FR2 (right target): Increases from 0 to 1 as time progresses
        - FR3 (left target): Increases from 0 to 1 as time progresses
        
        Returns:
        --------
        weights : array (P, n_query)
            Frame weights for each time point
        """
        n_query = len(time_values)
        weights = np.zeros((self.P, n_query))
        
        for t_idx, t in enumerate(time_values):
            # FR1 (body frame): Always active
            weights[0, t_idx] = 1.0
            
            # FR2 (right foot target): Increases with time
            # Use smooth transition (e.g., sigmoid or polynomial)
            weights[1, t_idx] = t**2  # Quadratic increase
            
            # FR3 (left foot target): Similar to FR2
            weights[2, t_idx] = t**2
        
        return weights
    
    def _get_time_varying_transforms(self, A_frames_base, b_frames_base, 
                                    weights, current_time):
        """
        Apply time-varying scaling to frame transformations
        
        Parameters:
        -----------
        A_frames_base : list
            Base rotation matrices
        b_frames_base : list
            Base translation vectors
        weights : array (P,)
            Frame weights at current time
        current_time : float
            Current time value (0-1)
        
        Returns:
        --------
        A_frames_t : list
            Time-adjusted rotation matrices
        b_frames_t : list
            Time-adjusted translation vectors
        """
        A_frames_t = []
        b_frames_t = []
        
        for p in range(self.P):
            if p == 0:
                # FR1 (body): No time scaling needed
                A_frames_t.append(A_frames_base[p])
                b_frames_t.append(b_frames_base[p])
            else:
                # FR2 and FR3: Apply time-varying scaling to translation
                A_frames_t.append(A_frames_base[p])
                
                # Scale translation based on time
                # At t=0: minimal translation (foot at body)
                # At t=1: full translation (foot at target)
                b_scaled = b_frames_base[p] * weights[p]
                b_frames_t.append(b_scaled)
        
        return A_frames_t, b_frames_t


def generate_trajectories_with_time_varying_adaptation(
    model, fr2_x_values, fr2_y_values=None, fr2_theta_values=None):
    """
    Generate trajectories with time-varying FR2 adaptation
    """
    print("\n" + "="*70)
    print("GENERATING TRAJECTORIES WITH TIME-VARYING ADAPTATION")
    print("="*70)
    
    # Time query points
    n_query = 200
    time_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    # Input and output dimensions
    input_dims = [10]  # Time
    output_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All other dimensions
    
    # Use the time-varying GMR
    gmr = OptimizedGMR_TimeVarying(model)
    
    # Create transformation matrices (same as before)
    from your_utils import create_transformation_matrix_2d
    
    # 1. BASELINE (No Adaptation)
    print("\n1. Generating baseline trajectory...")
    A_FR1, b_FR1 = create_transformation_matrix_2d(0, 0, 0)
    A_FR2, b_FR2 = create_transformation_matrix_2d(0, 0, 0)
    A_FR3, b_FR3 = create_transformation_matrix_2d(0, 0, 0)
    
    A_frames_baseline = [A_FR1, A_FR2, A_FR3]
    b_frames_baseline = [b_FR1, b_FR2, b_FR3]
    
    baseline_gmr, _ = gmr.predict(
        time_query, input_dims, output_dims, 
        A_frames_baseline, b_frames_baseline
    )
    
    # 2. ADAPTED TRAJECTORIES - X-POSITION
    print(f"\n2. Generating adapted trajectories (X-position)...")
    adapted_trajectories_x = []
    
    for fr2_x in fr2_x_values:
        # FR1 and FR3 unchanged
        A_FR1_adapt, b_FR1_adapt = create_transformation_matrix_2d(0, 0, 0)
        A_FR3_adapt, b_FR3_adapt = create_transformation_matrix_2d(0, 0, 0)
        
        # FR2 with x-displacement
        A_FR2_adapt, b_FR2_adapt = create_transformation_matrix_2d(fr2_x, 0, 0)
        
        A_frames_adapt = [A_FR1_adapt, A_FR2_adapt, A_FR3_adapt]
        b_frames_adapt = [b_FR1_adapt, b_FR2_adapt, b_FR3_adapt]
        
        # Generate with time-varying adaptation
        adapted_gmr, _ = gmr.predict(
            time_query, input_dims, output_dims,
            A_frames_adapt, b_frames_adapt
        )
        
        adapted_trajectories_x.append(adapted_gmr)
        
        print(f"   FR2_x = {fr2_x:+.3f}m → Generated")
        
        # Verify: Initial point should be same, endpoint different
        print(f"      Initial point: {adapted_gmr[0, :2]}")
        print(f"      Final point: {adapted_gmr[-1, :2]}")
    
    return baseline_gmr, adapted_trajectories_x


# Example of how to integrate this into your existing code:
if __name__ == "__main__":
    print("\nDemonstration of time-varying vs time-constant transformations:")
    print("-" * 60)
    
    # Create dummy data for illustration
    time_points = np.linspace(0, 1, 5)
    fr2_translation = np.array([0.5, 0.0])  # 0.5m horizontal displacement
    
    print("\nTime-Constant Transformation (current implementation):")
    print("FR2 translation applied uniformly at all time steps")
    for t in time_points:
        displaced_pos = fr2_translation  # Same displacement at all times
        print(f"  t={t:.2f}: displacement = [{displaced_pos[0]:.2f}, {displaced_pos[1]:.2f}]")
    
    print("\nTime-Varying Transformation (corrected implementation):")
    print("FR2 translation scaled by time (t^2 weighting)")
    for t in time_points:
        weight = t**2  # Quadratic increase
        displaced_pos = fr2_translation * weight
        print(f"  t={t:.2f}: displacement = [{displaced_pos[0]:.2f}, {displaced_pos[1]:.2f}]")
    
    print("\nKey difference:")
    print("- At t=0.0: No displacement (foot starts at body)")
    print("- At t=1.0: Full displacement (foot reaches target)")
    print("-" * 60)
