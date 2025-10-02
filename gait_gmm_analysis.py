import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class GaitGMM:
    """
    Gait analysis using Gaussian Mixture Models and Gaussian Mixture Regression
    with configurable reference frames and variables.
    """
    
    def __init__(self, n_components: int = 5):
        """
        Initialize the GaitGMM model.
        
        Args:
            n_components: Number of Gaussian components in the mixture model
        """
        self.n_components = n_components
        self.gmm = None
        self.data = None
        self.reference_frame = 'FR1'
        self.variables = ['position']  # 'position', 'velocity', 'acceleration'
        
    def load_data(self, file_path: str) -> None:
        """Load gait data from JSON file."""
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} demonstrations")
        
    def set_reference_frame(self, frame: str) -> None:
        """
        Set the reference frame to use.
        
        Args:
            frame: Reference frame ('FR1', 'FR2', 'FR3')
        """
        if frame not in ['FR1', 'FR2', 'FR3']:
            raise ValueError("Reference frame must be 'FR1', 'FR2', or 'FR3'")
        self.reference_frame = frame
        print(f"Reference frame set to {frame}")
        
    def set_variables(self, variables: List[str]) -> None:
        """
        Set which variables to use in the model.
        
        Args:
            variables: List of variables to use.
                      Options: ['position'], ['position', 'velocity'], 
                              ['position', 'velocity', 'acceleration']
        """
        valid_vars = ['position', 'velocity', 'acceleration']
        if not all(var in valid_vars for var in variables):
            raise ValueError(f"Variables must be from {valid_vars}")
        self.variables = variables
        print(f"Variables set to {variables}")
        
    def extract_features(self, demo_data: dict) -> np.ndarray:
        """
        Extract features from a single demonstration based on current settings.
        
        Args:
            demo_data: Single demonstration data dictionary
            
        Returns:
            Feature matrix with time and selected ankle variables
        """
        features = []
        
        # Add time (normalized 0-1)
        time = np.array(demo_data['time']).flatten()
        features.append(time)
        
        # Define possible ankle variable patterns
        ankle_patterns = [
            f'ankle_pos_{self.reference_frame}',
            f'right_ankle_pos_{self.reference_frame}', 
            f'left_ankle_pos_{self.reference_frame}'
        ]
        
        # Add position if requested
        if 'position' in self.variables:
            for pattern in ankle_patterns:
                if pattern in demo_data:
                    pos_data = np.array(demo_data[pattern])
                    features.extend([pos_data[:, 0], pos_data[:, 1]])  # x, y coordinates
                    print(f"Added position data from {pattern}")
            
        # Add velocity if requested  
        if 'velocity' in self.variables:
            vel_patterns = [
                f'ankle_pos_{self.reference_frame}_velocity',
                f'right_ankle_vel_{self.reference_frame}',
                f'left_ankle_vel_{self.reference_frame}'
            ]
            for pattern in vel_patterns:
                if pattern in demo_data:
                    vel_data = np.array(demo_data[pattern])
                    features.extend([vel_data[:, 0], vel_data[:, 1]])  # vx, vy
                    print(f"Added velocity data from {pattern}")
                
        # Add acceleration if requested
        if 'acceleration' in self.variables:
            acc_patterns = [
                f'right_ankle_acc_{self.reference_frame}',
                f'left_ankle_acc_{self.reference_frame}'
            ]
            
            # First try direct acceleration data
            found_acc = False
            for pattern in acc_patterns:
                if pattern in demo_data:
                    acc_data = np.array(demo_data[pattern])
                    features.extend([acc_data[:, 0], acc_data[:, 1]])  # ax, ay
                    print(f"Added acceleration data from {pattern}")
                    found_acc = True
            
            # If no direct acceleration, compute from velocity
            if not found_acc:
                vel_patterns = [
                    f'ankle_pos_{self.reference_frame}_velocity',
                    f'right_ankle_vel_{self.reference_frame}',
                    f'left_ankle_vel_{self.reference_frame}'
                ]
                for pattern in vel_patterns:
                    if pattern in demo_data:
                        vel_data = np.array(demo_data[pattern])
                        acc_x = np.gradient(vel_data[:, 0])
                        acc_y = np.gradient(vel_data[:, 1])
                        features.extend([acc_x, acc_y])
                        print(f"Computed acceleration from {pattern}")
                
        return np.column_stack(features)
    
    def prepare_training_data(self) -> np.ndarray:
        """
        Prepare training data from all demonstrations.
        
        Returns:
            Combined feature matrix from all demonstrations
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        all_features = []
        for demo in self.data:
            features = self.extract_features(demo)
            all_features.append(features)
            
        # Concatenate all demonstrations
        training_data = np.vstack(all_features)
        print(f"Training data shape: {training_data.shape}")
        return training_data
        
    def train_gmm(self) -> None:
        """Train the Gaussian Mixture Model."""
        training_data = self.prepare_training_data()
        
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            init_params='kmeans',  # Explicitly use k-means for initialization
            random_state=42
        )
        
        self.gmm.fit(training_data)
        
        # Calculate BIC and AIC scores
        bic_score = self.gmm.bic(training_data)
        aic_score = self.gmm.aic(training_data)
        log_likelihood = self.gmm.score(training_data)
        
        print(f"GMM trained with {self.n_components} components")
        print(f"Log-likelihood: {log_likelihood:.2f}")
        print(f"BIC score: {bic_score:.2f}")
        print(f"AIC score: {aic_score:.2f}")
        
    def optimize_components(self, max_components: int = 15, criterion: str = 'bic') -> int:
        """
        Optimize the number of components using BIC or AIC.
        
        Args:
            max_components: Maximum number of components to test
            criterion: 'bic' or 'aic' for optimization criterion
            
        Returns:
            Optimal number of components
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        training_data = self.prepare_training_data()
        
        scores = []
        components_range = range(3, max_components + 1)  # Start from 3 components
        
        print(f"Optimizing number of components using {criterion.upper()}...")
        
        for n_comp in components_range:
            gmm_temp = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                init_params='kmeans',
                random_state=42
            )
            gmm_temp.fit(training_data)
            
            if criterion == 'bic':
                score = gmm_temp.bic(training_data)
            else:  # aic
                score = gmm_temp.aic(training_data)
                
            scores.append(score)
            print(f"  {n_comp} components: {criterion.upper()} = {score:.2f}")
        
        # Find optimal (minimum BIC/AIC)
        optimal_idx = np.argmin(scores)
        optimal_components = components_range[optimal_idx]
        
        print(f"\nOptimal number of components: {optimal_components}")
        print(f"Best {criterion.upper()} score: {scores[optimal_idx]:.2f}")
        
        # Update the model with optimal components
        self.n_components = optimal_components
        
        return optimal_components
        
    def predict_trajectory(self, time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict ankle trajectory using Gaussian Mixture Regression.
        
        Args:
            time_points: Time points for prediction (normalized 0-1)
            
        Returns:
            Tuple of (predicted_means, predicted_covariances)
        """
        if self.gmm is None:
            raise ValueError("Model not trained. Call train_gmm() first.")
            
        predictions = []
        covariances = []
        
        for t in time_points:
            # Create input with just time
            query_point = np.array([[t] + [0] * (self.gmm.means_.shape[1] - 1)])
            
            # Get responsibilities (posterior probabilities)
            responsibilities = self.gmm.predict_proba(query_point)[0]
            
            # Compute weighted prediction
            weighted_mean = np.zeros(self.gmm.means_.shape[1])
            weighted_cov = np.zeros((self.gmm.means_.shape[1], self.gmm.means_.shape[1]))
            
            for k in range(self.n_components):
                weighted_mean += responsibilities[k] * self.gmm.means_[k]
                weighted_cov += responsibilities[k] * self.gmm.covariances_[k]
                
            predictions.append(weighted_mean)
            covariances.append(weighted_cov)
            
        return np.array(predictions), np.array(covariances)
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot the original data and GMM predictions."""
        if self.data is None or self.gmm is None:
            raise ValueError("Data and model must be loaded and trained first.")
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot original trajectories (two consecutive cycles)
        for demo in self.data:
            features = self.extract_features(demo)
            time = features[:, 0]
            if 'position' in self.variables:
                pos_x = features[:, 1]
                pos_y = features[:, 2]
                
                # First cycle (0-1)
                axes[0, 0].plot(time, pos_x, alpha=0.3, color='blue')
                axes[0, 1].plot(time, pos_y, alpha=0.3, color='blue')
                
                # Second cycle (1-2) - repeat the same data
                axes[0, 0].plot(time + 1, pos_x, alpha=0.3, color='blue')
                axes[0, 1].plot(time + 1, pos_y, alpha=0.3, color='blue')
                
                axes[1, 0].plot(pos_x, pos_y, alpha=0.3, color='blue')
        
        # Generate predictions for two consecutive cycles
        time_pred_single = np.linspace(0, 1, 100)  # Single cycle
        predictions_single, covariances_single = self.predict_trajectory(time_pred_single)
        
        # Create two consecutive cycles
        time_pred_double = np.concatenate([time_pred_single, time_pred_single + 1])
        predictions_double = np.concatenate([predictions_single, predictions_single])
        
        if 'position' in self.variables:
            pred_x = predictions_double[:, 1]
            pred_y = predictions_double[:, 2]
            
            axes[0, 0].plot(time_pred_double, pred_x, 'r-', linewidth=2, label='GMM Prediction')
            axes[0, 1].plot(time_pred_double, pred_y, 'r-', linewidth=2, label='GMM Prediction')
            axes[1, 0].plot(pred_x[:100], pred_y[:100], 'r-', linewidth=2, label='GMM Prediction')
        
        axes[0, 0].set_title('Ankle X Position vs Time')
        axes[0, 0].set_xlabel('Time (normalized)')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Ankle Y Position vs Time')
        axes[0, 1].set_xlabel('Time (normalized)')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Ankle Trajectory (X-Y)')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].legend()
        
        # Plot component means
        axes[1, 1].scatter(self.gmm.means_[:, 0], np.arange(len(self.gmm.means_)), 
                          c='red', s=100, alpha=0.7)
        axes[1, 1].set_title('GMM Component Timing')
        axes[1, 1].set_xlabel('Time (normalized)')
        axes[1, 1].set_ylabel('Component Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

def main():
    """Example usage of the GaitGMM class."""
    # Initialize model
    model = GaitGMM(n_components=5)
    
    # Load data
    data_path = "/home/jemajuinta/ws/Gait-analysis-coupled/TaskPaGMMM/examples/7days1/new_processed_gait_data#39_16.json"
    model.load_data(data_path)
    
    # Configure model
    model.set_reference_frame('FR1')  # Use Frame of Reference 1
    model.set_variables(['position', 'velocity'])  # Use position and velocity
    
    # Optimize number of components using BIC (test 3-15 components)
    optimal_components = model.optimize_components(max_components=15, criterion='bic')
    
    # Train with optimal components and visualize
    model.train_gmm()
    model.plot_results('gait_gmm_results.png')

if __name__ == "__main__":
    main()