#!/usr/bin/env python3
"""
Extract Frame Transformations for TPGMM

This script analyzes the training data to extract translation vectors between
the 3 coordinate frames (FR1, FR2, FR3) used in the TPGMM training.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add TaskPaGMMM to Python path
sys.path.append('TaskPaGMMM')

def load_model_data(model_path):
    """Load the trained TPGMM model data."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def analyze_frame_differences(all_trajectories, feature_names):
    """
    Analyze the differences between frames to extract translation vectors.
    
    Args:
        all_trajectories: List of 3 arrays, each with shape (num_cycles, num_points, num_features)
        feature_names: List of feature names
    
    Returns:
        dict: Translation vectors and analysis results
    """
    print("Analyzing frame differences...")
    
    # Get position feature indices (exclude time)
    pos_indices = [i for i, name in enumerate(feature_names) if 'pos' in name]
    print(f"Position feature indices: {pos_indices}")
    print(f"Position features: {[feature_names[i] for i in pos_indices]}")
    
    # Extract position data for each frame
    frame_positions = {}
    frame_means = {}
    
    for frame_idx, frame_name in enumerate(['FR1', 'FR2', 'FR3']):
        trajectories = all_trajectories[frame_idx]  # Shape: (num_cycles, num_points, num_features)
        
        # Extract position data (exclude time column)
        position_data = trajectories[:, :, pos_indices]  # Shape: (num_cycles, num_points, 4)
        
        # Flatten to get all position points
        flattened_positions = position_data.reshape(-1, len(pos_indices))  # Shape: (num_cycles*num_points, 4)
        
        frame_positions[frame_name] = flattened_positions
        frame_means[frame_name] = np.mean(flattened_positions, axis=0)
        
        print(f"{frame_name}: {flattened_positions.shape[0]} position points")
        print(f"  Mean positions: {frame_means[frame_name]}")
    
    # Calculate translation vectors (using FR1 as reference)
    translations = {}
    translations['FR1'] = np.zeros(len(pos_indices))  # Reference frame
    translations['FR2'] = frame_means['FR2'] - frame_means['FR1']
    translations['FR3'] = frame_means['FR3'] - frame_means['FR1']
    
    print("\nTranslation vectors (relative to FR1):")
    for frame_name, translation in translations.items():
        print(f"{frame_name}: {translation}")
    
    return {
        'translations': translations,
        'frame_positions': frame_positions,
        'frame_means': frame_means,
        'position_indices': pos_indices,
        'position_features': [feature_names[i] for i in pos_indices]
    }

def plot_frame_analysis(analysis_results, save_dir="plots"):
    """Plot frame analysis results."""
    os.makedirs(save_dir, exist_ok=True)
    
    frame_positions = analysis_results['frame_positions']
    frame_means = analysis_results['frame_means']
    translations = analysis_results['translations']
    pos_features = analysis_results['position_features']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'FR1': 'blue', 'FR2': 'red', 'FR3': 'green'}
    
    # Plot right ankle positions
    for frame_name, positions in frame_positions.items():
        color = colors[frame_name]
        
        # Right ankle (features 0, 1 in position data)
        ax1.scatter(positions[:, 0], positions[:, 1], c=color, alpha=0.3, s=1, label=f'{frame_name} data')
        ax1.scatter(frame_means[frame_name][0], frame_means[frame_name][1], 
                   c=color, s=100, marker='o', edgecolor='black', linewidth=2, label=f'{frame_name} mean')
    
    ax1.set_title('Right Ankle Position Distribution by Frame')
    ax1.set_xlabel(f'{pos_features[0]}')
    ax1.set_ylabel(f'{pos_features[1]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot left ankle positions
    for frame_name, positions in frame_positions.items():
        color = colors[frame_name]
        
        # Left ankle (features 2, 3 in position data)
        ax2.scatter(positions[:, 2], positions[:, 3], c=color, alpha=0.3, s=1, label=f'{frame_name} data')
        ax2.scatter(frame_means[frame_name][2], frame_means[frame_name][3], 
                   c=color, s=100, marker='o', edgecolor='black', linewidth=2, label=f'{frame_name} mean')
    
    ax2.set_title('Left Ankle Position Distribution by Frame')
    ax2.set_xlabel(f'{pos_features[2]}')
    ax2.set_ylabel(f'{pos_features[3]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot translation vectors
    frame_names = ['FR1', 'FR2', 'FR3']
    right_translations_x = [translations[name][0] for name in frame_names]
    right_translations_y = [translations[name][1] for name in frame_names]
    left_translations_x = [translations[name][2] for name in frame_names]
    left_translations_y = [translations[name][3] for name in frame_names]
    
    ax3.bar(frame_names, right_translations_x, alpha=0.7, label='Right Ankle X')
    ax3.bar(frame_names, right_translations_y, alpha=0.7, label='Right Ankle Y')
    ax3.set_title('Translation Vectors - Right Ankle')
    ax3.set_ylabel('Translation (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(frame_names, left_translations_x, alpha=0.7, label='Left Ankle X')
    ax4.bar(frame_names, left_translations_y, alpha=0.7, label='Left Ankle Y')
    ax4.set_title('Translation Vectors - Left Ankle')
    ax4.set_ylabel('Translation (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/frame_transformation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Frame analysis plot saved to {save_dir}/frame_transformation_analysis.png")

def main():
    """Main analysis pipeline."""
    print("=== Frame Transformation Extraction ===")
    
    # Load model data
    pkls_dir = "/home/jemajuinta/ws/Gait-analysis-coupled/pkls"
    model_path = os.path.join(pkls_dir, "gait_tpgmm_model_final.pkl")
    
    print(f"Loading model from: {model_path}")
    model_data = load_model_data(model_path)
    
    # Extract frame transformation data
    all_trajectories = model_data['all_trajectories']
    feature_names = model_data['feature_names']
    
    print(f"Number of frames: {len(all_trajectories)}")
    print(f"Features: {feature_names}")
    
    # Analyze frame differences
    analysis_results = analyze_frame_differences(all_trajectories, feature_names)
    
    # Plot analysis
    plot_frame_analysis(analysis_results)
    
    # Save analysis results
    analysis_save_path = os.path.join(pkls_dir, "frame_transformation_analysis.pkl")
    with open(analysis_save_path, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print(f"Analysis results saved to: {analysis_save_path}")
    
    # Print summary
    print("\n=== Translation Summary ===")
    translations = analysis_results['translations']
    pos_features = analysis_results['position_features']
    
    for frame_name, translation in translations.items():
        print(f"{frame_name}:")
        for i, (feat_name, trans_val) in enumerate(zip(pos_features, translation)):
            print(f"  {feat_name}: {trans_val:.6f}")
    
    print("\n=== Analysis Complete ===")
    return analysis_results

if __name__ == "__main__":
    main()