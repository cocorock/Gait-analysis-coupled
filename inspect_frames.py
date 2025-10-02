#!/usr/bin/env python3
"""
Script to inspect frame transformations in the trained TPGMM model
"""

import pickle
import numpy as np
import sys

# Add TaskPaGMMM to Python path
sys.path.append('TaskPaGMMM')

def inspect_model_frames():
    """Inspect the frame transformations stored in the trained model."""
    
    # Load the trained model
    with open('pkls/gait_tpgmm_model_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print("=== MODEL INSPECTION ===")
    print(f"Model keys: {list(model_data.keys())}")
    print()
    
    # Get trajectories from all frames
    all_trajectories = model_data['all_trajectories']
    feature_names = model_data['feature_names']
    
    print(f"Number of frames: {len(all_trajectories)}")
    print(f"Feature names: {feature_names}")
    print()
    
    # Analyze frame differences
    print("=== FRAME ANALYSIS ===")
    
    # For gait data, we need to understand what the 3 frames represent
    # Let's look at the first trajectory from each frame to understand the transformations
    for frame_idx in range(len(all_trajectories)):
        frame_data = all_trajectories[frame_idx]
        first_trajectory = frame_data[0]  # First trajectory in this frame
        
        print(f"Frame {frame_idx} (FR{frame_idx+1}):")
        print(f"  Shape: {first_trajectory.shape}")
        print(f"  Time range: {first_trajectory[:, 0].min():.3f} to {first_trajectory[:, 0].max():.3f}")
        
        # Look at the first and last points (potential reference points)
        start_point = first_trajectory[0, 1:]  # Exclude time
        end_point = first_trajectory[-1, 1:]   # Exclude time
        
        print(f"  Start point: {start_point}")
        print(f"  End point: {end_point}")
        print()
    
    # Calculate potential frame transformations
    print("=== FRAME TRANSFORMATION ANALYSIS ===")
    
    # Get reference points from the first trajectory of each frame
    frame_references = []
    for frame_idx in range(len(all_trajectories)):
        first_traj = all_trajectories[frame_idx][0]
        # Use start and end points as potential references
        start_ref = first_traj[0, 1:]  # Exclude time
        end_ref = first_traj[-1, 1:]   # Exclude time
        frame_references.append({'start': start_ref, 'end': end_ref})
    
    # For gait analysis, the frames might represent:
    # FR1: Original/Global frame
    # FR2: Referenced to start of gait cycle
    # FR3: Referenced to end of gait cycle
    
    # Calculate translations relative to FR1
    fr1_start = frame_references[0]['start']
    fr1_end = frame_references[0]['end']
    
    print("Reference points from FR1 (first trajectory):")
    print(f"  Start: {fr1_start}")
    print(f"  End: {fr1_end}")
    print()
    
    # Calculate what transformations would align other frames to FR1
    print("Estimated transformations to align frames to FR1:")
    for frame_idx in range(len(all_trajectories)):
        if frame_idx == 0:
            print(f"FR{frame_idx+1}: Identity (reference frame)")
            continue
            
        # For gait data, frames are likely translated versions
        # Let's calculate the translation that would align this frame to FR1
        frame_start = frame_references[frame_idx]['start'] 
        frame_end = frame_references[frame_idx]['end']
        
        # Translation from this frame to FR1
        translation_to_fr1 = fr1_start - frame_start
        
        print(f"FR{frame_idx+1}: Translation = {translation_to_fr1}")
    
    return model_data, frame_references

if __name__ == "__main__":
    model_data, frame_refs = inspect_model_frames()