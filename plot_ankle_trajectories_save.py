import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_gait_data(filepath):
    """Load the gait analysis data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_ankle_data(data):
    """Extract ankle position and velocity data for all frames"""
    frames = ['FR1', 'FR2', 'FR3']
    ankle_data = {
        'positions': {'left': [], 'right': []},
        'velocities': {'left': [], 'right': []}
    }
    
    for frame in frames:
        if frame in data['kinematics_data']:
            # Extract right ankle data
            right_leg = data['kinematics_data'][frame]['right_leg_kinematics'][0]
            right_pos = right_leg['right_ankle_pos']
            right_vel = right_leg['right_ankle_vel']
            
            ankle_data['positions']['right'].append({
                'x': right_pos[0],  # X coordinates
                'y': right_pos[1],  # Y coordinates
                'frame': frame
            })
            ankle_data['velocities']['right'].append({
                'x': right_vel[0],  # X velocities
                'y': right_vel[1],  # Y velocities
                'frame': frame
            })
            
            # Extract left ankle data
            left_pos = right_leg['left_ankle_pos']
            left_vel = right_leg['left_ankle_vel']
            
            ankle_data['positions']['left'].append({
                'x': left_pos[0],  # X coordinates
                'y': left_pos[1],  # Y coordinates
                'frame': frame
            })
            ankle_data['velocities']['left'].append({
                'x': left_vel[0],  # X velocities
                'y': left_vel[1],  # Y velocities
                'frame': frame
            })
    
    return ankle_data

def plot_ankle_positions(ankle_data, save_path="plots/ankle_positions.png"):
    """Plot ankle positions for left and right ankles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['blue', 'red', 'green']
    frame_names = ['FR1', 'FR2', 'FR3']
    
    # Plot left ankle positions
    ax1.set_title('Left Ankle Trajectory (X vs Y Position)', fontsize=16, fontweight='bold')
    for i, frame_data in enumerate(ankle_data['positions']['left']):
        ax1.plot(frame_data['x'], frame_data['y'], 
                color=colors[i], linewidth=2.5, alpha=0.8,
                label=f'{frame_data["frame"]}')
        # Mark start point
        ax1.plot(frame_data['x'][0], frame_data['y'][0], 'o', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
        # Mark end point
        ax1.plot(frame_data['x'][-1], frame_data['y'][-1], 's', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)
    
    # Plot right ankle positions
    ax2.set_title('Right Ankle Trajectory (X vs Y Position)', fontsize=16, fontweight='bold')
    for i, frame_data in enumerate(ankle_data['positions']['right']):
        ax2.plot(frame_data['x'], frame_data['y'], 
                color=colors[i], linewidth=2.5, alpha=0.8,
                label=f'{frame_data["frame"]}')
        # Mark start point
        ax2.plot(frame_data['x'][0], frame_data['y'][0], 'o', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
        # Mark end point
        ax2.plot(frame_data['x'][-1], frame_data['y'][-1], 's', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ankle position plot saved to: {save_path}")

def plot_ankle_velocities(ankle_data, save_path="plots/ankle_velocities.png"):
    """Plot ankle velocities for left and right ankles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['blue', 'red', 'green']
    frame_names = ['FR1', 'FR2', 'FR3']
    
    # Plot left ankle velocities
    ax1.set_title('Left Ankle Velocity (X vs Y Velocity)', fontsize=16, fontweight='bold')
    for i, frame_data in enumerate(ankle_data['velocities']['left']):
        ax1.plot(frame_data['x'], frame_data['y'], 
                color=colors[i], linewidth=2.5, alpha=0.8,
                label=f'{frame_data["frame"]}')
        # Mark start point
        ax1.plot(frame_data['x'][0], frame_data['y'][0], 'o', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
        # Mark end point
        ax1.plot(frame_data['x'][-1], frame_data['y'][-1], 's', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('X Velocity (m/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Velocity (m/s)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)
    
    # Plot right ankle velocities
    ax2.set_title('Right Ankle Velocity (X vs Y Velocity)', fontsize=16, fontweight='bold')
    for i, frame_data in enumerate(ankle_data['velocities']['right']):
        ax2.plot(frame_data['x'], frame_data['y'], 
                color=colors[i], linewidth=2.5, alpha=0.8,
                label=f'{frame_data["frame"]}')
        # Mark start point
        ax2.plot(frame_data['x'][0], frame_data['y'][0], 'o', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
        # Mark end point
        ax2.plot(frame_data['x'][-1], frame_data['y'][-1], 's', 
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('X Velocity (m/s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y Velocity (m/s)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ankle velocity plot saved to: {save_path}")

def main():
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load the data
    filepath = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    data = load_gait_data(filepath)
    
    # Extract ankle data
    ankle_data = extract_ankle_data(data)
    
    # Create plots
    print("Creating ankle position plots...")
    plot_ankle_positions(ankle_data, "plots/ankle_positions_subject35.png")
    
    print("Creating ankle velocity plots...")
    plot_ankle_velocities(ankle_data, "plots/ankle_velocities_subject35.png")
    
    # Print some statistics
    print("\n=== Data Summary ===")
    print(f"Number of frames: {len(ankle_data['positions']['left'])}")
    for i, frame_data in enumerate(ankle_data['positions']['left']):
        print(f"Frame {frame_data['frame']}: {len(frame_data['x'])} data points")
    
    # Print data ranges for each frame
    print("\n=== Position Ranges ===")
    for side in ['left', 'right']:
        print(f"\n{side.capitalize()} ankle positions:")
        for frame_data in ankle_data['positions'][side]:
            x_range = (min(frame_data['x']), max(frame_data['x']))
            y_range = (min(frame_data['y']), max(frame_data['y']))
            print(f"  {frame_data['frame']}: X range: {x_range[0]:.3f} to {x_range[1]:.3f} m, Y range: {y_range[0]:.3f} to {y_range[1]:.3f} m")

if __name__ == "__main__":
    main()