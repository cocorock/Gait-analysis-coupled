import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_gait_data(filepath):
    """Load the gait analysis data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_all_ankle_data(data):
    """Extract ankle position and velocity data for all frames and all trajectories"""
    frames = ['FR1', 'FR2', 'FR3']
    ankle_data = {
        'positions': {'left': {}, 'right': {}},
        'velocities': {'left': {}, 'right': {}}
    }
    
    for frame in frames:
        if frame in data['kinematics_data']:
            ankle_data['positions']['left'][frame] = []
            ankle_data['positions']['right'][frame] = []
            ankle_data['velocities']['left'][frame] = []
            ankle_data['velocities']['right'][frame] = []
            
            # Extract data for all trajectories in this frame
            trajectories = data['kinematics_data'][frame]['right_leg_kinematics']
            
            for traj_idx, trajectory in enumerate(trajectories):
                # Right ankle data
                right_pos = trajectory['right_ankle_pos']
                right_vel = trajectory['right_ankle_vel']
                
                ankle_data['positions']['right'][frame].append({
                    'x': right_pos[0],  # X coordinates
                    'y': right_pos[1],  # Y coordinates
                    'trajectory': traj_idx + 1
                })
                ankle_data['velocities']['right'][frame].append({
                    'x': right_vel[0],  # X velocities
                    'y': right_vel[1],  # Y velocities
                    'trajectory': traj_idx + 1
                })
                
                # Left ankle data
                left_pos = trajectory['left_ankle_pos']
                left_vel = trajectory['left_ankle_vel']
                
                ankle_data['positions']['left'][frame].append({
                    'x': left_pos[0],  # X coordinates
                    'y': left_pos[1],  # Y coordinates
                    'trajectory': traj_idx + 1
                })
                ankle_data['velocities']['left'][frame].append({
                    'x': left_vel[0],  # X velocities
                    'y': left_vel[1],  # Y velocities
                    'trajectory': traj_idx + 1
                })
    
    return ankle_data

def plot_ankle_positions(ankle_data, save_path="plots/ankle_positions.png"):
    """Plot ankle positions for left and right ankles, all trajectories"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Colors for different frames
    frame_colors = {'FR1': 'blue', 'FR2': 'red', 'FR3': 'green'}
    
    # Plot left ankle positions
    ax1.set_title('Left Ankle Trajectories (X vs Y Position)\nAll 11 Trajectories per Frame', 
                  fontsize=16, fontweight='bold')
    
    for frame in ['FR1', 'FR2', 'FR3']:
        trajectories = ankle_data['positions']['left'][frame]
        for traj_idx, traj_data in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.9  # Highlight first trajectory
            linewidth = 3 if traj_idx == 0 else 1.5
            label = f'{frame}' if traj_idx == 0 else ""
            
            ax1.plot(traj_data['x'], traj_data['y'], 
                    color=frame_colors[frame], linewidth=linewidth, alpha=alpha,
                    label=label)
            
            # Mark start point for first trajectory only
            if traj_idx == 0:
                ax1.plot(traj_data['x'][0], traj_data['y'][0], 'o', 
                        color=frame_colors[frame], markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)
    
    # Plot right ankle positions
    ax2.set_title('Right Ankle Trajectories (X vs Y Position)\nAll 11 Trajectories per Frame', 
                  fontsize=16, fontweight='bold')
    
    for frame in ['FR1', 'FR2', 'FR3']:
        trajectories = ankle_data['positions']['right'][frame]
        for traj_idx, traj_data in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.9  # Highlight first trajectory
            linewidth = 3 if traj_idx == 0 else 1.5
            label = f'{frame}' if traj_idx == 0 else ""
            
            ax2.plot(traj_data['x'], traj_data['y'], 
                    color=frame_colors[frame], linewidth=linewidth, alpha=alpha,
                    label=label)
            
            # Mark start point for first trajectory only
            if traj_idx == 0:
                ax2.plot(traj_data['x'][0], traj_data['y'][0], 'o', 
                        color=frame_colors[frame], markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
    
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
    """Plot ankle velocities for left and right ankles, all trajectories with different line styles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Colors and line styles for different frames
    frame_styles = {
        'FR1': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5},
        'FR2': {'color': 'red', 'linestyle': '--', 'linewidth': 2.5},
        'FR3': {'color': 'green', 'linestyle': '-.', 'linewidth': 2.5}
    }
    
    # Plot left ankle velocities
    ax1.set_title('Left Ankle Velocity Trajectories (X vs Y Velocity)\nAll 11 Trajectories per Frame', 
                  fontsize=16, fontweight='bold')
    
    for frame in ['FR1', 'FR2', 'FR3']:
        trajectories = ankle_data['velocities']['left'][frame]
        style = frame_styles[frame]
        
        for traj_idx, traj_data in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.9  # Highlight first trajectory
            linewidth = style['linewidth'] if traj_idx == 0 else style['linewidth'] * 0.6
            label = f'{frame}' if traj_idx == 0 else ""
            
            ax1.plot(traj_data['x'], traj_data['y'], 
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=linewidth, alpha=alpha, label=label)
            
            # Mark start point for first trajectory only
            if traj_idx == 0:
                ax1.plot(traj_data['x'][0], traj_data['y'][0], 'o', 
                        color=style['color'], markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('X Velocity (m/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y Velocity (m/s)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)
    
    # Plot right ankle velocities
    ax2.set_title('Right Ankle Velocity Trajectories (X vs Y Velocity)\nAll 11 Trajectories per Frame', 
                  fontsize=16, fontweight='bold')
    
    for frame in ['FR1', 'FR2', 'FR3']:
        trajectories = ankle_data['velocities']['right'][frame]
        style = frame_styles[frame]
        
        for traj_idx, traj_data in enumerate(trajectories):
            alpha = 0.6 if traj_idx > 0 else 0.9  # Highlight first trajectory
            linewidth = style['linewidth'] if traj_idx == 0 else style['linewidth'] * 0.6
            label = f'{frame}' if traj_idx == 0 else ""
            
            ax2.plot(traj_data['x'], traj_data['y'], 
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=linewidth, alpha=alpha, label=label)
            
            # Mark start point for first trajectory only
            if traj_idx == 0:
                ax2.plot(traj_data['x'][0], traj_data['y'][0], 'o', 
                        color=style['color'], markersize=8, 
                        markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('X Velocity (m/s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y Velocity (m/s)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ankle velocity plot saved to: {save_path}")

def create_legend_plot(save_path="plots/legend_explanation.png"):
    """Create a separate plot explaining the legend and line styles"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.9, 'Legend and Line Style Explanation', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Frame reference explanation
    ax.text(0.1, 0.75, 'Frame References:', fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.68, '• FR1: Frame Reference 1 (Blue)', fontsize=14, color='blue', transform=ax.transAxes)
    ax.text(0.1, 0.62, '• FR2: Frame Reference 2 (Red)', fontsize=14, color='red', transform=ax.transAxes)
    ax.text(0.1, 0.56, '• FR3: Frame Reference 3 (Green)', fontsize=14, color='green', transform=ax.transAxes)
    
    # Line styles for velocities
    ax.text(0.1, 0.42, 'Velocity Plot Line Styles:', fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.plot([0.1, 0.25], [0.35, 0.35], 'b-', linewidth=3, transform=ax.transAxes)
    ax.text(0.27, 0.34, 'FR1: Solid line', fontsize=14, transform=ax.transAxes)
    
    ax.plot([0.1, 0.25], [0.28, 0.28], 'r--', linewidth=3, transform=ax.transAxes)
    ax.text(0.27, 0.27, 'FR2: Dashed line', fontsize=14, transform=ax.transAxes)
    
    ax.plot([0.1, 0.25], [0.21, 0.21], 'g-.', linewidth=3, transform=ax.transAxes)
    ax.text(0.27, 0.20, 'FR3: Dash-dot line', fontsize=14, transform=ax.transAxes)
    
    # Data explanation
    ax.text(0.1, 0.07, 'Each frame contains 11 gait cycle trajectories\nThicker/brighter lines represent the first trajectory in each frame', 
            fontsize=12, transform=ax.transAxes)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Legend explanation saved to: {save_path}")

def main():
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load the data
    filepath = "/home/jemajuinta/ws/Gait-analysis-coupled/alpha/Gait Data/4D/gait_analysis_export_subject35.json"
    data = load_gait_data(filepath)
    
    # Extract ankle data
    ankle_data = extract_all_ankle_data(data)
    
    # Create plots
    print("Creating ankle position plots...")
    plot_ankle_positions(ankle_data, "plots/ankle_positions_all_trajectories.png")
    
    print("Creating ankle velocity plots...")
    plot_ankle_velocities(ankle_data, "plots/ankle_velocities_all_trajectories.png")
    
    print("Creating legend explanation...")
    create_legend_plot("plots/legend_explanation.png")
    
    # Print summary
    print("\n=== Data Summary ===")
    for frame in ['FR1', 'FR2', 'FR3']:
        num_trajectories = len(ankle_data['positions']['left'][frame])
        num_points = len(ankle_data['positions']['left'][frame][0]['x'])
        print(f"{frame}: {num_trajectories} trajectories, {num_points} points each")
    
    print(f"\nAll plots saved in the 'plots/' directory")

if __name__ == "__main__":
    main()