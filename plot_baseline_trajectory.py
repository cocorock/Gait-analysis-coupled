"""
BASELINE TRAJECTORY VISUALIZATION SCRIPT
=========================================

This script loads the 11D baseline trajectory from CSV and creates
comprehensive visualizations:

1. Right ankle trajectory (position x vs y)
2. Left ankle trajectory (position x vs y)
3. Both ankle trajectories comparison (x vs y)
4. Right ankle angle vs time
5. Left ankle angle vs time
6. Ankle angles comparison vs time
7. Right ankle velocity trajectory (vel_x vs vel_y)
8. Left ankle velocity trajectory (vel_x vs vel_y)

Author: Victor
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Set numpy print options
np.set_printoptions(precision=4, suppress=True)


def load_baseline_trajectory(csv_filepath):
    """
    Load baseline trajectory from CSV file
    
    Parameters:
    -----------
    csv_filepath : str
        Path to the baseline_trajectory_11D.csv file
    
    Returns:
    --------
    data : dict
        Dictionary containing all trajectory dimensions
    """
    print(f"\nLoading baseline trajectory from: {csv_filepath}")
    
    # First, read the header to see what columns we have
    with open(csv_filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print(f"✓ CSV columns found: {header}")
    
    # Read all data
    with open(csv_filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    if len(rows) == 0:
        raise ValueError("CSV file is empty!")
    
    # Initialize data dictionary with actual column names from CSV
    data = {}
    for col in header:
        data[col] = []
    
    # Fill data
    for row in rows:
        for col in header:
            try:
                value = float(row[col])
                data[col].append(value)
            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Could not parse value '{row.get(col, 'MISSING')}' in column '{col}'. Setting to 0.0")
                data[col].append(0.0)
    
    # Convert lists to numpy arrays
    for key in data.keys():
        data[key] = np.array(data[key])
    
    print(f"✓ Loaded {len(data[header[0]])} data points")
    print(f"✓ Time range: {data[header[0]][0]:.4f} to {data[header[0]][-1]:.4f}")
    
    # Create standardized aliases for easier access
    # Map actual column names to expected names
    standardized_data = {}
    
    # Try to map columns intelligently
    for key, value in data.items():
        key_lower = key.lower()
        standardized_data[key] = value  # Keep original
        
        # Create aliases
        if 'time' in key_lower and 'normalized' in key_lower:
            standardized_data['normalized_time'] = value
        elif 'right_ankle_pos_x' in key_lower or key == 'right_ankle_pos_x':
            standardized_data['right_ankle_pos_x'] = value
        elif 'right_ankle_pos_y' in key_lower or key == 'right_ankle_pos_y':
            standardized_data['right_ankle_pos_y'] = value
        elif 'right_ankle_vel_x' in key_lower or key == 'right_ankle_vel_x':
            standardized_data['right_ankle_vel_x'] = value
        elif 'right_ankle_vel_y' in key_lower or key == 'right_ankle_vel_y':
            standardized_data['right_ankle_vel_y'] = value
        elif 'right_ankle_angle' in key_lower or key == 'right_ankle_angle':
            standardized_data['right_ankle_angle'] = value
        elif 'left_ankle_pos_x' in key_lower or key == 'left_ankle_pos_x':
            standardized_data['left_ankle_pos_x'] = value
        elif 'left_ankle_pos_y' in key_lower or key == 'left_ankle_pos_y':
            standardized_data['left_ankle_pos_y'] = value
        elif 'left_ankle_vel_x' in key_lower or key == 'left_ankle_vel_x':
            standardized_data['left_ankle_vel_x'] = value
        elif 'left_ankle_vel_y' in key_lower or key == 'left_ankle_vel_y':
            standardized_data['left_ankle_vel_y'] = value
        elif 'left_ankle_angle' in key_lower or key == 'left_ankle_angle':
            standardized_data['left_ankle_angle'] = value
    
    return standardized_data


def plot_ankle_positions(data, output_folder):
    """
    Plot ankle position trajectories (x vs y)
    
    Creates 3 subplots:
    - Right ankle trajectory
    - Left ankle trajectory
    - Both ankles comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color map for trajectory (time-based)
    time_norm = data['normalized_time']
    colors = plt.cm.viridis(time_norm)
    
    # === SUBPLOT 1: Right Ankle Trajectory ===
    ax = axes[0]
    scatter = ax.scatter(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
                        c=time_norm, cmap='viridis', s=20, alpha=0.7)
    ax.plot(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
           'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(data['right_ankle_pos_x'][0], data['right_ankle_pos_y'][0],
           'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['right_ankle_pos_x'][-1], data['right_ankle_pos_y'][-1],
           'ro', markersize=12, label='End', markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Right Ankle Position Trajectory', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Time', fontweight='bold')
    
    # === SUBPLOT 2: Left Ankle Trajectory ===
    ax = axes[1]
    scatter = ax.scatter(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
                        c=time_norm, cmap='viridis', s=20, alpha=0.7)
    ax.plot(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
           'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(data['left_ankle_pos_x'][0], data['left_ankle_pos_y'][0],
           'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['left_ankle_pos_x'][-1], data['left_ankle_pos_y'][-1],
           'ro', markersize=12, label='End', markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Left Ankle Position Trajectory', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Time', fontweight='bold')
    
    # === SUBPLOT 3: Both Ankles Comparison ===
    ax = axes[2]
    ax.plot(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
           'b-', linewidth=2, label='Right Ankle', alpha=0.7)
    ax.plot(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
           'r-', linewidth=2, label='Left Ankle', alpha=0.7)
    
    # Mark start points
    ax.plot(data['right_ankle_pos_x'][0], data['right_ankle_pos_y'][0],
           'bo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['left_ankle_pos_x'][0], data['left_ankle_pos_y'][0],
           'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Mark end points
    ax.plot(data['right_ankle_pos_x'][-1], data['right_ankle_pos_y'][-1],
           'bs', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['left_ankle_pos_x'][-1], data['left_ankle_pos_y'][-1],
           'rs', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Both Ankles Comparison', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Overall title
    fig.suptitle('Baseline Trajectory: Ankle Position (X-Y Plane)', 
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_folder, 'baseline_ankle_positions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_ankle_angles(data, output_folder):
    """
    Plot ankle angles vs time
    
    Creates 3 subplots:
    - Right ankle angle vs time
    - Left ankle angle vs time
    - Both angles comparison
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data['normalized_time']
    
    # === SUBPLOT 1: Right Ankle Angle ===
    ax = axes[0]
    ax.plot(time, data['right_ankle_angle'], 'b-', linewidth=2.5, label='Right Ankle')
    ax.fill_between(time, data['right_ankle_angle'], alpha=0.3, color='blue')
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Right Ankle Angle vs Time', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add min/max annotations
    max_idx = np.argmax(data['right_ankle_angle'])
    min_idx = np.argmin(data['right_ankle_angle'])
    ax.plot(time[max_idx], data['right_ankle_angle'][max_idx], 'ro', markersize=8)
    ax.plot(time[min_idx], data['right_ankle_angle'][min_idx], 'go', markersize=8)
    ax.annotate(f'Max: {data["right_ankle_angle"][max_idx]:.2f}°',
               xy=(time[max_idx], data['right_ankle_angle'][max_idx]),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # === SUBPLOT 2: Left Ankle Angle ===
    ax = axes[1]
    ax.plot(time, data['left_ankle_angle'], 'r-', linewidth=2.5, label='Left Ankle')
    ax.fill_between(time, data['left_ankle_angle'], alpha=0.3, color='red')
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Left Ankle Angle vs Time', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add min/max annotations
    max_idx = np.argmax(data['left_ankle_angle'])
    min_idx = np.argmin(data['left_ankle_angle'])
    ax.plot(time[max_idx], data['left_ankle_angle'][max_idx], 'ro', markersize=8)
    ax.plot(time[min_idx], data['left_ankle_angle'][min_idx], 'go', markersize=8)
    ax.annotate(f'Max: {data["left_ankle_angle"][max_idx]:.2f}°',
               xy=(time[max_idx], data['left_ankle_angle'][max_idx]),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # === SUBPLOT 3: Both Angles Comparison ===
    ax = axes[2]
    ax.plot(time, data['right_ankle_angle'], 'b-', linewidth=2.5, 
           label='Right Ankle', alpha=0.8)
    ax.plot(time, data['left_ankle_angle'], 'r-', linewidth=2.5,
           label='Left Ankle', alpha=0.8)
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Ankle Angles Comparison', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Overall title
    fig.suptitle('Baseline Trajectory: Ankle Angles', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_folder, 'baseline_ankle_angles.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_ankle_velocities(data, output_folder):
    """
    Plot ankle velocity trajectories (vel_x vs vel_y)
    
    Creates 3 subplots:
    - Right ankle velocity trajectory
    - Left ankle velocity trajectory
    - Both velocities comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color map for trajectory (time-based)
    time_norm = data['normalized_time']
    
    # === SUBPLOT 1: Right Ankle Velocity ===
    ax = axes[0]
    scatter = ax.scatter(data['right_ankle_vel_x'], data['right_ankle_vel_y'],
                        c=time_norm, cmap='plasma', s=20, alpha=0.7)
    ax.plot(data['right_ankle_vel_x'], data['right_ankle_vel_y'],
           'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(data['right_ankle_vel_x'][0], data['right_ankle_vel_y'][0],
           'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['right_ankle_vel_x'][-1], data['right_ankle_vel_y'][-1],
           'ro', markersize=12, label='End', markeredgecolor='black', markeredgewidth=2)
    
    # Add origin
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('X Velocity (m/s)', fontweight='bold')
    ax.set_ylabel('Y Velocity (m/s)', fontweight='bold')
    ax.set_title('Right Ankle Velocity Trajectory', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Time', fontweight='bold')
    
    # === SUBPLOT 2: Left Ankle Velocity ===
    ax = axes[1]
    scatter = ax.scatter(data['left_ankle_vel_x'], data['left_ankle_vel_y'],
                        c=time_norm, cmap='plasma', s=20, alpha=0.7)
    ax.plot(data['left_ankle_vel_x'], data['left_ankle_vel_y'],
           'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(data['left_ankle_vel_x'][0], data['left_ankle_vel_y'][0],
           'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['left_ankle_vel_x'][-1], data['left_ankle_vel_y'][-1],
           'ro', markersize=12, label='End', markeredgecolor='black', markeredgewidth=2)
    
    # Add origin
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('X Velocity (m/s)', fontweight='bold')
    ax.set_ylabel('Y Velocity (m/s)', fontweight='bold')
    ax.set_title('Left Ankle Velocity Trajectory', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Time', fontweight='bold')
    
    # === SUBPLOT 3: Both Velocities Comparison ===
    ax = axes[2]
    ax.plot(data['right_ankle_vel_x'], data['right_ankle_vel_y'],
           'b-', linewidth=2, label='Right Ankle', alpha=0.7)
    ax.plot(data['left_ankle_vel_x'], data['left_ankle_vel_y'],
           'r-', linewidth=2, label='Left Ankle', alpha=0.7)
    
    # Mark start points
    ax.plot(data['right_ankle_vel_x'][0], data['right_ankle_vel_y'][0],
           'bo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(data['left_ankle_vel_x'][0], data['left_ankle_vel_y'][0],
           'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Add origin
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('X Velocity (m/s)', fontweight='bold')
    ax.set_ylabel('Y Velocity (m/s)', fontweight='bold')
    ax.set_title('Both Ankles Velocity Comparison', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    # Overall title
    fig.suptitle('Baseline Trajectory: Ankle Velocities (Phase Space)', 
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_folder, 'baseline_ankle_velocities.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_comprehensive_summary(data, output_folder):
    """
    Create a comprehensive 2x3 summary figure with all key plots
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    time = data['normalized_time']
    time_norm = time
    
    # === PLOT 1: Right Ankle Position ===
    ax = fig.add_subplot(gs[0, 0])
    scatter = ax.scatter(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
                        c=time_norm, cmap='viridis', s=15, alpha=0.6)
    ax.plot(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
           'k-', alpha=0.2, linewidth=0.8)
    ax.plot(data['right_ankle_pos_x'][0], data['right_ankle_pos_y'][0],
           'go', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(data['right_ankle_pos_x'][-1], data['right_ankle_pos_y'][-1],
           'ro', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Right Ankle Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # === PLOT 2: Left Ankle Position ===
    ax = fig.add_subplot(gs[0, 1])
    scatter = ax.scatter(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
                        c=time_norm, cmap='viridis', s=15, alpha=0.6)
    ax.plot(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
           'k-', alpha=0.2, linewidth=0.8)
    ax.plot(data['left_ankle_pos_x'][0], data['left_ankle_pos_y'][0],
           'go', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(data['left_ankle_pos_x'][-1], data['left_ankle_pos_y'][-1],
           'ro', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Left Ankle Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # === PLOT 3: Both Ankles Position ===
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(data['right_ankle_pos_x'], data['right_ankle_pos_y'],
           'b-', linewidth=2, label='Right', alpha=0.7)
    ax.plot(data['left_ankle_pos_x'], data['left_ankle_pos_y'],
           'r-', linewidth=2, label='Left', alpha=0.7)
    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Both Ankles Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    # === PLOT 4: Right Ankle Angle ===
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time, data['right_ankle_angle'], 'b-', linewidth=2)
    ax.fill_between(time, data['right_ankle_angle'], alpha=0.3, color='blue')
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Right Ankle Angle', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # === PLOT 5: Left Ankle Angle ===
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(time, data['left_ankle_angle'], 'r-', linewidth=2)
    ax.fill_between(time, data['left_ankle_angle'], alpha=0.3, color='red')
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Left Ankle Angle', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # === PLOT 6: Both Angles ===
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(time, data['right_ankle_angle'], 'b-', linewidth=2, label='Right', alpha=0.7)
    ax.plot(time, data['left_ankle_angle'], 'r-', linewidth=2, label='Left', alpha=0.7)
    ax.set_xlabel('Normalized Time', fontweight='bold')
    ax.set_ylabel('Angle (°)', fontweight='bold')
    ax.set_title('Both Ankle Angles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Overall title
    fig.suptitle('Baseline Trajectory: Comprehensive Summary', 
                fontsize=20, fontweight='bold')
    
    # Save figure
    save_path = os.path.join(output_folder, 'baseline_comprehensive_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.close()


def print_trajectory_statistics(data):
    """
    Print statistical summary of the trajectory
    """
    print("\n" + "="*70)
    print(" "*20 + "TRAJECTORY STATISTICS")
    print("="*70)
    
    print("\nRIGHT ANKLE:")
    print(f"  Position X: min={data['right_ankle_pos_x'].min():.4f}, max={data['right_ankle_pos_x'].max():.4f}, range={data['right_ankle_pos_x'].max()-data['right_ankle_pos_x'].min():.4f}")
    print(f"  Position Y: min={data['right_ankle_pos_y'].min():.4f}, max={data['right_ankle_pos_y'].max():.4f}, range={data['right_ankle_pos_y'].max()-data['right_ankle_pos_y'].min():.4f}")
    print(f"  Velocity X: min={data['right_ankle_vel_x'].min():.4f}, max={data['right_ankle_vel_x'].max():.4f}")
    print(f"  Velocity Y: min={data['right_ankle_vel_y'].min():.4f}, max={data['right_ankle_vel_y'].max():.4f}")
    print(f"  Angle: min={data['right_ankle_angle'].min():.2f}°, max={data['right_ankle_angle'].max():.2f}°, range={data['right_ankle_angle'].max()-data['right_ankle_angle'].min():.2f}°")
    
    print("\nLEFT ANKLE:")
    print(f"  Position X: min={data['left_ankle_pos_x'].min():.4f}, max={data['left_ankle_pos_x'].max():.4f}, range={data['left_ankle_pos_x'].max()-data['left_ankle_pos_x'].min():.4f}")
    print(f"  Position Y: min={data['left_ankle_pos_y'].min():.4f}, max={data['left_ankle_pos_y'].max():.4f}, range={data['left_ankle_pos_y'].max()-data['left_ankle_pos_y'].min():.4f}")
    print(f"  Velocity X: min={data['left_ankle_vel_x'].min():.4f}, max={data['left_ankle_vel_x'].max():.4f}")
    print(f"  Velocity Y: min={data['left_ankle_vel_y'].min():.4f}, max={data['left_ankle_vel_y'].max():.4f}")
    print(f"  Angle: min={data['left_ankle_angle'].min():.2f}°, max={data['left_ankle_angle'].max():.2f}°, range={data['left_ankle_angle'].max()-data['left_ankle_angle'].min():.2f}°")
    
    print("\n" + "="*70)


def main():
    """
    Main function to load and visualize baseline trajectory
    """
    print("\n" + "="*80)
    print(" "*20 + "BASELINE TRAJECTORY VISUALIZATION")
    print(" "*15 + "Lower Limb Exoskeleton - 11D GMR Trajectory")
    print("="*80)
    
    # ========== CONFIGURATION ==========
    csv_filepath = "adaptability_tests/3e-04/baseline_trajectory_11D.csv"
    output_folder = "baseline_visualization/"
    
    # Check if CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"\n❌ ERROR: CSV file not found: {csv_filepath}")
        print(f"\nPlease ensure you have run the adaptability test script first to generate the baseline CSV.")
        sys.exit(1)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder created: {output_folder}")
    
    # ========== LOAD DATA ==========
    data = load_baseline_trajectory(csv_filepath)
    
    # Verify all required keys are present
    required_keys = [
        'normalized_time', 'right_ankle_pos_x', 'right_ankle_pos_y',
        'right_ankle_vel_x', 'right_ankle_vel_y', 'right_ankle_angle',
        'left_ankle_pos_x', 'left_ankle_pos_y', 'left_ankle_vel_x',
        'left_ankle_vel_y', 'left_ankle_angle'
    ]
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"\n❌ ERROR: Missing required data columns: {missing_keys}")
        print(f"\nAvailable columns: {list(data.keys())}")
        print("\nPlease check that the CSV file has the correct column names.")
        sys.exit(1)
    
    print("✓ All required data columns found")
    
    # ========== PRINT STATISTICS ==========
    print_trajectory_statistics(data)
    
    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "="*70)
    print(" "*25 + "GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    print("Creating ankle position plots...")
    plot_ankle_positions(data, output_folder)
    
    print("Creating ankle angle plots...")
    plot_ankle_angles(data, output_folder)
    
    print("Creating ankle velocity plots...")
    plot_ankle_velocities(data, output_folder)
    
    print("Creating comprehensive summary...")
    plot_comprehensive_summary(data, output_folder)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print(" "*25 + "VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n✓ All visualizations generated successfully!")
    print(f"✓ Results saved in: {output_folder}/")
    print(f"\nGenerated files (4 total):")
    print(f"  1. {output_folder}/baseline_ankle_positions.png")
    print(f"  2. {output_folder}/baseline_ankle_angles.png")
    print(f"  3. {output_folder}/baseline_ankle_velocities.png")
    print(f"  4. {output_folder}/baseline_comprehensive_summary.png")
    print(f"\nFigure details:")
    print(f"  - Resolution: 300 DPI")
    print(f"  - Format: PNG")
    print(f"  - All coordinates in body (hip-centered) frame")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
