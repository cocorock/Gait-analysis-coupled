"""
FILTER AND PLOT GMR TRAJECTORIES
Apply low-pass filter to recovered trajectories and create comprehensive plots

This script:
1. Loads the recovered trajectory from GMR
2. Applies a low-pass Butterworth filter (6 Hz cutoff)
3. Creates comprehensive plots comparing filtered vs unfiltered
4. Saves all figures

Author: Victor
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import os

# Configure matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino Linotype', 'Palatino', 'Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data
    
    Parameters:
    -----------
    data : array (N, D)
        Input trajectory data
    cutoff : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order (default: 4)
    
    Returns:
    --------
    filtered_data : array (N, D)
        Filtered trajectory (same length as input)
    """
    # Design the Butterworth filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply forward-backward filter to avoid phase shift
    # filtfilt preserves the length of the signal
    filtered_data = np.zeros_like(data)
    
    for dim in range(data.shape[1]):
        filtered_data[:, dim] = signal.filtfilt(b, a, data[:, dim])
    
    return filtered_data


def compute_velocities(trajectory, dt):
    """
    Compute velocities from position trajectory
    
    Parameters:
    -----------
    trajectory : array (N, D)
        Position trajectory
    dt : float
        Time step
    
    Returns:
    --------
    velocities : array (N, D)
        Velocity trajectory
    """
    velocities = np.zeros_like(trajectory)
    
    # Forward difference for velocities
    velocities[:-1] = np.diff(trajectory, axis=0) / dt
    velocities[-1] = velocities[-2]  # Repeat last value
    
    return velocities


def compute_accelerations(trajectory, dt):
    """
    Compute accelerations from position trajectory
    
    Parameters:
    -----------
    trajectory : array (N, D)
        Position trajectory
    dt : float
        Time step
    
    Returns:
    --------
    accelerations : array (N, D)
        Acceleration trajectory
    """
    velocities = compute_velocities(trajectory, dt)
    accelerations = np.zeros_like(velocities)
    
    # Forward difference for accelerations
    accelerations[:-1] = np.diff(velocities, axis=0) / dt
    accelerations[-1] = accelerations[-2]  # Repeat last value
    
    return accelerations


def plot_position_comparison(time, original, filtered, output_folder):
    """
    Plot position trajectories: original vs filtered
    """
    feature_names = [
        'Right Ankle X', 'Right Ankle Y',
        'Right Ankle Vx', 'Right Ankle Vy',
        'Right Ankle Angle',
        'Left Ankle X', 'Left Ankle Y',
        'Left Ankle Vx', 'Left Ankle Vy',
        'Left Ankle Angle'
    ]
    
    # Extract only position and angle dimensions (0, 1, 4, 5, 6, 9)
    position_dims = [0, 1, 4, 5, 6, 9]
    position_names = [feature_names[i] for i in position_dims]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, dim in enumerate(position_dims):
        ax = axes[idx]
        
        ax.plot(time, original[:, dim], 'b-', linewidth=1.5, alpha=0.7, label='Original GMR')
        ax.plot(time, filtered[:, dim], 'r-', linewidth=2, label='Filtered (6 Hz)')
        
        ax.set_xlabel('Time (normalized)', fontweight='bold')
        ax.set_ylabel(position_names[idx], fontweight='bold')
        ax.set_title(f'{position_names[idx]} - Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'filtered_positions_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def plot_velocity_comparison(time, original_vel, filtered_vel, output_folder):
    """
    Plot velocity trajectories computed from position
    """
    feature_names = [
        'Right Ankle Vx', 'Right Ankle Vy',
        'Left Ankle Vx', 'Left Ankle Vy'
    ]
    
    # Velocity dimensions in the original trajectory
    vel_dims = [2, 3, 7, 8]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(vel_dims):
        ax = axes[idx]
        
        ax.plot(time, original_vel[:, dim], 'b-', linewidth=1.5, alpha=0.7, label='Original GMR')
        ax.plot(time, filtered_vel[:, dim], 'r-', linewidth=2, label='Filtered (6 Hz)')
        
        ax.set_xlabel('Time (normalized)', fontweight='bold')
        ax.set_ylabel(f'{feature_names[idx]} (m/s)', fontweight='bold')
        ax.set_title(f'{feature_names[idx]} - Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'filtered_velocities_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def plot_xy_trajectories(original, filtered, output_folder, cyclic=True):
    """
    Plot X-Y position trajectories for both ankles
    
    Parameters:
    -----------
    original : array (N, D)
        Original trajectory
    filtered : array (N, D)
        Filtered trajectory
    output_folder : str
        Output folder for saving plots
    cyclic : bool
        If True, close the trajectory by connecting last to first point
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Right ankle
    axes[0].plot(original[:, 0], original[:, 1], 'b-', linewidth=1.5, alpha=0.7, 
                label='Original GMR')
    axes[0].plot(filtered[:, 0], filtered[:, 1], 'r-', linewidth=2.5, 
                label='Filtered (6 Hz)', zorder=10)
    
    # Close the trajectory if cyclic
    if cyclic:
        axes[0].plot([filtered[-1, 0], filtered[0, 0]], 
                    [filtered[-1, 1], filtered[0, 1]], 
                    'r--', linewidth=2.5, alpha=0.8, label='Cycle closure', zorder=10)
        axes[0].plot([original[-1, 0], original[0, 0]], 
                    [original[-1, 1], original[0, 1]], 
                    'b--', linewidth=1.5, alpha=0.5, zorder=5)
    
    axes[0].plot(filtered[0, 0], filtered[0, 1], 'go', markersize=12, 
                label='Start', zorder=15, markeredgecolor='darkgreen', markeredgewidth=2)
    axes[0].plot(filtered[-1, 0], filtered[-1, 1], 'rs', markersize=12, 
                label='End', zorder=15, markeredgecolor='darkred', markeredgewidth=2)
    
    axes[0].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Right Ankle - X-Y Trajectory', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].set_aspect('equal', adjustable='box')
    
    # Left ankle
    axes[1].plot(original[:, 5], original[:, 6], 'b-', linewidth=1.5, alpha=0.7,
                label='Original GMR')
    axes[1].plot(filtered[:, 5], filtered[:, 6], 'r-', linewidth=2.5,
                label='Filtered (6 Hz)', zorder=10)
    
    # Close the trajectory if cyclic
    if cyclic:
        axes[1].plot([filtered[-1, 5], filtered[0, 5]], 
                    [filtered[-1, 6], filtered[0, 6]], 
                    'r--', linewidth=2.5, alpha=0.8, label='Cycle closure', zorder=10)
        axes[1].plot([original[-1, 5], original[0, 5]], 
                    [original[-1, 6], original[0, 6]], 
                    'b--', linewidth=1.5, alpha=0.5, zorder=5)
    
    axes[1].plot(filtered[0, 5], filtered[0, 6], 'go', markersize=12,
                label='Start', zorder=15, markeredgecolor='darkgreen', markeredgewidth=2)
    axes[1].plot(filtered[-1, 5], filtered[-1, 6], 'rs', markersize=12,
                label='End', zorder=15, markeredgecolor='darkred', markeredgewidth=2)
    
    axes[1].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Left Ankle - X-Y Trajectory', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'filtered_xy_trajectories.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def plot_acceleration_comparison(time, original_acc, filtered_acc, output_folder):
    """
    Plot acceleration comparison
    """
    feature_names = [
        'Right Ankle Ax', 'Right Ankle Ay',
        'Left Ankle Ax', 'Left Ankle Ay'
    ]
    
    # Acceleration for position dimensions
    acc_dims = [0, 1, 5, 6]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(acc_dims):
        ax = axes[idx]
        
        ax.plot(time, original_acc[:, dim], 'b-', linewidth=1.5, alpha=0.7, 
               label='Original GMR')
        ax.plot(time, filtered_acc[:, dim], 'r-', linewidth=2, 
               label='Filtered (6 Hz)')
        
        ax.set_xlabel('Time (normalized)', fontweight='bold')
        ax.set_ylabel(f'{feature_names[idx]} (m/s²)', fontweight='bold')
        ax.set_title(f'{feature_names[idx]} - Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'filtered_accelerations_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def plot_frequency_response(original, filtered, fs, output_folder):
    """
    Plot frequency spectrum comparison
    """
    from scipy.fft import fft, fftfreq
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Analyze right ankle X, Y and left ankle X, Y
    dims_to_analyze = [0, 1, 5, 6]
    dim_names = ['Right Ankle X', 'Right Ankle Y', 'Left Ankle X', 'Left Ankle Y']
    
    for idx, dim in enumerate(dims_to_analyze):
        ax = axes[idx]
        
        # Compute FFT
        N = len(original)
        
        # Original
        yf_orig = fft(original[:, dim])
        xf = fftfreq(N, 1/fs)[:N//2]
        power_orig = 2.0/N * np.abs(yf_orig[0:N//2])
        
        # Filtered
        yf_filt = fft(filtered[:, dim])
        power_filt = 2.0/N * np.abs(yf_filt[0:N//2])
        
        ax.semilogy(xf, power_orig, 'b-', linewidth=1.5, alpha=0.7, label='Original GMR')
        ax.semilogy(xf, power_filt, 'r-', linewidth=2, label='Filtered (6 Hz)')
        ax.axvline(x=6, color='k', linestyle='--', linewidth=1.5, label='Cutoff (6 Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Power', fontweight='bold')
        ax.set_title(f'{dim_names[idx]} - Frequency Spectrum', fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best')
        ax.set_xlim([0, 20])  # Show up to 20 Hz
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'frequency_spectrum_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def main(model_folder, cutoff_freq=6.0, sampling_freq=100.0, filter_order=4, cyclic=True):
    """
    Main function to filter and plot trajectories
    
    Parameters:
    -----------
    model_folder : str
        Folder containing the trained model
    cutoff_freq : float
        Cutoff frequency for low-pass filter (Hz)
    sampling_freq : float
        Sampling frequency (Hz) - estimated from trajectory
    filter_order : int
        Butterworth filter order
    cyclic : bool
        If True, close trajectories by connecting last to first point
    """
    print("\n" + "="*70)
    print("TRAJECTORY FILTERING AND VISUALIZATION")
    print("="*70)
    
    # Load the generated trajectory
    # Assuming you have saved mu_generated from apply_gmr_generate_trajectories.py
    model_path = os.path.join(model_folder, 'trained_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return
    
    # For this example, we need to regenerate the trajectory
    # You should modify this to load your actual generated trajectory
    print(f"\n⚠ Note: You need to run apply_gmr_generate_trajectories.py first")
    print(f"    and save the mu_generated trajectory.")
    print(f"\n✓ For now, I'll create example code structure...")
    
    # Example: Load from a saved trajectory file (you need to save this first)
    trajectory_file = os.path.join(model_folder, 'gmr_trajectory.npy')
    
    if not os.path.exists(trajectory_file):
        print(f"\n❌ Trajectory file not found: {trajectory_file}")
        print(f"\n💡 Solution: Modify apply_gmr_generate_trajectories.py to save:")
        print(f"    np.save(os.path.join(OUTPUT_FOLDER, 'gmr_trajectory.npy'), mu_generated)")
        print(f"    np.save(os.path.join(OUTPUT_FOLDER, 'time_query.npy'), time_query)")
        return
    
    # Load trajectory
    print(f"\n✓ Loading trajectory from: {trajectory_file}")
    mu_generated = np.load(trajectory_file)
    
    time_file = os.path.join(model_folder, 'time_query.npy')
    if os.path.exists(time_file):
        time_query = np.load(time_file)
        # Flatten in case it's 2D
        time_query = time_query.flatten()
    else:
        # Create default time vector
        time_query = np.linspace(0, 1, mu_generated.shape[0])
    
    # Estimate sampling frequency
    if len(time_query) > 1:
        dt = np.mean(np.diff(time_query))
        if dt > 0:
            estimated_fs = 1.0 / dt
        else:
            print(f"⚠ Warning: Invalid time step, using default fs")
            dt = 0.01
            estimated_fs = sampling_freq
    else:
        dt = 0.01
        estimated_fs = sampling_freq
    
    print(f"\n✓ Trajectory loaded:")
    print(f"  Shape: {mu_generated.shape}")
    print(f"  Duration: {float(time_query[-1]):.2f} s")
    print(f"  Time step: {float(dt):.4f} s")
    print(f"  Estimated sampling frequency: {float(estimated_fs):.2f} Hz")
    
    # Apply low-pass filter
    print(f"\n✓ Applying Butterworth low-pass filter:")
    print(f"  Cutoff frequency: {cutoff_freq} Hz")
    print(f"  Filter order: {filter_order}")
    print(f"  Using sampling frequency: {sampling_freq} Hz")
    
    mu_filtered = butter_lowpass_filter(mu_generated, cutoff_freq, sampling_freq, filter_order)
    
    print(f"\n✓ Filtered trajectory shape: {mu_filtered.shape}")
    print(f"  Length preserved: {mu_filtered.shape[0] == mu_generated.shape[0]}")
    
    # Compute velocities and accelerations
    print(f"\n✓ Computing velocities and accelerations...")
    original_acc = compute_accelerations(mu_generated, dt)
    filtered_acc = compute_accelerations(mu_filtered, dt)
    
    # Create plots
    print(f"\n✓ Creating plots...")
    
    # 1. Position comparison
    plot_position_comparison(time_query.flatten(), mu_generated, mu_filtered, model_folder)
    
    # 2. Velocity comparison (from stored velocities in trajectory)
    plot_velocity_comparison(time_query.flatten(), mu_generated, mu_filtered, model_folder)
    
    # 3. X-Y trajectories
    plot_xy_trajectories(mu_generated, mu_filtered, model_folder, cyclic=cyclic)
    
    # 4. Acceleration comparison
    plot_acceleration_comparison(time_query.flatten(), original_acc, filtered_acc, model_folder)
    
    # 5. Frequency spectrum
    plot_frequency_response(mu_generated, mu_filtered, sampling_freq, model_folder)
    
    # Save filtered trajectory
    filtered_file = os.path.join(model_folder, 'gmr_trajectory_filtered.npy')
    np.save(filtered_file, mu_filtered)
    print(f"\n✓ Saved filtered trajectory: {filtered_file}")
    
    print("\n" + "="*70)
    print("FILTERING AND VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n✓ All figures saved in: {model_folder}/")
    print(f"✓ Filtered trajectory saved: gmr_trajectory_filtered.npy")
    print("="*70)


if __name__ == "__main__":
    # Configuration
    model_folder = "train_tpgmm_model_reg5e-04"  # Update with your folder
    cutoff_frequency = 6.0  # Hz
    sampling_frequency = 200  # Hz (adjust based on your trajectory)
    filter_order = 4  # Butterworth filter order
    cyclic_trajectory = True  # Set to True for gait cycles (periodic motion)
    
    main(model_folder, cutoff_frequency, sampling_frequency, filter_order, cyclic_trajectory)
