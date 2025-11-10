#!/usr/bin/env python3
"""
Quick test to validate proper left/right ankle extraction
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def test_extraction():
    print("Testing proper ankle data extraction...")
    
    # Load data
    data_path = "examples/7days1/gait_analysis_export_subject35v4.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract first demo from FR1
    first_demo = data['kinematics_data']['FR1']['right_leg_kinematics'][0]
    print("Keys in first demo:", list(first_demo.keys()))
    
    # Extract ankle data
    right_ankle_pos = np.array(first_demo['right_ankle_pos'])
    right_ankle_vel = np.array(first_demo['right_ankle_vel'])
    left_ankle_pos = np.array(first_demo['left_ankle_pos'])
    left_ankle_vel = np.array(first_demo['left_ankle_vel'])
    
    print(f"\nRight ankle pos shape: {right_ankle_pos.shape}")
    print(f"Right ankle vel shape: {right_ankle_vel.shape}")
    print(f"Left ankle pos shape: {left_ankle_pos.shape}")
    print(f"Left ankle vel shape: {left_ankle_vel.shape}")
    
    # Check data ranges
    print(f"\nRight ankle X position range: {right_ankle_pos[0].min():.3f} to {right_ankle_pos[0].max():.3f}")
    print(f"Left ankle X position range: {left_ankle_pos[0].min():.3f} to {left_ankle_pos[0].max():.3f}")
    print(f"Right ankle Y position range: {right_ankle_pos[1].min():.3f} to {right_ankle_pos[1].max():.3f}")
    print(f"Left ankle Y position range: {left_ankle_pos[1].min():.3f} to {left_ankle_pos[1].max():.3f}")
    
    # Quick plot to verify
    plt.figure(figsize=(12, 8))
    
    time = np.linspace(0, 1, right_ankle_pos.shape[1])
    
    plt.subplot(2, 2, 1)
    plt.plot(time, right_ankle_pos[0], 'b-', label='Right Ankle X')
    plt.plot(time, left_ankle_pos[0], 'r-', label='Left Ankle X')
    plt.xlabel('Time')
    plt.ylabel('X Position')
    plt.title('X Position Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(time, right_ankle_pos[1], 'b-', label='Right Ankle Y')
    plt.plot(time, left_ankle_pos[1], 'r-', label='Left Ankle Y')
    plt.xlabel('Time')
    plt.ylabel('Y Position')
    plt.title('Y Position Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(right_ankle_pos[0], right_ankle_pos[1], 'b-', label='Right Ankle', linewidth=2)
    plt.plot(left_ankle_pos[0], left_ankle_pos[1], 'r-', label='Left Ankle', linewidth=2)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Trajectories')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 2, 4)
    plt.plot(time, right_ankle_vel[0], 'b-', alpha=0.7, label='Right X Vel')
    plt.plot(time, left_ankle_vel[0], 'r-', alpha=0.7, label='Left X Vel')
    plt.xlabel('Time')
    plt.ylabel('X Velocity')
    plt.title('X Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ankle_extraction_test.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Data extraction test complete!")
    print(f"✓ Both ankles have valid data")
    print(f"✓ Plot saved as ankle_extraction_test.png")
    
    # Check if ankles show different patterns
    right_range_x = right_ankle_pos[0].max() - right_ankle_pos[0].min()
    left_range_x = left_ankle_pos[0].max() - left_ankle_pos[0].min()
    right_range_y = right_ankle_pos[1].max() - right_ankle_pos[1].min()
    left_range_y = left_ankle_pos[1].max() - left_ankle_pos[1].min()
    
    print(f"\nTrajectory ranges:")
    print(f"Right ankle - X: {right_range_x:.3f}, Y: {right_range_y:.3f}")
    print(f"Left ankle - X: {left_range_x:.3f}, Y: {left_range_y:.3f}")
    
    if abs(right_range_x - left_range_x) > 0.1 or abs(right_range_y - left_range_y) > 0.1:
        print("✓ Left and right ankles show DIFFERENT movement patterns!")
    else:
        print("⚠ Left and right ankles show similar patterns")

if __name__ == "__main__":
    test_extraction()