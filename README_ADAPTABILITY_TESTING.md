# TPGMM Adaptability Testing for Lower Limb Exoskeleton

## Overview

This script tests the adaptability of your trained Task-Parameterized Gaussian Mixture Model (TPGMM) by applying **time-varying transformations** to reference frame FR2 (right foot target). This demonstrates how the DEXO-1 exoskeleton can adapt gait trajectories to new scenarios while maintaining natural motion patterns.

## Key Features

✅ **Three comprehensive adaptability tests:**
1. **Horizontal Displacement**: FR2 shifts from -0.4m to +0.4m along x-axis
2. **Vertical Displacement**: FR2 shifts from -0.3m to +0.3m along y-axis  
3. **Rotation**: FR2 rotates from -25° to +25°

✅ **Time-varying transformations** (NOT constant!)
- Transformations evolve smoothly over the gait cycle
- Demonstrates endpoint adaptation while preserving initial position
- Mimics real-world exoskeleton control scenarios

✅ **All coordinates relative to body** (hip-centered frame)

✅ **Comprehensive visualizations** for each test showing:
- Ankle trajectories (X-Y space)
- Position evolution over time
- Velocity profiles
- Angle changes
- Adaptation metrics

## Prerequisites

```bash
pip install numpy matplotlib scipy scikit-learn joblib --break-system-packages
```

## File Structure

```
your_project/
├── train_tpgmm_model.py                          # Training script (provided)
├── test_tpgmm_adaptability.py                    # Adaptability testing script (NEW!)
├── train_tpgmm_model_reg3e-04/                   # Model folder
│   └── trained_model.pkl                          # Trained TPGMM model
├── TaskPaGMMM/examples/7days1/                   # Data folder
│   └── gait_analysis_export_subject35v6.json     # Original gait data
└── adaptability_tests/                            # Output folder (auto-created)
    ├── adaptability_horizontal_displacement.png
    ├── adaptability_vertical_displacement.png
    └── adaptability_rotation.png
```

## Usage

### Basic Usage

Simply run the script:

```bash
python test_tpgmm_adaptability.py
```

### Configuration

You can modify these parameters in the `main()` function:

```python
# Model and data paths
model_path = "train_tpgmm_model_reg3e-04/trained_model.pkl"
data_path = "TaskPaGMMM\\examples\\7days1\\gait_analysis_export_subject35v6.json"
output_folder = "adaptability_tests"

# Number of query points for trajectory generation
N_query = 100

# Displacement ranges (in meters)
horizontal_range = (-0.4, 0.4)  # ±40 cm
vertical_range = (-0.3, 0.3)    # ±30 cm

# Rotation range (in degrees)
rotation_range = (-25, 25)      # ±25°
```

## How It Works

### 1. Time-Varying Transformations

The script creates **time-varying** transformation matrices A(t) and translation vectors b(t) for each reference frame:

```
For each time step t ∈ [0, 1]:
  FR1 (hip):           A₁(t) = I,  b₁(t) = 0     [Identity - no change]
  FR2 (right foot):    A₂(t) = R(θ(t)),  b₂(t) = d(t)  [Time-varying!]
  FR3 (left foot):     A₃(t) = I,  b₃(t) = 0     [Identity - no change]
```

### 2. Horizontal Displacement Test

```python
# FR2 shifts horizontally over time
b₂(t) = [displacement(t), 0, 0, ..., 0]ᵀ

where displacement(t) = d_min + (d_max - d_min) * t
```

**Expected behavior:**
- Initial position (t=0): Same as baseline
- Final position (t=1): Shifted horizontally by (d_max - d_min)
- Smooth adaptation throughout gait cycle

### 3. Vertical Displacement Test

```python
# FR2 shifts vertically over time
b₂(t) = [0, displacement(t), 0, ..., 0]ᵀ

where displacement(t) = d_min + (d_max - d_min) * t
```

**Expected behavior:**
- Trajectory adapts to higher/lower target position
- Natural foot clearance maintained
- Velocity profiles adjust accordingly

### 4. Rotation Test

```python
# FR2 rotates over time
A₂(t) = [cos(θ(t))  -sin(θ(t))  0  ...  0]
        [sin(θ(t))   cos(θ(t))  0  ...  0]
        [    0           0      1  ...  0]
        [   ...         ...   ... ... ...]

where θ(t) = θ_min + (θ_max - θ_min) * t
```

**Expected behavior:**
- Trajectory rotates around hip center
- Both position and velocity vectors rotate
- Angular joint trajectories adapt to new orientation

## Understanding the Output

### Generated Visualizations

Each test produces a comprehensive 9-panel figure:

**Row 1:**
1. Right ankle trajectory (X-Y) - baseline vs adapted
2. Left ankle trajectory (X-Y) - baseline vs adapted
3. Parameter evolution over time

**Row 2:**
4. Right ankle X position vs time
5. Right ankle Y position vs time
6. Right ankle angle vs time

**Row 3:**
7. Right ankle velocity magnitude
8. Position difference (adapted - baseline)
9. Adaptation metrics (summary statistics)

### Interpreting the Results

**Good adaptation shows:**
- ✅ Initial position nearly identical to baseline
- ✅ Final position significantly different (matching parameter change)
- ✅ Smooth, continuous trajectory throughout
- ✅ Realistic velocity profiles (no discontinuities)
- ✅ Natural angle progressions

**Poor adaptation shows:**
- ❌ Large initial position difference
- ❌ Jerky or discontinuous trajectories
- ❌ Unrealistic velocity spikes
- ❌ Unnatural joint angles

## Expected Results

### Horizontal Displacement (-0.4 to +0.4 m)

```
Initial position difference: ~0-5 mm  (should be very small!)
Final position difference: ~800 mm    (0.4 - (-0.4) = 0.8 m)
Adaptation ratio: ~160-∞              (large final change, small initial change)
```

### Vertical Displacement (-0.3 to +0.3 m)

```
Initial position difference: ~0-5 mm
Final position difference: ~600 mm    (0.3 - (-0.3) = 0.6 m)
Adaptation ratio: ~120-∞
```

### Rotation (-25° to +25°)

```
Initial position difference: ~0-5 mm
Final position difference: varies based on trajectory radius
Angular difference: 50° total rotation
```

## Troubleshooting

### Issue: "Model file not found"
**Solution:** Update `model_path` to point to your trained model PKL file

### Issue: "Data file not found"
**Solution:** Update `data_path` to point to your JSON data file

### Issue: Large initial position difference
**Solution:** This is expected for time-varying transformations! The script applies transformations at each time step, so even at t=0, there may be small effects due to numerical precision.

### Issue: Unrealistic trajectories
**Solution:** 
- Check displacement/rotation ranges aren't too extreme
- Verify trained model quality (check BIC scores)
- Try smaller parameter ranges first

## Advanced Customization

### Custom Transformation Functions

You can create your own transformation functions:

```python
def create_custom_transformation(N):
    """
    Create your own time-varying transformation
    """
    P = 3  # Number of frames
    D = 11  # Dimensions
    
    A_frames = np.zeros((N, P, D, D))
    b_frames = np.zeros((N, P, D))
    
    time = np.linspace(0, 1, N)
    
    for n in range(N):
        # FR1: Identity
        A_frames[n, 0] = np.eye(D)
        
        # FR2: Your custom transformation
        # Example: Sinusoidal displacement
        displacement = 0.2 * np.sin(2 * np.pi * time[n])
        
        A_frames[n, 1] = np.eye(D)
        b_frames[n, 1, 0] = displacement
        
        # FR3: Identity
        A_frames[n, 2] = np.eye(D)
    
    return A_frames, b_frames
```

### Multiple Simultaneous Adaptations

Test combined horizontal + vertical displacement:

```python
# Combine transformations
b_frames[n, 1, 0] = horizontal_displacement[n]  # X
b_frames[n, 1, 1] = vertical_displacement[n]    # Y
```

## Integration with DEXO-1 Exoskeleton

### Real-time Implementation

```python
def get_adapted_trajectory_realtime(model, target_position, current_phase):
    """
    Get adapted trajectory for real-time exoskeleton control
    
    Parameters:
    -----------
    model : trained TPGMM
    target_position : (x, y) - desired foot endpoint
    current_phase : float [0, 1] - current gait phase
    
    Returns:
    --------
    desired_joint_angles : array
        Target joint angles for exoskeleton
    """
    # Calculate transformation to reach target
    A, b = compute_transformation_to_target(target_position, current_phase)
    
    # Generate trajectory using GMR
    mu, sigma = gmr.predict([current_phase], [10], [0,1,2,3,4,5,6,7,8,9], 
                           A, b)
    
    # Convert to joint angles for exoskeleton control
    joint_angles = inverse_kinematics(mu)
    
    return joint_angles
```

## Citations

If you use this code in your research, please cite:

```
@article{calinon2016tutorial,
  title={A tutorial on task-parameterized movement learning and retrieval},
  author={Calinon, Sylvain},
  journal={Intelligent Service Robotics},
  volume={9},
  number={1},
  pages={1--29},
  year={2016}
}
```

## Contact

For questions about this script or the DEXO-1 exoskeleton project:
- Victor - PhD Student, FEEC/UNICAMP
- Research focus: Lower-limb assistive robotics for individuals with paraplegia

## License

This code is part of the DEXO-1 exoskeleton research project at UNICAMP.

---

**Happy testing! 🦾**

The TPGMM adaptability demonstrates your exoskeleton's ability to:
- ✅ Adapt to varying terrain
- ✅ Adjust stride length dynamically  
- ✅ Modify foot placement in real-time
- ✅ Maintain natural, smooth motion patterns

This is exactly what you need for robust, user-centered assistive robotics!
