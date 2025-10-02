# Task-Parameterized Gaussian Mixture Model Analysis

## Complete Step-by-Step Analysis of `example_program.py`

### Phase 1: Data Loading & Preprocessing

1. **Load Trajectories** (lines 188-197)
   - Load trajectory data from text files
   - Extract 3D position data (x, y, z coordinates)
   - Result: List of trajectory arrays

2. **Apply Filters** (lines 227-229)
   - **DiffFilter**: Remove trajectories with large jumps (threshold: 0.1353)
   - **GroundFilter**: Keep only trajectories near ground (z < 0.3)

3. **Generate Synthetic Data** (lines 200-217)
   - Use base trajectory to create 120 synthetic variations
   - Add Gaussian noise (σ = 0.01) to create realistic variations

### Phase 2: Data Preprocessing & Frame Setup

4. **Unify Trajectory Lengths** (lines 102-142)
   - Use spline interpolation to normalize all trajectories to same length (200 points)
   - Sample trajectories at uniform intervals

5. **📍 KEY: Frame of Reference Transformation** (lines 144-157)
   ```python
   def translate_trajectories(self, trajectories: ndarray) -> ndarray:
       # Extract start [0] and end [-1] points as reference frames
       start_end_translation = trajectories[:, [0, -1], :]
       
       # Create 2 frames: start frame and end frame
       result = np.empty((2, *trajectories.shape))
       for frame_idx in range(2):
           # Transform each trajectory relative to its start/end point
           points = trajectories.swapaxes(0, 1) - start_end_translation[:, frame_idx] 
           result[frame_idx] = points.swapaxes(0, 1)
   ```
   **This creates TWO coordinate frames:**
   - **Frame 0**: All trajectories relative to their START points
   - **Frame 1**: All trajectories relative to their END points

6. **Add Time Dimension** (lines 159-186)
   - Add timestamp to each trajectory point
   - Final shape: `(2_frames, num_trajectories, num_points, 4_features)`

### Phase 3: Task-Parameterized GMM Training

7. **Prepare Data for TPGMM** (lines 276-279)
   ```python
   concatenated_trajectories = np.reshape(
       pre_processed_data, (2, num_trajectories * num_samples, 4)
   )
   ```
   - Concatenate all trajectories within each frame
   - Shape: `(2_frames, total_points, 4_features)`

8. **Fit TPGMM Model** (lines 274-282)
   ```python
   tpgmm = TPGMM(6, verbose=True)  # 6 Gaussian components
   tpgmm.fit(concatenated_trajectories)
   ```
   - Learn Gaussian parameters for each frame using EM algorithm
   - Results in: `tpgmm.means_[frame_idx]` and `tpgmm.covariances_[frame_idx]`

### Phase 4: Gaussian Mixture Regression Setup

9. **Create GMR Object** (line 314)
   ```python
   gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, [3])  # Use time as input
   ```

10. **📍 KEY: Define New Frame Transformations** (lines 316-321)
    ```python
    # NEW trajectory frame transformations
    translation = np.array([
        [0.64892812, -0.00127319, 0.27268653],  # New start point  
        [3.25658728, 4.49575041, 0.95660171]   # New end point
    ])
    rotation_matrix = np.eye(3)[None].repeat(2, axis=0)  # Identity rotations
    ```

11. **📍 KEY: Apply Frame Transformations** (line 323)
    ```python
    gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
    ```
    **This applies Equations 5 & 6:**
    - **Equation 5**: Transform learned Gaussians to new coordinate frames
    - **Equation 6**: Combine predictions from both frames using Gaussian products

### Phase 5: Trajectory Generation & Prediction

12. **Generate Time Input** (line 327)
    ```python
    input_data = np.linspace(0, 2, 200)[:, None]  # Time from 0 to 2 seconds
    ```

13. **📍 KEY: Predict New Trajectory** (line 328)
    ```python
    mu, cov = gmr.predict(input_data)
    ```
    **This uses Equations 13-16:**
    - **Equation 13**: Mix predictions from multiple Gaussians
    - **Equation 14**: Conditional mean for each Gaussian
    - **Equation 15**: Conditional covariance
    - **Equation 16**: Mixture weights/responsibilities

### Phase 6: Visualization & Validation

14. **Visualize Results** (lines 331-341)
    - Plot the generated trajectory
    - Show uncertainty ellipsoids
    - Display start/end reference points

15. **Validate Performance** (lines 344-348)
    ```python
    start_distance = np.linalg.norm(translation[0] - mu[0])
    end_distance = np.linalg.norm(translation[1] - mu[-1])
    ```

## Frame of Reference Transformation Details

### Training Phase - Creating Reference Frames:
```python
# Step 1: Extract start/end points as frame references
start_end_translation = trajectories[:, [0, -1], :]

# Step 2: Transform trajectories relative to each frame
for frame_idx in range(2):
    # All trajectories become relative to their start (frame 0) or end (frame 1)
    points = trajectories - start_end_translation[:, frame_idx]
```

### Prediction Phase - Applying Learned Transformations:
```python
# Step 1: Define new task parameters (where we want trajectory to go)
translation = np.array([[start_x, start_y, start_z], 
                       [end_x, end_y, end_z]])

# Step 2: Transform learned model to new situation
gmr.fit(translation=translation, rotation_matrix=rotation_matrix)

# Step 3: Generate trajectory adapted to new constraints
mu, cov = gmr.predict(time_input)
```

## Equations Used in TP-GMM Framework

### Core TP-GMM Equations (5-6) from Calinon Paper

**Equation 5** - Frame Transformations:
```
ξ̂^(j)_t,i = A_t,j μ^(j)_i + b_t,j
Σ̂^(j)_t,i = A_t,j Σ^(j)_i A^T_t,j
```

**Equation 6** - Gaussian Products:
```
Σ̂_t,i = (Σ_{j=1}^P Σ̂^(j)_t,i^{-1})^{-1}
ξ̂_t,i = Σ̂_t,i Σ_{j=1}^P Σ̂^(j)_t,i^{-1} ξ̂^(j)_t,i
```

### GMR Equations (13-16) from TaskPaGMMM Library

**Equation 13** - GMR Prediction:
```
P(φ_t^O | φ_t^I) ~ ∑_{i=1}^K h_i(φ_t^I) N(μ̂_t^O(φ_t^I), Σ̂_t^O)
```

**Equation 14** - Conditional Mean:
```
μ̂_i^O(φ_t^I) = μ_i^O + Σ_i^{OI} Σ_i^{I,-1} (φ_t^I - μ_i^I)
```

**Equation 15** - Conditional Covariance:
```
Σ̂_t^O = Σ_i^O - Σ_i^{OI} Σ_i^{I,-1} Σ_i^{IO}
```

**Equation 16** - Responsibility/Weight Calculation:
```
h_i(φ_t^I) = (π_i N(φ_t^I | μ_i^I, Σ_i^I)) / (∑_k π_k N(φ_t^I | μ_k^I, Σ_k^I))
```

## Comparison: `example_program.py` vs `gait_theory_gmr.py`

| Aspect | `example_program.py` | `gait_theory_gmr.py` |
|--------|---------------------|---------------------|
| **Implementation** | Uses TaskPaGMMM library (practical) | Custom theoretical implementation |
| **Equations Used** | Equations 5-6 + 13-16 | Only Equations 5-6 |
| **Frame Transformations** | Real start/end point frames | Identity transformations |
| **Purpose** | Practical trajectory generation | Theoretical framework demonstration |
| **Gaussian Products** | Library implementation | Manual implementation (lines 79-112) |
| **GMR Method** | Standard GMR (Eq. 13-16) | Custom theoretical GMR |
| **Frame Setup** | Meaningful coordinate frames | Identity matrices for simplicity |

## Key Insights

### What `example_program.py` Does:
✅ **Complete practical TP-GMM workflow**  
✅ **Real frame transformations with meaningful coordinates**  
✅ **Uses both core TP-GMM (5-6) and GMR equations (13-16)**  
✅ **Generates trajectories for new task parameters**  

### What `gait_theory_gmr.py` Does:
✅ **Demonstrates theoretical TP-GMM mathematics**  
✅ **Manual implementation of Gaussian products**  
✅ **Focus on equations 5-6 from Calinon paper**  
✅ **Educational/research-oriented approach**  

### Summary
The example program shows the **complete practical workflow** of TP-GMM with real frame transformations, while `gait_theory_gmr.py` focuses on the **theoretical mathematical foundations**. Both are valuable but serve different purposes in understanding and applying Task-Parameterized Gaussian Mixture Models.