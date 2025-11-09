# GMR Investigation Summary

## Problem Statement

The GMR (Gaussian Mixture Regression) produces viable trajectories when using the same frame twice (FR1=FR1) but fails when using different frames (FR1!=FR2). You suspected that the `np.einsum` operations in `_equation_5` and `_equation_6` might be causing issues.

## Investigation Results

### Key Finding: The einsum implementation is MATHEMATICALLY CORRECT

After detailed analysis and testing, I found that:

1. **The `_equation_5` and `_equation_6` implementations correctly follow Calinon's equations**
2. **Both einsum and explicit loop implementations produce IDENTICAL results** (verified with numerical tests)
3. **The problem likely lies elsewhere, NOT in these core GMR functions**

### Test Results

```
Testing Equation 5...
Equation 5 - Mean difference: 0.0
Equation 5 - Covariance difference: 4.440892098500626e-16
[PASS] Equation 5: Both implementations match!

Testing Equation 6...
Equation 6 - Mean difference: 2.220446049250313e-16
Equation 6 - Covariance difference: 2.220446049250313e-16
[PASS] Equation 6: Both implementations match!
```

The tiny differences (1e-16) are just floating-point rounding errors, confirming both implementations are equivalent.

## Files Created

### 1. [gmr_analysis.py](gmr_analysis.py)

This file contains:
- Detailed mathematical analysis comparing einsum operations with Calinon's paper
- Original einsum-based implementations
- New explicit loop implementations (without einsum)
- Test function to verify both produce identical results
- Comprehensive documentation of what each implementation does

### 2. [tpgmm/gmr/gmr_fixed.py](tpgmm/gmr/gmr_fixed.py)

This is a modified version of the GMR library with:
- `_equation_5` reimplemented using explicit loops instead of einsum
- `_equation_6` reimplemented using explicit loops instead of einsum
- Extensive comments explaining each step in relation to Calinon's equations
- All other functions remain unchanged

You can test this version by modifying your notebook to import from `gmr_fixed` instead of `gmr`.

## What the Equations Do

### Equation 5: Frame Transformation

**Calinon's Equation:**
- xi_hat(j)_t,i = A_t,j × mu(j)_i + b_t,j
- Sigma_hat(j)_t,i = A_t,j × Sigma(j)_i × A_t,j^T

**What it does:**
- Takes Gaussians (mean and covariance) from LOCAL frame coordinates
- Transforms them to GLOBAL coordinate system using rotation matrix A and translation vector b
- Does this for EACH frame and EACH component

### Equation 6: Product of Gaussians

**Calinon's Equation:**
- Sigma_hat_t,i = (Sum_j Sigma_hat(j)_t,i^(-1))^(-1)
- xi_hat_t,i = Sigma_hat_t,i × Sum_j (Sigma_hat(j)_t,i^(-1) × xi_hat(j)_t,i)

**What it does:**
- FUSES Gaussians from all frames into a single Gaussian Mixture Model
- Uses the "product of Gaussians" formula (precision-weighted fusion)
- For each component, combines information from all frames

## Where the Real Problem Likely Is

Since equations 5 and 6 are correct, the issue with FR1!=FR2 is likely in:

### 1. **How rotation_matrix and translation are computed** (in cell-34 of your notebook)

The transformation matrices need to represent:
- WHERE each frame is located in space
- HOW each frame is oriented

Current approach in cell-34:
```python
# FR1: identity transformation (base frame at origin)
rotation_fr1 = np.eye(6)
translation_fr1 = np.zeros(6)

# FR2: rotation by -rotation_angle_rad
rotation_fr2 = rotation_matrix_2d_forward  # Applied to position and velocity
translation_fr2 = ?  # This might be the issue!
```

**Key question:** What should `translation_fr2` be?
- Should it be zeros (no translation)?
- Should it be the mean position difference between FR1 and FR2?
- Should it be something else?

### 2. **Coordinate frame conventions**

The relationship between FR1 and FR2 depends on what they represent:
- Are they body-fixed frames that move with the gait cycle?
- Are they world-fixed frames at different spatial locations?
- Are they temporal frames at different points in time?

### 3. **The `_pad` operation** (lines 295-323 in gmr.py)

This pads the rotation/translation to include identity transformations for input features (time):
```python
# Creates: [[I_input, 0], [0, rotation_matrix]]
# This means: don't transform time, only transform output features
```

This is likely correct, but worth verifying.

### 4. **Feature ordering** (`_sort_by_input` and `_revoke_sort_by_input`)

These reorder features to put input features (time) first. The operations look correct but could have subtle bugs.

## Recommended Next Steps

### Option 1: Test with gmr_fixed.py

Modify your notebook to use the explicit implementation:

```python
# In your notebook, change the import:
from tpgmm.gmr.gmr_fixed import GaussianMixtureRegression

# Everything else stays the same
```

If the problem persists (which I expect it will, since both implementations are equivalent), then we know the issue is NOT in equations 5 or 6.

### Option 2: Investigate transformation setup (cell-34)

Focus on cell-34 where you compute rotation_fr2 and translation_fr2. Specifically:

1. **Print the actual values:**
   ```python
   print("Rotation FR1:\n", rotation_fr1)
   print("Translation FR1:\n", translation_fr1)
   print("Rotation FR2:\n", rotation_fr2)
   print("Translation FR2:\n", translation_fr2)
   ```

2. **Verify what these represent physically:**
   - Do they represent WHERE the frames are in space?
   - Do they represent HOW to transform data FROM frame TO global?
   - Do they represent HOW to transform data FROM global TO frame?

3. **Compare with the working example.ipynb:**
   - How are transformations computed there?
   - What are the actual numerical values?

### Option 3: Debug with single component

Simplify the problem:
1. Train TPGMM with just 1 component (n_components=1)
2. Use only 2 frames (FR1 and FR2)
3. Manually compute what equation 5 and 6 should produce
4. Compare with actual output

### Option 4: Visualize the frames

Create a visualization showing:
1. Original trajectory in FR1 coordinates
2. Original trajectory in FR2 coordinates
3. After equation 5: both trajectories transformed to global frame (should align)
4. After equation 6: fused trajectory

This will help you see where things go wrong.

## Technical Details

### Why einsum and explicit loops are equivalent:

**Einsum version (equation 5 mean):**
```python
xi_hat_ = np.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)
```

**What it does:**
- `i`: frame index
- `k,l`: feature indices (for matrix multiplication)
- `j`: component index
- Computes: xi_hat[i,j,k] = Sum_l rotation[i,k,l] * means[i,j,l]
- Equivalent to: rotation[i,:,:] @ means[i,j,:]

**Explicit version:**
```python
for frame_idx in range(num_frames):
    for comp_idx in range(num_components):
        xi_hat_[frame_idx, comp_idx] = rotation_matrix[frame_idx] @ sorted_means[frame_idx, comp_idx]
```

Both do EXACTLY the same computation, just with different syntax.

## Conclusion

The GMR library's `_equation_5` and `_equation_6` functions are implemented correctly. The issue with FR1!=FR2 not working must be in:

1. How the transformation matrices are computed (cell-34)
2. The physical interpretation of what these transformations represent
3. Possibly in the data preparation or frame definition

I recommend focusing your investigation on the transformation setup in cell-34, particularly:
- The computation of `translation_fr2`
- The meaning of the rotation angle
- How these relate to the actual spatial/temporal relationship between FR1 and FR2 in your gait data
