"""
Quick test to verify gmr_fixed.py works with the TPGMM pipeline
"""

import sys
import numpy as np

# Import directly to avoid matplotlib dependency
sys.path.insert(0, 'D:\\Github\\Gait-analysis-coupled\\TaskPaGMMM')
from tpgmm.tpgmm.tpgmm import TPGMM
from tpgmm.gmr.gmr import GaussianMixtureRegression as GMR_Original
from tpgmm.gmr.gmr_fixed import GaussianMixtureRegression as GMR_Fixed

# Create synthetic test data
np.random.seed(42)
num_frames = 2
num_demos = 5
num_points = 50
num_features = 4  # x, y, velocity, time

# Generate synthetic trajectories for 2 frames
trajectories = np.random.randn(num_frames, num_demos * num_points, num_features)

# Make time monotonic
for frame_idx in range(num_frames):
    trajectories[frame_idx, :, -1] = np.sort(trajectories[frame_idx, :, -1])

print("=" * 80)
print("Testing GMR Original vs GMR Fixed")
print("=" * 80)
print(f"Trajectories shape: {trajectories.shape}")
print()

# Train TPGMM
print("Training TPGMM...")
tpgmm = TPGMM(n_components=3, reg_factor=1e-5)
tpgmm.fit(trajectories)
print(f"TPGMM trained: {tpgmm.n_components} components")
print()

# Create rotation and translation for 2 frames
rotation_matrix = np.array([
    np.eye(3),  # FR1: identity
    np.array([[0.87, -0.5, 0],
              [0.5, 0.87, 0],
              [0, 0, 1]])  # FR2: 30 degree rotation
])
translation = np.array([
    [0, 0, 0],  # FR1: no translation
    [1, 2, 0]   # FR2: offset by (1, 2)
])

print("Testing GMR Original...")
gmr_orig = GMR_Original.from_tpgmm(tpgmm, input_idx=[3])
gmr_orig.fit(translation, rotation_matrix)
print("GMR Original fit complete")
print()

print("Testing GMR Fixed...")
gmr_fixed = GMR_Fixed.from_tpgmm(tpgmm, input_idx=[3])
gmr_fixed.fit(translation, rotation_matrix)
print("GMR Fixed fit complete")
print()

# Compare the results
print("Comparing xi_ (means):")
print(f"  Max difference: {np.max(np.abs(gmr_orig.xi_ - gmr_fixed.xi_))}")
print(f"  Are they close? {np.allclose(gmr_orig.xi_, gmr_fixed.xi_)}")
print()

print("Comparing sigma_ (covariances):")
print(f"  Max difference: {np.max(np.abs(gmr_orig.sigma_ - gmr_fixed.sigma_))}")
print(f"  Are they close? {np.allclose(gmr_orig.sigma_, gmr_fixed.sigma_)}")
print()

# Test prediction
test_times = np.linspace(trajectories[0, :, -1].min(), trajectories[0, :, -1].max(), 20).reshape(-1, 1)

print("Testing prediction on 20 time points...")
mu_orig, cov_orig = gmr_orig.predict(test_times)
mu_fixed, cov_fixed = gmr_fixed.predict(test_times)

print("Comparing predicted means:")
print(f"  Max difference: {np.max(np.abs(mu_orig - mu_fixed))}")
print(f"  Are they close? {np.allclose(mu_orig, mu_fixed)}")
print()

print("Comparing predicted covariances:")
print(f"  Max difference: {np.max(np.abs(cov_orig - cov_fixed))}")
print(f"  Are they close? {np.allclose(cov_orig, cov_fixed)}")
print()

if np.allclose(gmr_orig.xi_, gmr_fixed.xi_) and \
   np.allclose(gmr_orig.sigma_, gmr_fixed.sigma_) and \
   np.allclose(mu_orig, mu_fixed) and \
   np.allclose(cov_orig, cov_fixed):
    print("=" * 80)
    print("[SUCCESS] GMR Fixed produces identical results to GMR Original!")
    print("=" * 80)
    print()
    print("This confirms that:")
    print("1. The explicit loop implementation is correct")
    print("2. Both versions are mathematically equivalent")
    print("3. The FR1!=FR2 issue is NOT in _equation_5 or _equation_6")
    print()
    print("Next step: Investigate the transformation setup in your notebook (cell-34)")
else:
    print("=" * 80)
    print("[WARNING] GMR Fixed produces different results!")
    print("=" * 80)
    print("This suggests a bug in the reimplementation.")
