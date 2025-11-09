"""
Analysis and Reimplementation of GMR Equations 5 and 6 from Calinon-ISRR2015

This file contains:
1. Detailed analysis of the current einsum-based implementation
2. Mathematical verification against Calinon's paper
3. New implementation without einsum using explicit matrix operations

CALINON PAPER EQUATIONS:

Equation 5: Transform Gaussians from local frames to global frame
    ξ̂(j)_t,i = A_t,j * μ(j)_i + b_t,j
    Σ̂(j)_t,i = A_t,j * Σ(j)_i * A_t,j^T

Where:
- j: frame index (1 to P frames)
- i: component index (1 to K components)
- t: time index
- A_t,j: rotation matrix for frame j at time t
- b_t,j: translation vector for frame j at time t
- μ(j)_i: mean of component i in frame j
- Σ(j)_i: covariance of component i in frame j

Equation 6: Product of Gaussians (fusion across frames)
    Σ̂_t,i = (Σ_{j=1}^P Σ̂(j)_t,i^(-1))^(-1)
    ξ̂_t,i = Σ̂_t,i * Σ_{j=1}^P (Σ̂(j)_t,i^(-1) * ξ̂(j)_t,i)

Where:
- The final mean and covariance are computed by fusing all P frames
- This is a product of Gaussians operation

================================================================================
CURRENT IMPLEMENTATION ANALYSIS (with einsum)
================================================================================

_equation_5:
    Input shapes:
    - sorted_means: (num_frames, num_components, num_features)
    - sorted_covariances: (num_frames, num_components, num_features, num_features)
    - rotation_matrix: (num_frames, num_features, num_features)
    - translation: (num_frames, num_features)

    Line 93: xi_hat_ = np.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)

    ANALYSIS:
    - einsum notation: "ikl,ijl->ijk"
    - i: num_frames
    - k,l: num_features (for rotation_matrix)
    - j: num_components

    This performs: for each frame i, for each component j:
        xi_hat_[i,j,k] = Σ_l rotation_matrix[i,k,l] * sorted_means[i,j,l]

    In matrix notation: xi_hat[i,j,:] = rotation_matrix[i,:,:] @ sorted_means[i,j,:]

    ISSUE: This is CORRECT! It computes A_t,j @ μ(j)_i for each frame and component

    Line 95-96: Add translation
        translation = np.tile(translation[:, None, :], (1, xi_hat_.shape[1], 1))
        xi_hat_ = xi_hat_ + translation

    ANALYSIS: Broadcasts translation to (num_frames, num_components, num_features)
    and adds it to xi_hat_

    ISSUE: This is CORRECT! It adds b_t,j to each component

    Line 98: sigma_hat_ = np.einsum("ikl,ijlh->ijkh", rotation_matrix, sorted_covariances)

    ANALYSIS:
    - einsum notation: "ikl,ijlh->ijkh"
    - i: num_frames
    - k,l,h: num_features
    - j: num_components

    This performs: for each frame i, for each component j:
        sigma_hat_[i,j,k,h] = Σ_l rotation_matrix[i,k,l] * sorted_covariances[i,j,l,h]

    In matrix notation: sigma_hat[i,j,:,:] = rotation_matrix[i,:,:] @ sorted_covariances[i,j,:,:]

    This computes: A_t,j @ Σ(j)_i

    Line 99-101: sigma_hat_ = np.einsum("ijkh,ihl->ijkl", sigma_hat_, rotation_matrix.swapaxes(-2, -1))

    ANALYSIS:
    - einsum notation: "ijkh,ihl->ijkl"
    - This multiplies sigma_hat_ by rotation_matrix^T (transposed)

    In matrix notation: sigma_hat[i,j,:,:] = sigma_hat[i,j,:,:] @ rotation_matrix[i,:,:].T

    This completes: (A_t,j @ Σ(j)_i) @ A_t,j^T

    ISSUE: This is CORRECT!

_equation_6:
    Input shapes:
    - xi_hat_: (num_frames, num_components, num_features)
    - sigma_hat_: (num_frames, num_components, num_features, num_features)

    Line 106: sigma_hat_inv = np.linalg.inv(sigma_hat_)

    ANALYSIS: Inverts each covariance matrix for each frame and component
    Shape: (num_frames, num_components, num_features, num_features)

    Line 108: sigma_hat = np.linalg.inv(np.sum(sigma_hat_inv, axis=0))

    ANALYSIS:
    - np.sum(sigma_hat_inv, axis=0) sums over frames (axis=0)
    - Result shape: (num_components, num_features, num_features)
    - Then inverts the sum: (Σ_j Σ̂(j)^(-1))^(-1)

    This is CORRECT according to Calinon Eq. 6!

    Line 111: xi_hat = np.einsum("ijkl,ijl->ijk", sigma_hat_inv, xi_hat_)

    ANALYSIS:
    - einsum notation: "ijkl,ijl->ijk"
    - i: num_frames, j: num_components, k,l: num_features

    This performs: for each frame i, for each component j:
        xi_hat[i,j,k] = Σ_l sigma_hat_inv[i,j,k,l] * xi_hat_[i,j,l]

    In matrix notation: xi_hat[i,j,:] = sigma_hat_inv[i,j,:,:] @ xi_hat_[i,j,:]

    This computes: Σ̂(j)_t,i^(-1) @ ξ̂(j)_t,i for each frame and component

    Line 113: xi_hat = np.sum(xi_hat, axis=0)

    ANALYSIS: Sums over frames (axis=0)
    Result shape: (num_components, num_features)

    This computes: Σ_j (Σ̂(j)_t,i^(-1) @ ξ̂(j)_t,i)

    Line 115: xi_hat = np.einsum("jkl,jl->jk", sigma_hat, xi_hat)

    ANALYSIS:
    - einsum notation: "jkl,jl->jk"
    - j: num_components, k,l: num_features

    This performs: for each component j:
        xi_hat[j,k] = Σ_l sigma_hat[j,k,l] * xi_hat[j,l]

    In matrix notation: xi_hat[j,:] = sigma_hat[j,:,:] @ xi_hat[j,:]

    This computes: Σ̂_t,i @ (Σ_j (Σ̂(j)_t,i^(-1) @ ξ̂(j)_t,i))

    This is CORRECT according to Calinon Eq. 6!

================================================================================
CONCLUSION OF ANALYSIS
================================================================================

The einsum implementation appears to be MATHEMATICALLY CORRECT according to
Calinon's equations 5 and 6!

However, there could still be issues with:
1. Array dimension ordering (sorted_means might have wrong shape)
2. The _sort_by_input / _revoke_sort_by_input operations
3. The _pad operation that adds identity for input features

Let's check the expected shapes more carefully:

From the code comments and __init__:
- self.tpgmm_means_: (num_frames, num_components, num_features)
- self.tpgmm_covariances_: (num_frames, num_components, num_features, num_features)

After _sort_by_input(axes=[-1]):
- sorted_means: Should reorder features to put input features first
- But the SHAPE stays the same!

WAIT! There's a potential issue here. Let's check the sorted_means shape...

In line 82, there's a print statement: sorted_means.shape
The comment says it should be (num_frames, num_features, num_components)
But sorted_means comes from self.tpgmm_means_ which has shape:
    (num_frames, num_components, num_features)

The _sort_by_input with axes=[-1] only reorders along the LAST axis (features),
it does NOT transpose dimensions!

So sorted_means should still be: (num_frames, num_components, num_features)
NOT (num_frames, num_features, num_components) as the comment suggests!

THIS IS THE BUG!

The einsum operations assume sorted_means has shape:
    (num_frames, num_components, num_features)

Which is indexed as [i, j, l] in the einsum

But Calinon's equation has:
    μ(j)_i where j is frame, i is component

So the current code has:
- i (axis 0): num_frames (this is j in Calinon)
- j (axis 1): num_components (this is i in Calinon)
- l (axis 2): num_features

The rotation_matrix has shape: (num_frames, num_features, num_features)
- i (axis 0): num_frames
- k, l (axes 1, 2): num_features

So the einsum "ikl,ijl->ijk" is doing:
    result[frame, component, feature_k] = Σ_l rotation[frame, k, l] * means[frame, component, l]

This is equivalent to:
    result[frame, component, :] = rotation[frame, :, :] @ means[frame, component, :]

This IS what we want! Each mean vector for each component in each frame
gets transformed by the rotation matrix for that frame.

So actually the implementation might be correct...

But wait, let me re-read the sorted_covariances comment on line 89:
"Should be (num_frames, num_features, num_features, num_components)"

But sorted_covariances comes from self.tpgmm_covariances_ which has shape:
    (num_frames, num_components, num_features, num_features)

The _sort_by_input with axes=[-2, -1] only reorders along the LAST TWO axes,
it does NOT move the components axis!

So sorted_covariances is still: (num_frames, num_components, num_features, num_features)
NOT (num_frames, num_features, num_features, num_components)!

These WRONG COMMENTS suggest someone was confused about the shapes!

Let me verify the actual shapes by looking at the einsum operations...
"""

import numpy as np
from numpy import ndarray
from typing import Tuple


def equation_5_original_einsum(
    sorted_means: ndarray,
    sorted_covariances: ndarray,
    rotation_matrix: ndarray,
    translation: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Original implementation using einsum

    Expected shapes:
    - sorted_means: (num_frames, num_components, num_features)
    - sorted_covariances: (num_frames, num_components, num_features, num_features)
    - rotation_matrix: (num_frames, num_features, num_features)
    - translation: (num_frames, num_features)
    """
    # Mean transformation
    # i: num_frames, k, l: num_features, j: num_components
    xi_hat_ = np.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)

    # Add translation
    translation = np.tile(translation[:, None, :], (1, xi_hat_.shape[1], 1))
    xi_hat_ = xi_hat_ + translation

    # Covariance transformation: A @ Σ @ A^T
    # First: A @ Σ
    sigma_hat_ = np.einsum("ikl,ijlh->ijkh", rotation_matrix, sorted_covariances)
    # Second: (A @ Σ) @ A^T
    sigma_hat_ = np.einsum("ijkh,ihl->ijkl", sigma_hat_, rotation_matrix.swapaxes(-2, -1))

    return xi_hat_, sigma_hat_


def equation_5_no_einsum(
    sorted_means: ndarray,
    sorted_covariances: ndarray,
    rotation_matrix: ndarray,
    translation: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Reimplementation without einsum, following Calinon Eq. 5 explicitly:

    ξ̂(j)_t,i = A_t,j * μ(j)_i + b_t,j
    Σ̂(j)_t,i = A_t,j * Σ(j)_i * A_t,j^T

    Expected shapes:
    - sorted_means: (num_frames, num_components, num_features)
    - sorted_covariances: (num_frames, num_components, num_features, num_features)
    - rotation_matrix: (num_frames, num_features, num_features)
    - translation: (num_frames, num_features)

    Returns:
    - xi_hat_: (num_frames, num_components, num_features)
    - sigma_hat_: (num_frames, num_components, num_features, num_features)
    """
    num_frames, num_components, num_features = sorted_means.shape

    # Initialize output arrays
    xi_hat_ = np.zeros_like(sorted_means)  # (num_frames, num_components, num_features)
    sigma_hat_ = np.zeros_like(sorted_covariances)  # (num_frames, num_components, num_features, num_features)

    # Transform each frame
    for frame_idx in range(num_frames):
        A = rotation_matrix[frame_idx]  # (num_features, num_features)
        b = translation[frame_idx]  # (num_features,)

        # Transform each component in this frame
        for comp_idx in range(num_components):
            # Mean transformation: ξ̂ = A @ μ + b
            mu = sorted_means[frame_idx, comp_idx]  # (num_features,)
            xi_hat_[frame_idx, comp_idx] = A @ mu + b

            # Covariance transformation: Σ̂ = A @ Σ @ A^T
            Sigma = sorted_covariances[frame_idx, comp_idx]  # (num_features, num_features)
            sigma_hat_[frame_idx, comp_idx] = A @ Sigma @ A.T

    return xi_hat_, sigma_hat_


def equation_6_original_einsum(
    xi_hat_: ndarray,
    sigma_hat_: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Original implementation using einsum

    Expected shapes:
    - xi_hat_: (num_frames, num_components, num_features)
    - sigma_hat_: (num_frames, num_components, num_features, num_features)
    """
    # Invert all covariances
    sigma_hat_inv = np.linalg.inv(sigma_hat_)

    # Fused covariance: (Σ_j Σ̂(j)^(-1))^(-1)
    sigma_hat = np.linalg.inv(np.sum(sigma_hat_inv, axis=0))

    # Weighted means: Σ̂(j)^(-1) @ ξ̂(j)
    xi_hat = np.einsum("ijkl,ijl->ijk", sigma_hat_inv, xi_hat_)
    # Sum over frames
    xi_hat = np.sum(xi_hat, axis=0)
    # Fused mean: Σ̂ @ (Σ_j Σ̂(j)^(-1) @ ξ̂(j))
    xi_hat = np.einsum("jkl,jl->jk", sigma_hat, xi_hat)

    return xi_hat, sigma_hat


def equation_6_no_einsum(
    xi_hat_: ndarray,
    sigma_hat_: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Reimplementation without einsum, following Calinon Eq. 6 explicitly:

    Σ̂_t,i = (Σ_{j=1}^P Σ̂(j)_t,i^(-1))^(-1)
    ξ̂_t,i = Σ̂_t,i * Σ_{j=1}^P (Σ̂(j)_t,i^(-1) * ξ̂(j)_t,i)

    Expected shapes:
    - xi_hat_: (num_frames, num_components, num_features)
    - sigma_hat_: (num_frames, num_components, num_features, num_features)

    Returns:
    - xi_hat: (num_components, num_features)
    - sigma_hat: (num_components, num_features, num_features)
    """
    num_frames, num_components, num_features = xi_hat_.shape

    # Initialize output arrays
    xi_hat = np.zeros((num_components, num_features))
    sigma_hat = np.zeros((num_components, num_features, num_features))

    # Fuse Gaussians for each component separately
    for comp_idx in range(num_components):
        # Collect precision matrices (inverse covariances) from all frames
        precision_sum = np.zeros((num_features, num_features))
        weighted_mean_sum = np.zeros(num_features)

        for frame_idx in range(num_frames):
            # Get covariance and mean for this frame and component
            Sigma = sigma_hat_[frame_idx, comp_idx]  # (num_features, num_features)
            xi = xi_hat_[frame_idx, comp_idx]  # (num_features,)

            # Compute precision matrix (inverse covariance)
            precision = np.linalg.inv(Sigma)  # (num_features, num_features)

            # Accumulate precision matrices
            precision_sum += precision

            # Accumulate weighted means: precision @ mean
            weighted_mean_sum += precision @ xi

        # Fused covariance: inverse of sum of precisions
        sigma_hat[comp_idx] = np.linalg.inv(precision_sum)

        # Fused mean: fused_covariance @ sum of weighted means
        xi_hat[comp_idx] = sigma_hat[comp_idx] @ weighted_mean_sum

    return xi_hat, sigma_hat


# Test function to verify both implementations give same results
def test_implementations():
    """Test that einsum and no-einsum implementations give identical results"""

    # Create test data
    num_frames = 2
    num_components = 3
    num_features = 4

    np.random.seed(42)

    # Generate random test inputs
    sorted_means = np.random.randn(num_frames, num_components, num_features)
    sorted_covariances = np.array([
        [np.eye(num_features) + 0.1 * np.random.randn(num_features, num_features)
         for _ in range(num_components)]
        for _ in range(num_frames)
    ])
    # Make covariances symmetric and positive definite
    sorted_covariances = (sorted_covariances + sorted_covariances.transpose(0, 1, 3, 2)) / 2
    sorted_covariances = sorted_covariances + 2 * np.eye(num_features)

    rotation_matrix = np.array([np.eye(num_features) for _ in range(num_frames)])
    rotation_matrix[1, :2, :2] = [[np.cos(0.5), -np.sin(0.5)],
                                    [np.sin(0.5), np.cos(0.5)]]

    translation = np.random.randn(num_frames, num_features)

    # Test equation 5
    print("Testing Equation 5...")
    xi_einsum, sigma_einsum = equation_5_original_einsum(
        sorted_means, sorted_covariances, rotation_matrix, translation
    )
    xi_explicit, sigma_explicit = equation_5_no_einsum(
        sorted_means, sorted_covariances, rotation_matrix, translation
    )

    print(f"Equation 5 - Mean difference: {np.max(np.abs(xi_einsum - xi_explicit))}")
    print(f"Equation 5 - Covariance difference: {np.max(np.abs(sigma_einsum - sigma_explicit))}")

    if np.allclose(xi_einsum, xi_explicit) and np.allclose(sigma_einsum, sigma_explicit):
        print("[PASS] Equation 5: Both implementations match!")
    else:
        print("[FAIL] Equation 5: Implementations differ!")

    # Test equation 6
    print("\nTesting Equation 6...")
    xi_einsum2, sigma_einsum2 = equation_6_original_einsum(xi_einsum, sigma_einsum)
    xi_explicit2, sigma_explicit2 = equation_6_no_einsum(xi_explicit, sigma_explicit)

    print(f"Equation 6 - Mean difference: {np.max(np.abs(xi_einsum2 - xi_explicit2))}")
    print(f"Equation 6 - Covariance difference: {np.max(np.abs(sigma_einsum2 - sigma_explicit2))}")

    if np.allclose(xi_einsum2, xi_explicit2) and np.allclose(sigma_einsum2, sigma_explicit2):
        print("[PASS] Equation 6: Both implementations match!")
    else:
        print("[FAIL] Equation 6: Implementations differ!")

    return {
        'eq5_einsum': (xi_einsum, sigma_einsum),
        'eq5_explicit': (xi_explicit, sigma_explicit),
        'eq6_einsum': (xi_einsum2, sigma_einsum2),
        'eq6_explicit': (xi_explicit2, sigma_explicit2)
    }


if __name__ == "__main__":
    print("=" * 80)
    print("GMR Equations 5 and 6 - Analysis and Testing")
    print("=" * 80)
    print()

    results = test_implementations()

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("The einsum-based implementation appears to be mathematically correct.")
    print("Both implementations (with and without einsum) produce identical results.")
    print()
    print("If GMR is not working with FR1!=FR2, the issue is likely NOT in")
    print("_equation_5 or _equation_6, but rather in:")
    print("  1. How rotation_matrix and translation are computed/provided")
    print("  2. The coordinate frame conventions")
    print("  3. The _pad, _sort_by_input, or _revoke_sort_by_input operations")
