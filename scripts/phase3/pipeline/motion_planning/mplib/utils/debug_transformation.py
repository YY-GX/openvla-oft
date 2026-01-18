#!/usr/bin/env python3
"""
Debug the transformation between controller and LIBERO frames.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# At reset pose:
print("="*70)
print("DATA AT RESET")
print("="*70)

R_ctrl_reset = np.array([
    [-4.91829687e-04,  9.98386745e-01, -5.67773315e-02],
    [ 9.99999879e-01,  4.92624356e-04,  0.00000000e+00],
    [ 2.79698964e-05, -5.67773246e-02, -9.98386866e-01]
])

R_libero_reset = np.array([
    [ 9.98386866e-01, -8.14845695e-19, -5.67773315e-02],
    [-2.79698964e-05,  9.99999879e-01, -4.91829687e-04],
    [ 5.67773246e-02,  4.92624356e-04,  9.98386745e-01]
])

print("Controller axis (reset):", np.degrees(R.from_matrix(R_ctrl_reset).as_rotvec()))
print("LIBERO axis (reset):", np.degrees(R.from_matrix(R_libero_reset).as_rotvec()))

# At target pose (from user's data):
print("\n" + "="*70)
print("DATA AT TARGET")
print("="*70)

# User says correct target is:
# Controller: axis=[148.311, 99.877, 14.229]°
# LIBERO: axis=[-170.484, 33.271, -10.450]°

R_ctrl_target = R.from_rotvec(np.deg2rad([148.311, 99.877, 14.229])).as_matrix()
R_libero_target = R.from_rotvec(np.deg2rad([-170.484, 33.271, -10.450])).as_matrix()

print("Controller axis (target):", np.degrees(R.from_matrix(R_ctrl_target).as_rotvec()))
print("LIBERO axis (target):", np.degrees(R.from_matrix(R_libero_target).as_rotvec()))

# Compute transformations from both examples
print("\n" + "="*70)
print("COMPUTE TRANSFORMATION MATRICES")
print("="*70)

# Method 1: From reset data
# If R_libero = R_ctrl @ T, then T = R_ctrl^T @ R_libero
T_from_reset = R_ctrl_reset.T @ R_libero_reset
print("\nT from reset (R_ctrl.T @ R_libero):")
print(T_from_reset)
print("Axis-angle:", np.degrees(R.from_matrix(T_from_reset).as_rotvec()))

# Method 2: From target data
T_from_target = R_ctrl_target.T @ R_libero_target
print("\nT from target (R_ctrl.T @ R_libero):")
print(T_from_target)
print("Axis-angle:", np.degrees(R.from_matrix(T_from_target).as_rotvec()))

# Check if they're the same
print(f"\nAre they the same? {np.allclose(T_from_reset, T_from_target, atol=1e-6)}")

# If not the same, try other relationships
if not np.allclose(T_from_reset, T_from_target, atol=1e-6):
    print("\n⚠️  Transformations don't match! Trying other relationships...")

    # Try: R_libero = T @ R_ctrl
    T_alt1_reset = R_libero_reset @ R_ctrl_reset.T
    T_alt1_target = R_libero_target @ R_ctrl_target.T
    print("\nAlternative 1: T = R_libero @ R_ctrl.T")
    print(f"  From reset: {np.degrees(R.from_matrix(T_alt1_reset).as_rotvec())}")
    print(f"  From target: {np.degrees(R.from_matrix(T_alt1_target).as_rotvec())}")
    print(f"  Match? {np.allclose(T_alt1_reset, T_alt1_target, atol=1e-6)}")

    # Try: R_ctrl = R_libero @ T
    T_alt2_reset = R_libero_reset.T @ R_ctrl_reset
    T_alt2_target = R_libero_target.T @ R_ctrl_target
    print("\nAlternative 2: T = R_libero.T @ R_ctrl")
    print(f"  From reset: {np.degrees(R.from_matrix(T_alt2_reset).as_rotvec())}")
    print(f"  From target: {np.degrees(R.from_matrix(T_alt2_target).as_rotvec())}")
    print(f"  Match? {np.allclose(T_alt2_reset, T_alt2_target, atol=1e-6)}")

    # Try: R_ctrl = T @ R_libero
    T_alt3_reset = R_ctrl_reset @ R_libero_reset.T
    T_alt3_target = R_ctrl_target @ R_libero_target.T
    print("\nAlternative 3: T = R_ctrl @ R_libero.T")
    print(f"  From reset: {np.degrees(R.from_matrix(T_alt3_reset).as_rotvec())}")
    print(f"  From target: {np.degrees(R.from_matrix(T_alt3_target).as_rotvec())}")
    print(f"  Match? {np.allclose(T_alt3_reset, T_alt3_target, atol=1e-6)}")

# Test the current implementation
print("\n" + "="*70)
print("TEST CURRENT IMPLEMENTATION")
print("="*70)

T_current = R_ctrl_reset.T @ R_libero_reset
print("Current T:")
print(T_current)

# Convert user's LIBERO target to controller using current method
# Current: R_ctrl = R_libero @ T.T
R_ctrl_converted = R_libero_target @ T_current.T
axis_converted = np.degrees(R.from_matrix(R_ctrl_converted).as_rotvec())

print(f"\nConversion test:")
print(f"  Input (LIBERO): {np.degrees(R.from_matrix(R_libero_target).as_rotvec())}")
print(f"  Output (converted): {axis_converted}")
print(f"  Expected (controller): {np.degrees(R.from_matrix(R_ctrl_target).as_rotvec())}")
print(f"  ❌ Match? {np.allclose(R_ctrl_converted, R_ctrl_target, atol=1e-6)}")

print("\n" + "="*70)
