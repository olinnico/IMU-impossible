#!/usr/bin/env python3
"""
four_imu_3d_ekf.py

Extended Kalman Filter for 4 IMUs with full 3D rigid body constraints.

Uses imusensor Kalman filter (from your SOP) for smooth attitude estimation,
then applies rigid body constraints to validate accelerometer measurements.

IMU Configuration:
- CH2 @ (0,0,0):     ±2g range
- CH3 @ (0,80,5):    ±4g range  
- CH0 @ (80,80,10):  ±8g range
- CH1 @ (80,0,15):   ±16g range

All positions in millimeters, converted to meters in code.

State vector includes:
- Global position and velocity (x, y, z, vx, vy, vz)
- Orientation (roll, pitch, yaw) - from imusensor Kalman filter
- Angular velocity (wx, wy, wz)
- Gyro biases for each IMU (4 × 3 = 12 bias states)
- Delta accel biases between IMUs (3 × 3 = 9 bias states, relative to CH2)

Measurements:
- 4 gyroscopes (12 measurements)
- 4 accelerometers differenced from CH2 (9 measurements)
- Total: 21 measurements

Dependencies:
- numpy
- hardware_interface.py (your IMU reader)
- imusensor (for attitude Kalman filter)
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from hardware_interface import IMUHardware
from imusensor.filters import kalman

# -----------------------------
# Utility functions
# -----------------------------

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3-vector for cross product.
    
    [v]× such that [v]× @ u = v × u
    
    Args:
        v: shape (3,) or (3,1)
    
    Returns:
        3×3 skew-symmetric matrix
    """
    v = np.asarray(v).flatten()
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute 3D rotation matrix from Euler angles (Z-Y-X convention).
    
    Args:
        roll: rotation about X axis (rad)
        pitch: rotation about Y axis (rad)
        yaw: rotation about Z axis (rad)
    
    Returns:
        3×3 rotation matrix R_GB (body to global)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr  ]
    ])
    return R


def std_zero_mean(samples: np.ndarray) -> np.ndarray:
    """
    Compute sample standard deviation after removing mean.
    
    Args:
        samples: shape (N,) or (N, d)
    
    Returns:
        std per dimension
    """
    s = np.asarray(samples, dtype=float)
    s0 = s - np.mean(s, axis=0, keepdims=True)
    return np.std(s0, axis=0, ddof=1)


# -----------------------------
# Attitude Estimation using imusensor Kalman Filter
# -----------------------------

class AttitudeFilter:
    """
    Manages attitude estimation for all 4 IMUs using imusensor Kalman filters.
    
    This replicates the approach from your SOP but for 4 IMUs simultaneously.
    Each IMU gets its own Kalman filter for smooth roll/pitch/yaw.
    """
    
    def __init__(self):
        """Initialize Kalman filters for all 4 IMUs."""
        # Create separate Kalman filter for each IMU
        self.kf_ch2 = kalman.Kalman()
        self.kf_ch3 = kalman.Kalman()
        self.kf_ch0 = kalman.Kalman()
        self.kf_ch1 = kalman.Kalman()
        
        print("Initialized imusensor Kalman filters for all 4 IMUs")
    
    def update_ch2(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray, dt: float) -> np.ndarray:
        """
        Update CH2 attitude using imusensor Kalman filter.
        
        Args:
            accel: acceleration (3,) in m/s²
            gyro: gyro (3,) in rad/s (will convert to deg/s internally)
            mag: magnetometer (3,) in µT
            dt: time step (seconds)
        
        Returns:
            attitude (3,): [roll, pitch, yaw] in radians
        """
        # imusensor expects deg/s for gyro
        gyro_deg = np.rad2deg(gyro)
        
        self.kf_ch2.computeAndUpdateRollPitchYaw(
            accel[0], accel[1], accel[2],
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            mag[0], mag[1], mag[2],
            dt
        )
        
        # Return in radians
        return np.array([
            np.deg2rad(self.kf_ch2.roll),
            np.deg2rad(self.kf_ch2.pitch),
            np.deg2rad(self.kf_ch2.yaw)
        ])
    
    def update_ch3(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray, dt: float) -> np.ndarray:
        """Update CH3 attitude."""
        gyro_deg = np.rad2deg(gyro)
        self.kf_ch3.computeAndUpdateRollPitchYaw(
            accel[0], accel[1], accel[2],
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            mag[0], mag[1], mag[2],
            dt
        )
        return np.array([
            np.deg2rad(self.kf_ch3.roll),
            np.deg2rad(self.kf_ch3.pitch),
            np.deg2rad(self.kf_ch3.yaw)
        ])
    
    def update_ch0(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray, dt: float) -> np.ndarray:
        """Update CH0 attitude."""
        gyro_deg = np.rad2deg(gyro)
        self.kf_ch0.computeAndUpdateRollPitchYaw(
            accel[0], accel[1], accel[2],
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            mag[0], mag[1], mag[2],
            dt
        )
        return np.array([
            np.deg2rad(self.kf_ch0.roll),
            np.deg2rad(self.kf_ch0.pitch),
            np.deg2rad(self.kf_ch0.yaw)
        ])
    
    def update_ch1(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray, dt: float) -> np.ndarray:
        """Update CH1 attitude."""
        gyro_deg = np.rad2deg(gyro)
        self.kf_ch1.computeAndUpdateRollPitchYaw(
            accel[0], accel[1], accel[2],
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            mag[0], mag[1], mag[2],
            dt
        )
        return np.array([
            np.deg2rad(self.kf_ch1.roll),
            np.deg2rad(self.kf_ch1.pitch),
            np.deg2rad(self.kf_ch1.yaw)
        ])
    
    def update_all(self, data: dict, dt: float) -> dict:
        """
        Update all 4 IMU attitudes.
        
        Args:
            data: dict with keys 'ch2', 'ch3', 'ch0', 'ch1'
                  each containing (accel, gyro, mag) tuples
            dt: time step
        
        Returns:
            dict with keys 'ch2', 'ch3', 'ch0', 'ch1'
            each containing attitude [roll, pitch, yaw] in radians
        """
        acc_ch2, gyr_ch2, mag_ch2 = data['ch2']
        acc_ch3, gyr_ch3, mag_ch3 = data['ch3']
        acc_ch0, gyr_ch0, mag_ch0 = data['ch0']
        acc_ch1, gyr_ch1, mag_ch1 = data['ch1']
        
        return {
            'ch2': self.update_ch2(acc_ch2, gyr_ch2, mag_ch2, dt),
            'ch3': self.update_ch3(acc_ch3, gyr_ch3, mag_ch3, dt),
            'ch0': self.update_ch0(acc_ch0, gyr_ch0, mag_ch0, dt),
            'ch1': self.update_ch1(acc_ch1, gyr_ch1, mag_ch1, dt)
        }
    
    def get_consensus_attitude(self, attitudes: dict) -> np.ndarray:
        """
        Compute consensus attitude from all 4 IMUs (median).
        
        Args:
            attitudes: dict from update_all()
        
        Returns:
            consensus attitude [roll, pitch, yaw] in radians
        """
        att_ch2 = attitudes['ch2']
        att_ch3 = attitudes['ch3']
        att_ch0 = attitudes['ch0']
        att_ch1 = attitudes['ch1']
        
        # Stack and take median
        all_att = np.array([att_ch2, att_ch3, att_ch0, att_ch1])
        consensus = np.median(all_att, axis=0)
        
        return consensus


# -----------------------------
# EKF Configuration
# -----------------------------

@dataclass
class FourIMU_EKF_Config:
    """Configuration for 4-IMU 3D EKF."""
    
    # IMU positions in body frame (meters)
    r_ch2: np.ndarray  # (3,) - origin
    r_ch3: np.ndarray  # (3,)
    r_ch0: np.ndarray  # (3,)
    r_ch1: np.ndarray  # (3,)
    
    # Alignment matrices (identity if pre-aligned)
    C_ch2: np.ndarray  # (3,3)
    C_ch3: np.ndarray  # (3,3)
    C_ch0: np.ndarray  # (3,3)
    C_ch1: np.ndarray  # (3,3)
    
    # Process noise variances
    q_v: float           # velocity random walk (m/s)^2 per step
    q_omega: float       # angular velocity random walk (rad/s)^2 per step
    q_att: float         # attitude random walk (rad)^2 per step
    q_bg: float          # gyro bias random walk (rad/s)^2 per step
    q_dba: float         # delta accel bias random walk (m/s^2)^2 per step
    
    # Measurement noise variances
    # Gyro noise (one per IMU per axis)
    r_gyro_ch2: np.ndarray  # (3,) variances for x,y,z
    r_gyro_ch3: np.ndarray  # (3,)
    r_gyro_ch0: np.ndarray  # (3,)
    r_gyro_ch1: np.ndarray  # (3,)
    
    # Delta accel noise (CH3-CH2, CH0-CH2, CH1-CH2)
    r_da_ch3: np.ndarray    # (3,)
    r_da_ch0: np.ndarray    # (3,)
    r_da_ch1: np.ndarray    # (3,)


# -----------------------------
# 4-IMU 3D EKF Class
# -----------------------------

class FourIMU_3D_EKF:
    """
    Extended Kalman Filter for 4 IMUs with full 3D rigid body constraints.
    
    State vector (36 dimensions):
        x = [x, y, z,                    # 0-2:   position (m)
             vx, vy, vz,                 # 3-5:   velocity (m/s)
             roll, pitch, yaw,           # 6-8:   attitude (rad)
             wx, wy, wz,                 # 9-11:  angular velocity (rad/s)
             bg_ch2_x, bg_ch2_y, bg_ch2_z,   # 12-14: gyro bias CH2
             bg_ch3_x, bg_ch3_y, bg_ch3_z,   # 15-17: gyro bias CH3
             bg_ch0_x, bg_ch0_y, bg_ch0_z,   # 18-20: gyro bias CH0
             bg_ch1_x, bg_ch1_y, bg_ch1_z,   # 21-23: gyro bias CH1
             dba_ch3_x, dba_ch3_y, dba_ch3_z,  # 24-26: delta accel bias CH3-CH2
             dba_ch0_x, dba_ch0_y, dba_ch0_z,  # 27-29: delta accel bias CH0-CH2
             dba_ch1_x, dba_ch1_y, dba_ch1_z]  # 30-32: delta accel bias CH1-CH2
    
    Measurements (21 dimensions):
        z = [gyro_ch2 (3),           # 0-2
             gyro_ch3 (3),           # 3-5
             gyro_ch0 (3),           # 6-8
             gyro_ch1 (3),           # 9-11
             delta_accel_ch3 (3),    # 12-14: CH3 - CH2
             delta_accel_ch0 (3),    # 15-17: CH0 - CH2
             delta_accel_ch1 (3)]    # 18-20: CH1 - CH2
    """
    
    def __init__(self, cfg: FourIMU_EKF_Config):
        self.cfg = cfg
        self.n = 33  # state dimension
        self.m = 21  # measurement dimension
        
        # Initialize state
        self.x = np.zeros((self.n, 1), dtype=float)
        
        # Initial covariance
        P0 = np.diag([
            10.0**2, 10.0**2, 10.0**2,           # x,y,z position (m^2)
            1.0**2, 1.0**2, 1.0**2,              # vx,vy,vz velocity
            np.deg2rad(30)**2, np.deg2rad(30)**2, np.deg2rad(30)**2,  # attitude
            np.deg2rad(30)**2, np.deg2rad(30)**2, np.deg2rad(30)**2,  # ang vel
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,     # bg_ch2
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,     # bg_ch3
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,     # bg_ch0
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,     # bg_ch1
            0.5**2, 0.5**2, 0.5**2,              # dba_ch3
            0.5**2, 0.5**2, 0.5**2,              # dba_ch0
            0.5**2, 0.5**2, 0.5**2,              # dba_ch1
        ])
        self.P = P0
        
        # Store previous omega for angular acceleration approximation
        self._omega_prev = np.zeros(3)
        
        # Store previous angular velocities (smoothing buffer)
        self._omega_history = [np.zeros(3) for _ in range(3)]
    
    def predict(self, dt: float):
        """
        EKF prediction step.
        
        Args:
            dt: time step (seconds)
        """
        dt = float(dt)
        if dt <= 0:
            return
        
        x = self.x.flatten()
        
        # Extract current state
        pos = x[0:3]
        vel = x[3:6]
        att = x[6:9]   # roll, pitch, yaw
        omega = x[9:12]
        # Biases remain at x[12:33]
        
        # Simple kinematic prediction
        pos_new = pos + vel * dt
        vel_new = vel  # Constant velocity model
        
        # Attitude integration (simple Euler integration)
        roll, pitch, yaw = att
        wx, wy, wz = omega
        
        # Euler angle rates from body angular velocity
        # [roll_dot, pitch_dot, yaw_dot]^T = T(roll, pitch) @ [wx, wy, wz]^T
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)
        
        roll_dot = wx + wy * sr * tp + wz * cr * tp
        pitch_dot = wy * cr - wz * sr
        yaw_dot = (wy * sr + wz * cr) / cp if abs(cp) > 0.01 else 0.0
        
        att_new = att + np.array([roll_dot, pitch_dot, yaw_dot]) * dt
        
        # Angular velocity and biases remain constant (random walk model)
        omega_new = omega
        biases_new = x[12:33]
        
        # Update state
        self.x = np.concatenate([
            pos_new, vel_new, att_new, omega_new, biases_new
        ]).reshape(-1, 1)
        
        # Jacobian F (linearized dynamics)
        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * dt  # pos depends on vel
        F[6:9, 9:12] = np.eye(3) * dt  # attitude depends on omega (simplified)
        
        # Process noise Q
        Q = np.zeros((self.n, self.n))
        Q[3:6, 3:6] = np.eye(3) * self.cfg.q_v * dt**2        # velocity
        Q[6:9, 6:9] = np.eye(3) * self.cfg.q_att * dt**2      # attitude
        Q[9:12, 9:12] = np.eye(3) * self.cfg.q_omega * dt**2  # angular vel
        Q[12:24, 12:24] = np.eye(12) * self.cfg.q_bg * dt     # gyro biases
        Q[24:33, 24:33] = np.eye(9) * self.cfg.q_dba * dt     # accel biases
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + Q
    
    def update(self, z: np.ndarray, dt: float):
        """
        EKF measurement update.
        
        Args:
            z: measurement vector (21,) containing:
               - gyro_ch2 (3)
               - gyro_ch3 (3)
               - gyro_ch0 (3)
               - gyro_ch1 (3)
               - delta_accel_ch3 (3)
               - delta_accel_ch0 (3)
               - delta_accel_ch1 (3)
            dt: time step (seconds)
        """
        z = np.asarray(z, dtype=float).reshape((self.m, 1))
        dt = float(dt)
        if dt <= 0:
            return
        
        x = self.x.flatten()
        
        # Extract state
        att = x[6:9]
        omega = x[9:12]
        bg_ch2 = x[12:15]
        bg_ch3 = x[15:18]
        bg_ch0 = x[18:21]
        bg_ch1 = x[21:24]
        dba_ch3 = x[24:27]
        dba_ch0 = x[27:30]
        dba_ch1 = x[30:33]
        
        # Compute angular acceleration (smoothed finite difference)
        self._omega_history.pop(0)
        self._omega_history.append(omega.copy())
        omega_smooth = np.mean(self._omega_history, axis=0)
        alpha = (omega_smooth - self._omega_prev) / dt
        self._omega_prev = omega_smooth.copy()
        
        # Rotation matrix (body to global)
        R_GB = rotation_matrix(att[0], att[1], att[2])
        
        # Position vectors from CH2 to other IMUs (in body frame)
        r_ch2_body = self.cfg.r_ch2
        r_ch3_body = self.cfg.r_ch3
        r_ch0_body = self.cfg.r_ch0
        r_ch1_body = self.cfg.r_ch1
        
        # Lever arms (relative to CH2)
        dr_ch3_body = r_ch3_body - r_ch2_body
        dr_ch0_body = r_ch0_body - r_ch2_body
        dr_ch1_body = r_ch1_body - r_ch2_body
        
        # Transform to global frame
        dr_ch3_global = R_GB @ dr_ch3_body
        dr_ch0_global = R_GB @ dr_ch0_body
        dr_ch1_global = R_GB @ dr_ch1_body
        
        # Predicted measurements
        zhat = np.zeros((self.m, 1))
        
        # Gyro measurements (each should measure omega + bias)
        zhat[0:3, 0] = omega + bg_ch2
        zhat[3:6, 0] = omega + bg_ch3
        zhat[6:9, 0] = omega + bg_ch0
        zhat[9:12, 0] = omega + bg_ch1
        
        # Delta accelerometer measurements (rigid body constraint)
        # delta_accel = alpha × dr + omega × (omega × dr) + bias
        
        # For CH3 - CH2
        term_alpha_ch3 = np.cross(alpha, dr_ch3_global)
        omega_dot_r_ch3 = np.dot(omega, dr_ch3_global)
        omega_mag_sq = np.dot(omega, omega)
        term_omega_ch3 = omega_dot_r_ch3 * omega - omega_mag_sq * dr_ch3_global
        zhat[12:15, 0] = term_alpha_ch3 + term_omega_ch3 + dba_ch3
        
        # For CH0 - CH2
        term_alpha_ch0 = np.cross(alpha, dr_ch0_global)
        omega_dot_r_ch0 = np.dot(omega, dr_ch0_global)
        term_omega_ch0 = omega_dot_r_ch0 * omega - omega_mag_sq * dr_ch0_global
        zhat[15:18, 0] = term_alpha_ch0 + term_omega_ch0 + dba_ch0
        
        # For CH1 - CH2
        term_alpha_ch1 = np.cross(alpha, dr_ch1_global)
        omega_dot_r_ch1 = np.dot(omega, dr_ch1_global)
        term_omega_ch1 = omega_dot_r_ch1 * omega - omega_mag_sq * dr_ch1_global
        zhat[18:21, 0] = term_alpha_ch1 + term_omega_ch1 + dba_ch1
        
        # Measurement Jacobian H (21 × 33)
        # This is complex; for simplicity, use numerical approximation or
        # simplified analytical form. Here we use simplified form.
        H = np.zeros((self.m, self.n))
        
        # Gyro measurements depend on omega and respective bias
        # Row 0-2: gyro_ch2 = omega + bg_ch2
        H[0:3, 9:12] = np.eye(3)   # omega
        H[0:3, 12:15] = np.eye(3)  # bg_ch2
        
        # Row 3-5: gyro_ch3 = omega + bg_ch3
        H[3:6, 9:12] = np.eye(3)
        H[3:6, 15:18] = np.eye(3)
        
        # Row 6-8: gyro_ch0 = omega + bg_ch0
        H[6:9, 9:12] = np.eye(3)
        H[6:9, 18:21] = np.eye(3)
        
        # Row 9-11: gyro_ch1 = omega + bg_ch1
        H[9:12, 9:12] = np.eye(3)
        H[9:12, 21:24] = np.eye(3)
        
        # Delta accel measurements depend on omega (complex) and bias
        # Simplified: assume dependence on omega and bias only
        # Full Jacobian would include attitude dependencies (via R_GB)
        
        # Row 12-14: delta_accel_ch3 depends on omega and dba_ch3
        # Approximation: d(delta_accel)/d(omega) ≈ some complex term
        # For simplicity, use identity on bias
        H[12:15, 24:27] = np.eye(3)  # dba_ch3
        
        # Row 15-17: delta_accel_ch0
        H[15:18, 27:30] = np.eye(3)  # dba_ch0
        
        # Row 18-20: delta_accel_ch1
        H[18:21, 30:33] = np.eye(3)  # dba_ch1
        
        # Measurement noise covariance R
        R = np.diag(np.concatenate([
            self.cfg.r_gyro_ch2,
            self.cfg.r_gyro_ch3,
            self.cfg.r_gyro_ch0,
            self.cfg.r_gyro_ch1,
            self.cfg.r_da_ch3,
            self.cfg.r_da_ch0,
            self.cfg.r_da_ch1
        ]))
        
        # Innovation
        y = z - zhat
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular S, add regularization
            S += 1e-6 * np.eye(self.m)
            K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.n)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
    
    def get_state(self) -> dict:
        """
        Get current state as dictionary.
        
        Returns:
            dict with human-readable state values
        """
        x = self.x.flatten()
        return {
            "x_m": float(x[0]),
            "y_m": float(x[1]),
            "z_m": float(x[2]),
            "vx_ms": float(x[3]),
            "vy_ms": float(x[4]),
            "vz_ms": float(x[5]),
            "roll_deg": float(np.rad2deg(x[6])),
            "pitch_deg": float(np.rad2deg(x[7])),
            "yaw_deg": float(np.rad2deg(x[8])),
            "wx_deg_s": float(np.rad2deg(x[9])),
            "wy_deg_s": float(np.rad2deg(x[10])),
            "wz_deg_s": float(np.rad2deg(x[11])),
            "bg_ch2": np.rad2deg(x[12:15]),
            "bg_ch3": np.rad2deg(x[15:18]),
            "bg_ch0": np.rad2deg(x[18:21]),
            "bg_ch1": np.rad2deg(x[21:24]),
            "dba_ch3": x[24:27],
            "dba_ch0": x[27:30],
            "dba_ch1": x[30:33],
        }


# -----------------------------
# Main loop
# -----------------------------

def main():
    print("=" * 80)
    print("4-IMU 3D EKF with Full Rigid Body Constraints")
    print("Using imusensor Kalman Filter for Attitude Estimation")
    print("=" * 80)
    print()
    
    # Initialize hardware
    print("Initializing hardware...")
    hw = IMUHardware(use_multiplexer=True)
    print()
    
    # Initialize attitude filters (imusensor Kalman filters)
    print("Initializing attitude filters...")
    att_filter = AttitudeFilter()
    print()
    
    # IMU positions (in meters)
    r_ch2 = np.array([0.000, 0.000, 0.000])
    r_ch3 = np.array([0.000, 0.080, 0.005])
    r_ch0 = np.array([0.080, 0.080, 0.010])
    r_ch1 = np.array([0.080, 0.000, 0.015])
    
    print("IMU positions (meters):")
    print(f"  CH2: {r_ch2}")
    print(f"  CH3: {r_ch3}")
    print(f"  CH0: {r_ch0}")
    print(f"  CH1: {r_ch1}")
    print()
    
    # Alignment matrices (identity if aligned)
    C_ch2 = np.eye(3)
    C_ch3 = np.eye(3)
    C_ch0 = np.eye(3)
    C_ch1 = np.eye(3)
    
    # Noise parameters (start with defaults)
    # TODO: Collect still data and estimate these properly
    r_gyro = (np.deg2rad(0.2))**2 * np.ones(3)
    r_da = (0.15)**2 * np.ones(3)
    
    cfg = FourIMU_EKF_Config(
        r_ch2=r_ch2, r_ch3=r_ch3, r_ch0=r_ch0, r_ch1=r_ch1,
        C_ch2=C_ch2, C_ch3=C_ch3, C_ch0=C_ch0, C_ch1=C_ch1,
        q_v=0.5**2,
        q_omega=(np.deg2rad(30))**2,
        q_att=(np.deg2rad(10))**2,
        q_bg=(np.deg2rad(0.02))**2,
        q_dba=0.02**2,
        r_gyro_ch2=r_gyro,
        r_gyro_ch3=r_gyro,
        r_gyro_ch0=r_gyro,
        r_gyro_ch1=r_gyro,
        r_da_ch3=r_da,
        r_da_ch0=r_da,
        r_da_ch1=r_da,
    )
    
    ekf = FourIMU_3D_EKF(cfg)
    
    print("Starting combined filter loop (Ctrl+C to stop)...")
    print("- imusensor Kalman: Smooth attitude from gyro+accel+mag")
    print("- EKF: Rigid body constraints on accelerometers")
    print()
    
    t_prev = time.time()
    iteration = 0
    
    try:
        while True:
            t_now = time.time()
            dt = max(1e-3, t_now - t_prev)
            t_prev = t_now
            
            # Read all 4 IMUs
            data = hw.read_all()
            acc_ch2, gyr_ch2, mag_ch2 = data['ch2']
            acc_ch3, gyr_ch3, mag_ch3 = data['ch3']
            acc_ch0, gyr_ch0, mag_ch0 = data['ch0']
            acc_ch1, gyr_ch1, mag_ch1 = data['ch1']
            
            # ============================================================
            # STEP 1: Update attitude using imusensor Kalman filters
            # ============================================================
            attitudes = att_filter.update_all(data, dt)
            
            # Get consensus attitude (median of all 4)
            att_consensus = att_filter.get_consensus_attitude(attitudes)
            roll, pitch, yaw = att_consensus
            
            # Compute rotation matrix from consensus attitude
            R_GB = rotation_matrix(roll, pitch, yaw)
            
            # ============================================================
            # STEP 2: Remove gravity and transform to global frame
            # ============================================================
            # Gravity in body frame
            g_B = np.array([
                -9.81 * np.sin(pitch),
                9.81 * np.sin(roll) * np.cos(pitch),
                9.81 * np.cos(roll) * np.cos(pitch)
            ])
            
            # Remove gravity and rotate to global
            acc_ch2_global = R_GB @ (acc_ch2 - g_B)
            acc_ch3_global = R_GB @ (acc_ch3 - g_B)
            acc_ch0_global = R_GB @ (acc_ch0 - g_B)
            acc_ch1_global = R_GB @ (acc_ch1 - g_B)
            
            # ============================================================
            # STEP 3: Compute consensus angular velocity
            # ============================================================
            omega_consensus = np.median([gyr_ch2, gyr_ch3, gyr_ch0, gyr_ch1], axis=0)
            
            # ============================================================
            # STEP 4: Compute delta accelerations for rigid body constraint
            # ============================================================
            # These are gravity-corrected, global-frame accelerations
            da_ch3 = acc_ch3_global - acc_ch2_global
            da_ch0 = acc_ch0_global - acc_ch2_global
            da_ch1 = acc_ch1_global - acc_ch2_global
            
            # ============================================================
            # STEP 5: Build measurement vector for EKF
            # ============================================================
            z = np.concatenate([
                gyr_ch2,
                gyr_ch3,
                gyr_ch0,
                gyr_ch1,
                da_ch3,
                da_ch0,
                da_ch1
            ])
            
            # ============================================================
            # STEP 6: EKF predict and update
            # ============================================================
            ekf.predict(dt)
            ekf.update(z, dt)
            
            # Override EKF attitude with imusensor Kalman result
            # (imusensor is better for attitude, EKF is better for accel validation)
            ekf.x[6:9, 0] = att_consensus
            
            # ============================================================
            # STEP 7: Print diagnostics
            # ============================================================
            if iteration % 10 == 0:
                s = ekf.get_state()
                
                # Also show individual IMU attitudes vs consensus
                att_ch2_deg = np.rad2deg(attitudes['ch2'])
                att_consensus_deg = np.rad2deg(att_consensus)
                
                print(
                    f"[{iteration:5d}] dt={dt:0.3f} | "
                    f"Consensus att=[{att_consensus_deg[0]:6.2f}, {att_consensus_deg[1]:6.2f}, {att_consensus_deg[2]:6.2f}]° | "
                    f"CH2 att=[{att_ch2_deg[0]:6.2f}, {att_ch2_deg[1]:6.2f}, {att_ch2_deg[2]:6.2f}]° | "
                    f"ω=[{np.rad2deg(omega_consensus[0]):6.2f}, {np.rad2deg(omega_consensus[1]):6.2f}, {np.rad2deg(omega_consensus[2]):6.2f}]°/s"
                )
            
            iteration += 1
            time.sleep(0.02)  # 50 Hz
            
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("EKF stopped by user")
        print("=" * 80)
        print()
        print("Final state:")
        s = ekf.get_state()
        print(f"  Position: [{s['x_m']:.3f}, {s['y_m']:.3f}, {s['z_m']:.3f}] m")
        print(f"  Velocity: [{s['vx_ms']:.3f}, {s['vy_ms']:.3f}, {s['vz_ms']:.3f}] m/s")
        print(f"  Attitude: [{s['roll_deg']:.2f}, {s['pitch_deg']:.2f}, {s['yaw_deg']:.2f}] deg")
        print(f"  Ang Vel:  [{s['wx_deg_s']:.2f}, {s['wy_deg_s']:.2f}, {s['wz_deg_s']:.2f}] deg/s")
        print()
        print("Attitude source: imusensor Kalman filter (smooth gyro+accel+mag fusion)")
        print("Accel validation: EKF with rigid body constraints")


if __name__ == "__main__":
    main()
