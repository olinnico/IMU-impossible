#!/usr/bin/env python3
"""
accel_rigidbody_kf.py

Assumptions:
- 4 IMUs are aligned with the body axes
- Units are SI (meters)
"""

from __future__ import annotations
import numpy as np
import time
import smbus

from imusensor.MPU9250 import MPU9250
from imusensor.filters import kalman

# ============================
# 4-IMU READ LAYER (TCA9546A @ 0x70)
# ============================

TCA_ADDR = 0x70
IMU_ADDR = 0x69          # change to 0x68 if your MPU9250s are strapped that way
IMU_CHANNELS = [0, 1, 2, 3]  # CH0..CH3

class TCA9546A:
    def __init__(self, bus: smbus.SMBus, address: int = TCA_ADDR):
        self.bus = bus
        self.address = address

    def select(self, channel: int) -> None:
        if channel < 0 or channel > 3:
            raise ValueError("TCA9546A channel must be 0..3")
        self.bus.write_byte(self.address, 1 << channel)

class IMUArray:
    def __init__(self, bus_num: int, imu_addr: int, channels: list[int], tca_addr: int = TCA_ADDR):
        self.bus = smbus.SMBus(bus_num)
        self.tca = TCA9546A(self.bus, tca_addr)
        self.channels = channels
        self.imu_addr = imu_addr

        self.imus: dict[int, MPU9250] = {}
        for ch in self.channels:
            self.tca.select(ch)
            imu = MPU9250(self.bus, self.imu_addr)
            imu.begin()
            self.imus[ch] = imu

    def read_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          accels: (4,3)
          gyros:  (4,3)  rad/s
          mags:   (4,3)
        Order matches IMU_CHANNELS: CH0,CH1,CH2,CH3
        """
        N = len(self.channels)
        accels = np.zeros((N, 3), dtype=float)
        gyros  = np.zeros((N, 3), dtype=float)
        mags   = np.zeros((N, 3), dtype=float)

        for i, ch in enumerate(self.channels):
            self.tca.select(ch)
            imu = self.imus[ch]
            imu.readSensor()

            accels[i, :] = [imu.AccelVals[0], imu.AccelVals[1], imu.AccelVals[2]]
            gyros[i,  :] = [imu.GyroVals[0],  imu.GyroVals[1],  imu.GyroVals[2]]
            mags[i,   :] = [imu.MagVals[0],   imu.MagVals[1],   imu.MagVals[2]]

        return accels, gyros, mags



# -----------------------------
# STEP 1 constants
# -----------------------------
FS_HZ: float = 50.0
DT: float = 1.0 / FS_HZ

# -----------------------------
# STEP 2: IMU geometry (meters)
# -----------------------------
R_IMU_B: np.ndarray = np.array(
    [
        [-0.040,  0.040, 0.010],  # CH0
        [-0.040, -0.040, 0.015],  # CH1
        [ 0.040, -0.040, 0.000],  # CH2
        [ 0.040,  0.040, 0.005],  # CH3
    ],
    dtype=float,
)
N_IMU = R_IMU_B.shape[0]

# -----------------------------
# STEP 3: Rigid-body accel model
# -----------------------------

def rigid_body_accel_at_point(a0_B, omega_B, alpha_B, r_i_B):


    """ Predict linear acceleration at IMU i due to rigid-body motion.
    Inputs are all 3-element vectors in BODY frame:
    a0_B = [ax, ay, az] reference-point linear acceleration
    omega_B = [wx, wy, wz] angular rate from gyro-based solution (rad/s)
    alpha_B = [ax, ay, az] angular acceleration (rad/s^2)
    r_i_B = [x, y, z] IMU position from body origin (meters)
    
    Equation: a_i = a0 + alpha x r_i + omega x (omega x r_i) """
    
    a0_B = np.array(a0_B, dtype=float)
    omega_B = np.array(omega_B, dtype=float)
    alpha_B = np.array(alpha_B, dtype=float)
    r_i_B = np.array(r_i_B, dtype=float)

    tangential = np.cross(alpha_B, r_i_B)  # alpha x r_i
    centripetal = np.cross(omega_B, np.cross(omega_B, r_i_B))  # omega x (omega x r_i)
    return a0_B + tangential + centripetal

def predict_all_imu_accels(a0_B, omega_B, alpha_B):
    preds = np.zeros((N_IMU, 3), dtype=float)
    for i in range(N_IMU):
        preds[i] = rigid_body_accel_at_point(a0_B, omega_B, alpha_B, R_IMU_B[i]) # rigid body for each IMU
    return preds



class AlphaFromOmega:
    def __init__(self, dt, gamma=0.2):
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.omega_prev = None
        self.alpha = np.zeros(3, dtype=float)

    def update(self, omega_B):
        omega_B = np.array(omega_B, dtype=float)
        
        if self.omega_prev is None:
            self.omega_prev = omega_B.copy()
            self.alpha[:] = 0.0
            return self.alpha.copy()

        alpha_raw = (omega_B - self.omega_prev) / self.dt
        self.alpha = (1.0 - self.gamma) * self.alpha + self.gamma * alpha_raw

        self.omega_prev = omega_B.copy()
        return self.alpha.copy()



def remove_gravity_from_accels(accels_body: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Remove gravity component from accelerometer readings based on body orientation.
    
    Args:
        accels_body: (N_IMU, 3) raw accelerometer readings in body frame (m/s^2)
        roll: body roll angle (radians)
        pitch: body pitch angle (radians)  
        yaw: body yaw angle (radians) - not actually needed for gravity but included for completeness
    
    Returns:
        accels_clean: (N_IMU, 3) accelerometer readings with gravity removed
    """
    # Gravity vector in world frame (NED convention: pointing down)
    # If using ENU convention, use [0, 0, 9.81] instead
    g_world = np.array([0, 0, 9.81], dtype=float)
    
    # Build rotation matrix from world to body frame (ZYX Euler angles)
    # This rotates the gravity vector into the body frame
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Rotation matrix: R_world_to_body
    R = np.array([
        [cy*cp, sy*cp, -sp],
        [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
        [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
    ], dtype=float)
    
    # Transform gravity to body frame
    g_body = R @ g_world  # (3,)
    
    # Remove gravity from all IMU measurements
    accels_clean = accels_body - g_body.reshape(1, 3)  # broadcast across all IMUs
    
    return accels_clean



# -----------------------------
# CALIBRATION: estimate sigma_meas and sigma_b_walk in-memory
# -----------------------------

def calibrate_noise_and_bias_walk(
    imu_array,
    fs_hz: float,
    n_seconds: float = 20.0,
    window_sec: float = 1.0,
) -> dict:
    
    """
    Static calibration: device must be held still.

    Returns:
      sigma_meas: (4,3) per-IMU per-axis measurement noise std (m/s^2)
      sigma_b_walk: (4,3) per-IMU per-axis bias random-walk std (m/s^2)/sqrt(s)

    Math:
      - sigma_meas from high-frequency residual around a window mean
      - sigma_b_walk from variance of consecutive window means (random walk rate)
    """
    
    dt = 1.0 / float(fs_hz)
    N = int(n_seconds * fs_hz)
    W = max(1, int(window_sec * fs_hz))     # samples per window
    dT = W * dt                             # seconds per window

    # Collect N samples in RAM
    acc = np.zeros((N, 4, 3), dtype=float)

    t0 = time.time()
    for k in range(N):
        a, g, m = imu_array.read_all()
        acc[k] = a
        # simple timing to target fs_hz (best-effort)
        target = t0 + (k + 1) * dt
        sleep = target - time.time()
        if sleep > 0:
            time.sleep(sleep)

    # 1) Measurement noise sigma_meas: within-window residual std
    # For each sample, subtract its window mean to remove slow drift + constants.
    sigma_meas = np.zeros((4, 3), dtype=float)

    # Compute window means for each IMU/axis
    n_win = N // W
    win_means = np.zeros((n_win, 4, 3), dtype=float)
    for j in range(n_win):
        sl = slice(j * W, (j + 1) * W)
        win_means[j] = np.mean(acc[sl], axis=0)

    # Expand window means back to sample rate and compute residuals
    resid = np.zeros_like(acc[:n_win * W])
    for j in range(n_win):
        sl = slice(j * W, (j + 1) * W)
        resid[sl] = acc[sl] - win_means[j]

    sigma_meas = np.std(resid.reshape(-1, 4, 3), axis=0, ddof=1)

    # 2) Bias random walk sigma_b_walk: variance of window-mean increments
    dmean = np.diff(win_means, axis=0)          # (n_win-1,4,3)
    var_dmean = np.var(dmean, axis=0, ddof=1)   # (4,3)
    sigma_b_walk = np.sqrt(var_dmean / dT)

    return {
        "sigma_meas": sigma_meas,
        "sigma_b_walk": sigma_b_walk,
        "window_sec": window_sec,
        "n_seconds": n_seconds,
        "fs_hz": fs_hz,
    }

def moving_average(data, window):
    """
    Compute moving average with given window size.
    Returns array same shape as input, using 'same' mode convolution.
    """
    if window < 1:
        return data.copy()
    
    kernel = np.ones(window) / window
    result = np.zeros_like(data)
    
    for axis in range(data.shape[1]):
        result[:, axis] = np.convolve(data[:, axis], kernel, mode='same')
    
    return result
# -----------------------------
# CALIBRATION: estimate sigma_a0_walk from a short motion routine (no CSV)
# -----------------------------

def calibrate_a0_walk_motion(
    imu_array,
    R_IMU_B: np.ndarray,
    fs_hz: float,
    n_seconds: float = 25.0,
    window_sec: float = 0.5,
    alpha_gamma: float = 0.2,
    use_gyro_channel: int = 0,
) -> dict:
    """
    Motion calibration: user gently moves the rigid body for n_seconds.

    Estimates sigma_a0_walk (m/s^2)/sqrt(s) per axis using:
      1) compute omega from chosen gyro channel (default CH0)
      2) compute alpha by differentiating omega (with LPF gamma)
      3) compute rigid-body rotation term c_i for each IMU
      4) form z_minus_c = accel - c
      5) robust consensus a0_hat = median across IMUs of z_minus_c
      6) window-average a0_hat to suppress measurement noise
      7) sigma_a0_walk = sqrt( Var(delta window_mean) / window_duration )

    Returns:
      sigma_a0_walk_axis: (3,)
      sigma_a0_walk_scalar_mean: float
    """
    dt = 1.0 / float(fs_hz)
    N = int(n_seconds * fs_hz)
    W = max(2, int(window_sec * fs_hz))   # at least 2 samples
    dT = W * dt

    n_imu = R_IMU_B.shape[0]

    # Rolling omega->alpha estimator (single instance)
    alpha_est = AlphaFromOmega(dt=dt, gamma=alpha_gamma)

    # Store window means of a0_hat
    n_win = N // W
    a0_win = np.zeros((n_win, 3), dtype=float)

    t0 = time.time()
    k = 0
    j = 0
    a0_accum = np.zeros(3, dtype=float)
    count = 0

    while j < n_win:
        accels, gyros, mags = imu_array.read_all()

        # omega from selected channel (CH0 recommended)
        omega_B = np.asarray(gyros[use_gyro_channel], dtype=float).reshape(3,)

        # keep alpha estimator synced to actual dt
        alpha_est.dt = dt
        alpha_B = alpha_est.update(omega_B)

        # compute c_i for each IMU
        c = np.zeros((n_imu, 3), dtype=float)
        for i in range(n_imu):
            r_i = R_IMU_B[i]
            c[i] = np.cross(alpha_B, r_i) + np.cross(omega_B, np.cross(omega_B, r_i))

        # z_minus_c for each IMU
        z_minus_c = accels - c  # (4,3)

        # robust consensus estimate of a0 (median across IMUs)
        a0_hat = np.median(z_minus_c, axis=0)  # (3,)

        # accumulate into a window mean
        a0_accum += a0_hat
        count += 1

        # finalize window
        if count >= W:
            a0_win[j] = a0_accum / count
            j += 1
            a0_accum[:] = 0.0
            count = 0

        # best-effort timing
        k += 1
        target = t0 + k * dt
        sleep = target - time.time()
        if sleep > 0:
            time.sleep(sleep)

    # Remove slow trend (bias-like) by subtracting a longer moving average over windows
    long_win = max(3, int(5.0 / window_sec))  # ~5 seconds worth of windows
    a0_trend = moving_average(a0_win, long_win)
    a0_hp = a0_win - a0_trend

    # Increments of window means
    da = np.diff(a0_hp, axis=0)                # (n_win-1,3)
    var_da = np.var(da, axis=0, ddof=1)        # (3,)
    sigma_a0_walk_axis = np.sqrt(var_da / dT)  # (m/s^2)/sqrt(s)

    return {
        "sigma_a0_walk_axis": sigma_a0_walk_axis,
        "sigma_a0_walk_scalar_mean": float(np.mean(sigma_a0_walk_axis)),
        "n_seconds": n_seconds,
        "window_sec": window_sec,
        "fs_hz": fs_hz,
        "use_gyro_channel": use_gyro_channel,
    }


# -----------------------------
# STEP 4: Pick the state (what we estimate)
# State x = [a0(3), b0(3), b1(3), b2(3), b3(3)]  -> 15 elements
# -----------------------------

STATE_A0_SLICE = slice(0, 3)

def state_bias_slice(i: int) -> slice:
    
    """
    Return the slice in the state vector corresponding to IMU i's bias b_i (3 elements).
    i must be 0..N_IMU-1, matching CH0..CH3 order.
    """
    
    if i < 0 or i >= N_IMU:
        raise ValueError(f"IMU index must be 0..{N_IMU-1}")
    start = 3 + 3*i    # first 3 slots are a_0 -> bias starts after 3
    return slice(start, start + 3)

NX = 3 + 3*N_IMU  # total state dimension = 15 for 4 IMUs

def pack_state(a0_B: np.ndarray, biases_B: np.ndarray) -> np.ndarray:
    
    """
    Build state vector x from:
      a0_B:    (3,)
      biases_B:(N_IMU,3)
    Returns:
      x: (NX,)
    """
    
    a0_B = np.asarray(a0_B, dtype=float).reshape(3,)
    biases_B = np.asarray(biases_B, dtype=float).reshape(N_IMU, 3)

    x = np.zeros(NX, dtype=float)
    x[STATE_A0_SLICE] = a0_B 
    for i in range(N_IMU):
        x[state_bias_slice(i)] = biases_B[i]
    return x

def unpack_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Split state vector x into:
      a0_B: (3,)
      biases_B: (N_IMU,3)
    """
    
    x = np.asarray(x, dtype=float).reshape(NX,)
    a0_B = x[STATE_A0_SLICE].copy()
    biases_B = np.zeros((N_IMU, 3), dtype=float)
    for i in range(N_IMU):
        biases_B[i] = x[state_bias_slice(i)]
    return a0_B, biases_B


# -----------------------------
# STEP 5: Choose process models (random walks)
# x_{k+1} = x_k + w_k
# w_k ~ N(0, Q)
# -----------------------------

# TUNING KNOBS (initial guesses)
# These are "per sqrt(second)" random-walk standard deviations.
# Bigger = state is allowed to change faster.

SIGMA_A0_WALK = 0.05
SIGMA_B_WALK  = 0.01


def build_Q_random_walk(dt: float,
                        sigma_a0_walk: float = SIGMA_A0_WALK,
                        sigma_b_walk: float = SIGMA_B_WALK) -> np.ndarray:
    """
    Build discrete-time Q for a random-walk state:
        a0_{k+1} = a0_k + w_a0
        b_i{k+1} = b_i{k} + w_bi

    If sigma is specified as "per sqrt(second)", then:
        Var(w) = sigma^2 * dt

    Returns:
        Q: (NX, NX) process noise covariance
    """
    
    dt = float(dt)

    q_a0 = (sigma_a0_walk ** 2) * dt   # variance per step for each a0 axis
    q_b  = (sigma_b_walk  ** 2) * dt   # variance per step for each bias axis

    Q = np.zeros((NX, NX), dtype=float)

    # a0 block (first 3 states)
    Q[STATE_A0_SLICE, STATE_A0_SLICE] = np.eye(3) * q_a0

    # bias blocks
    for i in range(N_IMU):
        s = state_bias_slice(i)
        Q[s, s] = np.eye(3) * q_b

    return Q


# -----------------------------
# STEP 6: Build F and Q (discrete-time)
# For random-walk states:
#   x_{k+1} = x_k + w_k
# so:
#   F = I
#   Q from Step 5 helper
# -----------------------------

def build_F_random_walk() -> np.ndarray:
    """
    Random-walk state transition:
      x_{k+1} = x_k + w_k

    Returns:
      F: (NX, NX) identity matrix
    """
    return np.eye(NX, dtype=float)

# Example usage (later in the filter object / main loop):
# F = build_F_random_walk()
# Q = build_Q_random_walk(dt)


# -----------------------------
# STEP 7: Measurement model (H and R)
# y = H x + v
# -----------------------------

NZ = 3 * N_IMU  # 3 axis * N_IMU (4 IMUS) -> 12 

def build_H_measurement() -> np.ndarray:
    """
    Measurement model:
      y_i = a0 + b_i

    Returns:
      H: (NZ, NX)
    """
    H = np.zeros((NZ, NX), dtype=float)

    for i in range(N_IMU):
        row = 3 * i

        # a0 contribution
        H[row:row+3, STATE_A0_SLICE] = np.eye(3)

        # bias contribution for IMU i
        H[row:row+3, state_bias_slice(i)] = np.eye(3)

    return H

def build_R_measurement(sigma_meas: np.ndarray) -> np.ndarray:
    
    """
    Build measurement noise covariance R.

    sigma_meas: (N_IMU,3) std dev per IMU per axis (m/s^2)

    Returns:
      R: (NZ, NZ)
    """
    
    R = np.zeros((NZ, NZ), dtype=float)

    for i in range(N_IMU):
        row = 3 * i
        Ri = np.diag(sigma_meas[i] ** 2)
        R[row:row+3, row:row+3] = Ri

    return R

# -----------------------------
# STEP 7.5: Build corrected measurement y = z - c
# This produces the measurement vector that matches H and R stacking.
# -----------------------------

def compute_c_terms(omega_B: np.ndarray, alpha_B: np.ndarray) -> np.ndarray:
    
    """
    Compute c_i = alpha x r_i + omega x (omega x r_i) for each IMU.

    Returns:
      c: (N_IMU, 3)
    """
    
    omega_B = np.asarray(omega_B, dtype=float).reshape(3,)
    alpha_B = np.asarray(alpha_B, dtype=float).reshape(3,)

    c = np.zeros((N_IMU, 3), dtype=float)
    for i in range(N_IMU):
        r_i = R_IMU_B[i]
        c[i] = np.cross(alpha_B, r_i) + np.cross(omega_B, np.cross(omega_B, r_i))
    return c

def build_y_corrected(z_accels_B: np.ndarray, omega_B: np.ndarray, alpha_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Build corrected measurement vector y (stacked 12x1) that matches H and R.

    Inputs:
      z_accels_B: (N_IMU,3) raw accel measurements in body frame
      omega_B: (3,) angular rate in body frame (rad/s)
      alpha_B: (3,) angular accel in body frame (rad/s^2)

    Returns:
      y: (NZ,)   stacked corrected measurements [y0;y1;y2;y3]
      c: (N_IMU,3) the rotation terms that were subtracted
    """
    
    z_accels_B = np.asarray(z_accels_B, dtype=float).reshape(N_IMU, 3)

    c = compute_c_terms(omega_B, alpha_B)      # (N_IMU,3)
    y_mat = z_accels_B - c                     # (N_IMU,3)
    y = y_mat.reshape(NZ,)                     # stack into (12,)

    return y, c

def predict_measurements(x_pred: np.ndarray, omega_B: np.ndarray, alpha_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Given predicted state x_pred, compute:
      y_hat: (NZ,) predicted corrected measurements (z - c)
      z_hat: (N_IMU,3) predicted raw accelerometer measurements z

    Uses:
      y_hat = H x_pred
      z_hat_i = a0 + c_i + b_i
    """
    
    a0_B, biases_B = unpack_state(x_pred)
    c = compute_c_terms(omega_B, alpha_B)             # (N_IMU,3)
    z_hat = a0_B.reshape(1,3) + c + biases_B          # (N_IMU,3)
    y_hat = (a0_B.reshape(1,3) + biases_B).reshape(NZ,)  # since y = z - c = a0 + b
    return y_hat, z_hat



# -----------------------------
# STEP 8: Implement the filter loop (predict + update)
# -----------------------------

class LinearKF:
    """
    Generic linear Kalman filter:
      x_{k+1} = F x_k + w,    w ~ N(0,Q)
      y_k     = H x_k + v,    v ~ N(0,R)

    For our accel-only problem:
      y_k is the corrected measurement built from y = z - c(omega, alpha).
    """

    def __init__(self, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
                 x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        self.F = np.asarray(F, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.nx = self.F.shape[0]
        self.nz = self.H.shape[0]

        # State estimate and covariance
        self.x = np.zeros(self.nx, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(self.nx,)
        self.P = np.eye(self.nx, dtype=float) * 10.0 if P0 is None else np.asarray(P0, dtype=float).reshape(self.nx, self.nx)

        self.Ix = np.eye(self.nx, dtype=float)

    def predict(self):
        """
        Time update:
          x <- F x
          P <- F P F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y: np.ndarray) -> np.ndarray:
        """
        Measurement update using measurement vector y.

        Returns:
          innovation nu = y - H x_pred  (shape (nz,))
        """
        y = np.asarray(y, dtype=float).reshape(self.nz,)

        # innovation
        nu = y - (self.H @ self.x)

        # innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain (use solve instead of inverse)
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(self.nz))

        # state and covariance update
        self.x = self.x + K @ nu
        self.P = (self.Ix - K @ self.H) @ self.P

        return nu



def main():
    print("Starting 4-IMU read + Kalman filter...")
    
    # Initialize hardware
    imu_array = IMUArray(bus_num=1, imu_addr=IMU_ADDR, channels=IMU_CHANNELS, tca_addr=TCA_ADDR)
    alpha_est = AlphaFromOmega(dt=DT, gamma=0.2)
    
    # Static calibration
    print("Static calibration: hold device still for 20 seconds...")
    cal_static = calibrate_noise_and_bias_walk(imu_array, fs_hz=FS_HZ, n_seconds=20.0, window_sec=1.0)
    sigma_meas = cal_static["sigma_meas"]       # (4,3)
    sigma_b_walk = cal_static["sigma_b_walk"]   # (4,3)
    print("sigma_meas (m/s^2):\n", sigma_meas)
    print("sigma_b_walk (m/s^2)/sqrt(s):\n", sigma_b_walk)
    
    # Motion calibration
    print("Motion calibration: move the device gently for 25 seconds...")
    cal_motion = calibrate_a0_walk_motion(
        imu_array=imu_array,
        R_IMU_B=R_IMU_B,
        fs_hz=FS_HZ,
        n_seconds=25.0,
        window_sec=0.5,
        alpha_gamma=0.2,
        use_gyro_channel=0,
    )
    print("sigma_a0_walk_axis (m/s^2)/sqrt(s):", cal_motion["sigma_a0_walk_axis"])
    print("sigma_a0_walk_mean:", cal_motion["sigma_a0_walk_scalar_mean"])
    
    # Set calibrated values
    sigma_a0_walk = cal_motion["sigma_a0_walk_scalar_mean"]
    sigma_b_walk_mean = np.mean(sigma_b_walk)
    
    # Build Kalman filter matrices
    H = build_H_measurement()
    R = build_R_measurement(sigma_meas)
    F = build_F_random_walk()
    Q = build_Q_random_walk(DT, sigma_a0_walk=sigma_a0_walk, sigma_b_walk=sigma_b_walk_mean)
    
    # Initialize Kalman filter
    kf = LinearKF(F=F, Q=Q, H=H, R=R)
    
    # Initialize sensor fusion for orientation
    sensorfusion = kalman.Kalman()
    
    print("\nStarting main loop...")
    prev_time = time.time()
    
    while True:
        now = time.time()
        dt = max(0.001, now - prev_time)
        prev_time = now
        
        # Read all 4 IMUs
        accels, gyros, mags = imu_array.read_all()
        
        # Use CH0 for attitude estimation
        idx = 0
        ax, ay, az = accels[idx]
        gx, gy, gz = gyros[idx]
        mx, my, mz = mags[idx]
        
        # Update orientation filter
        sensorfusion.computeAndUpdateRollPitchYaw(
            ax, ay, az,
            gx, gy, gz,
            mx, my, mz,
            dt
        )
        
        roll, pitch, yaw = sensorfusion.roll, sensorfusion.pitch, sensorfusion.yaw
        
        # Remove gravity from all accelerometer readings
        accels_no_gravity = remove_gravity_from_accels(accels, roll, pitch, yaw)
        
        # omega_B from CH0 gyro
        omega_B = np.array([gx, gy, gz], dtype=float)
        
        # Estimate angular acceleration
        alpha_B = alpha_est.update(omega_B)
        
        # Update Q with actual dt
        Q = build_Q_random_walk(dt, sigma_a0_walk=sigma_a0_walk, sigma_b_walk=sigma_b_walk_mean)
        kf.Q = Q
        
        # Kalman filter predict step
        kf.predict()
        
        # Build corrected measurement and update (using gravity-free accels)
        y, c = build_y_corrected(accels_no_gravity, omega_B, alpha_B)
        nu = kf.update(y)
        
        # Extract estimated states
        a0_B, biases_B = unpack_state(kf.x)
        
        # Compute cleaned accelerations (gravity already removed)
        accel_clean = accels_no_gravity - biases_B
        
        # Print results
        print(f"dt={dt:.3f} | R={roll:.2f} P={pitch:.2f} Y={yaw:.2f} | " +
              f"a0={a0_B} | accel_clean[0]={accel_clean[0]}")
        
        time.sleep(0.01)



        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
