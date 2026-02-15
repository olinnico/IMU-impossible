#!/usr/bin/env python3
"""
imu_fusion_pi.py

Hardware assumed (same as your sketch):
- TCA9548A I2C multiplexer at 0x70
- Four channels on the mux:
    ch0: MPU9250 (accel ±16g) + AK8963 mag, pos (80,80,10) mm
    ch1: MPU9250 (accel ±16g) + AK8963 mag, pos (80,0,15) mm
    ch2: MPU9050 (accel ±2g) + AK8963 mag, pos (0,80,5) mm
    ch3: MPU9250 (accel ±16g) + AK8963 mag, pos (0,0,0) mm

Addresses:
- MPU9250/MPU9050: 0x68
- AK8963: 0x0C (reachable only when BYPASS is enabled in the MPU9250)

Units:
- accel: m/s^2
- gyro: rad/s
- mag: microtesla (approx)

Install requirements:
- sudo apt-get install python3-pip
- pip3 install smbus2 numpy

Enable I2C on Pi:
- sudo raspi-config  -> Interface Options -> I2C -> enable

Run:
- python3 imu_fusion_pi.py

Anchor / Grid calibration mode (optional):
- Enable in Config with anchor_mode_enable = True.
- Commands via stdin while running:
    "a x y" or "a x y z" -> set anchor in grid units
    "zero"              -> anchor at (0,0,0)
    "print"             -> print current p, v, and last anchor
- Anchors only apply after the rig is stationary for anchor_stationary_hold_s.
- If anchor_prompt_calibration is True, press Enter to start calibration.
"""

import time
import math
import sys
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
from smbus2 import SMBus, i2c_msg


# ============================================================
# I2C / device constants (register-level)
# ============================================================

TCA_ADDR = 0x70

MPU_ADDR = 0x68
MAG_ADDR = 0x0C  # AK8963

# MPU registers (MPU9050/MPU9250 share these)
REG_PWR_MGMT_1   = 0x6B
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_INT_PIN_CFG  = 0x37  # bypass enable bit
REG_ACCEL_XOUT_H = 0x3B
REG_GYRO_XOUT_H  = 0x43

# AK8963 registers
AK8963_ST1  = 0x02
AK8963_XOUT_L = 0x03
AK8963_ST2  = 0x09
AK8963_CNTL1 = 0x0A


# ============================================================
# Low-level I2C helpers with retries to reduce "freezes"
# ============================================================

def i2c_write_byte(bus: SMBus, addr: int, reg: int, val: int, retries: int = 3) -> None:
    """Write one byte to (addr, reg). Retries help if I2C glitches."""
    for k in range(retries):
        try:
            bus.write_byte_data(addr, reg, val & 0xFF)
            return
        except Exception:
            time.sleep(0.005)
    raise IOError(f"I2C write failed addr=0x{addr:02X} reg=0x{reg:02X}")

def i2c_read_bytes(bus: SMBus, addr: int, reg: int, n: int, retries: int = 3) -> bytes:
    """
    Read n bytes starting at reg from addr.
    Implemented using write-then-read (repeated start) via smbus2 i2c_msg.
    This is closer to what Wire.endTransmission(false) does on Arduino.
    """
    for k in range(retries):
        try:
            write = i2c_msg.write(addr, [reg & 0xFF])
            read = i2c_msg.read(addr, n)
            bus.i2c_rdwr(write, read)
            return bytes(read)
        except Exception:
            time.sleep(0.005)
    raise IOError(f"I2C read failed addr=0x{addr:02X} reg=0x{reg:02X} n={n}")

def be16_to_i16(b_hi: int, b_lo: int) -> int:
    """Big-endian 16-bit -> signed int16."""
    v = (b_hi << 8) | b_lo
    return v - 65536 if v & 0x8000 else v

def le16_to_i16(b_lo: int, b_hi: int) -> int:
    """Little-endian 16-bit -> signed int16."""
    v = (b_hi << 8) | b_lo
    return v - 65536 if v & 0x8000 else v


# ============================================================
# TCA9548A multiplexer
# ============================================================

class TCA9548A:
    """
    Select exactly one channel at a time by writing (1<<ch) to the mux.
    """
    def __init__(self, bus: SMBus, addr: int = TCA_ADDR):
        self.bus = bus
        self.addr = addr
        self.current = None

    def select(self, ch: int) -> None:
        if ch < 0 or ch > 7:
            raise ValueError("TCA channel must be 0..7")
        if self.current == ch:
            return
        self.bus.write_byte(self.addr, 1 << ch)
        self.current = ch
        # Tiny settle time helps if you're hammering reads
        time.sleep(0.001)


# ============================================================
# MPU9050 / MPU9250 driver (what your .ino did)
# ============================================================

class MPUBase:
    """
    Driver for MPU9050 or MPU9250 main IMU registers.
    Note: the MPU9250 magnetometer AK8963 is a separate device read at 0x0C,
    and only accessible when bypass is enabled (INT_PIN_CFG bit1).
    """
    def __init__(self, bus: SMBus, mux: TCA9548A, ch: int, accel_16g: bool, has_mag: bool):
        self.bus = bus
        self.mux = mux
        self.ch = ch
        self.accel_16g = accel_16g
        self.has_mag = has_mag

        # Scaling factors
        self.g = 9.81
        self.deg2rad = math.pi / 180.0

        # Accel scale:
        # - ±2g   => 16384 LSB/g
        # - ±16g  => 2048  LSB/g
        self.acc_lsb_per_g = 2048.0 if accel_16g else 16384.0

        # Gyro scale in your sketch: ±250 dps => 131 LSB/(deg/s)
        self.gyro_lsb_per_dps = 131.0

        self.ready = False

    def init_device(self) -> None:
        """
        Equivalent to init_mpu_common + init_mpu9250_mag_bypass in your sketch.
        """
        self.mux.select(self.ch)

        # Wake device (PWR_MGMT_1 = 0)
        i2c_write_byte(self.bus, MPU_ADDR, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.05)

        # Gyro config: ±250 dps => 0x00
        i2c_write_byte(self.bus, MPU_ADDR, REG_GYRO_CONFIG, 0x00)
        time.sleep(0.01)

        # Accel config:
        # 0x1C AFS_SEL bits [4:3]
        # ±2g => 0x00, ±16g => 0x18
        i2c_write_byte(self.bus, MPU_ADDR, REG_ACCEL_CONFIG, 0x18 if self.accel_16g else 0x00)
        time.sleep(0.01)

        if self.has_mag:
            # Enable bypass: INT_PIN_CFG (0x37) BYPASS_EN = bit1 => 0x02
            i2c_write_byte(self.bus, MPU_ADDR, REG_INT_PIN_CFG, 0x02)
            time.sleep(0.01)

            # AK8963 setup:
            # Power down then continuous measurement mode 2 (100 Hz), 16-bit output
            i2c_write_byte(self.bus, MAG_ADDR, AK8963_CNTL1, 0x00)
            time.sleep(0.01)
            i2c_write_byte(self.bus, MAG_ADDR, AK8963_CNTL1, 0x16)
            time.sleep(0.01)

        self.ready = True

    def read_accel_gyro(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads accel XYZ and gyro XYZ from the MPU registers, converts to:
        accel m/s^2, gyro rad/s
        """
        self.mux.select(self.ch)
        if not self.ready:
            raise RuntimeError("IMU not initialized")

        # Accel (6 bytes) starting at 0x3B
        a_bytes = i2c_read_bytes(self.bus, MPU_ADDR, REG_ACCEL_XOUT_H, 6)
        ax = be16_to_i16(a_bytes[0], a_bytes[1])
        ay = be16_to_i16(a_bytes[2], a_bytes[3])
        az = be16_to_i16(a_bytes[4], a_bytes[5])

        # Gyro (6 bytes) starting at 0x43
        g_bytes = i2c_read_bytes(self.bus, MPU_ADDR, REG_GYRO_XOUT_H, 6)
        gx = be16_to_i16(g_bytes[0], g_bytes[1])
        gy = be16_to_i16(g_bytes[2], g_bytes[3])
        gz = be16_to_i16(g_bytes[4], g_bytes[5])

        # Convert raw accel counts -> g -> m/s^2
        a = np.array([ax, ay, az], dtype=float) / self.acc_lsb_per_g * self.g

        # Convert raw gyro counts -> deg/s -> rad/s
        g_dps = np.array([gx, gy, gz], dtype=float) / self.gyro_lsb_per_dps
        w = g_dps * self.deg2rad

        return a, w

    def read_mag_uT(self) -> np.ndarray:
        """
        Reads AK8963 magnetometer if present.
        If no mag, returns zeros.
        """
        if not self.has_mag:
            return np.zeros(3, dtype=float)

        self.mux.select(self.ch)

        # ST1 bit0 = data ready
        st1 = i2c_read_bytes(self.bus, MAG_ADDR, AK8963_ST1, 1)[0]
        if (st1 & 0x01) == 0:
            return np.zeros(3, dtype=float)

        # Read 6 data bytes + ST2 (7 bytes total) from 0x03
        m = i2c_read_bytes(self.bus, MAG_ADDR, AK8963_XOUT_L, 7)
        x = le16_to_i16(m[0], m[1])
        y = le16_to_i16(m[2], m[3])
        z = le16_to_i16(m[4], m[5])
        st2 = m[6]

        # Overflow bit in ST2 (bit3)
        if st2 & 0x08:
            return np.zeros(3, dtype=float)

        # Same approx scaling as your sketch:
        # 4912 uT full scale over 32760 counts => ~0.15 uT/LSB
        uT_per_lsb = 4912.0 / 32760.0
        return np.array([x, y, z], dtype=float) * uT_per_lsb


# ============================================================
# Math helpers (same as your fusion code)
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y, x,  0.0]], dtype=float)

def centripetal_term(omega: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.cross(omega, np.cross(omega, r))

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


# ============================================================
# Mahony AHRS (your code, fixed True/False)
# ============================================================

class MahonyAHRS:
    """
    Minimal Mahony-style AHRS.
    - Correct roll/pitch using accel direction (gravity)
    - Correct yaw lightly using mag (optional)
    """
    def __init__(self, kp: float = 2.5, ki: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # w,x,y,z
        self.e_int = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self.q[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.e_int[:] = 0.0

    def q_to_R(self) -> np.ndarray:
        """
        Rotation matrix from BODY to WORLD (consistent with how your code used it).
        """
        w, x, y, z = self.q
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
            [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
        ], dtype=float)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-9:
            return v
        return v / n

    def update(self, gyro_rad: np.ndarray, accel_mps2: np.ndarray, mag_uT: np.ndarray,
               dt: float, use_mag: bool = True) -> None:
        """
        Update quaternion estimate.

        gyro_rad: body angular velocity [rad/s]
        accel_mps2: body acceleration [m/s^2] (includes gravity)
        mag_uT: body magnetometer [uT]
        """
        a = accel_mps2
        a_n = np.linalg.norm(a)
        if a_n < 1e-6:
            return
        a_unit = a / a_n

        R = self.q_to_R()

        # predicted gravity direction in body:
        # world gravity direction is +Z in this formulation
        g_b = R.T @ np.array([0.0, 0.0, 1.0])

        # accel correction tries to align measured accel direction to gravity direction
        e = np.cross(a_unit, g_b)

        # Optional yaw correction from magnetometer (very light)
        if use_mag and np.linalg.norm(mag_uT) > 1e-6:
            m_unit = self.normalize(mag_uT)

            # Move mag into world frame, flatten to horizontal
            m_w = R @ m_unit
            mh = np.array([m_w[0], m_w[1], 0.0])
            if np.linalg.norm(mh) > 1e-6:
                mh_unit = self.normalize(mh)

                # bring expected horizontal mag back into body
                mh_b = R.T @ mh_unit

                # measured mag projected to horizontal in body
                mb_h = np.array([m_unit[0], m_unit[1], 0.0])
                if np.linalg.norm(mb_h) > 1e-6:
                    mb_h = self.normalize(mb_h)
                    e_mag = np.cross(mb_h, mh_b)
                    e += 0.25 * e_mag

        # Integral term (bias-ish correction)
        self.e_int += e * dt

        gyro_corr = gyro_rad + self.kp * e + self.ki * self.e_int

        # Quaternion derivative
        w, x, y, z = self.q
        gx, gy, gz = gyro_corr
        q_dot = 0.5 * np.array([
            -x*gx - y*gy - z*gz,
             w*gx + y*gz - z*gy,
             w*gy - x*gz + z*gx,
             w*gz + x*gy - y*gx
        ], dtype=float)

        self.q = self.q + q_dot * dt
        self.q = self.q / np.linalg.norm(self.q)


# ============================================================
# Config and fusion logic (same as you had, but Pi-friendly)
# ============================================================

@dataclass
class Config:
    # Sensor positions in meters (same numbers you used, just as meters)
    pos: Dict[int, np.ndarray]

    g: float = 9.81
    cal_frames: int = 150

    # Residual thresholds for rigid body consistency checks
    res_warn: float = 0.8
    res_hard: float = 1.6

    # Placeholder mag calibration
    mag_offset_uT: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_scale: np.ndarray = field(default_factory=lambda:np.ones(3))

    # ZUPT thresholds (you can tweak)
    zupt_enable: bool = True
    zupt_gyro: float = 0.06      # rad/s (stricter than before)
    zupt_acc_mag: float = 0.20   # m/s^2 difference from g (stricter)
    zupt_min_frames: int = 8     # frames required to enter "stationary"
    zupt_release_frames: int = 3 # frames required to exit "stationary"

    # Bias learning when stationary
    bias_learn_enable: bool = True
    bias_learn_rate: float = 0.05

    # Velocity damping (reduces drift when slightly moving)
    vel_damping: float = 0.25    # 1/s, exponential decay of velocity

    # Anchor / grid calibration mode
    anchor_mode_enable: bool = False
    anchor_prompt_calibration: bool = True
    anchor_units_in_meters: float = 0.01
    anchor_stationary_hold_s: float = 1.0
    anchor_pos_gain: float = 1.0
    anchor_bias_tune_enable: bool = False
    anchor_bias_gain: float = 0.02
    anchor_bias_max: float = 0.30

    # dt clamps
    dt_min: float = 0.002
    dt_max: float = 0.05


cfg = Config(
    pos={
        0: np.array([0.080, 0.080, 0.010], dtype=float),
        1: np.array([0.080, 0.000, 0.015], dtype=float),
        2: np.array([0.000, 0.080, 0.005], dtype=float),
        3: np.array([0.000, 0.000, 0.000], dtype=float),
    }
)

MAIN_IMU = 3


class AnchorInput:
    """
    Background stdin reader to avoid blocking the main loop.
    """
    def __init__(self):
        self.queue: "queue.Queue" = queue.Queue()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        for line in sys.stdin:
            self.queue.put(line.strip())

    def poll(self) -> Optional[str]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


class AnchorManager:
    """
    Handle anchor commands and apply position corrections when stationary.
    """
    def __init__(self):
        self.pending_anchor_m: Optional[np.ndarray] = None
        self.last_anchor_m: Optional[np.ndarray] = None
        self.last_delta_m: Optional[np.ndarray] = None
        self.last_applied: bool = False
        self.print_requested: bool = False

    def _parse_anchor(self, parts: List[str]) -> Optional[np.ndarray]:
        if len(parts) not in (3, 4):
            return None
        try:
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            return None
        if len(coords) == 2:
            coords.append(0.0)
        return np.array(coords, dtype=float) * cfg.anchor_units_in_meters

    def handle_command(self, cmd: str) -> None:
        parts = cmd.strip().split()
        if not parts:
            return
        key = parts[0].lower()
        if key == "a":
            anchor = self._parse_anchor(parts)
            if anchor is not None:
                self.pending_anchor_m = anchor
                print(f"[ANCHOR] pending set to {anchor}")
            else:
                print("[ANCHOR] invalid format. Use: a x y [z]")
        elif key == "zero":
            self.pending_anchor_m = np.zeros(3, dtype=float)
            print("[ANCHOR] pending set to [0,0,0]")
        elif key == "print":
            self.print_requested = True
        else:
            print(f"[ANCHOR] unknown command: {cmd}")

    def apply_if_ready(self, integ: "Integrator", stationary: bool, stationary_time: float) -> None:
        self.last_applied = False
        if self.pending_anchor_m is None:
            return
        if not (stationary and stationary_time >= cfg.anchor_stationary_hold_s):
            return
        desired = self.pending_anchor_m
        current = integ.p.copy()
        delta = desired - current
        gain = clamp(cfg.anchor_pos_gain, 0.0, 1.0)
        integ.p = current + gain * delta
        integ.v[:] = 0.0
        self.last_anchor_m = integ.p.copy()
        self.last_delta_m = delta
        self.last_applied = True
        self.pending_anchor_m = None
        if cfg.anchor_bias_tune_enable:
            bias_adjust = -cfg.anchor_bias_gain * delta
            integ.b_aw += np.clip(bias_adjust, -cfg.anchor_bias_max, cfg.anchor_bias_max)
        print(f"[ANCHOR] applied, delta={delta}, new p={integ.p}")


class BiasCal4:
    """
    Collect stationary samples to estimate accel/gyro bias per IMU.
    Same logic you had, but now input comes from Pi drivers directly.
    """
    def __init__(self, N: int):
        self.N = N
        self.k = 0
        self.done = False
        self.acc_sum = {i: np.zeros(3) for i in cfg.pos}
        self.gyr_sum = {i: np.zeros(3) for i in cfg.pos}
        self.acc_bias = {i: np.zeros(3) for i in cfg.pos}
        self.gyr_bias = {i: np.zeros(3) for i in cfg.pos}

    def add(self, acc: Dict[int, np.ndarray], gyr: Dict[int, np.ndarray]) -> None:
        if self.done:
            return
        self.k += 1
        for i in cfg.pos:
            self.acc_sum[i] += acc[i]
            self.gyr_sum[i] += gyr[i]

        if self.k >= self.N:
            for i in cfg.pos:
                acc_avg = self.acc_sum[i] / self.N
                gyr_avg = self.gyr_sum[i] / self.N

                # accel bias relative to expected [0,0,g] in BODY during calibration
                self.acc_bias[i] = acc_avg - np.array([0.0, 0.0, cfg.g])
                self.gyr_bias[i] = gyr_avg
            self.done = True

    def correct(self,
                acc: Dict[int, np.ndarray],
                gyr: Dict[int, np.ndarray],
                mag: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        acc2, gyr2, mag2 = {}, {}, {}
        for i in cfg.pos:
            acc2[i] = acc[i] - self.acc_bias[i]
            gyr2[i] = gyr[i] - self.gyr_bias[i]
            mag2[i] = (mag[i] - cfg.mag_offset_uT) * cfg.mag_scale
        return acc2, gyr2, mag2


class Fusion:
    """
    Use one "main" IMU as the primary source, and use the others to
    validate rigid-body consistency (estimate alpha + residuals).
    """
    @staticmethod
    def _relative_positions(main_idx: int) -> Dict[int, np.ndarray]:
        origin = cfg.pos[main_idx]
        return {i: (cfg.pos[i] - origin) for i in cfg.pos}

    def fuse_omega(self, gyr: Dict[int, np.ndarray], main_idx: int) -> np.ndarray:
        """
        Use the main IMU gyro directly.
        """
        return gyr[main_idx]

    def estimate_alpha(self, acc: Dict[int, np.ndarray], omega: np.ndarray, main_idx: int) -> np.ndarray:
        """
        Estimate angular acceleration using the non-main IMUs:

            acc_i - acc_main - omega x (omega x r_i) = alpha x r_i
        """
        rel = self._relative_positions(main_idx)
        A, y = [], []
        for i in cfg.pos:
            if i == main_idx:
                continue
            r = rel[i]
            c = centripetal_term(omega, r)
            yi = acc[i] - acc[main_idx] - c
            Ai = -skew(r)
            A.append(Ai)
            y.append(yi.reshape(3, 1))

        if not A:
            return np.zeros(3, dtype=float)

        A = np.vstack(A)
        y = np.vstack(y).ravel()
        alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        return alpha

    def residual_norms(self, acc: Dict[int, np.ndarray], omega: np.ndarray,
                       a0: np.ndarray, alpha: np.ndarray, main_idx: int) -> Dict[int, float]:
        """
        Compute per-IMU residual error for consistency scoring.
        """
        rel = self._relative_positions(main_idx)
        norms = {}
        for i in cfg.pos:
            if i == main_idx:
                norms[i] = 0.0
                continue
            r = rel[i]
            pred = a0 + np.cross(alpha, r) + centripetal_term(omega, r)
            norms[i] = float(np.linalg.norm(acc[i] - pred))
        return norms

    def fuse_mag(self, mag: Dict[int, np.ndarray], main_idx: int) -> np.ndarray:
        """
        Use the main IMU magnetometer directly.
        """
        return mag[main_idx]


class Integrator:
    """
    Convert body-frame accel -> world-frame linear accel, integrate to v and p.
    Includes:
    - gravity removal using attitude Rwb
    - learned world accel bias b_aw when stationary
    - ZUPT (zero velocity update) when stationary detection triggers
    - trapezoid integration for v (less noisy than Euler)
    """
    def __init__(self):
        self.v = np.zeros(3, dtype=float)
        self.p = np.zeros(3, dtype=float)
        self.b_aw = np.zeros(3, dtype=float)
        self.a_prev = np.zeros(3, dtype=float)
        self.stationary = False
        self.stationary_count = 0
        self.moving_count = 0

    def reset(self) -> None:
        self.v[:] = 0.0
        self.p[:] = 0.0
        self.b_aw[:] = 0.0
        self.a_prev[:] = 0.0
        self.stationary = False
        self.stationary_count = 0
        self.moving_count = 0

    def zupt(self, a_body: np.ndarray, omega: np.ndarray, residual_ok: bool) -> bool:
        """
        Stationary detection:
        - small omega
        - accel magnitude close to g
        - rigid-body residuals ok

        Uses a small hysteresis window so "still" doesn't flicker.
        If stationary, we clamp v=0.
        """
        if not cfg.zupt_enable:
            return False

        cond = (safe_norm(omega) < cfg.zupt_gyro and
                abs(safe_norm(a_body) - cfg.g) < cfg.zupt_acc_mag and
                residual_ok)
        if cond:
            self.stationary_count += 1
            self.moving_count = 0
        else:
            self.moving_count += 1
            self.stationary_count = 0

        if not self.stationary and self.stationary_count >= cfg.zupt_min_frames:
            self.stationary = True
        elif self.stationary and self.moving_count >= cfg.zupt_release_frames:
            self.stationary = False

        if self.stationary:
            self.v[:] = 0.0
        return self.stationary

    def step(self, a_body: np.ndarray, Rwb: np.ndarray, dt: float, stationary: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One integration step.

        a_body includes gravity, in BODY frame.
        Rwb maps BODY -> WORLD.

        Steps:
        1) Convert a_body to world
        2) Remove world gravity [0,0,g]
        3) Remove learned bias b_aw
        4) If stationary: update bias estimate so linear accel -> 0
        5) If not stationary: integrate v using trapezoid rule
        6) integrate p using v
        """
        a_world = Rwb @ a_body
        a_lin = a_world - np.array([0.0, 0.0, cfg.g], dtype=float)
        a_lin = a_lin - self.b_aw

        if stationary and cfg.bias_learn_enable:
            k = cfg.bias_learn_rate * dt
            self.b_aw += k * a_lin

        if not stationary:
            self.v += 0.5 * (self.a_prev + a_lin) * dt
            if cfg.vel_damping > 0.0:
                self.v *= math.exp(-cfg.vel_damping * dt)

        self.p += self.v * dt
        self.a_prev = a_lin
        return self.p.copy(), self.v.copy(), a_lin


# ============================================================
# Main loop (replaces App Lab Bridge loop)
# ============================================================

def main() -> None:
    print("=== PI IMU FUSION START ===")

    # Open I2C bus 1 (typical on Raspberry Pi)
    with SMBus(1) as bus:
        mux = TCA9548A(bus, TCA_ADDR)

        # Create the 4 IMU objects
        imus = {
            0: MPUBase(bus, mux, ch=0, accel_16g=True,  has_mag=True),
            1: MPUBase(bus, mux, ch=1, accel_16g=True,  has_mag=True),
            2: MPUBase(bus, mux, ch=2, accel_16g=False, has_mag=True),
            3: MPUBase(bus, mux, ch=3, accel_16g=True,  has_mag=True),
        }

        # -------- "Ping" equivalent ----------
        # On Pi, ping just means "can we talk to the mux and the IMUs"
        try:
            mux.select(0)
            print("Ping: mux channel select OK")
        except Exception as e:
            raise SystemExit(f"Ping failed: cannot talk to TCA9548A at 0x{TCA_ADDR:02X}: {e}")

        # Initialize each IMU channel
        for i in [0, 1, 2, 3]:
            try:
                print(f"Init IMU{i} on mux channel {imus[i].ch} (mag={imus[i].has_mag})")
                imus[i].init_device()
            except Exception as e:
                raise SystemExit(f"Init failed on IMU{i}: {e}")

        # Build algorithm objects
        cal = BiasCal4(cfg.cal_frames)
        fuse = Fusion()
        ahrs = MahonyAHRS(kp=2.5, ki=0.05)
        integ = Integrator()
        if cfg.anchor_mode_enable and cfg.anchor_prompt_calibration:
            print("Anchor mode enabled. Press Enter to start calibration...")
            try:
                input()
            except EOFError:
                pass

        anchor_input = AnchorInput() if cfg.anchor_mode_enable else None
        anchor_mgr = AnchorManager() if cfg.anchor_mode_enable else None

        print(f"Collecting {cfg.cal_frames} stationary frames for bias... keep still")

        # Timing
        t_prev = time.perf_counter()
        frame = 0
        stationary_time = 0.0

        # Main loop
        while True:
            frame += 1

            # dt (clamped like your code)
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            dt = clamp(dt, cfg.dt_min, cfg.dt_max)

            # Read all sensors each loop, but never let ONE bad channel kill the program
            acc_raw: Dict[int, np.ndarray] = {}
            gyr_raw: Dict[int, np.ndarray] = {}
            mag_raw: Dict[int, np.ndarray] = {}

            ok = True
            for i in [0, 1, 2, 3]:
                try:
                    a, w = imus[i].read_accel_gyro()
                    m = imus[i].read_mag_uT()
                    acc_raw[i] = a
                    gyr_raw[i] = w
                    mag_raw[i] = m
                except Exception as e:
                    print(f"[WARN] read fail IMU{i}: {e}")
                    ok = False
                    break

            if not ok:
                time.sleep(0.01)
                continue

            # Calibration phase
            if not cal.done:
                cal.add(acc_raw, gyr_raw)
                if frame % 25 == 0:
                    print(f"  cal {min(cal.k, cal.N)}/{cal.N}")
                continue

            # Bias-corrected sensor values
            acc, gyr, mag = cal.correct(acc_raw, gyr_raw, mag_raw)

            # Main IMU provides primary motion; others validate consistency
            omega = fuse.fuse_omega(gyr, MAIN_IMU)
            a0_body = acc[MAIN_IMU]
            alpha = fuse.estimate_alpha(acc, omega, MAIN_IMU)
            norms = fuse.residual_norms(acc, omega, a0_body, alpha, MAIN_IMU)

            # Mag from main IMU (optional yaw correction)
            m_fused = fuse.fuse_mag(mag, MAIN_IMU)
            use_mag = safe_norm(m_fused) > 1e-3

            # Attitude update uses a0_body (translation accel estimate)
            ahrs.update(omega, a0_body, m_fused, dt, use_mag=use_mag)
            Rwb = ahrs.q_to_R()

            # Stationary decision for ZUPT
            residual_ok = (max(norms.values()) < cfg.res_warn)
            stationary = integ.zupt(a0_body, omega, residual_ok)
            stationary_time = (stationary_time + dt) if stationary else 0.0

            if cfg.anchor_mode_enable and anchor_input and anchor_mgr:
                while True:
                    cmd = anchor_input.poll()
                    if cmd is None:
                        break
                    anchor_mgr.handle_command(cmd)
                anchor_mgr.apply_if_ready(integ, stationary, stationary_time)

            # Integrate
            p, v, a_lin_w = integ.step(a0_body, Rwb, dt, stationary=stationary)

            # Minimal print (every 10 frames)
            if frame % 10 == 0:
                if cfg.anchor_mode_enable and anchor_mgr:
                    anchor = anchor_mgr.last_anchor_m
                    anchor_str = "None" if anchor is None else f"[{anchor[0]:+7.3f},{anchor[1]:+7.3f},{anchor[2]:+7.3f}]"
                    print(
                        f"p [{p[0]:+7.3f},{p[1]:+7.3f},{p[2]:+7.3f}]  "
                        f"anchor {anchor_str}  still={stationary}  "
                        f"applied={anchor_mgr.last_applied}  dt={dt*1000:5.1f}ms"
                    )
                else:
                    # Report only what you said you care about:
                    # world-frame linear accel, velocity, position, plus "still" and dt
                    print(
                        # f"a_xy [{a_lin_w[0]:+7.3f},{a_lin_w[1]:+7.3f}]  "
                        # f"v_xy [{v[0]:+7.3f},{v[1]:+7.3f}]  "
                        f"p_xy [{p[0]:+7.3f},{p[1]:+7.3f}]  "
                        f"still={stationary}  dt={dt*1000:5.1f}ms"
                    )

            if cfg.anchor_mode_enable and anchor_mgr and anchor_mgr.print_requested:
                anchor_mgr.print_requested = False
                anchor = anchor_mgr.last_anchor_m
                anchor_str = "None" if anchor is None else f"[{anchor[0]:+7.3f},{anchor[1]:+7.3f},{anchor[2]:+7.3f}]"
                print(
                    f"[ANCHOR] p={p} v={v} anchor={anchor_str} still={stationary} "
                    f"stationary_s={stationary_time:.2f}"
                )

            time.sleep(0.005)


if __name__ == "__main__":
    main()
