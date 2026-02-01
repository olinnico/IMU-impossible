#!/usr/bin/env python3
"""
imu_x_accel_4x_mpu9250.py

Reads X-axis acceleration from four MPU9250 IMUs behind a TCA9548A I2C mux.

Assumed hardware (based on repo context):
- TCA9548A I2C multiplexer at 0x70
- Four MPU9250 IMUs at address 0x68
- IMUs connected to mux channels 0..3

Output:
- Prints X-axis acceleration (m/s^2) for each IMU.
- Optional +X/-X calibration saved to imu_x_calibration.json.
- Optional per-IMU gyro/mag readings and Kalman-filtered roll/pitch/yaw.

Notes:
- This script uses direct smbus2 I2C reads (not imusensor.MPU9250) to support the
  TCA9548A mux and per-channel mag reads without swapping bus objects.

Requirements:
- pip3 install smbus2
- I2C enabled on Raspberry Pi
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from smbus2 import SMBus, i2c_msg

TCA_ADDR = 0x70
MPU_ADDR = 0x68

REG_PWR_MGMT_1 = 0x6B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_GYRO_CONFIG = 0x1B
REG_GYRO_XOUT_H = 0x43
REG_INT_PIN_CFG = 0x37  # bypass enable bit

MAG_ADDR = 0x0C
AK8963_ST1 = 0x02
AK8963_XOUT_L = 0x03
AK8963_ST2 = 0x09
AK8963_CNTL1 = 0x0A

ACCEL_LSB_PER_G_16G = 2048.0
GYRO_LSB_PER_DPS_250 = 131.0
G_MPS2 = 9.81
DEG2RAD = math.pi / 180.0

CALIBRATION_PATH = Path(__file__).with_name("imu_x_calibration.json")


@dataclass
class XCalibration:
    offset: float = 0.0
    scale: float = 1.0


def i2c_write_byte(bus: SMBus, addr: int, reg: int, val: int, retries: int = 3) -> None:
    for _ in range(retries):
        try:
            bus.write_byte_data(addr, reg, val & 0xFF)
            return
        except Exception:
            time.sleep(0.005)
    raise IOError(f"I2C write failed addr=0x{addr:02X} reg=0x{reg:02X}")


def i2c_read_bytes(bus: SMBus, addr: int, reg: int, n: int, retries: int = 3) -> bytes:
    for _ in range(retries):
        try:
            write = i2c_msg.write(addr, [reg & 0xFF])
            read = i2c_msg.read(addr, n)
            bus.i2c_rdwr(write, read)
            return bytes(read)
        except Exception:
            time.sleep(0.005)
    raise IOError(f"I2C read failed addr=0x{addr:02X} reg=0x{reg:02X} n={n}")


def be16_to_i16(b_hi: int, b_lo: int) -> int:
    v = (b_hi << 8) | b_lo
    return v - 65536 if v & 0x8000 else v


def le16_to_i16(b_lo: int, b_hi: int) -> int:
    v = (b_hi << 8) | b_lo
    return v - 65536 if v & 0x8000 else v


class Kalman1D:
    def __init__(self, q_angle: float = 0.001, q_bias: float = 0.003, r_measure: float = 0.03):
        self.q_angle = q_angle
        self.q_bias = q_bias
        self.r_measure = r_measure
        self.angle = 0.0
        self.bias = 0.0
        self.rate = 0.0
        self.p = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, new_angle: float, new_rate: float, dt: float) -> float:
        self.rate = new_rate - self.bias
        self.angle += dt * self.rate

        self.p[0][0] += dt * (dt * self.p[1][1] - self.p[0][1] - self.p[1][0] + self.q_angle)
        self.p[0][1] -= dt * self.p[1][1]
        self.p[1][0] -= dt * self.p[1][1]
        self.p[1][1] += self.q_bias * dt

        s = self.p[0][0] + self.r_measure
        k0 = self.p[0][0] / s
        k1 = self.p[1][0] / s
        y = new_angle - self.angle
        self.angle += k0 * y
        self.bias += k1 * y

        p00 = self.p[0][0]
        p01 = self.p[0][1]
        self.p[0][0] -= k0 * p00
        self.p[0][1] -= k0 * p01
        self.p[1][0] -= k1 * p00
        self.p[1][1] -= k1 * p01
        return self.angle


class KalmanRPY:
    def __init__(self) -> None:
        self.roll = Kalman1D()
        self.pitch = Kalman1D()
        self.yaw = Kalman1D()

    def update(self, roll_meas: float, pitch_meas: float, yaw_meas: float,
               gyro_dps: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        roll = self.roll.update(roll_meas, gyro_dps[0], dt)
        pitch = self.pitch.update(pitch_meas, gyro_dps[1], dt)
        yaw = self.yaw.update(yaw_meas, gyro_dps[2], dt)
        return roll, pitch, yaw


def accel_to_roll_pitch_deg(accel: Tuple[float, float, float]) -> Tuple[float, float]:
    ax, ay, az = accel
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return roll, pitch


def mag_to_yaw_deg(mag: Tuple[float, float, float], roll_deg: float, pitch_deg: float) -> float:
    mx, my, mz = mag
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    xh = mx * math.cos(pitch) + mz * math.sin(pitch)
    yh = mx * math.sin(roll) * math.sin(pitch) + my * math.cos(roll) - mz * math.sin(roll) * math.cos(pitch)
    return math.degrees(math.atan2(-yh, xh))


class TCA9548A:
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
        time.sleep(0.001)


class MPU9250:
    def __init__(self, bus: SMBus, mux: TCA9548A, channel: int, enable_mag: bool = False):
        self.bus = bus
        self.mux = mux
        self.channel = channel
        self.enable_mag = enable_mag
        self.ready = False

    def init_device(self) -> None:
        self.mux.select(self.channel)
        i2c_write_byte(self.bus, MPU_ADDR, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.05)
        i2c_write_byte(self.bus, MPU_ADDR, REG_ACCEL_CONFIG, 0x18)
        time.sleep(0.01)
        i2c_write_byte(self.bus, MPU_ADDR, REG_GYRO_CONFIG, 0x00)
        time.sleep(0.01)
        if self.enable_mag:
            i2c_write_byte(self.bus, MPU_ADDR, REG_INT_PIN_CFG, 0x02)
            time.sleep(0.01)
            i2c_write_byte(self.bus, MAG_ADDR, AK8963_CNTL1, 0x00)
            time.sleep(0.01)
            i2c_write_byte(self.bus, MAG_ADDR, AK8963_CNTL1, 0x16)
            time.sleep(0.01)
        self.ready = True

    def read_accel_mps2(self) -> Tuple[float, float, float]:
        if not self.ready:
            raise RuntimeError("IMU not initialized")
        self.mux.select(self.channel)
        data = i2c_read_bytes(self.bus, MPU_ADDR, REG_ACCEL_XOUT_H, 6)
        ax = be16_to_i16(data[0], data[1])
        ay = be16_to_i16(data[2], data[3])
        az = be16_to_i16(data[4], data[5])
        return (
            (ax / ACCEL_LSB_PER_G_16G) * G_MPS2,
            (ay / ACCEL_LSB_PER_G_16G) * G_MPS2,
            (az / ACCEL_LSB_PER_G_16G) * G_MPS2,
        )

    def read_accel_x_mps2(self) -> float:
        ax, _, _ = self.read_accel_mps2()
        return ax

    def read_gyro_rad(self) -> Tuple[float, float, float]:
        if not self.ready:
            raise RuntimeError("IMU not initialized")
        self.mux.select(self.channel)
        data = i2c_read_bytes(self.bus, MPU_ADDR, REG_GYRO_XOUT_H, 6)
        gx = be16_to_i16(data[0], data[1])
        gy = be16_to_i16(data[2], data[3])
        gz = be16_to_i16(data[4], data[5])
        gx_dps = gx / GYRO_LSB_PER_DPS_250
        gy_dps = gy / GYRO_LSB_PER_DPS_250
        gz_dps = gz / GYRO_LSB_PER_DPS_250
        return gx_dps * DEG2RAD, gy_dps * DEG2RAD, gz_dps * DEG2RAD

    def read_mag_ut(self) -> Tuple[float, float, float]:
        if not self.enable_mag:
            return 0.0, 0.0, 0.0
        self.mux.select(self.channel)
        st1 = i2c_read_bytes(self.bus, MAG_ADDR, AK8963_ST1, 1)[0]
        if (st1 & 0x01) == 0:
            return 0.0, 0.0, 0.0
        data = i2c_read_bytes(self.bus, MAG_ADDR, AK8963_XOUT_L, 7)
        x = le16_to_i16(data[0], data[1])
        y = le16_to_i16(data[2], data[3])
        z = le16_to_i16(data[4], data[5])
        st2 = data[6]
        if st2 & 0x08:
            return 0.0, 0.0, 0.0
        ut_per_lsb = 4912.0 / 32760.0
        return x * ut_per_lsb, y * ut_per_lsb, z * ut_per_lsb


def init_imus(bus: SMBus, mux: TCA9548A, enable_mag: bool) -> Dict[int, MPU9250]:
    imus = {ch: MPU9250(bus, mux, ch, enable_mag=enable_mag) for ch in range(4)}
    for ch, imu in imus.items():
        print(f"Init IMU{ch} on mux channel {ch}")
        imu.init_device()
    return imus


def read_all_x(imus: Dict[int, MPU9250]) -> Dict[int, float]:
    values = {}
    for ch, imu in imus.items():
        values[ch] = imu.read_accel_x_mps2()
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read X-axis acceleration from four MPU9250 IMUs."
    )
    parser.add_argument(
        "--calibrate-x",
        action="store_true",
        help="Run +X/-X calibration and save to imu_x_calibration.json.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Samples to average per calibration pose (default: 200).",
    )
    parser.add_argument(
        "--disable-mag",
        action="store_true",
        help="Disable AK8963 magnetometer reads (enabled by default for MPU9250).",
    )
    return parser.parse_args()


def load_calibration() -> Dict[int, XCalibration]:
    if not CALIBRATION_PATH.exists():
        return {ch: XCalibration() for ch in range(4)}
    data = json.loads(CALIBRATION_PATH.read_text())
    cal = {}
    for key, entry in data.items():
        ch = int(key)
        cal[ch] = XCalibration(
            offset=float(entry.get("offset", 0.0)),
            scale=float(entry.get("scale", 1.0)),
        )
    return cal


def save_calibration(calibration: Dict[int, XCalibration]) -> None:
    payload = {
        str(ch): {"offset": cal.offset, "scale": cal.scale}
        for ch, cal in calibration.items()
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))


def average_samples(imus: Dict[int, MPU9250], samples: int) -> Dict[int, float]:
    totals = {ch: 0.0 for ch in imus}
    for _ in range(samples):
        values = read_all_x(imus)
        for ch, val in values.items():
            totals[ch] += val
        time.sleep(0.005)
    return {ch: totals[ch] / samples for ch in totals}


def calibrate_x(imus: Dict[int, MPU9250], samples: int) -> Dict[int, XCalibration]:
    print("Calibration step 1: Place +X up for all IMUs, then press Enter.")
    input()
    plus = average_samples(imus, samples)

    print("Calibration step 2: Place -X up for all IMUs, then press Enter.")
    input()
    minus = average_samples(imus, samples)

    calibration = {}
    for ch in sorted(imus):
        offset = (plus[ch] + minus[ch]) / 2.0
        scale = (2.0 * G_MPS2) / (plus[ch] - minus[ch])
        calibration[ch] = XCalibration(offset=offset, scale=scale)
        print(
            f"IMU{ch}: +X={plus[ch]:+.3f} -X={minus[ch]:+.3f} "
            f"-> offset={offset:+.3f} scale={scale:.5f}"
        )
    return calibration


def apply_calibration(values: Dict[int, float], calibration: Dict[int, XCalibration]) -> Dict[int, float]:
    corrected = {}
    for ch, raw in values.items():
        cal = calibration.get(ch, XCalibration())
        corrected[ch] = (raw - cal.offset) * cal.scale
    return corrected


def main() -> None:
    args = parse_args()
    print("=== MPU9250 X-AXIS ACCEL READ ===")
    with SMBus(1) as bus:
        mux = TCA9548A(bus, TCA_ADDR)
        try:
            mux.select(0)
            print("Mux OK")
        except Exception as exc:
            raise SystemExit(f"Mux init failed at 0x{TCA_ADDR:02X}: {exc}")

        imus = init_imus(bus, mux, not args.disable_mag)
        calibration = load_calibration()
        kalman_filters = {ch: KalmanRPY() for ch in imus}
        if args.calibrate_x:
            calibration = calibrate_x(imus, args.samples)
            save_calibration(calibration)
            print(f"Saved calibration to {CALIBRATION_PATH}")
            return
        t_prev = time.perf_counter()
        while True:
            t_now = time.perf_counter()
            dt = max(0.001, t_now - t_prev)
            t_prev = t_now
            lines = []
            for ch in sorted(imus):
                try:
                    ax, ay, az = imus[ch].read_accel_mps2()
                    gx, gy, gz = imus[ch].read_gyro_rad()
                    mx, my, mz = imus[ch].read_mag_ut()
                except Exception as exc:
                    print(f"Read error IMU{ch}: {exc}")
                    time.sleep(0.05)
                    continue
                cal = calibration.get(ch, XCalibration())
                ax = (ax - cal.offset) * cal.scale
                roll_meas, pitch_meas = accel_to_roll_pitch_deg((ax, ay, az))
                yaw_meas = mag_to_yaw_deg((mx, my, mz), roll_meas, pitch_meas)
                gyro_dps = (gx / DEG2RAD, gy / DEG2RAD, gz / DEG2RAD)
                roll_k, pitch_k, yaw_k = kalman_filters[ch].update(
                    roll_meas, pitch_meas, yaw_meas, gyro_dps, dt
                )
                lines.append(
                    f"IMU{ch}: ax={ax:+7.3f} m/s^2  "
                    f"kalman_rpy=[{roll_k:+6.1f},{pitch_k:+6.1f},{yaw_k:+6.1f}] deg"
                )
            if lines:
                print("  ".join(lines))
            time.sleep(0.05)


if __name__ == "__main__":
    main()
