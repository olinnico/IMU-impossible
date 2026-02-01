#!/usr/bin/env python3
"""
imu_x_accel_4x_mpu9050.py

Reads X-axis acceleration from four MPU9050 IMUs behind a TCA9548A I2C mux.

Assumed hardware (based on repo context):
- TCA9548A I2C multiplexer at 0x70
- Four MPU9050 IMUs at address 0x68
- IMUs connected to mux channels 0..3

Output:
- Prints X-axis acceleration (m/s^2) for each IMU.
- Optional +X/-X calibration saved to imu_x_calibration.json.

Requirements:
- pip3 install smbus2
- I2C enabled on Raspberry Pi
"""

import argparse
import json
import time
from typing import Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from smbus2 import SMBus, i2c_msg

TCA_ADDR = 0x70
MPU_ADDR = 0x68

REG_PWR_MGMT_1 = 0x6B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B

ACCEL_LSB_PER_G_2G = 16384.0
G_MPS2 = 9.81

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
@@ -93,53 +106,140 @@ class MPU9050:

    def read_accel_x_mps2(self) -> float:
        if not self.ready:
            raise RuntimeError("IMU not initialized")
        self.mux.select(self.channel)
        data = i2c_read_bytes(self.bus, MPU_ADDR, REG_ACCEL_XOUT_H, 2)
        ax_counts = be16_to_i16(data[0], data[1])
        return (ax_counts / ACCEL_LSB_PER_G_2G) * G_MPS2


def init_imus(bus: SMBus, mux: TCA9548A) -> Dict[int, MPU9050]:
    imus = {ch: MPU9050(bus, mux, ch) for ch in range(4)}
    for ch, imu in imus.items():
        print(f"Init IMU{ch} on mux channel {ch}")
        imu.init_device()
    return imus


def read_all_x(imus: Dict[int, MPU9050]) -> Dict[int, float]:
    values = {}
    for ch, imu in imus.items():
        values[ch] = imu.read_accel_x_mps2()
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read X-axis acceleration from four MPU9050 IMUs."
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


def average_samples(imus: Dict[int, MPU9050], samples: int) -> Dict[int, float]:
    totals = {ch: 0.0 for ch in imus}
    for _ in range(samples):
        values = read_all_x(imus)
        for ch, val in values.items():
            totals[ch] += val
        time.sleep(0.005)
    return {ch: totals[ch] / samples for ch in totals}


def calibrate_x(imus: Dict[int, MPU9050], samples: int) -> Dict[int, XCalibration]:
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
    print("=== MPU9050 X-AXIS ACCEL READ ===")
    with SMBus(1) as bus:
        mux = TCA9548A(bus, TCA_ADDR)
        try:
            mux.select(0)
            print("Mux OK")
        except Exception as exc:
            raise SystemExit(f"Mux init failed at 0x{TCA_ADDR:02X}: {exc}")

        imus = init_imus(bus, mux)
        calibration = load_calibration()
        if args.calibrate_x:
            calibration = calibrate_x(imus, args.samples)
            save_calibration(calibration)
            print(f"Saved calibration to {CALIBRATION_PATH}")
            return
        while True:
            try:
                values = read_all_x(imus)
                values = apply_calibration(values, calibration)
            except Exception as exc:
                print(f"Read error: {exc}")
                time.sleep(0.05)
                continue

            line = "  ".join(
                f"IMU{ch}: ax={values[ch]:+7.3f} m/s^2" for ch in sorted(values)
            )
            print(line)
            time.sleep(0.05)


if __name__ == "__main__":
    main()
