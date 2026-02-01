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

Requirements:
- pip3 install smbus2
- I2C enabled on Raspberry Pi
"""

import time
from typing import Dict, Tuple

from smbus2 import SMBus, i2c_msg

TCA_ADDR = 0x70
MPU_ADDR = 0x68

REG_PWR_MGMT_1 = 0x6B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B

ACCEL_LSB_PER_G_2G = 16384.0
G_MPS2 = 9.81


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


class MPU9050:
    def __init__(self, bus: SMBus, mux: TCA9548A, channel: int):
        self.bus = bus
        self.mux = mux
        self.channel = channel
        self.ready = False

    def init_device(self) -> None:
        self.mux.select(self.channel)
        i2c_write_byte(self.bus, MPU_ADDR, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.05)
        i2c_write_byte(self.bus, MPU_ADDR, REG_ACCEL_CONFIG, 0x00)
        time.sleep(0.01)
        self.ready = True

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


def main() -> None:
    print("=== MPU9050 X-AXIS ACCEL READ ===")
    with SMBus(1) as bus:
        mux = TCA9548A(bus, TCA_ADDR)
        try:
            mux.select(0)
            print("Mux OK")
        except Exception as exc:
            raise SystemExit(f"Mux init failed at 0x{TCA_ADDR:02X}: {exc}")

        imus = init_imus(bus, mux)
        while True:
            try:
                values = read_all_x(imus)
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
