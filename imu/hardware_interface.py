"""
hardware_interface.py

Hardware interface for 4 MPU9250 IMUs on Raspberry Pi.

Your IMU Configuration:
- CH2 @ (0,0,0):     I2C 0x68, ±2g range
- CH3 @ (0,80,5):    I2C 0x69, ±4g range  
- CH0 @ (80,80,10):  I2C 0x68 (via multiplexer?), ±8g range
- CH1 @ (80,0,15):   I2C 0x69 (via multiplexer?), ±16g range

NOTE: You have address conflicts (two 0x68, two 0x69). 
Solutions:
1. Use I2C multiplexer (TCA9548A) to isolate each IMU
2. Use two I2C buses (I2C1 and I2C0) if available
3. Reconfigure AD0 pins to get unique addresses

This implementation assumes Solution 1 (I2C multiplexer).
"""

import time
import numpy as np
import smbus
from imusensor.MPU9250 import MPU9250

# =============================================================================
# I2C MULTIPLEXER CONTROL (if using TCA9548A)
# =============================================================================

MUX_ADDRESS = 0x70  # Default TCA9548A address
MUX_CHANNEL_CH2 = 0  # Channel 0 for CH2
MUX_CHANNEL_CH3 = 1  # Channel 1 for CH3
MUX_CHANNEL_CH0 = 2  # Channel 2 for CH0
MUX_CHANNEL_CH1 = 3  # Channel 3 for CH1

def select_mux_channel(bus: smbus.SMBus, channel: int):
    """
    Select I2C multiplexer channel.
    
    Args:
        bus: SMBus object
        channel: Channel number (0-7)
    """
    if channel < 0 or channel > 7:
        raise ValueError(f"Invalid mux channel: {channel}")
    
    # Write channel select byte (1 << channel)
    bus.write_byte(MUX_ADDRESS, 1 << channel)
    time.sleep(0.001)  # Small delay for switching


# =============================================================================
# IMU HARDWARE CLASS
# =============================================================================

class IMUHardware:
    """
    Manages 4 MPU9250 IMUs with different configurations.
    """
    
    # I2C addresses (after multiplexer selects channel)
    ADDR_CH2 = 0x68  # ±2g
    ADDR_CH3 = 0x69  # ±4g
    ADDR_CH0 = 0x68  # ±8g (same as CH2 but on different mux channel)
    ADDR_CH1 = 0x69  # ±16g (same as CH3 but on different mux channel)
    
    # MPU9250 accelerometer range register values
    ACCEL_RANGE_2G  = 0x00  # ±2g
    ACCEL_RANGE_4G  = 0x08  # ±4g
    ACCEL_RANGE_8G  = 0x10  # ±8g
    ACCEL_RANGE_16G = 0x18  # ±16g
    
    # Conversion factors (LSB/g)
    ACCEL_SCALE_2G  = 16384.0  # LSB/g
    ACCEL_SCALE_4G  = 8192.0
    ACCEL_SCALE_8G  = 4096.0
    ACCEL_SCALE_16G = 2048.0
    
    # Gyroscope scale (assuming ±250°/s range, typical default)
    GYRO_SCALE = 131.0  # LSB/(°/s)
    
    def __init__(self, use_multiplexer: bool = True):
        """
        Initialize 4 IMUs.
        
        Args:
            use_multiplexer: If True, use I2C multiplexer. If False, assumes
                           you've solved address conflicts another way.
        """
        self.use_mux = use_multiplexer
        self.bus = smbus.SMBus(1)  # I2C bus 1 (primary on Raspberry Pi)
        
        # Initialize each IMU
        print("Initializing 4 IMUs...")
        
        # CH2: ±2g at (0,0,0)
        print("  CH2 (±2g)...", end=" ")
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH2)
        self.imu_ch2 = MPU9250(self.bus, self.ADDR_CH2)
        self.imu_ch2.begin()
        self._configure_accel_range(self.imu_ch2, self.ACCEL_RANGE_2G)
        print("OK")
        
        # CH3: ±4g at (0,80,5)
        print("  CH3 (±4g)...", end=" ")
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH3)
        self.imu_ch3 = MPU9250(self.bus, self.ADDR_CH3)
        self.imu_ch3.begin()
        self._configure_accel_range(self.imu_ch3, self.ACCEL_RANGE_4G)
        print("OK")
        
        # CH0: ±8g at (80,80,10)
        print("  CH0 (±8g)...", end=" ")
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH0)
        self.imu_ch0 = MPU9250(self.bus, self.ADDR_CH0)
        self.imu_ch0.begin()
        self._configure_accel_range(self.imu_ch0, self.ACCEL_RANGE_8G)
        print("OK")
        
        # CH1: ±16g at (80,0,15)
        print("  CH1 (±16g)...", end=" ")
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH1)
        self.imu_ch1 = MPU9250(self.bus, self.ADDR_CH1)
        self.imu_ch1.begin()
        self._configure_accel_range(self.imu_ch1, self.ACCEL_RANGE_16G)
        print("OK")
        
        print("All IMUs initialized successfully!\n")
    
    def _configure_accel_range(self, imu: MPU9250, range_setting: int):
        """
        Configure accelerometer range for an IMU.
        
        This writes directly to the MPU9250 ACCEL_CONFIG register (0x1C).
        
        Args:
            imu: MPU9250 object
            range_setting: One of ACCEL_RANGE_* constants
        """
        ACCEL_CONFIG_REG = 0x1C
        try:
            # Read current register value
            current = imu.bus.read_byte_data(imu.address, ACCEL_CONFIG_REG)
            # Clear bits 3-4 (range bits) and set new range
            new_value = (current & 0xE7) | range_setting
            # Write back
            imu.bus.write_byte_data(imu.address, ACCEL_CONFIG_REG, new_value)
            time.sleep(0.01)  # Allow sensor to update
        except Exception as e:
            print(f"Warning: Could not configure accel range: {e}")
    
    def read_ch2(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read CH2 (±2g at origin).
        
        Returns:
            (accel, gyro, mag) in m/s², rad/s, and µT
        """
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH2)
        
        self.imu_ch2.readSensor()
        
        # Convert to standard units
        accel = np.array([
            self.imu_ch2.AccelVals[0],  # Already in m/s² from library
            self.imu_ch2.AccelVals[1],
            self.imu_ch2.AccelVals[2]
        ], dtype=float)
        
        gyro = np.array([
            np.deg2rad(self.imu_ch2.GyroVals[0]),  # Convert deg/s -> rad/s
            np.deg2rad(self.imu_ch2.GyroVals[1]),
            np.deg2rad(self.imu_ch2.GyroVals[2])
        ], dtype=float)
        
        mag = np.array([
            self.imu_ch2.MagVals[0],  # Magnetometer in µT
            self.imu_ch2.MagVals[1],
            self.imu_ch2.MagVals[2]
        ], dtype=float)
        
        return accel, gyro, mag
    
    def read_ch3(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read CH3 (±4g at 0,80,5).
        
        Returns:
            (accel, gyro, mag) in m/s², rad/s, and µT
        """
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH3)
        
        self.imu_ch3.readSensor()
        
        accel = np.array([
            self.imu_ch3.AccelVals[0],
            self.imu_ch3.AccelVals[1],
            self.imu_ch3.AccelVals[2]
        ], dtype=float)
        
        gyro = np.array([
            np.deg2rad(self.imu_ch3.GyroVals[0]),
            np.deg2rad(self.imu_ch3.GyroVals[1]),
            np.deg2rad(self.imu_ch3.GyroVals[2])
        ], dtype=float)
        
        mag = np.array([
            self.imu_ch3.MagVals[0],
            self.imu_ch3.MagVals[1],
            self.imu_ch3.MagVals[2]
        ], dtype=float)
        
        return accel, gyro, mag
    
    def read_ch0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read CH0 (±8g at 80,80,10).
        
        Returns:
            (accel, gyro, mag) in m/s², rad/s, and µT
        """
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH0)
        
        self.imu_ch0.readSensor()
        
        accel = np.array([
            self.imu_ch0.AccelVals[0],
            self.imu_ch0.AccelVals[1],
            self.imu_ch0.AccelVals[2]
        ], dtype=float)
        
        gyro = np.array([
            np.deg2rad(self.imu_ch0.GyroVals[0]),
            np.deg2rad(self.imu_ch0.GyroVals[1]),
            np.deg2rad(self.imu_ch0.GyroVals[2])
        ], dtype=float)
        
        mag = np.array([
            self.imu_ch0.MagVals[0],
            self.imu_ch0.MagVals[1],
            self.imu_ch0.MagVals[2]
        ], dtype=float)
        
        return accel, gyro, mag
    
    def read_ch1(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read CH1 (±16g at 80,0,15).
        
        Returns:
            (accel, gyro, mag) in m/s², rad/s, and µT
        """
        if self.use_mux:
            select_mux_channel(self.bus, MUX_CHANNEL_CH1)
        
        self.imu_ch1.readSensor()
        
        accel = np.array([
            self.imu_ch1.AccelVals[0],
            self.imu_ch1.AccelVals[1],
            self.imu_ch1.AccelVals[2]
        ], dtype=float)
        
        gyro = np.array([
            np.deg2rad(self.imu_ch1.GyroVals[0]),
            np.deg2rad(self.imu_ch1.GyroVals[1]),
            np.deg2rad(self.imu_ch1.GyroVals[2])
        ], dtype=float)
        
        mag = np.array([
            self.imu_ch1.MagVals[0],
            self.imu_ch1.MagVals[1],
            self.imu_ch1.MagVals[2]
        ], dtype=float)
        
        return accel, gyro, mag
    
    def read_all(self) -> dict:
        """
        Read all 4 IMUs sequentially.
        
        Returns:
            dict with keys: 'ch2', 'ch3', 'ch0', 'ch1'
            Each value is tuple (accel, gyro, mag)
        """
        return {
            'ch2': self.read_ch2(),
            'ch3': self.read_ch3(),
            'ch0': self.read_ch0(),
            'ch1': self.read_ch1()
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Test reading from all 4 IMUs.
    """
    print("=" * 80)
    print("4-IMU Hardware Interface Test")
    print("=" * 80)
    print()
    
    # Initialize hardware
    hw = IMUHardware(use_multiplexer=True)
    
    print("Reading all 4 IMUs for 5 seconds...")
    print("(Keep platform stationary to verify gravity readings)")
    print()
    
    t_start = time.time()
    while time.time() - t_start < 5.0:
        # Read all IMUs
        data = hw.read_all()
        
        # Extract accelerations
        a_ch2, g_ch2, m_ch2 = data['ch2']
        a_ch3, g_ch3, m_ch3 = data['ch3']
        a_ch0, g_ch0, m_ch0 = data['ch0']
        a_ch1, g_ch1, m_ch1 = data['ch1']
        
        # Print magnitudes (should be ~9.81 m/s² when stationary)
        print(f"Accel magnitudes: "
              f"CH2={np.linalg.norm(a_ch2):6.3f} "
              f"CH3={np.linalg.norm(a_ch3):6.3f} "
              f"CH0={np.linalg.norm(a_ch0):6.3f} "
              f"CH1={np.linalg.norm(a_ch1):6.3f} m/s²")
        
        time.sleep(0.5)
    
    print()
    print("Test complete!")
    print()
    print("Expected: All magnitudes ≈ 9.81 m/s² when stationary")
    print("If values differ significantly, check:")
    print("  1. IMU mounting orientation")
    print("  2. I2C address conflicts")
    print("  3. Multiplexer channel assignments")
