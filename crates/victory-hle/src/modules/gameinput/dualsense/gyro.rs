//! 6-axis DualSense IMU calibration (`0x05`), 3 MHz hardware timestamp
//! integration, Player-Space (`Yaw + Roll`) synthesis, and right-stick
//! anti-deadzone mapping.

use victory_core::config::{DualSenseConfig, GyroMode, GyroSpace};

use super::protocol::{
    BT_CRC_SEED_FEATURE, FEATURE_CALIBRATION_ID, FEATURE_CALIBRATION_LEN, RawReport, Transport,
    crc32_seeded,
};

/// Hardware sensor clock: 3,000,000 ticks per second (`0.3333 µs` per tick).
const SENSOR_HZ: f32 = 3_000_000.0;
/// Nominal DualSense gyroscope LSB per deg/s (~2000 dps full-scale) when `0x05` is unavailable.
const DEFAULT_GYRO_LSB_PER_DPS: f32 = 16.4;
/// Nominal DualSense accelerometer LSB per g (~4g full-scale).
const DEFAULT_ACCEL_LSB_PER_G: f32 = 8192.0;
/// Angular velocity (deg/s) corresponding to full right-stick deflection at `sensitivity = 1.0`.
const FULL_STICK_DPS: f32 = 180.0;

#[derive(Clone, Copy, Debug)]
pub struct ImuCalibration {
    pub gyro_bias: [f32; 3],
    pub gyro_scale_dps: [f32; 3],
    pub accel_bias: [f32; 3],
    pub accel_scale_g: [f32; 3],
}

impl Default for ImuCalibration {
    fn default() -> Self {
        Self {
            gyro_bias: [0.0; 3],
            gyro_scale_dps: [1.0 / DEFAULT_GYRO_LSB_PER_DPS; 3],
            accel_bias: [0.0; 3],
            accel_scale_g: [1.0 / DEFAULT_ACCEL_LSB_PER_G; 3],
        }
    }
}

impl ImuCalibration {
    /// Parses Feature Report `0x05` (41 bytes on USB; 41 bytes with CRC-32 on BT).
    pub fn from_feature_report(buf: &[u8], transport: Transport) -> Option<Self> {
        if buf.len() < FEATURE_CALIBRATION_LEN || buf[0] != FEATURE_CALIBRATION_ID {
            return None;
        }
        if transport == Transport::Bluetooth {
            let body = &buf[..FEATURE_CALIBRATION_LEN - 4];
            let crc = u32::from_le_bytes(
                buf[FEATURE_CALIBRATION_LEN - 4..FEATURE_CALIBRATION_LEN]
                    .try_into()
                    .ok()?,
            );
            if crc32_seeded(BT_CRC_SEED_FEATURE, body) != crc {
                return None;
            }
        }
        let i16_at = |at: usize| f32::from(i16::from_le_bytes([buf[at], buf[at + 1]]));

        let pitch_bias = i16_at(1);
        let yaw_bias = i16_at(3);
        let roll_bias = i16_at(5);
        let pitch_plus = i16_at(7);
        let pitch_minus = i16_at(9);
        let yaw_plus = i16_at(11);
        let yaw_minus = i16_at(13);
        let roll_plus = i16_at(15);
        let roll_minus = i16_at(17);
        let speed_2x = i16_at(19) + i16_at(21);

        let gyro_axis_scale = |plus: f32, minus: f32| {
            let denom = plus - minus;
            if denom.abs() > 100.0 && speed_2x.abs() > 10.0 {
                speed_2x / denom
            } else {
                1.0 / DEFAULT_GYRO_LSB_PER_DPS
            }
        };

        let accel_axis = |at: usize| {
            let plus = i16_at(at);
            let minus = i16_at(at + 2);
            let range_2g = plus - minus;
            if range_2g.abs() > 500.0 {
                ((plus + minus) * 0.5, 2.0 / range_2g)
            } else {
                (0.0, 1.0 / DEFAULT_ACCEL_LSB_PER_G)
            }
        };
        let (ax_b, ax_s) = accel_axis(23);
        let (ay_b, ay_s) = accel_axis(27);
        let (az_b, az_s) = accel_axis(31);

        Some(Self {
            gyro_bias: [pitch_bias, yaw_bias, roll_bias],
            gyro_scale_dps: [
                gyro_axis_scale(pitch_plus, pitch_minus),
                gyro_axis_scale(yaw_plus, yaw_minus),
                gyro_axis_scale(roll_plus, roll_minus),
            ],
            accel_bias: [ax_b, ay_b, az_b],
            accel_scale_g: [ax_s, ay_s, az_s],
        })
    }
}

/// High-frequency IMU integrator shared between the native HID reader thread
/// (up to 1000 Hz+) and the GameInput poll consumer.
pub struct GyroTracker {
    calibration: ImuCalibration,
    runtime_dps_offset: [f32; 3],
    last_dps: [f32; 3],
    gravity: [f32; 3],
    last_sensor_ts: Option<u32>,
    /// Integrated `(horiz_deg, vert_deg)` and total `dt_sec` since last consume.
    accum_deg_x: f32,
    accum_deg_y: f32,
    accum_dt: f32,
}

impl Default for GyroTracker {
    fn default() -> Self {
        Self {
            calibration: ImuCalibration::default(),
            runtime_dps_offset: [0.0; 3],
            last_dps: [0.0; 3],
            // Default resting DualSense held slightly reclined (Y up, -Z forward).
            gravity: [0.0, 0.85, -0.52],
            last_sensor_ts: None,
            accum_deg_x: 0.0,
            accum_deg_y: 0.0,
            accum_dt: 0.0,
        }
    }
}

impl GyroTracker {
    pub fn set_calibration(&mut self, cal: ImuCalibration) {
        self.calibration = cal;
    }

    pub fn reset_stream(&mut self) {
        self.last_sensor_ts = None;
        self.accum_deg_x = 0.0;
        self.accum_deg_y = 0.0;
        self.accum_dt = 0.0;
    }

    /// Captures the latest calibrated gyro reading as the new zero-drift baseline
    /// (invoked on `PS + Touchpad Click`).
    pub fn recenter(&mut self) {
        for i in 0..3 {
            self.runtime_dps_offset[i] += self.last_dps[i];
        }
        self.accum_deg_x = 0.0;
        self.accum_deg_y = 0.0;
        self.accum_dt = 0.0;
    }

    /// Feeds one raw HID report on the reader thread.
    pub fn push_sample(&mut self, raw: &RawReport, space: GyroSpace) {
        let cal = &self.calibration;
        let mut dps = [0.0f32; 3];
        for (i, &g) in raw.gyro.iter().enumerate() {
            dps[i] = (f32::from(g) - cal.gyro_bias[i]) * cal.gyro_scale_dps[i]
                - self.runtime_dps_offset[i];
        }
        self.last_dps = dps;

        // Low-pass filter accelerometer for stable gravity pitch projection.
        let ax = (f32::from(raw.accel[0]) - cal.accel_bias[0]) * cal.accel_scale_g[0];
        let ay = (f32::from(raw.accel[1]) - cal.accel_bias[1]) * cal.accel_scale_g[1];
        let az = (f32::from(raw.accel[2]) - cal.accel_bias[2]) * cal.accel_scale_g[2];
        let norm = (ax * ax + ay * ay + az * az).sqrt();
        if (0.5..=1.8).contains(&norm) {
            const ALPHA: f32 = 0.06;
            self.gravity[0] += ALPHA * (ax / norm - self.gravity[0]);
            self.gravity[1] += ALPHA * (ay / norm - self.gravity[1]);
            self.gravity[2] += ALPHA * (az / norm - self.gravity[2]);
        }

        let dt = match self.last_sensor_ts.replace(raw.sensor_timestamp) {
            Some(prev) => {
                let ticks = raw.sensor_timestamp.wrapping_sub(prev);
                let sec = (ticks as f32) / SENSOR_HZ;
                if (0.000_1..=0.05).contains(&sec) {
                    sec
                } else {
                    0.001
                }
            }
            None => 0.001,
        };

        let pitch_dps = dps[0];
        let yaw_dps = dps[1];
        let roll_dps = dps[2];

        // DualSense axes: +pitch = nose up, +yaw = counter-clockwise (turn left),
        // +roll = right side down (clockwise). Turning right in world space is `-yaw`.
        let turn_right_dps = match space {
            GyroSpace::Yaw => -yaw_dps,
            GyroSpace::Roll => roll_dps,
            GyroSpace::PlayerSpace => {
                let gy = self.gravity[1];
                let gz = self.gravity[2];
                let plane = (gy * gy + gz * gz).sqrt().max(0.25);
                let cos_pitch = (gy.abs() / plane).clamp(0.0, 1.0);
                let sin_pitch = (gz.abs() / plane).clamp(0.0, 1.0);
                -yaw_dps * cos_pitch + roll_dps * sin_pitch
            }
        };

        self.accum_deg_x += turn_right_dps * dt;
        self.accum_deg_y += pitch_dps * dt;
        self.accum_dt += dt;
    }

    /// Consumes accumulated angular motion and blends it onto `(right_x, right_y)`
    /// with anti-deadzone compensation when gyro aiming is active.
    pub fn apply_to_right_stick(
        &mut self,
        raw: &RawReport,
        cfg: &DualSenseConfig,
        right_x: &mut f32,
        right_y: &mut f32,
    ) {
        let (dx, dy, dt) = (
            std::mem::take(&mut self.accum_deg_x),
            std::mem::take(&mut self.accum_deg_y),
            std::mem::take(&mut self.accum_dt),
        );

        let active = match cfg.gyro_mode {
            GyroMode::Off => false,
            GyroMode::Always => true,
            GyroMode::Ads => raw.l2 >= 38, // ~15% L2 pull
            GyroMode::Touch => raw.any_touch_active(),
            GyroMode::AdsOrTouch => raw.l2 >= 38 || raw.any_touch_active(),
        };
        if !active || dt <= 0.0 || cfg.gyro_sensitivity <= 0.0 {
            return;
        }

        let avg_dps_x = dx / dt;
        let avg_dps_y = dy / dt;
        // Suppress sub-0.35 deg/s sensor thermal noise at rest.
        let speed = (avg_dps_x * avg_dps_x + avg_dps_y * avg_dps_y).sqrt();
        if speed < 0.35 {
            return;
        }

        let scale = cfg.gyro_sensitivity / FULL_STICK_DPS;
        let mut gx = avg_dps_x * scale;
        let mut gy = avg_dps_y * scale;

        // When the physical right thumbstick is at rest, lift small gyro vectors
        // onto `[deadzone_floor .. 1.0]` so the title's inner stick deadzone does
        // not swallow fine micro-aiming.
        let stick_mag = (*right_x * *right_x + *right_y * *right_y).sqrt();
        let gyro_mag = (gx * gx + gy * gy).sqrt();
        let floor = cfg.gyro_deadzone_floor.clamp(0.0, 0.45);
        if stick_mag < 0.08 && gyro_mag > 1e-4 && floor > 0.0 {
            let lifted = (floor + gyro_mag.min(1.0) * (1.0 - floor)).min(1.0);
            let factor = lifted / gyro_mag;
            gx *= factor;
            gy *= factor;
        }

        *right_x = (*right_x + gx).clamp(-1.0, 1.0);
        *right_y = (*right_y + gy).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_report_parses_and_gyro_lifts_above_deadzone_floor() {
        let mut feat = [0u8; FEATURE_CALIBRATION_LEN];
        feat[0] = FEATURE_CALIBRATION_ID;
        // Zero biases, +/- 16384 for 1000 dps (+/- 500 -> speed_2x = 1000).
        for at in [7usize, 11, 15] {
            feat[at..at + 2].copy_from_slice(&8192i16.to_le_bytes());
            feat[at + 2..at + 4].copy_from_slice(&(-8192i16).to_le_bytes());
        }
        feat[19..21].copy_from_slice(&500i16.to_le_bytes());
        feat[21..23].copy_from_slice(&500i16.to_le_bytes());
        for at in [23usize, 27, 31] {
            feat[at..at + 2].copy_from_slice(&8192i16.to_le_bytes());
            feat[at + 2..at + 4].copy_from_slice(&(-8192i16).to_le_bytes());
        }

        let cal = ImuCalibration::from_feature_report(&feat, Transport::Usb).unwrap();
        let mut tracker = GyroTracker::default();
        tracker.set_calibration(cal);

        let mut sample = RawReport {
            l2: 200,             // ADS active
            accel: [0, 8192, 0], // flat controller (gy = 1.0)
            sensor_timestamp: 3_000,
            ..Default::default()
        };
        tracker.push_sample(&sample, GyroSpace::PlayerSpace);

        // Gently turn right (`yaw` negative -> `turn_right` positive) over 1 ms (3000 ticks).
        sample.gyro = [0, -160, 0];
        sample.sensor_timestamp = 6_000;
        tracker.push_sample(&sample, GyroSpace::PlayerSpace);

        let cfg = DualSenseConfig::default();
        let (mut rx, mut ry) = (0.0, 0.0);
        tracker.apply_to_right_stick(&sample, &cfg, &mut rx, &mut ry);
        assert!(
            rx > cfg.gyro_deadzone_floor,
            "expected anti-deadzone boost > {}, got {rx}",
            cfg.gyro_deadzone_floor
        );
        assert_eq!(ry, 0.0);

        // Recenter zeroes subsequent stationary drift at the same raw rate.
        tracker.recenter();
        sample.sensor_timestamp = 9_000;
        tracker.push_sample(&sample, GyroSpace::PlayerSpace);
        let (mut rx2, mut ry2) = (0.0, 0.0);
        tracker.apply_to_right_stick(&sample, &cfg, &mut rx2, &mut ry2);
        assert_eq!((rx2, ry2), (0.0, 0.0));
    }
}
