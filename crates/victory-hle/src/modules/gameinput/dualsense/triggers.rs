//! Xbox impulse trigger (`[left_trigger, right_trigger]` in `0.0..=1.0`) to
//! DualSense adaptive trigger translation and output-report rate limiting.

use std::time::{Duration, Instant};

use victory_core::config::TriggerMode;

use super::protocol::{OUTPUT_COMMON_LEN, OutputCommand};

/// Minimum interval between non-stopping output reports (~100 Hz) so continuous
/// game rumble updates never saturate Pico 2 W `DS5Dongle` Bluetooth ACL links.
pub const MIN_REPORT_INTERVAL: Duration = Duration::from_millis(10);

/// Mode `0x00`: releases stepper motor tension completely.
pub const TRIGGER_OFF: [u8; 11] = [0; 11];

/// Mode `0x21`: continuous positional resistance across zones.
const MODE_FEEDBACK: u8 = 0x21;
/// Mode `0x26`: positional vibration across 10 trigger zones with Hz control.
const MODE_VIBRATION: u8 = 0x26;

/// Compiles a normalized Xbox trigger rumble scalar (`0.0..=1.0`) into an
/// 11-byte DualSense adaptive trigger command block.
pub fn encode_trigger_effect(level: f32, mode: TriggerMode, strength: f32) -> [u8; 11] {
    let intensity = (level.clamp(0.0, 1.0) * strength.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    if mode == TriggerMode::Off || intensity <= 0.01 {
        return TRIGGER_OFF;
    }

    match mode {
        TriggerMode::Off => TRIGGER_OFF,
        TriggerMode::Vibration => encode_zone_vibration(intensity, 35.0, 130.0),
        TriggerMode::Hybrid => {
            if intensity < 0.28 {
                let force = (1 + (intensity / 0.28 * 2.0).round() as u8).min(3);
                encode_zone_feedback(0x03FC, force)
            } else {
                let norm = ((intensity - 0.28) / 0.72).clamp(0.0, 1.0);
                encode_zone_vibration(0.35 + norm * 0.65, 45.0, 135.0)
            }
        }
    }
}

/// Packs 3-bit force/amplitude `amp` (`1..=7`) across active bits of `zone_mask` (`0..10`).
fn pack_zone_forces(zone_mask: u16, amp: u8) -> u32 {
    let amp = u32::from(amp.min(7));
    let mut packed = 0u32;
    for zone in 0..10u16 {
        if zone_mask & (1 << zone) != 0 {
            packed |= amp << (zone * 3);
        }
    }
    packed
}

fn encode_zone_feedback(zone_mask: u16, force: u8) -> [u8; 11] {
    let mut out = [0u8; 11];
    out[0] = MODE_FEEDBACK;
    out[1..3].copy_from_slice(&zone_mask.to_le_bytes());
    let packed = pack_zone_forces(zone_mask, force);
    out[3..7].copy_from_slice(&packed.to_le_bytes());
    out
}

fn encode_zone_vibration(intensity: f32, min_hz: f32, max_hz: f32) -> [u8; 11] {
    // Activate zones 1..=9 (`0x03FE`), leaving initial ~10% pre-travel crisp and slack-free.
    let zone_mask: u16 = 0x03FE;
    let amp = (1 + (intensity * 6.0).round() as u8).clamp(1, 7);
    let freq = (min_hz + intensity * (max_hz - min_hz))
        .round()
        .clamp(1.0, 255.0) as u8;
    let packed = pack_zone_forces(zone_mask, amp);

    let mut out = [0u8; 11];
    out[0] = MODE_VIBRATION;
    out[1..3].copy_from_slice(&zone_mask.to_le_bytes());
    out[3..7].copy_from_slice(&packed.to_le_bytes());
    out[9] = freq;
    out
}

/// Deduplicates and rate-limits outgoing DualSense output reports while
/// guaranteeing that stop/release packets (`0.0` rumble & `Off` triggers) are
/// flushed immediately.
#[derive(Default)]
pub struct ReportLimiter {
    last_common: Option<[u8; OUTPUT_COMMON_LEN]>,
    last_sent_at: Option<Instant>,
}

impl ReportLimiter {
    pub fn reset(&mut self) {
        self.last_common = None;
        self.last_sent_at = None;
    }

    pub fn should_send(&mut self, cmd: &OutputCommand, now: Instant, force: bool) -> bool {
        let common = cmd.encode_common();
        if !force && self.last_common == Some(common) {
            return false;
        }
        let immediate = force || cmd.is_actuators_zero() || cmd.update_leds || cmd.update_mic_led;
        if !immediate
            && let Some(prev) = self.last_sent_at
            && now.saturating_duration_since(prev) < MIN_REPORT_INTERVAL
        {
            return false;
        }
        self.last_common = Some(common);
        self.last_sent_at = Some(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::FIRMWARE_VIBRATION2_MIN;
    use super::*;

    #[test]
    fn trigger_modes_encode_off_vibration_and_hybrid() {
        assert_eq!(
            encode_trigger_effect(0.0, TriggerMode::Vibration, 1.0),
            TRIGGER_OFF
        );
        assert_eq!(
            encode_trigger_effect(0.8, TriggerMode::Off, 1.0),
            TRIGGER_OFF
        );

        let light = encode_trigger_effect(0.2, TriggerMode::Vibration, 1.0);
        let heavy = encode_trigger_effect(1.0, TriggerMode::Vibration, 1.0);
        assert_eq!(light[0], MODE_VIBRATION);
        assert_eq!(heavy[0], MODE_VIBRATION);
        assert!(heavy[9] > light[9], "frequency scales with intensity");

        let hybrid_low = encode_trigger_effect(0.15, TriggerMode::Hybrid, 1.0);
        let hybrid_high = encode_trigger_effect(0.9, TriggerMode::Hybrid, 1.0);
        assert_eq!(hybrid_low[0], MODE_FEEDBACK);
        assert_eq!(hybrid_high[0], MODE_VIBRATION);
    }

    #[test]
    fn rate_limiter_throttles_updates_but_flushes_zero_stop_immediately() {
        let mut limiter = ReportLimiter::default();
        let t0 = Instant::now();

        let mut active = OutputCommand::neutral(FIRMWARE_VIBRATION2_MIN, false);
        active.motor_left = 100;
        active.right_trigger = encode_trigger_effect(0.5, TriggerMode::Vibration, 1.0);
        assert!(limiter.should_send(&active, t0, false));
        // Duplicate state is dropped.
        assert!(!limiter.should_send(&active, t0 + Duration::from_millis(20), false));

        // Rapid change 2 ms later is throttled.
        active.motor_left = 180;
        assert!(!limiter.should_send(&active, t0 + Duration::from_millis(2), false));
        // After 10 ms window it passes.
        assert!(limiter.should_send(&active, t0 + Duration::from_millis(12), false));

        // Zeroing rumble/triggers bypasses the 10 ms throttle immediately.
        let stop = OutputCommand::neutral(FIRMWARE_VIBRATION2_MIN, false);
        assert!(limiter.should_send(&stop, t0 + Duration::from_millis(13), false));
    }
}
