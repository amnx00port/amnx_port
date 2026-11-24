//! Host input arbitration: combines DualSense HID controllers (wired USB,
//! Pico 2 W `DS5Dongle`, and Bluetooth `0x31`) with XInput controllers across
//! host slots `0..4`, deduplicating Steam Input (`0x28DE`) / ViGEm virtual
//! XInput mirrors while preserving genuine Xbox + PlayStation multi-pad play.

use victory_core::config::{InputConfig, VirtualXInputPolicy};

use super::dualsense::{DualSenseManager, MAX_DUALSENSE, SlotSnapshot};
use super::{BatteryState, KeyState, PADS, PadReading, PadState, Source};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSlot {
    DualSense(usize),
    XInput(usize),
}

/// Decides whether XInput pads should be suppressed to prevent double-input
/// when a DualSense HID controller is active alongside a virtual XInput wrapper.
pub fn should_suppress_xinput(
    policy: VirtualXInputPolicy,
    any_dualsense_active: bool,
    virtual_wrapper_detected: bool,
    buttons_mirror_dualsense: bool,
) -> bool {
    if !any_dualsense_active {
        return false;
    }
    match policy {
        VirtualXInputPolicy::Always => true,
        VirtualXInputPolicy::Never => false,
        VirtualXInputPolicy::Auto => virtual_wrapper_detected || buttons_mirror_dualsense,
    }
}

/// Assigns DualSense and XInput devices to persistent host slots `0..PADS` so
/// hot-plugging a second controller never swaps Player 1's slot index.
#[derive(Default)]
pub struct SlotArbitrator {
    owners: [Option<BackendSlot>; PADS],
    mirror_hits: [u8; PADS],
}

impl SlotArbitrator {
    pub fn owner(&self, host_slot: usize) -> Option<BackendSlot> {
        self.owners.get(host_slot).copied().flatten()
    }

    pub fn resolve(
        &mut self,
        ds: &[Option<SlotSnapshot>; MAX_DUALSENSE],
        xinput: &[Option<PadState>; PADS],
        policy: VirtualXInputPolicy,
        virtual_wrapper_detected: bool,
    ) -> [Option<PadReading>; PADS] {
        let any_ds = ds.iter().any(Option::is_some);

        // Heuristic mirror check: if an XInput slot presses the exact same non-zero
        // buttons as an active DualSense over multiple consecutive polls, it is
        // a software wrapper (DS4Windows / DSX / Steam) without HidHide.
        for (xi, xpad) in xinput.iter().enumerate() {
            if let Some(xp) = xpad
                && xp.buttons != 0
                && ds.iter().flatten().any(|d| d.pad.buttons == xp.buttons)
            {
                self.mirror_hits[xi] = self.mirror_hits[xi].saturating_add(1);
            } else if xpad.is_none() {
                self.mirror_hits[xi] = 0;
            }
        }

        // Release slots whose backing device disconnected or became suppressed.
        for slot in &mut self.owners {
            let keep = match *slot {
                None => false,
                Some(BackendSlot::DualSense(i)) => ds.get(i).and_then(|x| *x).is_some(),
                Some(BackendSlot::XInput(i)) => {
                    let present = xinput.get(i).and_then(|x| *x).is_some();
                    let mirrored = self.mirror_hits[i] >= 2;
                    present
                        && !should_suppress_xinput(
                            policy,
                            any_ds,
                            virtual_wrapper_detected,
                            mirrored,
                        )
                }
            };
            if !keep {
                *slot = None;
            }
        }

        // Bind newly connected DualSense pads first.
        for (i, snap) in ds.iter().enumerate() {
            if snap.is_some()
                && !self.owners.contains(&Some(BackendSlot::DualSense(i)))
                && let Some(free) = self.owners.iter_mut().find(|o| o.is_none())
            {
                *free = Some(BackendSlot::DualSense(i));
            }
        }

        // Bind genuine (non-suppressed) XInput pads to remaining free slots.
        for (i, pad) in xinput.iter().enumerate() {
            let mirrored = self.mirror_hits[i] >= 2;
            if pad.is_some()
                && !should_suppress_xinput(policy, any_ds, virtual_wrapper_detected, mirrored)
                && !self.owners.contains(&Some(BackendSlot::XInput(i)))
                && let Some(free) = self.owners.iter_mut().find(|o| o.is_none())
            {
                *free = Some(BackendSlot::XInput(i));
            }
        }

        std::array::from_fn(|slot| match self.owners[slot]? {
            BackendSlot::DualSense(i) => {
                let s = ds[i]?;
                Some(PadReading {
                    pad: s.pad,
                    battery: s.battery,
                    system_buttons: s.system_buttons,
                })
            }
            BackendSlot::XInput(i) => Some(PadReading {
                pad: xinput[i]?,
                battery: BatteryState::unknown(),
                system_buttons: 0,
            }),
        })
    }
}

#[cfg(windows)]
pub struct Host {
    config: InputConfig,
    dualsense: DualSenseManager,
    arbitrator: SlotArbitrator,
    empty_until: [Option<std::time::Instant>; PADS],
    resolved: [Option<PadReading>; PADS],
    was_focused: bool,
}

#[cfg(windows)]
impl Default for Host {
    fn default() -> Self {
        let cfg = InputConfig::default();
        Self {
            dualsense: DualSenseManager::new(cfg.dualsense.clone()),
            config: cfg,
            arbitrator: SlotArbitrator::default(),
            empty_until: [None; PADS],
            resolved: [None; PADS],
            was_focused: true,
        }
    }
}

#[cfg(windows)]
impl Host {
    fn refresh(&mut self) {
        let focused = imp::focused();
        let active_focus = !self.config.mute_unfocused || focused;
        if self.was_focused && !active_focus {
            for xi in 0..PADS {
                imp::xinput_rumble(xi, 0, 0);
            }
        }
        self.was_focused = active_focus;

        let ds = self.dualsense.poll(active_focus);
        let xinput: [Option<PadState>; PADS] = std::array::from_fn(|xi| {
            let mut pad = imp::poll_xinput(xi, &mut self.empty_until[xi])?;
            if !active_focus {
                pad = PadState::default();
            }
            Some(pad)
        });
        self.resolved = self.arbitrator.resolve(
            &ds,
            &xinput,
            self.config.suppress_virtual_xinput,
            super::dualsense::virtual_xinput_detected(),
        );
    }
}

#[cfg(windows)]
impl Source for Host {
    fn begin_poll(&mut self) {
        self.refresh();
    }

    fn pad(&mut self, slot: usize) -> Option<PadReading> {
        self.resolved.get(slot).copied().flatten()
    }

    fn keys(&mut self) -> Vec<KeyState> {
        imp::poll_keys()
    }

    fn rumble(&mut self, slot: usize, motors: [f32; 4]) {
        let active_focus = !self.config.mute_unfocused || imp::focused();
        let motors = if active_focus { motors } else { [0.0; 4] };
        match self.arbitrator.owner(slot) {
            Some(BackendSlot::DualSense(i)) => self.dualsense.set_rumble(i, motors),
            Some(BackendSlot::XInput(i)) => {
                let speed = |v: f32| (v.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
                imp::xinput_rumble(i, speed(motors[0]), speed(motors[1]));
            }
            None => {}
        }
    }

    fn set_config(&mut self, config: InputConfig) {
        self.dualsense.set_config(config.dualsense.clone());
        self.config = config;
    }
}

#[cfg(windows)]
mod imp {
    #[cfg(feature = "diagnostics")]
    use std::sync::LazyLock;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId,
    };

    use crate::modules::gameinput::{
        A, B, DOWN, KeyState, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT,
        RIGHT_SHOULDER, RIGHT_STICK, UP, VIEW, X, Y,
    };

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    pub(super) fn poll_xinput(slot: usize, empty_until: &mut Option<Instant>) -> Option<PadState> {
        if empty_until.is_some_and(|until| Instant::now() < until) {
            return None;
        }
        let mut state = XINPUT_STATE::default();
        // SAFETY: room for the state.
        if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
            *empty_until = Some(Instant::now() + EMPTY_SLOT_RECHECK);
            return None;
        }
        *empty_until = None;
        Some(convert(&state.Gamepad))
    }

    pub(super) fn xinput_rumble(slot: usize, low: u16, high: u16) {
        let speeds = XINPUT_VIBRATION {
            wLeftMotorSpeed: low,
            wRightMotorSpeed: high,
        };
        // SAFETY: a valid vibration; an empty slot just fails.
        unsafe {
            let _ = XInputSetState(slot as u32, &speeds);
        }
    }

    pub(super) fn poll_keys() -> Vec<KeyState> {
        #[cfg(feature = "diagnostics")]
        let mut down = scripted();
        #[cfg(not(feature = "diagnostics"))]
        let mut down = Vec::new();
        if focused() {
            // Left/right modifier keys only, as devices report them; no
            // mouse buttons (below 0x08).
            let held: Vec<u8> = (0x08..=0xFE_u8)
                .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                // SAFETY: plain queries.
                .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                .collect();
            down.extend(held);
        }
        down.into_iter()
            .map(|vk| KeyState {
                // SAFETY: plain query.
                scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                virtual_key: vk,
                // shortcut: no characters (code point 0); Grand Theft
                // Auto VI reads scan codes and virtual keys only.
                ..Default::default()
            })
            .collect()
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts, with or without focus.
    #[cfg(feature = "diagnostics")]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                Some((
                    Duration::try_from_secs_f64(at.trim().parse().ok()?).ok()?,
                    vk,
                ))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(at, _)| (at..at + Duration::from_millis(200)).contains(&now))
            .map(|&(_, vk)| vk)
            .collect()
    }

    /// The foreground window is one of ours (the title's or our fallback).
    pub(super) fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::gameinput::{A, B};

    #[test]
    fn suppresses_steam_virtual_xinput_while_allowing_mixed_real_pads() {
        let mut arb = SlotArbitrator::default();
        let ds_snap = SlotSnapshot {
            connected: true,
            transport: Some(super::super::dualsense::protocol::Transport::Usb),
            pad: PadState {
                buttons: A,
                ..Default::default()
            },
            battery: BatteryState::default(),
            system_buttons: 0,
        };
        let mut ds = [None; MAX_DUALSENSE];
        ds[0] = Some(ds_snap);

        // Steam Input (`VID_28DE` detected) exposes a duplicate virtual XInput pad on slot 0.
        let mut xi = [None; PADS];
        xi[0] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        let resolved = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, true);
        assert!(resolved[0].is_some());
        assert!(
            resolved[1].is_none(),
            "Steam virtual XInput mirror must be suppressed"
        );

        // With no virtual wrapper (`false`), a real Xbox pad on XInput 0 coexists on host slot 1!
        xi[0] = Some(PadState {
            buttons: B,
            ..Default::default()
        });
        let mixed = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, false);
        assert_eq!(mixed[0].unwrap().pad.buttons, A);
        assert_eq!(mixed[1].unwrap().pad.buttons, B);
    }
}
