//! DualSense / DualSense Edge device discovery, overlapped Win32 HID I/O,
//! `DS5Dongle` (Pico 2 W) 1.5s liveness monitoring, PS button chord filter,
//! and lock-free emergency hardware reset on focus loss, exit, or crash.

pub mod audio_haptics;
pub mod gyro;
pub mod protocol;
pub mod triggers;

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use victory_core::config::{DualSenseConfig, HapticsMode};

use super::{BatteryState, PadState, SYSTEM_BUTTON_GUIDE};
use audio_haptics::AudioHapticsEngine;
use gyro::{GyroTracker, ImuCalibration};
use protocol::{ControllerModel, FIRMWARE_VIBRATION2_MIN, OutputCommand, RawReport, Transport};
use triggers::{ReportLimiter, encode_trigger_effect};

/// Maximum DualSense slots managed concurrently (`0..4`).
pub const MAX_DUALSENSE: usize = 4;

/// If `sensor_timestamp` stalls or packets cease for longer than 1.5 s (e.g.
/// `DS5Dongle` plugged into USB while the wireless PS5 controller is turned off
/// or out of range), the slot transitions back to disconnected so keyboard
/// fallback (`WASD`) resumes and stuck buttons are cleared.
pub const DONGLE_FREEZE_TIMEOUT: Duration = Duration::from_millis(1500);

/// Short-press ceiling for PS Guide button events; holds `>= 600 ms` (such as
/// 2-5 s power-off or multi-device Bluetooth slot switching) are never intercepted.
const PS_SHORT_PRESS_MAX: Duration = Duration::from_millis(600);

/// Detected presence of a virtual XInput wrapper (e.g. Steam Input `VID_28DE`
/// or ViGEm virtual bus) during HID interface enumeration.
static VIRTUAL_XINPUT_SEEN: AtomicBool = AtomicBool::new(false);

/// Lock-free emergency handle table used to release adaptive triggers and
/// rumble from crash/panic/Ctrl+C handlers without acquiring any mutex.
static EMERGENCY_HANDLES: [AtomicUsize; MAX_DUALSENSE] =
    [const { AtomicUsize::new(0) }; MAX_DUALSENSE];
static EMERGENCY_TRANSPORTS: [AtomicU8; MAX_DUALSENSE] =
    [const { AtomicU8::new(0) }; MAX_DUALSENSE];
static EMERGENCY_OUT_LENS: [AtomicU16; MAX_DUALSENSE] =
    [const { AtomicU16::new(48) }; MAX_DUALSENSE];

pub fn virtual_xinput_detected() -> bool {
    VIRTUAL_XINPUT_SEEN.load(Ordering::Acquire)
}

/// Lock-free, allocation-free hardware reset safe to invoke inside unhandled
/// exception filters, panic hooks, `SetConsoleCtrlHandler`, and `end_title`.
pub fn emergency_reset_all() {
    #[cfg(windows)]
    imp::emergency_reset_all();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LivenessEvent {
    Unchanged,
    Connected,
    Disconnected,
}

/// Tracks whether a USB/Dongle/BT DualSense handle actually has a live
/// controller behind it (`sensor_timestamp` advancing within 1.5 s).
#[derive(Default)]
pub struct LivenessMonitor {
    connected: bool,
    last_sensor_ts: Option<u32>,
    last_advance_at: Option<Instant>,
}

impl LivenessMonitor {
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn on_report(&mut self, raw: &RawReport, now: Instant) -> LivenessEvent {
        if !raw.has_live_telemetry() {
            return self.check_timeout(now);
        }
        let advanced = match self.last_sensor_ts {
            None => true,
            Some(prev) => prev != raw.sensor_timestamp,
        };
        if advanced {
            self.last_sensor_ts = Some(raw.sensor_timestamp);
            self.last_advance_at = Some(now);
            if !self.connected {
                self.connected = true;
                return LivenessEvent::Connected;
            }
            LivenessEvent::Unchanged
        } else {
            self.check_timeout(now)
        }
    }

    pub fn check_timeout(&mut self, now: Instant) -> LivenessEvent {
        if self.connected
            && self
                .last_advance_at
                .is_some_and(|at| now.saturating_duration_since(at) >= DONGLE_FREEZE_TIMEOUT)
        {
            self.connected = false;
            self.last_sensor_ts = None;
            self.last_advance_at = None;
            return LivenessEvent::Disconnected;
        }
        LivenessEvent::Unchanged
    }

    pub fn force_disconnect(&mut self) {
        self.connected = false;
        self.last_sensor_ts = None;
        self.last_advance_at = None;
    }
}

/// Tracks PS button short-press vs chord/long-hold and Mic-Mute button toggles.
#[derive(Default)]
pub struct SystemButtonFilter {
    ps_down_since: Option<Instant>,
    ps_chorded: bool,
    recenter_latched: bool,
    guide_pulse_until: Option<Instant>,
    prev_mic_down: bool,
    pub mic_muted: bool,
}

pub struct FilterOutcome {
    pub recenter_gyro: bool,
    pub mic_led_changed: bool,
    pub system_buttons: u32,
}

impl SystemButtonFilter {
    pub fn reset(&mut self) {
        self.ps_down_since = None;
        self.ps_chorded = false;
        self.recenter_latched = false;
        self.guide_pulse_until = None;
        self.prev_mic_down = false;
    }

    pub fn update(&mut self, raw: &RawReport, other_buttons: u32, now: Instant) -> FilterOutcome {
        let mut recenter_gyro = false;
        let mut mic_led_changed = false;

        let mic_down = raw.mic_mute_pressed();
        if mic_down && !self.prev_mic_down {
            self.mic_muted = !self.mic_muted;
            mic_led_changed = true;
        }
        self.prev_mic_down = mic_down;

        if raw.ps_pressed() {
            let since = *self.ps_down_since.get_or_insert(now);
            if raw.touchpad_clicked() && !self.recenter_latched {
                self.recenter_latched = true;
                self.ps_chorded = true;
                recenter_gyro = true;
            }
            if other_buttons != 0
                || raw.any_face_button_pressed()
                || now.saturating_duration_since(since) >= PS_SHORT_PRESS_MAX
            {
                self.ps_chorded = true;
            }
        } else {
            if let Some(since) = self.ps_down_since.take()
                && !self.ps_chorded
                && now.saturating_duration_since(since) < PS_SHORT_PRESS_MAX
            {
                // Hold the virtual Guide bit for 80 ms so the poll loop delivers press + release.
                self.guide_pulse_until = Some(now + Duration::from_millis(80));
            }
            self.ps_chorded = false;
            self.recenter_latched = false;
        }

        let system_buttons = if self.guide_pulse_until.is_some_and(|until| now < until) {
            SYSTEM_BUTTON_GUIDE
        } else {
            self.guide_pulse_until = None;
            0
        };

        FilterOutcome {
            recenter_gyro,
            mic_led_changed,
            system_buttons,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SlotSnapshot {
    pub connected: bool,
    pub transport: Option<Transport>,
    pub pad: PadState,
    pub battery: BatteryState,
    pub system_buttons: u32,
}

pub(crate) struct SharedSlot {
    pub path: String,
    pub stop: AtomicBool,
    pub state: Mutex<WorkerState>,
}

pub(crate) struct WorkerState {
    pub snapshot: SlotSnapshot,
    pub raw_latest: Option<RawReport>,
    pub gyro: GyroTracker,
    pub firmware_version: u16,
    pub rumble: [f32; 4],
    pub focused: bool,
    pub leds_dirty: bool,
    pub mic_led_dirty: bool,
    pub mic_muted: bool,
    pub host_slot: usize,
    pub config: DualSenseConfig,
}

impl WorkerState {
    fn new(host_slot: usize, config: DualSenseConfig) -> Self {
        Self {
            snapshot: SlotSnapshot::default(),
            raw_latest: None,
            gyro: GyroTracker::default(),
            firmware_version: FIRMWARE_VIBRATION2_MIN,
            rumble: [0.0; 4],
            focused: true,
            leds_dirty: true,
            mic_led_dirty: false,
            mic_muted: false,
            host_slot,
            config,
        }
    }
}

pub struct DualSenseManager {
    slots: [Option<Arc<SharedSlot>>; MAX_DUALSENSE],
    config: DualSenseConfig,
    audio_haptics: AudioHapticsEngine,
    #[cfg(windows)]
    last_scan: Option<Instant>,
}

impl DualSenseManager {
    pub fn new(config: DualSenseConfig) -> Self {
        Self {
            slots: [const { None }; MAX_DUALSENSE],
            config,
            audio_haptics: AudioHapticsEngine::default(),
            #[cfg(windows)]
            last_scan: None,
        }
    }

    pub fn set_config(&mut self, config: DualSenseConfig) {
        self.config = config.clone();
        for slot in self.slots.iter().flatten() {
            if let Ok(mut st) = slot.state.lock() {
                st.config = config.clone();
                st.leds_dirty = true;
            }
        }
    }

    /// Scans for hot-plugged controllers (throttled internally) and reads the
    /// latest state of all active DualSense slots.
    pub fn poll(&mut self, focused: bool) -> [Option<SlotSnapshot>; MAX_DUALSENSE] {
        if !self.config.enabled {
            self.audio_haptics.stop();
            return [None; MAX_DUALSENSE];
        }

        #[cfg(windows)]
        self.scan_if_due();

        let mut out = [None; MAX_DUALSENSE];
        let mut any_usb_connected = false;
        let mut primary_rumble = [0.0f32; 4];
        let mut primary_muted = false;

        for (idx, entry) in self.slots.iter_mut().enumerate() {
            let remove = entry
                .as_ref()
                .is_some_and(|s| s.stop.load(Ordering::Acquire));
            if remove {
                *entry = None;
                continue;
            }
            let Some(slot) = entry.as_ref() else {
                continue;
            };
            let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if st.focused != focused {
                st.focused = focused;
                if focused {
                    st.leds_dirty = true;
                }
            }
            if !st.snapshot.connected {
                continue;
            }
            if st.snapshot.transport == Some(Transport::Usb) {
                any_usb_connected = true;
                primary_rumble = st.rumble;
                primary_muted = st.mic_muted;
            }

            let mut snap = st.snapshot;
            if let Some(raw) = st.raw_latest {
                let cfg = st.config.clone();
                st.gyro.apply_to_right_stick(
                    &raw,
                    &cfg,
                    &mut snap.pad.right_x,
                    &mut snap.pad.right_y,
                );
            }
            if !focused {
                snap.pad = PadState::default();
                snap.system_buttons = 0;
            }
            out[idx] = Some(snap);
        }

        let want_audio_haptics = any_usb_connected
            && focused
            && matches!(
                self.config.haptics_mode,
                HapticsMode::Audio4Ch | HapticsMode::Hybrid
            );
        if want_audio_haptics {
            self.audio_haptics
                .control()
                .set_options(self.config.speaker_mirror, primary_muted);
            self.audio_haptics.control().set_levels(
                primary_rumble[0],
                primary_rumble[1],
                primary_rumble[2],
                primary_rumble[3],
            );
            self.audio_haptics.ensure_started();
        } else {
            self.audio_haptics.control().clear_levels();
            if !any_usb_connected
                || !matches!(
                    self.config.haptics_mode,
                    HapticsMode::Audio4Ch | HapticsMode::Hybrid
                )
            {
                self.audio_haptics.stop();
            }
        }

        out
    }

    pub fn set_rumble(&mut self, index: usize, motors: [f32; 4]) {
        if index >= MAX_DUALSENSE {
            return;
        }
        if let Some(slot) = &self.slots[index] {
            let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            st.rumble = motors;
        }
    }

    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| {
                s.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .snapshot
                    .connected
            })
            .count()
    }

    #[cfg(windows)]
    fn scan_if_due(&mut self) {
        const RESCAN: Duration = Duration::from_millis(1000);
        if self.last_scan.is_some_and(|t| t.elapsed() < RESCAN) {
            return;
        }
        self.last_scan = Some(Instant::now());
        imp::scan_and_spawn(&mut self.slots, &self.config);
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use victory_core::config::{DualSenseConfig, HapticsMode};
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GET_DEVICE_INTERFACE_LIST_PRESENT, CM_Get_Device_Interface_List_SizeW,
        CM_Get_Device_Interface_ListW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HIDP_CAPS, HIDP_STATUS_SUCCESS, HidD_FreePreparsedData,
        HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid, HidD_GetPreparsedData,
        HidD_SetOutputReport, HidP_GetCaps, PHIDP_PREPARSED_DATA,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_IO_INCOMPLETE, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE,
        GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
        ReadFile, WriteFile,
    };
    use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
    use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};

    use super::{
        ControllerModel, EMERGENCY_HANDLES, EMERGENCY_OUT_LENS, EMERGENCY_TRANSPORTS,
        ImuCalibration, LivenessEvent, LivenessMonitor, MAX_DUALSENSE, OutputCommand,
        ReportLimiter, SharedSlot, SlotSnapshot, SystemButtonFilter, Transport,
        VIRTUAL_XINPUT_SEEN, WorkerState, encode_trigger_effect,
        protocol::{
            BT_REPORT_LEN, FEATURE_CALIBRATION_ID, FEATURE_CALIBRATION_LEN, FEATURE_FIRMWARE_ID,
            FEATURE_FIRMWARE_LEN, FIRMWARE_VIBRATION2_MIN, SONY_VID, USAGE_GAMEPAD,
            USAGE_PAGE_GENERIC_DESKTOP, parse_firmware_version, parse_input_report, to_pad_state,
        },
    };

    struct Discovered {
        path: String,
        path_wide: Vec<u16>,
        handle: HANDLE,
        model: ControllerModel,
        transport: Transport,
        in_len: usize,
        out_len: usize,
        feat_len: usize,
    }

    // SAFETY: Win32 file/device handles are process-wide kernel handles safe to move to worker threads.
    unsafe impl Send for Discovered {}

    pub(super) fn emergency_reset_all() {
        for slot in 0..MAX_DUALSENSE {
            let raw = EMERGENCY_HANDLES[slot].load(Ordering::Acquire);
            if raw == 0 {
                continue;
            }
            let handle = raw as HANDLE;
            let transport = EMERGENCY_TRANSPORTS[slot].load(Ordering::Acquire);
            let out_len = usize::from(EMERGENCY_OUT_LENS[slot].load(Ordering::Acquire));
            let cmd = OutputCommand::neutral(FIRMWARE_VIBRATION2_MIN, true);
            if transport == 1 {
                let bt = cmd.encode_bt();
                // SAFETY: valid open HID handle; stack buffer lives across the synchronous control transfer.
                unsafe {
                    let _ = HidD_SetOutputReport(handle, bt.as_ptr().cast(), bt.len() as u32);
                }
            } else {
                let mut usb = [0u8; 64];
                let len = out_len.clamp(48, 64);
                usb[0] = super::protocol::USB_OUTPUT_REPORT_ID;
                let common = cmd.encode_common();
                usb[1..48].copy_from_slice(&common);
                // SAFETY: valid open HID handle; stack buffer lives across the synchronous control transfer.
                unsafe {
                    let _ = HidD_SetOutputReport(handle, usb.as_ptr().cast(), len as u32);
                }
            }
        }
    }

    pub(super) fn scan_and_spawn(
        slots: &mut [Option<Arc<SharedSlot>>; MAX_DUALSENSE],
        config: &DualSenseConfig,
    ) {
        let paths = enumerate_hid_interfaces();
        let mut virtual_xinput = false;
        for (_, lower) in &paths {
            // Valve Steam Input virtual Xbox pad (`VID_28DE`) or ViGEm virtual bus.
            if (lower.contains("ig_") && lower.contains("vid_28de"))
                || lower.contains("vigem")
                || (lower.contains("ig_") && lower.contains("root#"))
            {
                virtual_xinput = true;
            }
        }
        VIRTUAL_XINPUT_SEEN.store(virtual_xinput, Ordering::Release);

        for (wide, lower) in paths {
            if !lower.contains("054c") {
                continue;
            }
            if slots
                .iter()
                .flatten()
                .any(|s| !s.stop.load(Ordering::Acquire) && s.path == lower)
            {
                continue;
            }
            let Some(free_slot) = slots.iter().position(Option::is_none) else {
                break;
            };
            let Some(dev) = try_open_dualsense(lower, wide) else {
                continue;
            };
            let shared = Arc::new(SharedSlot {
                path: dev.path.clone(),
                stop: AtomicBool::new(false),
                state: std::sync::Mutex::new(WorkerState::new(free_slot, config.clone())),
            });
            slots[free_slot] = Some(Arc::clone(&shared));
            let worker_shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new()
                .name(format!("victory dualsense {free_slot}"))
                .spawn(move || run_device_worker(free_slot, dev, worker_shared));
        }
    }

    use std::sync::atomic::AtomicBool;

    fn enumerate_hid_interfaces() -> Vec<(Vec<u16>, String)> {
        let mut guid = windows_sys::core::GUID {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        // SAFETY: writes the system HID interface class GUID.
        unsafe { HidD_GetHidGuid(&mut guid) };

        let mut len = 0u32;
        // SAFETY: queries required UTF-16 multi-string length.
        let cr = unsafe {
            CM_Get_Device_Interface_List_SizeW(
                &mut len,
                &guid,
                null(),
                CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
            )
        };
        if cr != CR_SUCCESS || len <= 1 {
            return Vec::new();
        }

        let mut buf = vec![0u16; len as usize + 16];
        // SAFETY: `buf` holds room for the queried multi-string list.
        let cr = unsafe {
            CM_Get_Device_Interface_ListW(
                &guid,
                null(),
                buf.as_mut_ptr(),
                buf.len() as u32,
                CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
            )
        };
        if cr != CR_SUCCESS {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut start = 0usize;
        while start < buf.len() && buf[start] != 0 {
            let end = buf[start..]
                .iter()
                .position(|&c| c == 0)
                .map_or(buf.len(), |p| start + p);
            let mut wide = buf[start..end].to_vec();
            let lower = String::from_utf16_lossy(&wide).to_ascii_lowercase();
            wide.push(0);
            out.push((wide, lower));
            start = end + 1;
        }
        out
    }

    fn try_open_dualsense(path: String, path_wide: Vec<u16>) -> Option<Discovered> {
        // Always open with FILE_SHARE_READ | FILE_SHARE_WRITE so DS5Dongle's
        // browser WebHID configurator and companion tools are never locked out.
        // SAFETY: NUL-terminated UTF-16 device interface path.
        let handle = unsafe {
            CreateFileW(
                path_wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE || handle.is_null() {
            return None;
        }

        let mut attrs = HIDD_ATTRIBUTES {
            Size: size_of::<HIDD_ATTRIBUTES>() as u32,
            VendorID: 0,
            ProductID: 0,
            VersionNumber: 0,
        };
        // SAFETY: valid open HID handle and sized `HIDD_ATTRIBUTES`.
        let ok_attr = unsafe { HidD_GetAttributes(handle, &mut attrs) };
        if !ok_attr || attrs.VendorID != SONY_VID {
            // SAFETY: closing our newly opened handle.
            unsafe { CloseHandle(handle) };
            return None;
        }
        let Some(model) = ControllerModel::from_pid(attrs.ProductID) else {
            // SAFETY: closing non-DualSense Sony device handle.
            unsafe { CloseHandle(handle) };
            return None;
        };

        let mut preparsed: PHIDP_PREPARSED_DATA = 0;
        // SAFETY: queries preparsed HID descriptor for `handle`.
        if !unsafe { HidD_GetPreparsedData(handle, &mut preparsed) } || preparsed == 0 {
            // SAFETY: closing handle.
            unsafe { CloseHandle(handle) };
            return None;
        }
        // SAFETY: `HIDP_CAPS` is a plain C integer struct valid when zero-initialized.
        let mut caps: HIDP_CAPS = unsafe { std::mem::zeroed() };
        // SAFETY: valid preparsed descriptor from `HidD_GetPreparsedData`.
        let status = unsafe { HidP_GetCaps(preparsed, &mut caps) };
        // SAFETY: frees preparsed descriptor allocated by `HidD_GetPreparsedData`.
        unsafe {
            let _ = HidD_FreePreparsedData(preparsed);
        }

        // Filter strictly for Generic Desktop (0x01) / Game Pad (0x05) so we never
        // bind DS5Dongle's vendor/WebHID config interface or audio HID controls.
        if status != HIDP_STATUS_SUCCESS
            || caps.UsagePage != USAGE_PAGE_GENERIC_DESKTOP
            || caps.Usage != USAGE_GAMEPAD
        {
            // SAFETY: closing non-gamepad HID collection handle.
            unsafe { CloseHandle(handle) };
            return None;
        }

        let in_len = usize::from(caps.InputReportByteLength).max(64);
        let out_len = usize::from(caps.OutputReportByteLength).max(48);
        let feat_len = usize::from(caps.FeatureReportByteLength).max(64);
        let transport = if caps.InputReportByteLength >= BT_REPORT_LEN as u16 {
            Transport::Bluetooth
        } else {
            Transport::Usb
        };

        Some(Discovered {
            path,
            path_wide,
            handle,
            model,
            transport,
            in_len,
            out_len,
            feat_len,
        })
    }

    fn query_features(dev: &Discovered) -> (Option<ImuCalibration>, Option<u16>) {
        // Match exact descriptor FeatureReportByteLength where larger than report size
        // so TinyUSB endpoints on DS5Dongle never stall on short buffers.
        let mut cal_buf = vec![0u8; dev.feat_len.max(FEATURE_CALIBRATION_LEN)];
        cal_buf[0] = FEATURE_CALIBRATION_ID;
        // SAFETY: valid HID handle and buffer starting with report ID 0x05; on Bluetooth
        // reading 0x05 also switches the DualSense into enhanced 0x31 report mode.
        let got_cal = unsafe {
            HidD_GetFeature(
                dev.handle,
                cal_buf.as_mut_ptr().cast(),
                FEATURE_CALIBRATION_LEN.min(cal_buf.len()) as u32,
            )
        };
        let cal = got_cal
            .then(|| ImuCalibration::from_feature_report(&cal_buf, dev.transport))
            .flatten();

        let mut fw_buf = vec![0u8; dev.feat_len.max(FEATURE_FIRMWARE_LEN)];
        fw_buf[0] = FEATURE_FIRMWARE_ID;
        // SAFETY: valid HID handle and buffer starting with report ID 0x20.
        let got_fw = unsafe {
            HidD_GetFeature(
                dev.handle,
                fw_buf.as_mut_ptr().cast(),
                FEATURE_FIRMWARE_LEN.min(fw_buf.len()) as u32,
            )
        };
        let fw = got_fw
            .then(|| parse_firmware_version(&fw_buf, dev.transport))
            .flatten();

        (cal, fw)
    }

    fn send_output(dev: &Discovered, write_event: HANDLE, cmd: &OutputCommand) {
        let packet: Vec<u8> = match dev.transport {
            Transport::Usb => cmd.encode_usb(dev.out_len),
            Transport::Bluetooth => cmd.encode_bt().to_vec(),
        };
        // SAFETY: `OVERLAPPED` is a plain C struct valid when zero-initialized.
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = write_event;
        // SAFETY: valid manual-reset event and overlapped HID handle.
        unsafe {
            let _ = ResetEvent(write_event);
            let mut written = 0u32;
            let ok = WriteFile(
                dev.handle,
                packet.as_ptr(),
                packet.len() as u32,
                &mut written,
                &mut ov,
            );
            if ok == 0 && GetLastError() == ERROR_IO_PENDING {
                if WaitForSingleObject(write_event, 25) == WAIT_OBJECT_0 {
                    let _ = GetOverlappedResult(dev.handle, &ov, &mut written, 0);
                } else {
                    let _ = CancelIoEx(dev.handle, &ov);
                }
            } else if ok == 0 {
                let _ = HidD_SetOutputReport(
                    dev.handle,
                    packet.as_ptr().cast::<c_void>(),
                    packet.len() as u32,
                );
            }
        }
    }

    fn run_device_worker(slot_idx: usize, dev: Discovered, shared: Arc<SharedSlot>) {
        let _ = &dev.path_wide;
        EMERGENCY_TRANSPORTS[slot_idx].store(
            u8::from(dev.transport == Transport::Bluetooth),
            Ordering::Release,
        );
        EMERGENCY_OUT_LENS[slot_idx].store(dev.out_len as u16, Ordering::Release);
        EMERGENCY_HANDLES[slot_idx].store(dev.handle as usize, Ordering::Release);

        // SAFETY: manual-reset unnamed Win32 events for overlapped ReadFile / WriteFile.
        let read_event = unsafe { CreateEventW(null(), 1, 0, null()) };
        // SAFETY: as above.
        let write_event = unsafe { CreateEventW(null(), 1, 0, null()) };

        // Initial feature probe (also triggers Bluetooth 0x01 -> 0x31 enhanced mode switch).
        let (init_cal, init_fw) = query_features(&dev);
        {
            let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cal) = init_cal {
                st.gyro.set_calibration(cal);
            }
            if let Some(fw) = init_fw {
                st.firmware_version = fw;
            }
        }

        let mut liveness = LivenessMonitor::default();
        let mut sys_filter = SystemButtonFilter::default();
        let mut limiter = ReportLimiter::default();
        let mut read_buf = vec![0u8; dev.in_len.max(BT_REPORT_LEN)];
        // SAFETY: `OVERLAPPED` is a plain C struct valid when zero-initialized.
        let mut read_ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        read_ov.hEvent = read_event;
        let mut read_pending = false;

        loop {
            if shared.stop.load(Ordering::Acquire) {
                break;
            }

            if !read_pending {
                // SAFETY: resets manual-reset event and starts overlapped HID interrupt IN read.
                unsafe {
                    let _ = ResetEvent(read_event);
                    let mut got = 0u32;
                    let ok = ReadFile(
                        dev.handle,
                        read_buf.as_mut_ptr(),
                        read_buf.len() as u32,
                        &mut got,
                        &mut read_ov,
                    );
                    if ok != 0 {
                        handle_packet(
                            slot_idx,
                            &dev,
                            &shared,
                            &read_buf[..got as usize],
                            &mut liveness,
                            &mut sys_filter,
                            &mut limiter,
                        );
                        flush_pending_output(&dev, write_event, &shared, &liveness, &mut limiter);
                        continue;
                    }
                    if GetLastError() != ERROR_IO_PENDING {
                        break;
                    }
                    read_pending = true;
                }
            }

            // Wait up to 20 ms so output commands and DS5Dongle 1.5s freeze checks
            // run promptly even when a wireless controller turns off mid-session.
            // SAFETY: valid event handle.
            let wait = unsafe { WaitForSingleObject(read_event, 20) };
            if wait == WAIT_OBJECT_0 {
                let mut got = 0u32;
                // SAFETY: queries completed overlapped `ReadFile`.
                let ok = unsafe { GetOverlappedResult(dev.handle, &read_ov, &mut got, 0) };
                read_pending = false;
                if ok == 0 {
                    // SAFETY: plain GetLastError check.
                    if unsafe { GetLastError() } == ERROR_IO_INCOMPLETE {
                        read_pending = true;
                    } else {
                        break;
                    }
                } else {
                    handle_packet(
                        slot_idx,
                        &dev,
                        &shared,
                        &read_buf[..got as usize],
                        &mut liveness,
                        &mut sys_filter,
                        &mut limiter,
                    );
                }
            } else if liveness.check_timeout(Instant::now()) == LivenessEvent::Disconnected {
                tracing::info!(
                    slot = slot_idx,
                    "DualSense wireless link idle/disconnected (restoring fallback)"
                );
                sys_filter.reset();
                limiter.reset();
                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                st.snapshot = SlotSnapshot::default();
                st.raw_latest = None;
                st.gyro.reset_stream();
            }

            flush_pending_output(&dev, write_event, &shared, &liveness, &mut limiter);
        }

        // Cleanly cancel any in-flight overlapped read and release trigger tension before closing.
        // SAFETY: valid handles owned by this worker thread.
        unsafe {
            if read_pending {
                let _ = CancelIoEx(dev.handle, &read_ov);
                let mut dummy = 0u32;
                let _ = GetOverlappedResult(dev.handle, &read_ov, &mut dummy, 1);
            }
            let fw = shared
                .state
                .lock()
                .map_or(FIRMWARE_VIBRATION2_MIN, |s| s.firmware_version);
            send_output(&dev, write_event, &OutputCommand::neutral(fw, true));
            EMERGENCY_HANDLES[slot_idx].store(0, Ordering::Release);
            let _ = CloseHandle(read_event);
            let _ = CloseHandle(write_event);
            let _ = CloseHandle(dev.handle);
        }

        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.snapshot = SlotSnapshot::default();
        st.raw_latest = None;
        drop(st);
        shared.stop.store(true, Ordering::Release);
    }

    fn handle_packet(
        slot_idx: usize,
        dev: &Discovered,
        shared: &SharedSlot,
        packet: &[u8],
        liveness: &mut LivenessMonitor,
        sys_filter: &mut SystemButtonFilter,
        limiter: &mut ReportLimiter,
    ) {
        let Some(raw) = parse_input_report(packet, dev.transport) else {
            return;
        };
        let now = Instant::now();
        match liveness.on_report(&raw, now) {
            LivenessEvent::Connected => {
                // Re-query calibration and firmware version when DS5Dongle's Bluetooth
                // controller wakes up after the dongle was already plugged into USB.
                let (cal, fw) = query_features(dev);
                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(cal) = cal {
                    st.gyro.set_calibration(cal);
                }
                if let Some(fw) = fw {
                    st.firmware_version = fw;
                }
                st.leds_dirty = true;
                limiter.reset();
                tracing::info!(
                    slot = slot_idx,
                    model = ?dev.model,
                    transport = ?dev.transport,
                    firmware = format_args!("{:#06x}", st.firmware_version),
                    "DualSense active"
                );
            }
            LivenessEvent::Disconnected => {
                sys_filter.reset();
                limiter.reset();
                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                st.snapshot = SlotSnapshot::default();
                st.raw_latest = None;
                st.gyro.reset_stream();
                return;
            }
            LivenessEvent::Unchanged => {
                if !liveness.is_connected() {
                    return;
                }
            }
        }

        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let pad = to_pad_state(&raw, dev.model, &st.config);
        let filter = sys_filter.update(&raw, pad.buttons, now);
        let space = st.config.gyro_space;
        st.gyro.push_sample(&raw, space);
        if filter.recenter_gyro {
            st.gyro.recenter();
            st.leds_dirty = true;
            tracing::info!(slot = slot_idx, "DualSense gyroscope recentered");
        }
        if filter.mic_led_changed {
            st.mic_muted = sys_filter.mic_muted;
            st.mic_led_dirty = true;
        }
        st.raw_latest = Some(raw);
        st.snapshot = SlotSnapshot {
            connected: true,
            transport: Some(dev.transport),
            pad,
            battery: raw.battery_state(),
            system_buttons: filter.system_buttons,
        };
    }

    fn flush_pending_output(
        dev: &Discovered,
        write_event: HANDLE,
        shared: &SharedSlot,
        liveness: &LivenessMonitor,
        limiter: &mut ReportLimiter,
    ) {
        if !liveness.is_connected() {
            return;
        }
        let cmd = {
            let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if !st.focused {
                let neutral = OutputCommand::neutral(st.firmware_version, false);
                if !limiter.should_send(&neutral, Instant::now(), false) {
                    return;
                }
                neutral
            } else {
                let cfg = &st.config;
                let enable_hid_rumble = !matches!(
                    (dev.transport, cfg.haptics_mode),
                    (_, HapticsMode::Off) | (Transport::Usb, HapticsMode::Audio4Ch)
                );
                let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                let low_batt = st
                    .raw_latest
                    .is_some_and(|r| r.charging_state == 0 && r.battery_level <= 1);
                let rgb = if low_batt {
                    [0xFF, 0x10, 0x00]
                } else {
                    cfg.lightbar_color
                };
                let cmd = OutputCommand {
                    firmware_version: st.firmware_version,
                    enable_hid_rumble,
                    motor_left: if enable_hid_rumble {
                        to_u8(st.rumble[0])
                    } else {
                        0
                    },
                    motor_right: if enable_hid_rumble {
                        to_u8(st.rumble[1])
                    } else {
                        0
                    },
                    left_trigger: encode_trigger_effect(
                        st.rumble[2],
                        cfg.adaptive_triggers,
                        cfg.trigger_strength,
                    ),
                    right_trigger: encode_trigger_effect(
                        st.rumble[3],
                        cfg.adaptive_triggers,
                        cfg.trigger_strength,
                    ),
                    update_mic_led: st.mic_led_dirty,
                    mic_muted_led: st.mic_muted,
                    update_leds: st.leds_dirty,
                    release_leds: false,
                    slot: st.host_slot,
                    lightbar_rgb: rgb,
                    brightness: cfg.led_brightness,
                };
                let force = st.leds_dirty || st.mic_led_dirty;
                if !limiter.should_send(&cmd, Instant::now(), force) {
                    return;
                }
                st.leds_dirty = false;
                st.mic_led_dirty = false;
                cmd
            }
        };
        send_output(dev, write_event, &cmd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dongle_liveness_ignores_standby_and_disconnects_on_frozen_timestamp() {
        let mut mon = LivenessMonitor::default();
        let t0 = Instant::now();

        // Unpaired/standby DS5Dongle frame with zero timestamp and zero battery stays disconnected.
        let idle = RawReport::default();
        assert_eq!(mon.on_report(&idle, t0), LivenessEvent::Unchanged);
        assert!(!mon.is_connected());

        // Advancing timestamp marks the controller connected.
        let mut live = RawReport {
            sensor_timestamp: 10_000,
            battery_level: 7,
            charging_state: 0,
            ..Default::default()
        };
        assert_eq!(mon.on_report(&live, t0), LivenessEvent::Connected);
        assert!(mon.is_connected());

        // If the controller turns off and DS5Dongle repeats a frozen packet (or stalls)
        // for >= 1.5s, liveness drops back to disconnected.
        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(800)),
            LivenessEvent::Unchanged
        );
        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(1550)),
            LivenessEvent::Disconnected
        );
        assert!(!mon.is_connected());

        // Waking the controller back up triggers Connected again.
        live.sensor_timestamp = 25_000;
        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(2000)),
            LivenessEvent::Connected
        );
    }

    #[test]
    fn ps_button_short_press_fires_guide_while_long_hold_and_chords_do_not() {
        let mut f = SystemButtonFilter::default();
        let t0 = Instant::now();
        let mut raw = RawReport::default();

        // Short PS press (< 600 ms) emits Guide pulse on release.
        raw.buttons[2] = 0x01;
        assert_eq!(f.update(&raw, 0, t0).system_buttons, 0);
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, t0 + Duration::from_millis(150))
                .system_buttons,
            SYSTEM_BUTTON_GUIDE
        );
        assert_eq!(
            f.update(&raw, 0, t0 + Duration::from_millis(300))
                .system_buttons,
            0
        );

        // Long hold (>= 2 s for power-off / pairing) must NOT emit Guide.
        raw.buttons[2] = 0x01;
        let _ = f.update(&raw, 0, t0 + Duration::from_secs(1));
        let _ = f.update(&raw, 0, t0 + Duration::from_secs(3));
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, t0 + Duration::from_millis(3100))
                .system_buttons,
            0
        );

        // PS + Touchpad click triggers gyro recenter and suppresses Guide.
        raw.buttons[2] = 0x01 | 0x02;
        let out = f.update(&raw, 0, t0 + Duration::from_secs(4));
        assert!(out.recenter_gyro);
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, t0 + Duration::from_millis(4100))
                .system_buttons,
            0
        );
    }
}
