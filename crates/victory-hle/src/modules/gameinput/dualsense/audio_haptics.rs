//! Optional 4-channel WASAPI DualSense voice-coil haptics synthesizer
//! (`haptics_mode = "audio_4ch" | "hybrid"`).
//!
//! Wired DualSense and Pico 2 W `DS5Dongle` expose a 4-channel 48 kHz audio
//! endpoint (`FL, FR, RL, RR`): channels 1-2 feed the controller speaker /
//! headset jack (kept silent unless `speaker_mirror` is enabled) and channels
//! 3-4 drive the left and right voice-coil actuators. Game rumble scalars are
//! modulated onto dual resonant carriers (`58 Hz` heavy + `162 Hz` crisp) and
//! blended with low-passed audio captured exclusively from our own process
//! tree (`VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`), never system-wide audio.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Shared lock-free control block updated by `gameinput` and read by the
/// WASAPI render thread.
pub struct AudioHapticsControl {
    running: AtomicBool,
    speaker_mirror: AtomicBool,
    muted: AtomicBool,
    low: AtomicU32,
    high: AtomicU32,
    left_trigger: AtomicU32,
    right_trigger: AtomicU32,
}

impl Default for AudioHapticsControl {
    fn default() -> Self {
        Self {
            running: AtomicBool::new(false),
            speaker_mirror: AtomicBool::new(false),
            muted: AtomicBool::new(false),
            low: AtomicU32::new(0.0f32.to_bits()),
            high: AtomicU32::new(0.0f32.to_bits()),
            left_trigger: AtomicU32::new(0.0f32.to_bits()),
            right_trigger: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

impl AudioHapticsControl {
    pub fn set_levels(&self, low: f32, high: f32, lt: f32, rt: f32) {
        self.low
            .store(low.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.high
            .store(high.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.left_trigger
            .store(lt.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.right_trigger
            .store(rt.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn clear_levels(&self) {
        self.set_levels(0.0, 0.0, 0.0, 0.0);
    }

    pub fn set_options(&self, speaker_mirror: bool, muted: bool) {
        self.speaker_mirror.store(speaker_mirror, Ordering::Relaxed);
        self.muted.store(muted, Ordering::Relaxed);
    }

    fn levels(&self) -> [f32; 4] {
        [
            f32::from_bits(self.low.load(Ordering::Relaxed)),
            f32::from_bits(self.high.load(Ordering::Relaxed)),
            f32::from_bits(self.left_trigger.load(Ordering::Relaxed)),
            f32::from_bits(self.right_trigger.load(Ordering::Relaxed)),
        ]
    }
}

/// Pure DSP synthesizer (unit-testable without audio hardware).
pub struct TactileSynth {
    phase_low: f32,
    phase_high: f32,
    lp_l1: f32,
    lp_l2: f32,
    lp_r1: f32,
    lp_r2: f32,
}

impl Default for TactileSynth {
    fn default() -> Self {
        Self {
            phase_low: 0.0,
            phase_high: 0.0,
            lp_l1: 0.0,
            lp_l2: 0.0,
            lp_r1: 0.0,
            lp_r2: 0.0,
        }
    }
}

impl TactileSynth {
    /// Generates one 4-channel frame `[FL, FR, HapticL, HapticR]` at `rate` Hz.
    pub fn next_frame(
        &mut self,
        rate: f32,
        levels: [f32; 4],
        game_stereo: [f32; 2],
        speaker_mirror: bool,
        muted: bool,
    ) -> [f32; 4] {
        if muted {
            return [0.0; 4];
        }
        let [low, high, lt, rt] = levels;
        let tau = std::f32::consts::TAU;
        self.phase_low = (self.phase_low + 58.0 / rate).fract();
        self.phase_high = (self.phase_high + 162.0 / rate).fract();

        let carrier_low = (self.phase_low * tau).sin();
        let carrier_high = (self.phase_high * tau).sin();

        // Cascaded 1-pole low-pass (~135 Hz @ 48 kHz) extracting tactile sub-bass
        // from the title's own process-loopback stereo stream.
        let alpha = (tau * 135.0 / rate).clamp(0.005, 0.25);
        self.lp_l1 += alpha * (game_stereo[0] - self.lp_l1);
        self.lp_l2 += alpha * (self.lp_l1 - self.lp_l2);
        self.lp_r1 += alpha * (game_stereo[1] - self.lp_r1);
        self.lp_r2 += alpha * (self.lp_r1 - self.lp_r2);

        let rumble_l = (low * 0.75 + lt * 0.35) * carrier_low + (high * 0.30) * carrier_high;
        let rumble_r = (low * 0.45) * carrier_low + (high * 0.75 + rt * 0.35) * carrier_high;

        let soft_clip = |x: f32| (x / (1.0 + 0.5 * x.abs())).clamp(-0.92, 0.92);
        let haptic_l = soft_clip(rumble_l + self.lp_l2 * 0.55);
        let haptic_r = soft_clip(rumble_r + self.lp_r2 * 0.55);

        let (fl, fr) = if speaker_mirror {
            (
                (game_stereo[0] * 0.6).clamp(-0.9, 0.9),
                (game_stereo[1] * 0.6).clamp(-0.9, 0.9),
            )
        } else {
            (0.0, 0.0)
        };
        [fl, fr, haptic_l, haptic_r]
    }
}

pub struct AudioHapticsEngine {
    control: Arc<AudioHapticsControl>,
}

impl Default for AudioHapticsEngine {
    fn default() -> Self {
        Self {
            control: Arc::new(AudioHapticsControl::default()),
        }
    }
}

impl AudioHapticsEngine {
    pub fn control(&self) -> &Arc<AudioHapticsControl> {
        &self.control
    }

    pub fn ensure_started(&self) {
        #[cfg(windows)]
        imp::ensure_started(&self.control);
    }

    pub fn stop(&self) {
        self.control.clear_levels();
        self.control.running.store(false, Ordering::Release);
    }
}

#[cfg(windows)]
mod imp {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
    use windows::Win32::Media::Audio::{
        AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
        AUDCLNT_STREAMFLAGS_LOOPBACK, AUDIOCLIENT_ACTIVATION_PARAMS,
        AUDIOCLIENT_ACTIVATION_PARAMS_0, AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, ActivateAudioInterfaceAsync, DEVICE_STATE_ACTIVE,
        IActivateAudioInterfaceAsyncOperation, IActivateAudioInterfaceCompletionHandler,
        IActivateAudioInterfaceCompletionHandler_Impl, IAudioCaptureClient, IAudioClient,
        IAudioRenderClient, IMMDevice, IMMDeviceEnumerator,
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        WAVE_FORMAT_PCM, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eRender,
    };
    use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
    use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    use windows::Win32::System::Com::StructuredStorage::PropVariantToString;
    use windows::Win32::System::Com::{
        CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, STGM_READ,
    };
    use windows::core::{GUID, HRESULT, IUnknown, Interface, Ref, implement};

    use super::{AudioHapticsControl, TactileSynth};

    const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID::from_u128(0xbcde0395_e52f_467c_8e3d_c4579291692e);
    const RATE: u32 = 48_000;

    #[implement(IActivateAudioInterfaceCompletionHandler)]
    struct LoopbackActivated;

    impl IActivateAudioInterfaceCompletionHandler_Impl for LoopbackActivated_Impl {
        fn ActivateCompleted(
            &self,
            _: Ref<IActivateAudioInterfaceAsyncOperation>,
        ) -> windows::core::Result<()> {
            Ok(())
        }
    }

    pub(super) fn ensure_started(control: &Arc<AudioHapticsControl>) {
        if control.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let ctrl = Arc::clone(control);
        let _ = std::thread::Builder::new()
            .name("victory dualsense haptics".into())
            .spawn(move || {
                if let Err(e) = run_worker(&ctrl) {
                    tracing::debug!("DualSense 4ch audio haptics inactive: {e}");
                }
                ctrl.running.store(false, Ordering::Release);
            });
    }

    fn find_dualsense_endpoint() -> windows::core::Result<Option<IMMDevice>> {
        // SAFETY: COM initialized on this worker thread before calling here.
        unsafe {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&CLSID_MMDEVICE_ENUMERATOR, None, CLSCTX_ALL)?;
            let collection = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)?;
            let count = collection.GetCount()?;
            for i in 0..count {
                let Ok(device) = collection.Item(i) else {
                    continue;
                };
                let Ok(store) = device.OpenPropertyStore(STGM_READ) else {
                    continue;
                };
                let Ok(prop) = store.GetValue(&PKEY_Device_FriendlyName) else {
                    continue;
                };
                let mut chars = [0u16; 256];
                if PropVariantToString(&prop, &mut chars).is_err() {
                    continue;
                }
                let end = chars.iter().position(|&c| c == 0).unwrap_or(chars.len());
                let name = String::from_utf16_lossy(&chars[..end]).to_ascii_lowercase();
                if name.contains("wireless controller")
                    || name.contains("dualsense")
                    || name.contains("ds5")
                {
                    tracing::info!(endpoint = %name, "found DualSense 4ch audio endpoint");
                    return Ok(Some(device));
                }
            }
            Ok(None)
        }
    }

    fn open_process_loopback() -> Option<(IAudioClient, IAudioCaptureClient)> {
        // SAFETY: activates a process-scoped loopback capture client on our own PID.
        unsafe {
            let params = AUDIOCLIENT_ACTIVATION_PARAMS {
                ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
                Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                    ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                        TargetProcessId: std::process::id(),
                        ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                    },
                },
            };
            #[repr(C)]
            struct Blob {
                vt: u16,
                reserved: [u16; 3],
                size: u32,
                data: *const AUDIOCLIENT_ACTIVATION_PARAMS,
            }
            let blob = Blob {
                vt: 0x41,
                reserved: [0; 3],
                size: size_of_val(&params) as u32,
                data: &params,
            };
            let handler: IActivateAudioInterfaceCompletionHandler = LoopbackActivated.into();
            let op = ActivateAudioInterfaceAsync(
                VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                &IAudioClient::IID,
                Some((&raw const blob).cast()),
                &handler,
            )
            .ok()?;
            let client = (0..50).find_map(|_| {
                let (mut hr, mut unk) = (HRESULT(0), None::<IUnknown>);
                if op.GetActivateResult(&mut hr, &mut unk).is_ok() && hr.is_ok() {
                    unk?.cast::<IAudioClient>().ok()
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                    None
                }
            })?;
            let fmt = WAVEFORMATEX {
                wFormatTag: WAVE_FORMAT_PCM as u16,
                nChannels: 2,
                nSamplesPerSec: RATE,
                nAvgBytesPerSec: RATE * 4,
                nBlockAlign: 4,
                wBitsPerSample: 16,
                cbSize: 0,
            };
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                    400_000, // 40 ms
                    0,
                    &fmt,
                    None,
                )
                .ok()?;
            let capture: IAudioCaptureClient = client.GetService().ok()?;
            client.Start().ok()?;
            Some((client, capture))
        }
    }

    fn run_worker(control: &AudioHapticsControl) -> Result<(), Box<dyn std::error::Error>> {
        // SAFETY: initializes multithreaded COM for this dedicated audio thread.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let Some(device) = find_dualsense_endpoint()? else {
                return Err("no DualSense USB audio endpoint found".into());
            };
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
            // Quadraphonic 48 kHz 32-bit float (`FL | FR | BL | BR` = 0x33).
            let quad = WAVEFORMATEXTENSIBLE {
                Format: WAVEFORMATEX {
                    wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
                    nChannels: 4,
                    nSamplesPerSec: RATE,
                    nAvgBytesPerSec: RATE * 16,
                    nBlockAlign: 16,
                    wBitsPerSample: 32,
                    cbSize: 22,
                },
                Samples: windows::Win32::Media::Audio::WAVEFORMATEXTENSIBLE_0 {
                    wValidBitsPerSample: 32,
                },
                dwChannelMask: 0x33,
                SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
            };
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                300_000, // 30 ms shared buffer for low latency
                0,
                &quad.Format,
                None,
            )?;
            let buffer_frames = client.GetBufferSize()?;
            let render: IAudioRenderClient = client.GetService()?;
            let loopback = open_process_loopback();
            client.Start()?;

            let mut synth = TactileSynth::default();
            let mut loop_ring: VecDeque<[f32; 2]> = VecDeque::with_capacity(4096);

            while control.running.load(Ordering::Acquire) {
                if let Some((_, ref capture)) = loopback {
                    while let Ok(pkt) = capture.GetNextPacketSize()
                        && pkt > 0
                    {
                        let (mut ptr, mut n, mut flags) = (std::ptr::null_mut(), 0u32, 0u32);
                        if capture
                            .GetBuffer(&mut ptr, &mut n, &mut flags, None, None)
                            .is_ok()
                        {
                            let silent = flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0;
                            if silent || ptr.is_null() {
                                for _ in 0..n {
                                    if loop_ring.len() < 4096 {
                                        loop_ring.push_back([0.0, 0.0]);
                                    }
                                }
                            } else {
                                let samples =
                                    std::slice::from_raw_parts(ptr.cast::<i16>(), n as usize * 2);
                                for pair in samples.as_chunks::<2>().0 {
                                    if loop_ring.len() == 4096 {
                                        loop_ring.pop_front();
                                    }
                                    loop_ring.push_back([
                                        f32::from(pair[0]) / 32768.0,
                                        f32::from(pair[1]) / 32768.0,
                                    ]);
                                }
                            }
                            let _ = capture.ReleaseBuffer(n);
                        } else {
                            break;
                        }
                    }
                }

                let padding = client.GetCurrentPadding().unwrap_or(buffer_frames);
                let avail = buffer_frames.saturating_sub(padding);
                if avail > 0
                    && let Ok(dst) = render.GetBuffer(avail)
                {
                    let out = std::slice::from_raw_parts_mut(dst.cast::<f32>(), avail as usize * 4);
                    let levels = control.levels();
                    let mirror = control.speaker_mirror.load(Ordering::Relaxed);
                    let muted = control.muted.load(Ordering::Relaxed);
                    for chunk in out.as_chunks_mut::<4>().0 {
                        let game = loop_ring.pop_front().unwrap_or([0.0, 0.0]);
                        let frame = synth.next_frame(RATE as f32, levels, game, mirror, muted);
                        chunk.copy_from_slice(&frame);
                    }
                    let _ = render.ReleaseBuffer(avail, 0);
                }
                std::thread::sleep(Duration::from_millis(4));
            }

            let _ = client.Stop();
            if let Some((lb_client, _)) = loopback {
                let _ = lb_client.Stop();
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_keeps_speaker_silent_by_default_and_modulates_voice_coils() {
        let mut synth = TactileSynth::default();
        let mut peak_haptic = 0.0f32;
        for _ in 0..480 {
            let f = synth.next_frame(48_000.0, [0.8, 0.5, 0.4, 0.0], [0.5, -0.5], false, false);
            assert_eq!(
                (f[0], f[1]),
                (0.0, 0.0),
                "Ch1/2 must stay silent when speaker_mirror=false"
            );
            peak_haptic = peak_haptic.max(f[2].abs()).max(f[3].abs());
        }
        assert!(peak_haptic > 0.2 && peak_haptic <= 0.92);

        let muted = synth.next_frame(48_000.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0], true, true);
        assert_eq!(muted, [0.0; 4]);
    }
}
