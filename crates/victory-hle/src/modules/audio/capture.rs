//! Debugging aid: `VICTORY_AUDIO_CAPTURE=<file.wav>` records what the title
//! plays, 48 kHz stereo 16-bit, through process loopback: our own process
//! tree only, never the rest of the host's mix. Stretches where nothing was
//! played are written as silence, so the file keeps real time.
//!
//! The recorder runs in a child copy of this executable: an audio client
//! in the title's process raises that process's timer resolution, which
//! changes how the title's own waits behave.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_LOOPBACK,
    AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVE_FORMAT_PCM, WAVEFORMATEX,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_SYNCHRONIZE, Sleep, WaitForSingleObject,
};
use windows::core::{HRESULT, IUnknown, Interface, Ref, implement};

const RATE: u32 = 48_000;
const FRAME_BYTES: usize = 4;
const MAX_WAV_DATA: u32 = u32::MAX - 36;

struct ProcessHandle(HANDLE);

impl Drop for ProcessHandle {
    fn drop(&mut self) {
        // SAFETY: owned process handle opened by `OpenProcess`.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Activation completes asynchronously; the recording thread polls for it.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Activated;

impl IActivateAudioInterfaceCompletionHandler_Impl for Activated_Impl {
    fn ActivateCompleted(
        &self,
        _: Ref<IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

/// Names the process to record, in the recorder's environment.
const TARGET: &str = "VICTORY_AUDIO_CAPTURE_OF";

/// Starts the recorder if `VICTORY_AUDIO_CAPTURE` names a file. With this
/// module's debug log, also logs once a second what a `Sleep(1)` takes in
/// this process (the title's millisecond waits get the same).
pub fn start() {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let _ = std::thread::Builder::new()
            .name("victory timing probe".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let at = Instant::now();
                    // SAFETY: plain wait.
                    unsafe { Sleep(1) };
                    let ms = at.elapsed().as_secs_f32() * 1000.0;
                    tracing::debug!(ms, "Sleep(1) took");
                }
            });
    }
    if std::env::var_os("VICTORY_AUDIO_CAPTURE").is_some_and(|v| !v.is_empty())
        && let Err(e) = std::env::current_exe().and_then(|exe| {
            std::process::Command::new(exe)
                .env(TARGET, std::process::id().to_string())
                .spawn()
        })
    {
        tracing::warn!("audio capture not started: {e}");
    }
}

/// In the recorder (a child started by [`start`]): records until the
/// target exits and returns true; false in any other process.
pub fn child() -> bool {
    let (Some(pid), Some(path)) = (
        std::env::var(TARGET).ok().and_then(|p| p.parse().ok()),
        std::env::var_os("VICTORY_AUDIO_CAPTURE").filter(|v| !v.is_empty()),
    ) else {
        return false;
    };
    if let Err(e) = record(pid, Path::new(&path)) {
        tracing::warn!("audio capture stopped: {e}");
    }
    true
}

fn header(frames: u64) -> [u8; 44] {
    let data = u32::try_from(frames.saturating_mul(FRAME_BYTES as u64))
        .unwrap_or(MAX_WAV_DATA)
        .min(MAX_WAV_DATA);
    let mut h = [0u8; 44];
    let fields: [(&[u8], usize); 11] = [
        (b"RIFF", 0),
        (&(36 + data).to_le_bytes(), 4),
        (b"WAVEfmt ", 8),
        (&16u32.to_le_bytes(), 16),
        (&1u16.to_le_bytes(), 20),
        (&2u16.to_le_bytes(), 22),
        (&RATE.to_le_bytes(), 24),
        (&(RATE * FRAME_BYTES as u32).to_le_bytes(), 28),
        (&(FRAME_BYTES as u16).to_le_bytes(), 32),
        (&16u16.to_le_bytes(), 34),
        (b"data", 36),
    ];
    for (bytes, at) in fields {
        h[at..at + bytes.len()].copy_from_slice(bytes);
    }
    h[40..].copy_from_slice(&data.to_le_bytes());
    h
}

/// Computes the target frame offset from a 100 ns QPC timestamp without
/// underflowing on retrograde timestamps or overflowing `u64` multiplication.
fn packet_frame_offset(qpc: u64, first: &mut Option<u64>, timestamp_error: bool, written: u64) -> u64 {
    if timestamp_error {
        return written;
    }
    let start = *first.get_or_insert(qpc);
    let delta = qpc.saturating_sub(start);
    let frames = (u128::from(delta) * u128::from(RATE)) / 10_000_000;
    u64::try_from(frames).unwrap_or(u64::MAX)
}

/// Writes `frames` silent stereo 16-bit frames in fixed-size chunks so long
/// idle gaps never trigger large heap allocations.
fn write_silence(writer: &mut impl Write, mut frames: u64) -> std::io::Result<()> {
    const CHUNK_FRAMES: usize = 1024;
    const ZEROS: [u8; CHUNK_FRAMES * FRAME_BYTES] = [0u8; CHUNK_FRAMES * FRAME_BYTES];
    while frames > 0 {
        let take = frames.min(CHUNK_FRAMES as u64) as usize;
        writer.write_all(&ZEROS[..take * FRAME_BYTES])?;
        frames -= take as u64;
    }
    Ok(())
}

fn record(pid: u32, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: COM calls with parameters built here; buffers are used only
    // between `GetBuffer` and `ReleaseBuffer`.
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let target = ProcessHandle(OpenProcess(PROCESS_SYNCHRONIZE, false, pid)?);
        let params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: pid,
                    ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                },
            },
        };
        /// A `PROPVARIANT` holding a blob (`VT_BLOB`).
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
        let handler: IActivateAudioInterfaceCompletionHandler = Activated.into();
        let operation = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some((&raw const blob).cast()),
            &handler,
        )?;
        let client = loop {
            let (mut hr, mut client) = (HRESULT(0), None::<IUnknown>);
            // Fails (`E_ILLEGAL_METHOD_CALL`) until activation completes.
            if operation.GetActivateResult(&mut hr, &mut client).is_ok() {
                hr.ok()?;
                break client.ok_or("no audio client")?.cast::<IAudioClient>()?;
            }
            if WaitForSingleObject(target.0, 0) == WAIT_OBJECT_0 {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let format = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: 2,
            nSamplesPerSec: RATE,
            nAvgBytesPerSec: RATE * FRAME_BYTES as u32,
            nBlockAlign: FRAME_BYTES as u16,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            10_000_000,
            0,
            &format,
            None,
        )?;
        let capture: IAudioCaptureClient = client.GetService()?;
        let mut file = File::create(path)?;
        file.write_all(&header(0))?;
        client.Start()?;
        tracing::info!(path = %path.display(), "recording the title's audio");

        let (mut written, mut first) = (0u64, None);
        let mut patched = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(10));
            let ended = WaitForSingleObject(target.0, 0) == WAIT_OBJECT_0;
            while capture.GetNextPacketSize()? > 0 {
                let (mut data, mut frames, mut flags, mut qpc) = (std::ptr::null_mut(), 0, 0, 0);
                capture.GetBuffer(&mut data, &mut frames, &mut flags, None, Some(&mut qpc))?;
                // Where this packet belongs, from its time stamp (100 ns).
                let ts_error = flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32 != 0;
                let at = packet_frame_offset(qpc, &mut first, ts_error, written);
                if at > written.saturating_add(u64::from(RATE / 500)) {
                    tracing::debug!(
                        ms = (at - written) * 1000 / u64::from(RATE),
                        "audio capture: nothing played"
                    );
                    write_silence(&mut file, at - written)?;
                    written = at;
                }
                if frames > 0 {
                    if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 || data.is_null() {
                        write_silence(&mut file, u64::from(frames))?;
                    } else {
                        let bytes = frames as usize * FRAME_BYTES;
                        file.write_all(std::slice::from_raw_parts(data, bytes))?;
                    }
                }
                if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 && written > 0 {
                    tracing::debug!("audio capture: discontinuity");
                }
                written = written.saturating_add(u64::from(frames));
                capture.ReleaseBuffer(frames)?;
            }
            if ended || patched.elapsed() >= Duration::from_secs(1) {
                // Valid sizes at any time: runs end by being killed.
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&header(written))?;
                file.seek(SeekFrom::End(0))?;
                patched = Instant::now();
            }
            if ended {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_encodes_48khz_stereo_16bit_pcm() {
        let h = header(100);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), 36 + 400);
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(h[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(h[20..22].try_into().unwrap()), 1); // PCM
        assert_eq!(u16::from_le_bytes(h[22..24].try_into().unwrap()), 2); // stereo
        assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), 48_000);
        assert_eq!(
            u32::from_le_bytes(h[28..32].try_into().unwrap()),
            48_000 * 4
        );
        assert_eq!(u16::from_le_bytes(h[32..34].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(h[34..36].try_into().unwrap()), 16);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes(h[40..44].try_into().unwrap()), 400);
    }

    #[test]
    fn wav_header_clamps_near_u32_max_without_overflow() {
        // Previously `frames * 4` fit in `u32` (u32::MAX - 31) and `36 + data` overflowed.
        let near_max_frames = u64::from(u32::MAX - 31) / FRAME_BYTES as u64;
        for frames in [near_max_frames, u64::from(u32::MAX), u64::MAX] {
            let h = header(frames);
            let riff = u32::from_le_bytes(h[4..8].try_into().unwrap());
            let data = u32::from_le_bytes(h[40..44].try_into().unwrap());
            assert_eq!(data, MAX_WAV_DATA);
            assert_eq!(riff, u32::MAX);
        }
    }

    #[test]
    fn packet_frame_offset_handles_monotonic_retrograde_and_large_qpc() {
        let mut first = None;
        // First timestamp establishes origin (0 frames).
        assert_eq!(packet_frame_offset(10_000_000, &mut first, false, 0), 0);
        assert_eq!(first, Some(10_000_000));

        // 1 second (10_000_000 * 100 ns) later -> 48_000 frames.
        assert_eq!(
            packet_frame_offset(20_000_000, &mut first, false, 480),
            48_000
        );

        // Retrograde QPC timestamp saturates to 0 instead of underflowing.
        assert_eq!(packet_frame_offset(5_000_000, &mut first, false, 48_000), 0);

        // Large delta that would overflow `delta * 48_000` in u64 still computes accurately.
        let huge_delta: u64 = 500_000_000_000_000;
        let expected = ((u128::from(huge_delta) * u128::from(RATE)) / 10_000_000) as u64;
        assert_eq!(
            packet_frame_offset(10_000_000 + huge_delta, &mut first, false, 0),
            expected
        );
    }

    #[test]
    fn packet_frame_offset_ignores_timestamp_error_flag() {
        let mut first = None;
        // Error on first packet must not poison `first` and keeps current written offset.
        assert_eq!(packet_frame_offset(0, &mut first, true, 123), 123);
        assert_eq!(first, None);

        assert_eq!(packet_frame_offset(50_000_000, &mut first, false, 123), 0);
        assert_eq!(packet_frame_offset(0, &mut first, true, 999), 999);
    }

    #[test]
    fn write_silence_handles_zero_single_and_multi_chunk_spans() {
        let mut buf = Vec::new();
        write_silence(&mut buf, 0).unwrap();
        assert!(buf.is_empty());

        write_silence(&mut buf, 1).unwrap();
        assert_eq!(buf.len(), FRAME_BYTES);
        assert!(buf.iter().all(|&b| b == 0));

        // Span multiple 1024-frame chunks plus a partial remainder.
        buf.clear();
        let frames = 2500u64;
        write_silence(&mut buf, frames).unwrap();
        assert_eq!(buf.len(), frames as usize * FRAME_BYTES);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn child_returns_false_without_capture_env() {
        assert!(!child());
    }
}
