//! `XSpeechSynthesizer`: text-to-speech synthesis into WAV streams.
//!
//! On Windows, synthesis delegates to the system SAPI 5 engine (`SpVoice` +
//! `SpStream` over an `HGLOBAL` `IStream`, loaded dynamically from `ole32.dll`),
//! producing 16 kHz mono 16-bit PCM wrapped in a standard RIFF/WAVE container.
//! Plain text (slot `0x68`) uses `SPF_IS_NOT_XML`; SSML (slot `0xa8`) uses
//! `SPF_IS_XML | SPF_PARSE_SSML`. If SAPI is unavailable (for example on a
//! stripped/headless host or non-Windows test runs), streams fall back cleanly
//! to a valid silent 44-byte RIFF/WAVE header so titles keep running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put};

pub const PROVIDER: Guid = Guid::new(
    0x27be_76b2,
    0xdba1,
    0x4bd8,
    [0xa0, 0x05, 0x5c, 0x0f, 0x15, 0x16, 0x18, 0x1a],
);
/// As Grand Theft Auto VI (2026) asks for it; the slots it calls (0x40-0x80,
/// 0xa8) take the public API's arguments in the 2020 order.
pub const IID_2026: Guid = Guid::new(
    0x611e_c004,
    0xd326,
    0x4580,
    [0xab, 0x90, 0xc3, 0x90, 0x39, 0x99, 0x3d, 0x3a],
);

pub static OBJECT: Object = Object::new("speech", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID_2026, &OBJECT)];

const E_INSUFFICIENT_BUFFER: HRESULT = 0x8007_007A_u32 as i32;

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;

type This = *const Object;

/// Owned synthesizer instance handed to titles by `create` (`0x48`).
struct Synthesizer {
    voice_id: Option<String>,
}

/// Owned synthesized WAV buffer handed to titles by `create_stream` (`0x68`/`0xa8`).
struct Stream {
    wav: Vec<u8>,
}

/// Matches the GDK `XSpeechSynthesizerVoiceInformation` struct layout.
#[repr(C)]
struct VoiceInfo {
    description: *const c_char,
    display_name: *const c_char,
    gender: u32,
    voice_id: *const c_char,
    language: *const c_char,
}

const DEFAULT_DESCRIPTION: &[u8] = b"Microsoft Default Voice\0";
const DEFAULT_DISPLAY_NAME: &[u8] = b"Default\0";
const DEFAULT_VOICE_ID: &[u8] = b"victory-default\0";
const DEFAULT_LANGUAGE: &[u8] = b"en-US\0";

type VoiceCallback = extern "system" fn(info: *const VoiceInfo, context: *mut c_void) -> bool;

/// Wraps raw interleaved little-endian 16-bit PCM bytes in a 44-byte RIFF/WAVE
/// container. If `pcm` already begins with `"RIFF"`, returns it verbatim.
fn wrap_wav_pcm16(pcm: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    if pcm.len() >= 12 && &pcm[..4] == b"RIFF" && &pcm[8..12] == b"WAVE" {
        return pcm.to_vec();
    }
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX - 36);
    let pcm = &pcm[..data_len as usize];
    let riff_len = 36u32.saturating_add(data_len);
    let block_align = channels * (BITS_PER_SAMPLE / 8);
    let byte_rate = sample_rate * u32::from(block_align);

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Reads a NUL-terminated UTF-8 C string from title memory, returning an empty
/// string if null or not valid UTF-8.
fn read_cstr(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: title passes a NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Synthesizes `input` (plain text or SSML) to a RIFF/WAVE buffer.
fn synthesize_wav(input: &str, ssml: bool) -> Vec<u8> {
    if !input.trim().is_empty() {
        #[cfg(windows)]
        if let Some(pcm) = sapi::speak_pcm16(input, ssml) {
            return wrap_wav_pcm16(&pcm, SAMPLE_RATE, CHANNELS);
        }
        #[cfg(not(windows))]
        let _ = ssml;
    }
    wrap_wav_pcm16(&[], SAMPLE_RATE, CHANNELS)
}

/// `cb(info, context) -> bool` (continue). Reports our default system voice.
extern "system" fn enumerate_voices(
    _: This,
    context: *mut c_void,
    callback: Option<VoiceCallback>,
) -> HRESULT {
    let Some(cb) = callback else {
        return E_INVALIDARG;
    };
    let info = VoiceInfo {
        description: DEFAULT_DESCRIPTION.as_ptr().cast(),
        display_name: DEFAULT_DISPLAY_NAME.as_ptr().cast(),
        gender: 0,
        voice_id: DEFAULT_VOICE_ID.as_ptr().cast(),
        language: DEFAULT_LANGUAGE.as_ptr().cast(),
    };
    let _ = cb(&info, context);
    S_OK
}

extern "system" fn create(_: This, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let synth = Box::into_raw(Box::new(Synthesizer { voice_id: None }));
    // SAFETY: checked non-null; the title passes room for the handle.
    unsafe { out.write(synth.cast()) };
    S_OK
}

extern "system" fn close_synthesizer(_: This, handle: *mut c_void) -> HRESULT {
    if !handle.is_null() {
        // SAFETY: handle was allocated by `create` via `Box::into_raw`.
        drop(unsafe { Box::from_raw(handle.cast::<Synthesizer>()) });
    }
    S_OK
}

extern "system" fn set_default_voice(_: This, handle: *mut c_void) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: live handle from `create`.
    unsafe { (*handle.cast::<Synthesizer>()).voice_id = None };
    S_OK
}

extern "system" fn set_custom_voice(
    _: This,
    handle: *mut c_void,
    voice_id: *const c_char,
) -> HRESULT {
    if handle.is_null() || voice_id.is_null() {
        return E_INVALIDARG;
    }
    let id = read_cstr(voice_id);
    // SAFETY: live handle from `create`.
    unsafe { (*handle.cast::<Synthesizer>()).voice_id = Some(id) };
    S_OK
}

fn make_stream(
    handle: *mut c_void,
    input: *const c_char,
    out: *mut *mut c_void,
    ssml: bool,
) -> HRESULT {
    if handle.is_null() || input.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let text = read_cstr(input);
    let wav = synthesize_wav(&text, ssml);
    let stream = Box::into_raw(Box::new(Stream { wav }));
    // SAFETY: checked non-null; the title passes room for the stream handle.
    unsafe { out.write(stream.cast()) };
    S_OK
}

extern "system" fn create_stream(
    _: This,
    handle: *mut c_void,
    text: *const c_char,
    out: *mut *mut c_void,
) -> HRESULT {
    make_stream(handle, text, out, false)
}

extern "system" fn create_stream_ssml(
    _: This,
    handle: *mut c_void,
    ssml: *const c_char,
    out: *mut *mut c_void,
) -> HRESULT {
    make_stream(handle, ssml, out, true)
}

extern "system" fn close_stream(_: This, stream: *mut c_void) -> HRESULT {
    if !stream.is_null() {
        // SAFETY: handle was allocated by `make_stream` via `Box::into_raw`.
        drop(unsafe { Box::from_raw(stream.cast::<Stream>()) });
    }
    S_OK
}

extern "system" fn stream_size(_: This, stream: *mut c_void, size: *mut usize) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: live handle from `make_stream`.
    let len = unsafe { &*stream.cast::<Stream>() }.wav.len();
    put(size, len)
}

extern "system" fn stream_data(
    _: This,
    stream: *mut c_void,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    if stream.is_null() || buffer.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: live handle from `make_stream`.
    let wav = &unsafe { &*stream.cast::<Stream>() }.wav;
    if size < wav.len() {
        if !used.is_null() {
            // SAFETY: checked non-null.
            unsafe { used.write(wav.len()) };
        }
        return E_INSUFFICIENT_BUFFER;
    }
    // SAFETY: the title's buffer holds at least `wav.len()` bytes.
    unsafe { std::ptr::copy_nonoverlapping(wav.as_ptr(), buffer, wav.len()) };
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(wav.len()) };
    }
    S_OK
}

/// Matches GDK `XClosedCaptionProperties` (`XAccessibility.h`, #865).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosedCaptionProperties {
    pub background_color: u32,
    pub font_color: u32,
    pub window_color: u32,
    pub font_edge_attribute: u32,
    pub font_style: u32,
    pub font_scale: f32,
    pub enabled: u8,
}

const _: () = assert!(size_of::<ClosedCaptionProperties>() == 28);
const _: () = assert!(align_of::<ClosedCaptionProperties>() == 4);

static CAPTIONS: AtomicBool = AtomicBool::new(false);
static HYPOTHESIS_ID: AtomicU32 = AtomicU32::new(1);

extern "system" fn closed_caption_get_properties(
    _: This,
    out: *mut ClosedCaptionProperties,
) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    put(
        out,
        ClosedCaptionProperties {
            background_color: 0xFF00_0000,
            font_color: 0xFFFF_FFFF,
            window_color: 0x0000_0000,
            font_edge_attribute: 0,
            font_style: 0,
            font_scale: 1.0,
            enabled: u8::from(CAPTIONS.load(Ordering::Acquire)),
        },
    )
}

extern "system" fn closed_caption_set_enabled(_: This, enabled: bool) -> HRESULT {
    CAPTIONS.store(enabled, Ordering::Release);
    S_OK
}

extern "system" fn high_contrast_get_mode(_: This, out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    put(out, 0u32) // XHighContrastMode::Off
}

extern "system" fn stt_position_hint(_: This, position: u32) -> HRESULT {
    tracing::debug!(position, "speech-to-text position hint");
    S_OK
}

extern "system" fn stt_send_string(
    _: This,
    speaker: *const c_char,
    content: *const c_char,
    kind: u32,
) -> HRESULT {
    tracing::debug!(
        speaker = read_cstr(speaker),
        content = read_cstr(content),
        kind,
        "speech-to-text send"
    );
    S_OK
}

extern "system" fn stt_begin_hypothesis(
    _: This,
    speaker: *const c_char,
    content: *const c_char,
    kind: u32,
    out: *mut u32,
) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let mut id = HYPOTHESIS_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        id = HYPOTHESIS_ID.fetch_add(1, Ordering::Relaxed);
    }
    tracing::debug!(
        id,
        speaker = read_cstr(speaker),
        content = read_cstr(content),
        kind,
        "speech-to-text begin hypothesis"
    );
    put(out, id)
}

extern "system" fn stt_update_hypothesis(_: This, id: u32, content: *const c_char) -> HRESULT {
    tracing::debug!(id, content = read_cstr(content), "speech-to-text update");
    S_OK
}

extern "system" fn stt_finalize_hypothesis(_: This, id: u32, content: *const c_char) -> HRESULT {
    tracing::debug!(id, content = read_cstr(content), "speech-to-text finalize");
    S_OK
}

extern "system" fn stt_cancel_hypothesis(_: This, id: u32) -> HRESULT {
    tracing::debug!(id, "speech-to-text cancel");
    S_OK
}

/// Slots by the 2020 order; Grand Theft Auto VI calls 0x40-0x80 and 0xa8.
static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    closed_caption_get_properties as *const (), // 0x18
    closed_caption_set_enabled as *const (),    // 0x20
    high_contrast_get_mode as *const (),        // 0x28
    stt_position_hint as *const (),             // 0x30
    stt_send_string as *const (),               // 0x38
    enumerate_voices as *const (),              // 0x40
    create as *const (),                        // 0x48
    close_synthesizer as *const (),             // 0x50
    set_default_voice as *const (),             // 0x58
    set_custom_voice as *const (),              // 0x60
    create_stream as *const (),                 // 0x68 text
    close_stream as *const (),                  // 0x70
    stream_size as *const (),                   // 0x78
    stream_data as *const (),                   // 0x80
    stt_begin_hypothesis as *const (),          // 0x88
    stt_update_hypothesis as *const (),         // 0x90
    stt_finalize_hypothesis as *const (),       // 0x98
    stt_cancel_hypothesis as *const (),         // 0xa0
    create_stream_ssml as *const (),            // 0xa8 SSML
]);

/// Windows SAPI 5 speech synthesis via dynamically loaded `ole32.dll` COM
/// entry points and raw `ISpVoice`/`ISpStream` vtable calls.
#[cfg(windows)]
mod sapi {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};
    use windows_sys::core::{PCSTR, w};

    use super::{BITS_PER_SAMPLE, CHANNELS, SAMPLE_RATE};
    use crate::modules::xgameruntime::{Guid, HRESULT};

    const COINIT_MULTITHREADED: u32 = 0;
    const CLSCTX_INPROC_SERVER: u32 = 1;
    const RPC_E_CHANGED_MODE: HRESULT = 0x8001_0106_u32 as i32;

    const SPF_IS_XML: u32 = 8;
    const SPF_IS_NOT_XML: u32 = 16;
    const SPF_PARSE_SSML: u32 = 256;

    const CLSID_SP_VOICE: Guid = Guid::new(
        0x9674_9377,
        0x3391,
        0x11d2,
        [0x9e, 0xe3, 0x00, 0xc0, 0x4f, 0x79, 0x73, 0x96],
    );
    const IID_ISP_VOICE: Guid = Guid::new(
        0x6c44_df74,
        0x72b9,
        0x4992,
        [0xa1, 0xec, 0xef, 0x99, 0x6e, 0x04, 0x22, 0xd4],
    );
    const CLSID_SP_STREAM: Guid = Guid::new(
        0x715d_9c59,
        0x4442,
        0x11d2,
        [0x96, 0x05, 0x00, 0xc0, 0x4f, 0x8e, 0xe6, 0x28],
    );
    const IID_ISP_STREAM: Guid = Guid::new(
        0x12e3_cca9,
        0x7518,
        0x44c5,
        [0xa5, 0xe7, 0xba, 0x5a, 0x79, 0xcb, 0x92, 0x9e],
    );
    const SPDFID_WAVE_FORMAT_EX: Guid = Guid::new(
        0xc31a_dbae,
        0x527f,
        0x4ff5,
        [0xa2, 0x30, 0xf6, 0x2b, 0xb6, 0x1f, 0xf7, 0x0c],
    );

    #[repr(C)]
    struct WaveFormatEx {
        format_tag: u16,
        channels: u16,
        samples_per_sec: u32,
        avg_bytes_per_sec: u32,
        block_align: u16,
        bits_per_sample: u16,
        cb_size: u16,
    }

    struct Ole32 {
        co_init: unsafe extern "system" fn(*const c_void, u32) -> HRESULT,
        co_uninit: unsafe extern "system" fn(),
        co_create: unsafe extern "system" fn(
            *const Guid,
            *mut c_void,
            u32,
            *const Guid,
            *mut *mut c_void,
        ) -> HRESULT,
        create_stream: unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> HRESULT,
        get_hglobal: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT,
    }

    fn ole32() -> Option<&'static Ole32> {
        static LIB: OnceLock<Option<Ole32>> = OnceLock::new();
        LIB.get_or_init(|| {
            // SAFETY: loading standard system `ole32.dll` and resolving documented exports.
            unsafe {
                let module = LoadLibraryW(w!("ole32.dll"));
                if module.is_null() {
                    return None;
                }
                let sym = |name: PCSTR| GetProcAddress(module, name);
                type FarProc = unsafe extern "system" fn() -> isize;
                Some(Ole32 {
                    co_init: std::mem::transmute::<
                        FarProc,
                        unsafe extern "system" fn(*const c_void, u32) -> HRESULT,
                    >(sym(c"CoInitializeEx".as_ptr().cast())?),
                    co_uninit: std::mem::transmute::<FarProc, unsafe extern "system" fn()>(
                        sym(c"CoUninitialize".as_ptr().cast())?,
                    ),
                    co_create: std::mem::transmute::<
                        FarProc,
                        unsafe extern "system" fn(
                            *const Guid,
                            *mut c_void,
                            u32,
                            *const Guid,
                            *mut *mut c_void,
                        ) -> HRESULT,
                    >(sym(c"CoCreateInstance".as_ptr().cast())?),
                    create_stream: std::mem::transmute::<
                        FarProc,
                        unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> HRESULT,
                    >(sym(c"CreateStreamOnHGlobal".as_ptr().cast())?),
                    get_hglobal: std::mem::transmute::<
                        FarProc,
                        unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT,
                    >(sym(c"GetHGlobalFromStream".as_ptr().cast())?),
                })
            }
        })
        .as_ref()
    }

    /// Calls `IUnknown::Release` (vtable slot 2) on a non-null COM interface pointer.
    unsafe fn com_release(obj: *mut c_void) {
        if obj.is_null() {
            return;
        }
        // SAFETY: `obj` is a live COM interface pointer whose slot 2 is `Release`.
        unsafe {
            let vtbl = *(obj as *const *const *const c_void);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(*vtbl.add(2));
            release(obj);
        }
    }

    /// RAII guard for a COM interface pointer.
    struct ComPtr(*mut c_void);
    impl Drop for ComPtr {
        fn drop(&mut self) {
            // SAFETY: we own one reference to this COM interface pointer.
            unsafe { com_release(self.0) };
        }
    }

    /// Synthesizes `text` using SAPI 5 into raw 16 kHz mono 16-bit PCM bytes.
    pub(super) fn speak_pcm16(text: &str, ssml: bool) -> Option<Vec<u8>> {
        let ole = ole32()?;
        // SAFETY: COM apartment initialization for this thread.
        let init_hr = unsafe { (ole.co_init)(std::ptr::null(), COINIT_MULTITHREADED) };
        let should_uninit = init_hr >= 0;
        if !should_uninit && init_hr != RPC_E_CHANGED_MODE {
            return None;
        }
        struct CoGuard<'a>(&'a Ole32, bool);
        impl Drop for CoGuard<'_> {
            fn drop(&mut self) {
                if self.1 {
                    // SAFETY: balanced with a successful `CoInitializeEx`.
                    unsafe { (self.0.co_uninit)() };
                }
            }
        }
        let _co = CoGuard(ole, should_uninit);

        let mut mem_stream_raw = std::ptr::null_mut();
        // SAFETY: `CreateStreamOnHGlobal(NULL, TRUE, &out)` allocates an HGLOBAL-backed IStream.
        if unsafe { (ole.create_stream)(std::ptr::null_mut(), 1, &mut mem_stream_raw) } < 0
            || mem_stream_raw.is_null()
        {
            return None;
        }
        let mem_stream = ComPtr(mem_stream_raw);

        let mut sp_stream_raw = std::ptr::null_mut();
        // SAFETY: instantiating standard in-process SAPI `SpStream`.
        if unsafe {
            (ole.co_create)(
                &CLSID_SP_STREAM,
                std::ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_ISP_STREAM,
                &mut sp_stream_raw,
            )
        } < 0
            || sp_stream_raw.is_null()
        {
            return None;
        }
        let sp_stream = ComPtr(sp_stream_raw);

        let mut voice_raw = std::ptr::null_mut();
        // SAFETY: instantiating standard in-process SAPI `SpVoice`.
        if unsafe {
            (ole.co_create)(
                &CLSID_SP_VOICE,
                std::ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_ISP_VOICE,
                &mut voice_raw,
            )
        } < 0
            || voice_raw.is_null()
        {
            return None;
        }
        let voice = ComPtr(voice_raw);

        let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);
        let wfx = WaveFormatEx {
            format_tag: 1, // WAVE_FORMAT_PCM
            channels: CHANNELS,
            samples_per_sec: SAMPLE_RATE,
            avg_bytes_per_sec: SAMPLE_RATE * u32::from(block_align),
            block_align,
            bits_per_sample: BITS_PER_SAMPLE,
            cb_size: 0,
        };

        // ISpStream::SetBaseStream is slot 15 (3 IUnknown + 2 ISequentialStream
        // + 9 IStream + 1 ISpStreamFormat).
        // SAFETY: calling verified COM vtable slot 15 on live `ISpStream`.
        let set_base_hr = unsafe {
            let vtbl = *(sp_stream.0 as *const *const *const c_void);
            let set_base: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const Guid,
                *const WaveFormatEx,
            ) -> HRESULT = std::mem::transmute(*vtbl.add(15));
            set_base(sp_stream.0, mem_stream.0, &SPDFID_WAVE_FORMAT_EX, &wfx)
        };
        if set_base_hr < 0 {
            return None;
        }

        // ISpVoice::SetOutput is slot 13 (3 IUnknown + 7 ISpNotifySource + 3 ISpEventSource).
        // SAFETY: calling verified COM vtable slot 13 on live `ISpVoice`.
        let set_out_hr = unsafe {
            let vtbl = *(voice.0 as *const *const *const c_void);
            let set_output: unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> HRESULT =
                std::mem::transmute(*vtbl.add(13));
            set_output(voice.0, sp_stream.0, 0)
        };
        if set_out_hr < 0 {
            return None;
        }

        let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
        let flags = if ssml {
            SPF_IS_XML | SPF_PARSE_SSML
        } else {
            SPF_IS_NOT_XML
        };

        // ISpVoice::Speak is slot 20 (13 SetOutput + 7 methods).
        // SAFETY: calling verified COM vtable slot 20 with NUL-terminated UTF-16 text.
        let speak_hr = unsafe {
            let vtbl = *(voice.0 as *const *const *const c_void);
            let speak: unsafe extern "system" fn(
                *mut c_void,
                *const u16,
                u32,
                *mut u32,
            ) -> HRESULT = std::mem::transmute(*vtbl.add(20));
            speak(voice.0, wide.as_ptr(), flags, std::ptr::null_mut())
        };
        if speak_hr < 0 {
            return None;
        }

        // ISpStream::Close is slot 18.
        // SAFETY: calling verified COM vtable slot 18 to flush written audio to `mem_stream`.
        let _ = unsafe {
            let vtbl = *(sp_stream.0 as *const *const *const c_void);
            let close: unsafe extern "system" fn(*mut c_void) -> HRESULT =
                std::mem::transmute(*vtbl.add(18));
            close(sp_stream.0)
        };

        let mut hglobal = std::ptr::null_mut();
        // SAFETY: `mem_stream` was created by `CreateStreamOnHGlobal`.
        if unsafe { (ole.get_hglobal)(mem_stream.0, &mut hglobal) } < 0 || hglobal.is_null() {
            return None;
        }

        // SAFETY: locking the stream's HGLOBAL memory block to copy out synthesized PCM.
        let bytes = unsafe {
            let size = GlobalSize(hglobal);
            if size == 0 {
                return None;
            }
            let ptr = GlobalLock(hglobal);
            if ptr.is_null() {
                return None;
            }
            let copied = std::slice::from_raw_parts(ptr.cast::<u8>(), size).to_vec();
            let _ = GlobalUnlock(hglobal);
            copied
        };
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_wav_builds_valid_16khz_mono_pcm_header() {
        let pcm = [0x12u8, 0x34, 0x56, 0x78];
        let wav = wrap_wav_pcm16(&pcm, 16_000, 1);
        assert_eq!(wav.len(), 48);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 40);
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1); // PCM
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1); // mono
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 32_000);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
        assert_eq!(&wav[44..48], &pcm);
    }

    #[test]
    fn wrap_wav_preserves_existing_riff_header() {
        let existing = wrap_wav_pcm16(&[1, 2], 16_000, 1);
        assert_eq!(wrap_wav_pcm16(&existing, 16_000, 1), existing);
    }

    #[test]
    fn synthesizer_and_stream_lifecycle_handles_text_and_ssml() {
        let mut synth = std::ptr::null_mut();
        assert_eq!(create(&OBJECT, &mut synth), S_OK);
        assert!(!synth.is_null());

        assert_eq!(
            set_custom_voice(&OBJECT, synth, DEFAULT_VOICE_ID.as_ptr().cast()),
            S_OK
        );
        assert_eq!(set_default_voice(&OBJECT, synth), S_OK);

        let mut stream = std::ptr::null_mut();
        assert_eq!(
            create_stream(&OBJECT, synth, c"Hello Los Santos".as_ptr(), &mut stream),
            S_OK
        );
        assert!(!stream.is_null());

        let mut size = 0usize;
        assert_eq!(stream_size(&OBJECT, stream, &mut size), S_OK);
        assert!(size >= 44);

        let mut tiny = [0u8; 8];
        let mut needed = 0usize;
        assert_eq!(
            stream_data(&OBJECT, stream, tiny.len(), tiny.as_mut_ptr(), &mut needed),
            E_INSUFFICIENT_BUFFER
        );
        assert_eq!(needed, size);

        let mut buf = vec![0u8; size];
        let mut written = 0usize;
        assert_eq!(
            stream_data(&OBJECT, stream, buf.len(), buf.as_mut_ptr(), &mut written),
            S_OK
        );
        assert_eq!(written, size);
        assert_eq!(&buf[0..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");

        assert_eq!(close_stream(&OBJECT, stream), S_OK);

        let mut ssml_stream = std::ptr::null_mut();
        assert_eq!(
            create_stream_ssml(
                &OBJECT,
                synth,
                c"<speak version='1.0'>Test</speak>".as_ptr(),
                &mut ssml_stream
            ),
            S_OK
        );
        assert_eq!(close_stream(&OBJECT, ssml_stream), S_OK);
        assert_eq!(close_synthesizer(&OBJECT, synth), S_OK);
    }

    #[test]
    fn enumerate_voices_reports_default_voice() {
        extern "system" fn collect(info: *const VoiceInfo, ctx: *mut c_void) -> bool {
            // SAFETY: test passes a valid pointer to `usize`.
            let count = unsafe { &mut *ctx.cast::<usize>() };
            // SAFETY: callback receives a live `VoiceInfo`.
            let info = unsafe { &*info };
            assert!(!info.voice_id.is_null());
            assert!(!info.language.is_null());
            assert!(!info.display_name.is_null());
            assert!(!info.description.is_null());
            *count += 1;
            true
        }
        let mut count = 0usize;
        assert_eq!(
            enumerate_voices(&OBJECT, (&raw mut count).cast(), Some(collect)),
            S_OK
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn accessibility_captions_high_contrast_and_stt_slots_work() {
        use crate::modules::xgameruntime::method;

        type GetCc = extern "system" fn(This, *mut ClosedCaptionProperties) -> HRESULT;
        type SetCc = extern "system" fn(This, bool) -> HRESULT;
        type GetHc = extern "system" fn(This, *mut u32) -> HRESULT;
        type Begin =
            extern "system" fn(This, *const c_char, *const c_char, u32, *mut u32) -> HRESULT;
        type Update = extern "system" fn(This, u32, *const c_char) -> HRESULT;
        type Cancel = extern "system" fn(This, u32) -> HRESULT;

        let mut props = ClosedCaptionProperties {
            background_color: 0,
            font_color: 0,
            window_color: 0,
            font_edge_attribute: 0,
            font_style: 0,
            font_scale: 0.0,
            enabled: 9,
        };
        assert_eq!(method::<GetCc>(&OBJECT, 0x18)(&OBJECT, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(method::<SetCc>(&OBJECT, 0x20)(&OBJECT, true), S_OK);
        assert_eq!(method::<GetCc>(&OBJECT, 0x18)(&OBJECT, &mut props), S_OK);
        assert_eq!(props.enabled, 1);
        assert_eq!(props.font_scale, 1.0);
        assert_eq!(method::<SetCc>(&OBJECT, 0x20)(&OBJECT, false), S_OK);

        let mut hc = 7u32;
        assert_eq!(method::<GetHc>(&OBJECT, 0x28)(&OBJECT, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(method::<GetHc>(&OBJECT, 0x28)(&OBJECT, &mut hc), S_OK);
        assert_eq!(hc, 0);

        let (mut id1, mut id2) = (0u32, 0u32);
        let begin = method::<Begin>(&OBJECT, 0x88);
        assert_eq!(begin(&OBJECT, c"P1".as_ptr(), c"hi".as_ptr(), 0, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(begin(&OBJECT, c"P1".as_ptr(), c"hi".as_ptr(), 0, &mut id1), S_OK);
        assert_eq!(begin(&OBJECT, c"P1".as_ptr(), c"yo".as_ptr(), 0, &mut id2), S_OK);
        assert!(id1 > 0 && id2 > id1);
        assert_eq!(method::<Update>(&OBJECT, 0x90)(&OBJECT, id1, c"hi!".as_ptr()), S_OK);
        assert_eq!(method::<Update>(&OBJECT, 0x98)(&OBJECT, id1, c"hi!".as_ptr()), S_OK);
        assert_eq!(method::<Cancel>(&OBJECT, 0xa0)(&OBJECT, id2), S_OK);
    }
}
