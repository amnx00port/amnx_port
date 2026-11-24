//! Accessibility: speech synthesis for the title's narration, spoken by
//! the host's WinRT speech synthesizer. When the host cannot produce a
//! voice (no voices installed, WinRT unavailable), streams fall back to a
//! valid WAVE file of no length, so the title still plays them and goes on.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};

use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

/// Closed captions, high contrast, speech to text and synthesis.
pub const PROVIDER: Guid = Guid::new(
    0x3e24_1977,
    0x6237,
    0x41e9,
    [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1],
);
/// Unchanged since 2020.
pub const IID: Guid = Guid::new(
    0xd722_b373,
    0x8c4d,
    0x4692,
    [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d],
);

pub static OBJECT: Object = Object::new("accessibility", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT)];

/// A WAVE file of 16 kHz mono 16-bit PCM with no samples: RIFF size 36,
/// a 16-byte format chunk (PCM, 1 channel, 16,000 Hz, 32,000 bytes/s,
/// block 2, 16 bits), an empty data chunk. The fallback for text the host
/// cannot speak.
const SILENCE: [u8; 44] =
    *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";

/// The synthesizer handle: one default voice, chosen by the host. Streams
/// are boxed and reclaimed by `close`.
static SYNTHESIZER: u8 = 0;

/// One synthesized stream: the WAVE bytes the title asks for.
struct Stream {
    wav: Vec<u8>,
}

#[cfg(windows)]
mod tts {
    //! The host's WinRT speech synthesizer as our voice.
    use std::sync::OnceLock;
    use std::time::Duration;

    use windows::Media::SpeechSynthesis::SpeechSynthesizer;
    use windows::Storage::Streams::DataReader;
    use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
    use windows_future::AsyncStatus;

    /// One-time WinRT initialization; `false` when the thread model is
    /// already taken or WinRT is unavailable.
    fn ready() -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| {
            // SAFETY: no parameters; every later call is a plain API use.
            // S_FALSE (already initialized, same model) counts as ready.
            unsafe { RoInitialize(RO_INIT_MULTITHREADED) }.is_ok()
        })
    }

    /// Synthesize `input` (plain text, or SSML when `ssml`) to RIFF WAVE
    /// bytes, or `None` when the host has no voice for it.
    pub fn synthesize(input: &str, ssml: bool) -> Option<Vec<u8>> {
        if input.trim().is_empty() || !ready() {
            return None;
        }
        let synth = SpeechSynthesizer::new().ok()?;
        let input = windows::core::HSTRING::from(input);
        let op = if ssml {
            synth.SynthesizeSsmlToStreamAsync(&input)
        } else {
            synth.SynthesizeTextToStreamAsync(&input)
        }
        .ok()?;
        // Narration strings are short; wait for the operation inline.
        while matches!(op.Status(), Ok(AsyncStatus::Started)) {
            std::thread::sleep(Duration::from_millis(1));
        }
        if op.Status().ok()? != AsyncStatus::Completed {
            return None;
        }
        let stream = op.GetResults().ok()?;
        let size = stream.Size().ok()?;
        if size == 0 {
            return None;
        }
        let input = stream.GetInputStreamAt(0).ok()?;
        let reader = DataReader::CreateDataReader(&input).ok()?;
        let load = reader.LoadAsync(size as u32).ok()?;
        while matches!(load.Status(), Ok(AsyncStatus::Started)) {
            std::thread::sleep(Duration::from_millis(1));
        }
        load.GetResults().ok()?;
        let mut wav = vec![0u8; size as usize];
        reader.ReadBytes(&mut wav).ok()?;
        (!wav.is_empty()).then_some(wav)
    }
}

#[cfg(not(windows))]
mod tts {
    //! No host synthesizer off Windows: streams stay silent.
    pub fn synthesize(_input: &str, _ssml: bool) -> Option<Vec<u8>> {
        None
    }
}

type This = *const Object;

/// No title of ours enumerates voices; the host's default speaks.
extern "system" fn enumerate_voices(
    _: This,
    _context: *mut c_void,
    _callback: *const c_void,
) -> HRESULT {
    S_OK
}
extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    put(out, &raw const SYNTHESIZER)
}
extern "system" fn close(_: This, handle: *const u8) -> HRESULT {
    // SAFETY: a handle we handed out; a stream is a box we reclaim here,
    // the synthesizer handle is a static that must survive.
    if !handle.is_null() && handle != &raw const SYNTHESIZER {
        // SAFETY: a live stream handle we handed out as `Box<Stream>`;
        // the title closes it once.
        drop(unsafe { Box::from_raw(handle.cast_mut().cast::<Stream>()) });
    }
    S_OK
}
extern "system" fn set_default_voice(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_custom_voice(_: This, _: *const u8, _voice: *const c_char) -> HRESULT {
    S_OK
}

fn stream_for(said: &str, ssml: bool) -> Box<Stream> {
    let wav = tts::synthesize(said, ssml).unwrap_or_else(|| SILENCE.to_vec());
    tracing::debug!(ssml, bytes = wav.len(), text = said, "speech");
    Box::new(Stream { wav })
}

extern "system" fn create_stream(
    _: This,
    _: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    let said = text(words).unwrap_or_default();
    put(
        out,
        Box::into_raw(stream_for(&said, false))
            .cast::<u8>()
            .cast_const(),
    )
}
extern "system" fn create_stream_ssml(
    _: This,
    _: *const u8,
    markup: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    let said = text(markup).unwrap_or_default();
    put(
        out,
        Box::into_raw(stream_for(&said, true))
            .cast::<u8>()
            .cast_const(),
    )
}
extern "system" fn stream_size(_: This, stream: *const u8, out: *mut usize) -> HRESULT {
    // SAFETY: `stream` is a live handle we handed out.
    let size = unsafe { &*stream.cast::<Stream>() }.wav.len();
    put(out, size)
}
extern "system" fn stream_data(
    _: This,
    stream: *const u8,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    // SAFETY: `stream` is a live handle we handed out.
    let wav = &unsafe { &*stream.cast::<Stream>() }.wav;
    if buffer.is_null() || size < wav.len() {
        return super::xasync::E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe { buffer.copy_from_nonoverlapping(wav.as_ptr(), wav.len()) };
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(wav.len()) };
    }
    S_OK
}

static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),       // 0x18 closed captions
    missing::<4> as *const (),       // 0x20
    missing::<5> as *const (),       // 0x28 high contrast
    missing::<6> as *const (),       // 0x30 speech to text
    missing::<7> as *const (),       // 0x38
    enumerate_voices as *const (),   // 0x40
    create as *const (),             // 0x48
    close as *const (),              // 0x50
    set_default_voice as *const (),  // 0x58
    set_custom_voice as *const (),   // 0x60
    create_stream as *const (),      // 0x68 from text
    close as *const (),              // 0x70 a stream
    stream_size as *const (),        // 0x78
    stream_data as *const (),        // 0x80
    missing::<17> as *const (),      // 0x88 speech to text
    missing::<18> as *const (),      // 0x90
    missing::<19> as *const (),      // 0x98
    missing::<20> as *const (),      // 0xa0
    create_stream_ssml as *const (), // 0xa8 from SSML
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    /// The host synthesizer speaks "Hello" into a non-empty WAVE stream,
    /// or leaves it a valid empty one when no voice is installed.
    #[test]
    fn streams_are_wave_files() {
        let this: This = &OBJECT;
        type Create = extern "system" fn(This, *mut *const u8) -> HRESULT;
        type Stream = extern "system" fn(This, *const u8, *const c_char, *mut *const u8) -> HRESULT;
        type Size = extern "system" fn(This, *const u8, *mut usize) -> HRESULT;
        type Data = extern "system" fn(This, *const u8, usize, *mut u8, *mut usize) -> HRESULT;
        let (mut synth, mut stream, mut size) = (std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let say = method::<Stream>(&OBJECT, 0x68);
        assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
        assert!(!stream.is_null() && stream != &raw const SYNTHESIZER);
        assert_eq!(method::<Size>(&OBJECT, 0x78)(this, stream, &mut size), S_OK);
        assert!(size >= 44);
        let mut wav = vec![0u8; size];
        let mut used = 0;
        let data = method::<Data>(&OBJECT, 0x80);
        assert_eq!(data(this, stream, size, wav.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(used, size);
        assert_eq!((&wav[..4], &wav[8..12]), (&b"RIFF"[..], &b"WAVE"[..]));
        // Chunks are not at fixed offsets (the host's stream may carry a
        // `fact` or `list` chunk and an extended `fmt `), so walk them.
        let (mut fmt, mut data) = (None, None);
        let mut at = 12;
        while at + 8 <= wav.len() {
            let id = &wav[at..at + 4];
            let len = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
            if id == b"fmt " {
                fmt = Some(at + 8);
            }
            if id == b"data" {
                data = Some(len);
            }
            at += 8 + len + (len & 1);
        }
        let fmt = fmt.expect("fmt chunk");
        assert_eq!(u16::from_le_bytes(wav[fmt..fmt + 2].try_into().unwrap()), 1); // PCM
        assert_eq!(
            u16::from_le_bytes(wav[fmt + 2..fmt + 4].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(wav[fmt + 4..fmt + 8].try_into().unwrap()),
            16_000
        );
        assert_eq!(
            u16::from_le_bytes(wav[fmt + 14..fmt + 16].try_into().unwrap()),
            16
        );
        assert!(data.expect("data chunk") > 0, "speech must not be silent");
        method::<extern "system" fn(This, *const u8)>(&OBJECT, 0x70)(this, stream);
    }
}
