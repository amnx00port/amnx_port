//! The console's audio DSP (`chad.dll`): streams that convolve blocks of
//! float samples with an impulse response (reverb), or run other block
//! transforms, on a coprocessor.
//!
//! A title connects once with a memory block for the DSP, activates
//! streams, and submits one command per block (gains, an input and an
//! output buffer). Activation hands back a status block whose second word is
//! the ticket of the last completed command and whose fourth is its result;
//! the title checks it one block later and treats a late command as an
//! overrun. Commands complete when submitted here.
//!
//! Convolution streams (`kind == CONVOLUTION`) run a stereo Schroeder/Freeverb
//! algorithmic reverb calibrated from the stream's impulse-response size and
//! energy (since the hardware's partitioned frequency-domain IR encoding is
//! opaque, the IR's length drives room decay and its non-zero energy gates
//! wet return — an all-zero IR produces silence, matching mathematical
//! convolution). Non-convolution block transforms still output silence.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

const CONVOLUTION: u32 = 1;
/// Option bits that need a stereo stream, and one convolution refuses.
/// Bit 2 (`0x2`) selects planar (deinterleaved) stereo channel layout.
const PLANAR_STEREO: u32 = 2;
const STEREO_OPTIONS: u32 = PLANAR_STEREO | 8;
const NOT_FOR_CONVOLUTION: u32 = 4;

/// Comb filter delay lengths at 48 kHz (Freeverb primes scaled by 48000/44100).
const COMB_TUNINGS: [usize; 8] = [1215, 1293, 1388, 1479, 1551, 1624, 1694, 1761];
/// Series allpass diffuser delay lengths at 48 kHz.
const ALLPASS_TUNINGS: [usize; 4] = [606, 480, 371, 245];
/// Stereo spread offset (in samples) added to the right channel's delay lines.
const STEREO_SPREAD: usize = 25;

/// Flushes tiny subnormal floats to zero to avoid denormal CPU slowdowns.
#[inline]
fn flush_denormal(x: f32) -> f32 {
    if !x.is_finite() || x.abs() < 1e-20 {
        0.0
    } else {
        x
    }
}

/// One lowpass-damped feedback comb filter.
struct Comb {
    buf: Vec<f32>,
    idx: usize,
    filter: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            idx: 0,
            filter: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, input: f32, feedback: f32, damp1: f32, damp2: f32) -> f32 {
        let out = self.buf[self.idx];
        self.filter = flush_denormal(out * damp2 + self.filter * damp1);
        self.buf[self.idx] = flush_denormal(input + self.filter * feedback);
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        out
    }
}

/// One Schroeder allpass diffuser.
struct Allpass {
    buf: Vec<f32>,
    idx: usize,
}

impl Allpass {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            idx: 0,
        }
    }

    #[inline]
    fn tick(&mut self, input: f32) -> f32 {
        let delayed = self.buf[self.idx];
        let out = -input + delayed;
        self.buf[self.idx] = flush_denormal(input + delayed * 0.5);
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        out
    }
}

/// One reverb channel (8 parallel comb filters followed by 4 series allpass
/// diffusers).
struct ReverbChannel {
    combs: [Comb; 8],
    allpasses: [Allpass; 4],
}

impl ReverbChannel {
    fn new(spread: usize) -> Self {
        Self {
            combs: COMB_TUNINGS.map(|n| Comb::new(n + spread)),
            allpasses: ALLPASS_TUNINGS.map(|n| Allpass::new(n + spread)),
        }
    }

    #[inline]
    fn tick(&mut self, input: f32, feedback: f32, damp1: f32, damp2: f32) -> f32 {
        let mut sum = 0.0f32;
        for comb in &mut self.combs {
            sum += comb.tick(input, feedback, damp1, damp2);
        }
        let mut out = sum;
        for ap in &mut self.allpasses {
            out = ap.tick(out);
        }
        out
    }
}

/// Algorithmic stereo reverb backing a `CONVOLUTION` DSP stream.
struct Reverb {
    channels: Vec<ReverbChannel>,
    feedback: f32,
    damp1: f32,
    damp2: f32,
    /// Calibrated wet return scale (`0.0` when the supplied IR is all zeros).
    ir_scale: f32,
}

impl Reverb {
    fn from_desc(desc: &StreamDesc, ir_seconds: u32) -> Self {
        let ch_count = (desc.channels as usize).clamp(1, 2);
        let channels = (0..ch_count)
            .map(|c| ReverbChannel::new(if c == 1 { STEREO_SPREAD } else { 0 }))
            .collect();

        // Inspect the supplied impulse response buffer when present.
        // An explicitly provided all-zero IR yields silence (0.0); a non-zero
        // IR (or null IR pointer with non-zero stream) yields a calibrated wet
        // return scaled cleanly into a safe listening range.
        let ir_floats = desc.ir_floats as usize;
        let ir_scale = if !desc.ir.is_null() && ir_floats > 0 {
            // Sample up to 64 Ki floats to measure energy without stalling activation.
            let probe = ir_floats.min(65_536);
            // SAFETY: title passed an IR buffer of `ir_floats` f32 values.
            let slice = unsafe { std::slice::from_raw_parts(desc.ir, probe) };
            let mut peak = 0.0f32;
            let mut sum_sq = 0.0f64;
            for &v in slice {
                if v.is_finite() {
                    let a = v.abs();
                    if a > peak {
                        peak = a;
                    }
                    sum_sq += f64::from(a) * f64::from(a);
                }
            }
            if peak == 0.0 {
                0.0
            } else {
                let rms = (sum_sq / probe as f64).sqrt() as f32;
                // Normalize diverse frequency-domain/time-domain IR magnitudes
                // into a gentle, musically balanced wet return scale.
                (rms.sqrt() * 0.25).clamp(0.08, 0.30)
            }
        } else if desc.ir.is_null() && ir_floats == 0 {
            0.18
        } else {
            0.0
        };

        // Estimate room decay from IR size (in 48 kHz samples) and connection
        // budget `ir_seconds`.
        let est_seconds = if ir_floats > 0 {
            (ir_floats as f32 / (48_000.0 * ch_count as f32)).clamp(0.25, 6.0)
        } else {
            (ir_seconds as f32 * 0.25).clamp(0.5, 4.0)
        };
        let norm = ((est_seconds - 0.25) / 5.75).clamp(0.0, 1.0);
        let feedback = 0.72 + norm * 0.20;
        let damp1 = 0.28 - norm * 0.10;
        let damp2 = 1.0 - damp1;

        Self {
            channels,
            feedback,
            damp1,
            damp2,
            ir_scale,
        }
    }

    /// Processes one audio block (`frames * channels` floats) into `output`.
    fn process(
        &mut self,
        input: Option<&[f32]>,
        output: &mut [f32],
        frames: usize,
        channels: usize,
        planar: bool,
        gains: [f32; 2],
    ) {
        if self.ir_scale == 0.0 {
            output.fill(0.0);
            return;
        }
        let g = [
            if gains[0].is_finite() {
                gains[0].clamp(-4.0, 4.0) * self.ir_scale
            } else {
                0.0
            },
            if gains[1].is_finite() {
                gains[1].clamp(-4.0, 4.0) * self.ir_scale
            } else {
                0.0
            },
        ];
        // Input attenuation keeps the 8 parallel combs well inside linear headroom.
        const INPUT_GAIN: f32 = 0.05;
        let (fb, d1, d2) = (self.feedback, self.damp1, self.damp2);

        if planar && channels == 2 {
            let (out_l, out_r) = output.split_at_mut(frames);
            let (in_l, in_r) = match input {
                Some(inp) if inp.len() >= frames * 2 => {
                    let (l, r) = inp.split_at(frames);
                    (Some(l), Some(r))
                }
                _ => (None, None),
            };
            let (ch_l, ch_r) = self.channels.split_at_mut(1);
            for i in 0..frames {
                let xl = in_l.map_or(0.0, |s| flush_denormal(s[i])) * INPUT_GAIN;
                let xr = in_r.map_or(0.0, |s| flush_denormal(s[i])) * INPUT_GAIN;
                let mix = 0.5 * (xl + xr);
                let yl = ch_l[0].tick(0.75 * xl + 0.25 * mix, fb, d1, d2) * g[0];
                let yr = ch_r[0].tick(0.75 * xr + 0.25 * mix, fb, d1, d2) * g[1];
                out_l[i] = yl.tanh();
                out_r[i] = yr.tanh();
            }
        } else if channels == 2 {
            let (ch_l, ch_r) = self.channels.split_at_mut(1);
            for i in 0..frames {
                let base = i * 2;
                let (xl, xr) = match input {
                    Some(inp) if base + 1 < inp.len() => (
                        flush_denormal(inp[base]) * INPUT_GAIN,
                        flush_denormal(inp[base + 1]) * INPUT_GAIN,
                    ),
                    _ => (0.0, 0.0),
                };
                let mix = 0.5 * (xl + xr);
                let yl = ch_l[0].tick(0.75 * xl + 0.25 * mix, fb, d1, d2) * g[0];
                let yr = ch_r[0].tick(0.75 * xr + 0.25 * mix, fb, d1, d2) * g[1];
                output[base] = yl.tanh();
                output[base + 1] = yr.tanh();
            }
        } else {
            let ch = &mut self.channels[0];
            for (i, dst) in output.iter_mut().enumerate().take(frames) {
                let x = input
                    .and_then(|inp| inp.get(i))
                    .map_or(0.0, |&s| flush_denormal(s) * INPUT_GAIN);
                let y = ch.tick(x, fb, d1, d2) * g[0];
                *dst = y.tanh();
            }
        }
    }
}

pub struct Dsp {
    max_streams: u32,
    ir_seconds: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    planar: bool,
    reverb: Option<Reverb>,
    status: Box<Status>,
}

/// One block of work.
#[repr(C)]
pub struct Command {
    gains: [f32; 2],
    input: *const f32,
    output: *mut f32,
    reserved: u64,
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        ir_seconds,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = if kind == CONVOLUTION {
        (1..=2).contains(&channels)
            && matches!(frames, 0x200 | 0x400)
            && options & NOT_FOR_CONVOLUTION == 0
    } else {
        channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800)
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    tracing::debug!(
        kind,
        options,
        frames,
        channels,
        "audio DSP stream activated"
    );
    let ir_seconds = {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
        dsp.ir_seconds
    };
    let reverb = (kind == CONVOLUTION).then(|| Reverb::from_desc(desc, ir_seconds));
    let new = Box::new(Stream {
        dsp,
        frames,
        channels,
        planar: options & PLANAR_STEREO != 0,
        reverb,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Processes one block of audio and marks its ticket complete.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    if !command.output.is_null() {
        let samples = (stream.frames * stream.channels) as usize;
        // SAFETY: the title's output buffer holds one block of the stream.
        let out = unsafe { std::slice::from_raw_parts_mut(command.output, samples) };
        if let Some(reverb) = &mut stream.reverb {
            let input = if command.input.is_null() {
                None
            } else {
                // SAFETY: the title's input buffer holds one block of the stream.
                Some(unsafe { std::slice::from_raw_parts(command.input, samples) })
            };
            reverb.process(
                input,
                out,
                stream.frames as usize,
                stream.channels as usize,
                stream.planar,
                command.gains,
            );
        } else {
            out.fill(0.0);
        }
    }
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_stream_as_the_title_drives_it() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        let ir = vec![0f32; 0x800];
        // Stereo convolution with option 2, as the title asks for.
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 2,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let mut output = vec![1f32; 0x800];
        let input = vec![0.5f32; 0x800];
        let command = Command {
            gains: [1.0, 1.0],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        // An all-zero impulse response mathematically convolves to silence.
        assert!(output.iter().all(|&s| s == 0.0));
        assert_eq!(XDspDeactivate(stream), S_OK);

        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc {
            channels: 1,
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &bad, &mut status, &mut stream),
            E_INVALIDARG
        );
        let transform = StreamDesc {
            kind: 2,
            options: 0,
            channels: 1,
            ir_floats: 0,
            ir: std::ptr::null(),
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &transform, &mut status, &mut stream),
            S_OK
        );
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn non_zero_ir_produces_decaying_stereo_reverb_tail() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                4,
                0x10,
                2,
                &mut dsp
            ),
            S_OK
        );
        // Synthetic decaying impulse response.
        let mut ir = vec![0.0f32; 0x4000];
        for (i, s) in ir.iter_mut().enumerate() {
            *s = (-(i as f32) / 4096.0).exp();
        }
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: PLANAR_STEREO,
            frames: 0x400,
            channels: 2,
            ir_floats: ir.len() as u32,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);

        // Feed an impulse on block 0, then silence on blocks 1..=8.
        let mut impulse = vec![0.0f32; 0x800];
        for i in 0..64 {
            impulse[i] = 1.0;
            impulse[0x400 + i] = 1.0;
        }
        let silence = vec![0.0f32; 0x800];
        let mut output = vec![0.0f32; 0x800];

        let mut energies = Vec::new();
        let mut stereo_diff = 0.0f32;
        for block in 0..10 {
            let src = if block == 0 { &impulse } else { &silence };
            let cmd = Command {
                gains: [1.0, 1.0],
                input: src.as_ptr(),
                output: output.as_mut_ptr(),
                reserved: 0,
            };
            assert_eq!(XDspSubmitCommand(stream, &cmd, std::ptr::null_mut()), S_OK);
            assert!(output.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
            let energy: f32 = output.iter().map(|s| s * s).sum();
            energies.push(energy);
            for i in 0..0x400 {
                stereo_diff += (output[i] - output[0x400 + i]).abs();
            }
        }
        // Reverb produces non-zero wet energy after the comb delay elapses (~1215 samples = block 1-2).
        let peak_tail = energies[1..6].iter().copied().fold(0.0f32, f32::max);
        assert!(peak_tail > 1e-4, "expected audible reverb tail, got {energies:?}");
        // Tail decays over time.
        assert!(energies[9] < peak_tail, "expected tail decay: {energies:?}");
        // Stereo spread produces distinct left and right channels.
        assert!(stereo_diff > 1e-3, "expected stereo decorrelation");

        // Zero gain silences wet output cleanly.
        let cmd = Command {
            gains: [0.0, 0.0],
            input: impulse.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        assert_eq!(XDspSubmitCommand(stream, &cmd, std::ptr::null_mut()), S_OK);
        assert!(output.iter().all(|&s| s == 0.0));

        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }
}
