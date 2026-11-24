//! XMA2, the console's hardware audio codec: mono or stereo WMA Pro frames of
//! 512 samples per channel, packed into 2048-byte packets.
//!
//! A packet starts with a 32-bit header (MSB first): the number of frames
//! that begin in it (6 bits), the number of bits that finish the previous
//! frame before the first new one (15), metadata (3) and how many packets of
//! other streams follow before this stream's next packet (8). Every frame
//! starts with its length in bits (15) and ends with a "more frames" bit.
//!
//! The frame decoder follows FFmpeg's `wmaprodec.c` (LGPL-2.1-or-later, used
//! under GPL-3.0-or-later) restricted to what XMA needs: at most two
//! channels, four subframe sizes (512 down to 128), no LFE channel.

mod tables;

use std::sync::OnceLock;

use tables::*;

pub const PACKET_BYTES: usize = 2048;
const PACKET_BITS: usize = PACKET_BYTES * 8;
const HEADER_BITS: usize = 32;
/// Samples per channel in one frame.
pub const FRAME_SAMPLES: usize = 512;
/// Samples per channel in one subframe, the unit of skips and loop ends.
pub const SUBFRAME_SAMPLES: usize = 128;
const LENGTH_BITS: u32 = 15;
/// 512 >> 2: the smallest subframe.
const MIN_SUBFRAME: usize = SUBFRAME_SAMPLES;
const BLOCK_SIZES: usize = 3;
const MAX_BANDS: usize = 29;
const MAX_SUBFRAMES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XMA: {}", self.0)
    }
}

impl std::error::Error for Error {}

// Packets.

/// A packet's header fields.
#[derive(Debug, Clone, Copy)]
pub struct Packet {
    pub frames: u32,
    pub first_frame_bit: u32,
    pub skip: u32,
}

pub fn packet(data: &[u8], index: usize) -> Option<Packet> {
    let start = index.checked_mul(PACKET_BYTES)?;
    let b = data.get(start..)?.get(..4)?;
    let h = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    Some(Packet {
        frames: h >> 26,
        first_frame_bit: (h >> 11) & 0x7fff,
        skip: h & 0xff,
    })
}

/// One frame's bits gathered out of its packets.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The frame from its length field on, MSB first.
    pub data: Vec<u8>,
    pub bits: usize,
    /// Bit offset in the buffer where it starts, and where the next frame
    /// starts (`None` when the buffer has no further frame).
    pub start: usize,
    pub next: Option<usize>,
}

/// Bit positions in a buffer of packets, skipping each packet's header and
/// the packets of other streams.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn bit(&mut self) -> Option<u32> {
        if self.pos.is_multiple_of(PACKET_BITS) {
            // Leaving a packet: jump over the other streams' packets.
            let previous = self.pos / PACKET_BITS;
            if previous > 0 {
                let skip = packet(self.data, previous - 1)?.skip as usize;
                self.pos = (previous + skip) * PACKET_BITS;
            }
            self.pos += HEADER_BITS;
        }
        let byte = *self.data.get(self.pos / 8)?;
        let bit = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(bit as u32)
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        (0..n).try_fold(0, |v, _| Some(v << 1 | self.bit()?))
    }
}

/// Reads the frame starting at bit `start` (a frame start inside a packet's
/// payload) of `data`, a buffer of whole packets of one stream.
pub fn read_frame(data: &[u8], start: usize) -> Option<Frame> {
    let mut cursor = Cursor { data, pos: start };
    let bits = cursor.bits(LENGTH_BITS)? as usize;
    if bits <= LENGTH_BITS as usize || bits == 0x7fff {
        return None;
    }
    let mut out = vec![0u8; bits.div_ceil(8)];
    let mut cursor = Cursor { data, pos: start };
    for i in 0..bits {
        out[i / 8] |= (cursor.bit()? as u8) << (7 - i % 8);
    }
    let end = cursor.pos;
    let next = next_frame(data, start, end);
    Some(Frame {
        data: out,
        bits,
        start,
        next,
    })
}

/// Where the frame after the one at `start`..`end` begins: right after it,
/// unless it was the last frame to begin in its packet and the rest of that
/// packet is padding.
fn next_frame(data: &[u8], start: usize, end: usize) -> Option<usize> {
    let index = start / PACKET_BITS;
    let header = packet(data, index)?;
    let mut at = index * PACKET_BITS + HEADER_BITS + header.first_frame_bit as usize;
    let mut number = 0;
    while at < start {
        let mut cursor = Cursor { data, pos: at };
        let bits = cursor.bits(LENGTH_BITS)? as usize;
        // The same rule as `read_frame`: a length that does not cover its own
        // field is not a frame. A zero length would never move `at`, so a
        // damaged packet would loop here for good.
        if bits <= LENGTH_BITS as usize {
            return None;
        }
        at += bits;
        number += 1;
    }
    let last = number + 1 >= header.frames;
    let mut end = end;
    if end.is_multiple_of(PACKET_BITS) || (last && end / PACKET_BITS == index) {
        // The next frame begins in the next packet of this stream.
        let next = (end - 1) / PACKET_BITS + 1 + packet(data, (end - 1) / PACKET_BITS)?.skip as usize;
        let header = packet(data, next)?;
        end = next * PACKET_BITS + HEADER_BITS + header.first_frame_bit as usize;
    }
    (end / 8 < data.len()).then_some(end)
}

// Bitstream reading inside a frame.

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl Bits<'_> {
    fn peek32(&self) -> u32 {
        let mut v = 0u64;
        let byte = self.pos / 8;
        for i in 0..5 {
            v = v << 8 | *self.data.get(byte + i).unwrap_or(&0) as u64;
        }
        (v << (24 + self.pos % 8) >> 32) as u32
    }

    fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek32() >> (32 - n);
        self.pos += n as usize;
        v
    }

    fn bit(&mut self) -> bool {
        self.read(1) != 0
    }

    fn signed(&mut self, n: u32) -> i32 {
        ((self.read(n) << (32 - n)) as i32) >> (32 - n)
    }

    fn consumed(&self) -> usize {
        self.pos
    }

    fn overread(&self) -> bool {
        self.pos > self.end
    }
}

/// A prefix code built like FFmpeg's `ff_vlc_init_from_lengths`: codes are
/// handed out in table order, each the next free one of its length.
struct Vlc {
    /// Left-aligned code, length, symbol; sorted by code.
    codes: Vec<(u32, u8, i32)>,
}

impl Vlc {
    fn new(entries: impl Iterator<Item = (i32, u8)>) -> Vlc {
        let mut code = 0u64;
        let mut codes = Vec::new();
        for (symbol, len) in entries {
            codes.push((code as u32, len, symbol));
            code += 1 << (32 - len);
        }
        Vlc { codes }
    }

    fn read(&self, bits: &mut Bits) -> i32 {
        let v = bits.peek32();
        let i = self.codes.partition_point(|c| c.0 <= v) - 1;
        let (_, len, symbol) = self.codes[i];
        bits.pos += len as usize;
        symbol
    }
}

struct Codes {
    scale: Vlc,
    scale_rl: Vlc,
    coef: [Vlc; 2],
    vec4: Vlc,
    vec2: Vlc,
    vec1: Vlc,
}

fn codes() -> &'static Codes {
    static CODES: OnceLock<Codes> = OnceLock::new();
    CODES.get_or_init(|| {
        let pairs = |t: &'static [(u16, u8)], offset: i32| {
            Vlc::new(t.iter().map(move |&(s, l)| (s as i32 + offset, l)))
        };
        Codes {
            scale: pairs(&SCALE, -60),
            scale_rl: pairs(&SCALE_RL, 0),
            coef: [
                Vlc::new(COEF0_SYMS.iter().zip(COEF0_LENS).map(|(&s, l)| (s as i32, l))),
                pairs(&COEF1, 0),
            ],
            vec4: Vlc::new(VEC4_SYMS.iter().zip(VEC4_LENS).map(|(&s, l)| (s as i32 - 1, l))),
            vec2: pairs(&VEC2, -1),
            vec1: pairs(&VEC1, 0),
        }
    })
}

/// `ff_wma_get_large_val`.
fn large_value(bits: &mut Bits) -> u32 {
    let mut n = 8;
    if bits.bit() {
        n += 8;
        if bits.bit() {
            n += 8;
            if bits.bit() {
                n += 7;
            }
        }
    }
    bits.read(n)
}

fn log2(v: usize) -> u32 {
    usize::BITS - 1 - v.max(1).leading_zeros()
}

// The inverse transform.

/// Half-length inverse MDCT as FFmpeg defines it (`av_tx` MDCT, inverse):
/// `n` coefficients to the middle `n` samples of the 2n-sample window,
/// negated and scaled by `scale`. Computed as a DCT-IV through an
/// n/2-point complex FFT.
fn imdct_half(input: &[f32], out: &mut [f32], scale: f32) {
    let n = input.len();
    let m = n / 2;
    let mut re = vec![0f64; m];
    let mut im = vec![0f64; m];
    for k in 0..m {
        let (a, b) = (input[2 * k] as f64, input[n - 1 - 2 * k] as f64);
        let (s, c) = (-std::f64::consts::PI * k as f64 / n as f64).sin_cos();
        (re[k], im[k]) = (a * c - b * s, a * s + b * c);
    }
    fft(&mut re, &mut im);
    for k in 0..m {
        let (s, c) = (-std::f64::consts::PI * (4 * k + 1) as f64 / (4 * n) as f64).sin_cos();
        let (wr, wi) = (re[k] * c - im[k] * s, re[k] * s + im[k] * c);
        // DCT-IV: c[2k] = Re, c[n-1-2k] = -Im; output j is c[n-1-j].
        out[n - 1 - 2 * k] = (wr as f32) * scale;
        out[2 * k] = -(wi as f32) * scale;
    }
}

/// In-place radix-2 forward DFT (`exp(-2πi nk/N)`).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let (ws, wc) = (-2.0 * std::f64::consts::PI / len as f64).sin_cos();
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                (re[b], im[b]) = (re[a] - tr, im[a] - ti);
                (re[a], im[a]) = (re[a] + tr, im[a] + ti);
                (cr, ci) = (cr * wc - ci * ws, cr * ws + ci * wc);
            }
        }
        len <<= 1;
    }
}

/// `ff_sine_windows`: the rising half of a sine window of length `n`.
fn sine_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f64 + 0.5) * std::f64::consts::PI / (2.0 * n as f64)).sin() as f32)
        .collect()
}

// The frame decoder.

#[derive(Clone)]
struct Channel {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [usize; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: usize,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    /// Which of the two saved sets is in use for this subframe.
    scale_factors: usize,
    table_idx: usize,
    num_vec_coeffs: usize,
    coeffs: [f32; FRAME_SAMPLES],
    /// The frame's output, then half a frame of overlap for the next one.
    out: [f32; FRAME_SAMPLES + FRAME_SAMPLES / 2],
}

impl Channel {
    fn new() -> Self {
        Channel {
            prev_block_len: FRAME_SAMPLES,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 1,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            scale_factors: 0,
            table_idx: 0,
            num_vec_coeffs: 0,
            coeffs: [0.0; FRAME_SAMPLES],
            out: [0.0; FRAME_SAMPLES + FRAME_SAMPLES / 2],
        }
    }
}

/// Decodes the frames of one mono or stereo stream.
#[derive(Clone)]
pub struct Decoder {
    channels: usize,
    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[usize; MAX_BANDS + 1]; BLOCK_SIZES],
    sf_offsets: [[[usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],
    ch: [Channel; 2],
    // Per subframe.
    subframe_len: usize,
    channels_for_cur_subframe: usize,
    channel_indexes: [usize; 2],
    num_bands: usize,
    table_idx: usize,
    esc_len: u32,
    transmit_num_vec_coeffs: bool,
    transform: bool,
    transform_band: [bool; MAX_BANDS],
    parsed_all_subframes: bool,
}

/// What the decoder needs from the stream format: the channel count and
/// the sample rate, which sets the scale factor band layout.
impl Decoder {
    pub fn new(channels: usize, sample_rate: u32) -> Decoder {
        let channels = channels.clamp(1, 2);
        let rate = match sample_rate {
            r if r > 44_100 => 48_000,
            r if r > 32_000 => 44_100,
            r if r > 24_000 => 32_000,
            _ => 24_000,
        };
        let mut sfb_offsets = [[0usize; MAX_BANDS + 1]; BLOCK_SIZES];
        let mut num_sfb = [0usize; BLOCK_SIZES];
        for i in 0..BLOCK_SIZES {
            let subframe_len = FRAME_SAMPLES >> i;
            let offsets = &mut sfb_offsets[i];
            let mut band = 1;
            for &freq in CRITICAL_FREQ.iter().take(MAX_BANDS - 1) {
                if offsets[band - 1] >= subframe_len {
                    break;
                }
                let offset = ((subframe_len * 2 * freq as usize) / rate + 2) & !3;
                if offset > offsets[band - 1] {
                    offsets[band] = offset;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
            }
            offsets[band - 1] = subframe_len;
            num_sfb[i] = band - 1;
        }
        let mut sf_offsets = [[[0usize; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES];
        for i in 0..BLOCK_SIZES {
            for b in 0..num_sfb[i] {
                let offset = ((sfb_offsets[i][b] + sfb_offsets[i][b + 1] - 1) << i) >> 1;
                for x in 0..BLOCK_SIZES {
                    let mut v = 0;
                    while v + 1 < MAX_BANDS && (sfb_offsets[x][v + 1] << x) < offset {
                        v += 1;
                    }
                    sf_offsets[i][x][b] = v;
                }
            }
        }
        Decoder {
            channels,
            num_sfb,
            sfb_offsets,
            sf_offsets,
            ch: [Channel::new(), Channel::new()],
            subframe_len: FRAME_SAMPLES,
            channels_for_cur_subframe: 0,
            channel_indexes: [0, 1],
            num_bands: 0,
            table_idx: 0,
            esc_len: 0,
            transmit_num_vec_coeffs: false,
            transform: false,
            transform_band: [false; MAX_BANDS],
            parsed_all_subframes: false,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Forgets the overlap with earlier frames (a new stream or a seek).
    pub fn reset(&mut self) {
        self.ch = [Channel::new(), Channel::new()];
    }

    /// Decodes one frame (`frame.data`, `frame.bits`) and appends
    /// `FRAME_SAMPLES` interleaved samples per channel to `out`. A `bits`
    /// past the end of `data` counts as the end of `data`.
    pub fn decode(&mut self, data: &[u8], bits: usize, out: &mut Vec<i16>) -> Result<(), Error> {
        // Reads past `data` give zeros, and a stereo tile header of zeros
        // never advances: only running past `end` stops it.
        let mut b = Bits {
            data,
            pos: 0,
            end: bits.min(data.len().saturating_mul(8)),
        };
        let len = b.read(LENGTH_BITS) as usize;
        let result = self.frame(&mut b, len);
        if result.is_err() {
            // Whatever the frame held is lost; keep the stream's timing.
            for c in 0..self.channels {
                self.ch[c].out = [0.0; FRAME_SAMPLES + FRAME_SAMPLES / 2];
            }
        }
        for i in 0..FRAME_SAMPLES {
            for c in 0..self.channels {
                out.push(to_i16(self.ch[c].out[i]));
            }
        }
        for c in 0..self.channels {
            self.ch[c].out.copy_within(FRAME_SAMPLES.., 0);
        }
        result
    }

    /// The overlap left after the last frame, then silence: half a frame of
    /// samples per channel.
    pub fn flush(&mut self, out: &mut Vec<i16>) {
        for i in 0..FRAME_SAMPLES / 2 {
            for c in 0..self.channels {
                out.push(to_i16(self.ch[c].out[i]));
            }
        }
        self.reset();
    }

    fn frame(&mut self, b: &mut Bits, len: usize) -> Result<(), Error> {
        self.tile_header(b)?;
        // Post-processing transform.
        if self.channels > 1 && b.bit() && b.bit() {
            b.read((self.channels * self.channels * 4) as u32);
        }
        // Dynamic range compression gain, unused.
        b.read(8);
        if b.bit() {
            // Trim at start and end, which only WMA Pro files apply.
            if b.bit() {
                b.read(log2(FRAME_SAMPLES * 2));
            }
            if b.bit() {
                b.read(log2(FRAME_SAMPLES * 2));
            }
        }
        self.parsed_all_subframes = false;
        for c in 0..self.channels {
            let ch = &mut self.ch[c];
            ch.decoded_samples = 0;
            ch.cur_subframe = 0;
            ch.reuse_sf = false;
        }
        while !self.parsed_all_subframes {
            self.subframe(b)?;
        }
        // FFmpeg flags `len != consumed + 2` but keeps the samples; so do we.
        if b.overread() || b.consumed() > len {
            return Err(Error("frame overread"));
        }
        Ok(())
    }

    fn subframe_length(&self, b: &mut Bits, offset: usize) -> Result<usize, Error> {
        if offset == FRAME_SAMPLES - MIN_SUBFRAME {
            return Ok(MIN_SUBFRAME);
        }
        // Four subframe sizes: a 1-bit "not maximal", then 1 bit of shift.
        let shift = if b.bit() { 1 + b.read(1) } else { 0 };
        let len = FRAME_SAMPLES >> shift;
        if len < MIN_SUBFRAME {
            return Err(Error("subframe length"));
        }
        Ok(len)
    }

    fn tile_header(&mut self, b: &mut Bits) -> Result<(), Error> {
        let mut num_samples = [0usize; 2];
        let mut contains = [false; 2];
        let mut channels_for_cur_subframe = self.channels;
        let mut min_channel_len = 0;
        for c in 0..self.channels {
            self.ch[c].num_subframes = 0;
        }
        let fixed = b.bit();
        loop {
            for c in 0..self.channels {
                contains[c] = num_samples[c] == min_channel_len
                    && (fixed
                        || channels_for_cur_subframe == 1
                        || min_channel_len == FRAME_SAMPLES - MIN_SUBFRAME
                        || b.bit());
            }
            let subframe_len = self.subframe_length(b, min_channel_len)?;
            min_channel_len += subframe_len;
            for c in 0..self.channels {
                let ch = &mut self.ch[c];
                if contains[c] {
                    if ch.num_subframes >= MAX_SUBFRAMES {
                        return Err(Error("too many subframes"));
                    }
                    ch.subframe_len[ch.num_subframes] = subframe_len;
                    num_samples[c] += subframe_len;
                    ch.num_subframes += 1;
                    if num_samples[c] > FRAME_SAMPLES {
                        return Err(Error("channel longer than a frame"));
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur_subframe = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur_subframe += 1;
                }
            }
            if min_channel_len >= FRAME_SAMPLES || b.overread() {
                break;
            }
        }
        if b.overread() {
            return Err(Error("tile header"));
        }
        Ok(())
    }

    fn channel_transform(&mut self, b: &mut Bits) -> Result<(), Error> {
        self.transform = false;
        if self.channels == 1 {
            return Ok(());
        }
        if b.bit() {
            return Err(Error("channel transform bit"));
        }
        // One group holding every channel of the subframe.
        if self.channels_for_cur_subframe == 2 {
            if b.bit() {
                if b.bit() {
                    return Err(Error("channel transform type"));
                }
            } else {
                self.transform = true;
            }
        }
        if self.transform {
            if !b.bit() {
                for i in 0..self.num_bands {
                    self.transform_band[i] = b.bit();
                }
            } else {
                self.transform_band = [true; MAX_BANDS];
            }
        }
        Ok(())
    }

    fn coefficients(&mut self, b: &mut Bits, c: usize) -> Result<(), Error> {
        let codes = codes();
        let table = b.bit() as usize;
        let (run, level): (&[u16], &[f32]) = if table == 1 {
            (&COEF1_RUN, &COEF1_LEVEL)
        } else {
            (&COEF0_RUN, &COEF0_LEVEL)
        };
        let subframe_len = self.subframe_len;
        let ch = &mut self.ch[c];
        let mut rl_mode = false;
        let mut cur = 0;
        let mut num_zeros = 0;
        while (self.transmit_num_vec_coeffs || !rl_mode) && cur + 3 < ch.num_vec_coeffs {
            let mut vals = [0u32; 4];
            let idx = codes.vec4.read(b);
            if idx < 0 {
                for i in (0..4).step_by(2) {
                    let idx = codes.vec2.read(b);
                    if idx < 0 {
                        for v in &mut vals[i..i + 2] {
                            let mut x = codes.vec1.read(b) as u32;
                            if x as usize == VEC1.len() - 1 {
                                x += large_value(b);
                            }
                            *v = x;
                        }
                    } else {
                        vals[i] = (idx >> 4) as u32;
                        vals[i + 1] = (idx & 0xf) as u32;
                    }
                }
            } else {
                vals = [
                    (idx >> 12) as u32,
                    ((idx >> 8) & 0xf) as u32,
                    ((idx >> 4) & 0xf) as u32,
                    (idx & 0xf) as u32,
                ];
            }
            for v in vals {
                if v != 0 {
                    let negative = !b.bit();
                    ch.coeffs[cur] = if negative { -(v as f32) } else { v as f32 };
                    num_zeros = 0;
                } else {
                    ch.coeffs[cur] = 0.0;
                    num_zeros += 1;
                    rl_mode |= num_zeros > subframe_len >> 8;
                }
                cur += 1;
            }
            if b.overread() {
                return Err(Error("vector coefficients"));
            }
        }
        if cur < subframe_len {
            ch.coeffs[cur..subframe_len].fill(0.0);
            // `ff_wma_run_level_decode`, version 1.
            let mask = subframe_len - 1;
            let mut offset = cur;
            while offset < subframe_len {
                let code = codes.coef[table].read(b);
                if code > 1 {
                    offset += run[code as usize] as usize;
                    let negative = !b.bit();
                    let v = level[code as usize];
                    ch.coeffs[offset & mask] = if negative { -v } else { v };
                } else if code == 1 {
                    break;
                } else {
                    let value = large_value(b) as i32;
                    if b.bit() {
                        if b.bit() {
                            if b.bit() {
                                // A broken escape ends the channel's
                                // coefficients, as in FFmpeg.
                                break;
                            }
                            offset += b.read(self.esc_len) as usize + 4;
                        } else {
                            offset += b.read(2) as usize + 1;
                        }
                    }
                    let negative = !b.bit();
                    ch.coeffs[offset & mask] = if negative { -value } else { value } as f32;
                }
                if b.overread() {
                    return Err(Error("run level coefficients"));
                }
                offset += 1;
            }
        }
        Ok(())
    }

    fn scale_factors(&mut self, b: &mut Bits) -> Result<(), Error> {
        let codes = codes();
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes[i];
            let sf_offsets = self.sf_offsets[self.table_idx][self.ch[c].table_idx];
            let ch = &mut self.ch[c];
            let target = 1 - ch.scale_factor_idx;
            ch.scale_factors = target;
            if ch.reuse_sf {
                let from = ch.saved_scale_factors[ch.scale_factor_idx];
                for (sf, &offset) in ch.saved_scale_factors[target].iter_mut().zip(&sf_offsets).take(self.num_bands) {
                    *sf = from[offset];
                }
            }
            if ch.cur_subframe == 0 || b.bit() {
                if !ch.reuse_sf {
                    ch.scale_factor_step = b.read(2) as i32 + 1;
                    let mut val = 45 / ch.scale_factor_step;
                    for band in 0..self.num_bands {
                        val += codes.scale.read(b);
                        ch.saved_scale_factors[target][band] = val;
                    }
                } else {
                    let mut band = 0;
                    while band < self.num_bands {
                        let idx = codes.scale_rl.read(b);
                        let (skip, val, negative) = match idx {
                            0 => {
                                let code = b.read(14);
                                ((code & 0x3f) >> 1, (code >> 6) as i32, code & 1 == 0)
                            }
                            1 => break,
                            _ => (
                                SCALE_RL_RUN[idx as usize] as u32,
                                SCALE_RL_LEVEL[idx as usize] as i32,
                                !b.bit(),
                            ),
                        };
                        band += skip as usize;
                        if band >= self.num_bands {
                            return Err(Error("scale factor run"));
                        }
                        ch.saved_scale_factors[target][band] += if negative { -val } else { val };
                        band += 1;
                    }
                }
                ch.scale_factor_idx = target;
                ch.table_idx = self.table_idx;
                ch.reuse_sf = true;
            }
            let sf = &ch.saved_scale_factors[ch.scale_factors][..self.num_bands];
            ch.max_scale_factor = sf.iter().copied().max().unwrap_or(0);
        }
        Ok(())
    }

    fn subframe(&mut self, b: &mut Bits) -> Result<(), Error> {
        let mut offset = FRAME_SAMPLES;
        let mut subframe_len = FRAME_SAMPLES;
        let mut total_samples = FRAME_SAMPLES * self.channels;
        for c in 0..self.channels {
            let ch = &self.ch[c];
            if offset > ch.decoded_samples {
                offset = ch.decoded_samples;
                subframe_len = ch.subframe_len[ch.cur_subframe.min(MAX_SUBFRAMES - 1)];
            }
        }
        self.channels_for_cur_subframe = 0;
        for c in 0..self.channels {
            let ch = &mut self.ch[c];
            total_samples -= ch.decoded_samples;
            if offset == ch.decoded_samples
                && ch.cur_subframe < ch.num_subframes
                && subframe_len == ch.subframe_len[ch.cur_subframe]
            {
                total_samples -= subframe_len;
                ch.decoded_samples += subframe_len;
                self.channel_indexes[self.channels_for_cur_subframe] = c;
                self.channels_for_cur_subframe += 1;
            }
        }
        if self.channels_for_cur_subframe == 0 {
            return Err(Error("no channel in subframe"));
        }
        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }
        self.table_idx = log2(FRAME_SAMPLES / subframe_len) as usize;
        self.num_bands = self.num_sfb[self.table_idx];
        self.subframe_len = subframe_len;
        self.esc_len = log2(subframe_len - 1) + 1;

        // Extended header: fill bits.
        if b.bit() {
            let mut fill = b.read(2) as usize;
            if fill == 0 {
                let len = b.read(4);
                fill = b.read(len) as usize + 1;
            }
            b.pos += fill;
        }
        if b.bit() {
            return Err(Error("reserved bit"));
        }
        self.channel_transform(b)?;

        let mut transmit_coeffs = false;
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes[i];
            self.ch[c].transmit_coefs = b.bit();
            transmit_coeffs |= self.ch[c].transmit_coefs;
        }
        if transmit_coeffs {
            self.transmit_num_vec_coeffs = b.bit();
            if self.transmit_num_vec_coeffs {
                let num_bits = log2(subframe_len.div_ceil(4)) + 1;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes[i];
                    let n = (b.read(num_bits) << 2) as usize;
                    if n > subframe_len {
                        return Err(Error("vector coefficient count"));
                    }
                    self.ch[c].num_vec_coeffs = n;
                }
            } else {
                for i in 0..self.channels_for_cur_subframe {
                    self.ch[self.channel_indexes[i]].num_vec_coeffs = subframe_len;
                }
            }
            // Quantisation step (16-bit samples: 90 * 16 >> 4).
            let mut quant_step = 90;
            let mut step = b.signed(6);
            quant_step += step;
            if step == -32 || step == 31 {
                let negative = step == -32;
                let mut quant = 0;
                while b.pos + 5 < b.end {
                    step = b.read(5) as i32;
                    if step != 31 {
                        break;
                    }
                    quant += 31;
                }
                quant_step += if negative { -(quant + step) } else { quant + step };
            }
            if self.channels_for_cur_subframe == 1 {
                self.ch[self.channel_indexes[0]].quant_step = quant_step;
            } else {
                let modifier_len = b.read(3);
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes[i];
                    self.ch[c].quant_step = quant_step;
                    if b.bit() {
                        self.ch[c].quant_step += if modifier_len > 0 {
                            b.read(modifier_len) as i32 + 1
                        } else {
                            1
                        };
                    }
                }
            }
            self.scale_factors(b)?;
        }

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes[i];
            if self.ch[c].transmit_coefs && b.pos < b.end {
                self.coefficients(b, c)?;
            } else {
                self.ch[c].coeffs[..subframe_len].fill(0.0);
            }
        }

        // Where this subframe's samples go in the channel output.
        let at = offset + FRAME_SAMPLES / 2;
        if transmit_coeffs {
            self.inverse_channel_transform();
            // Output in [-1, 1] for 16-bit streams.
            let scale = 2.0 / subframe_len as f32 / 32768.0;
            let sfb = self.sfb_offsets[self.table_idx];
            let mut tmp = [0f32; FRAME_SAMPLES];
            for i in 0..self.channels_for_cur_subframe {
                let ch = &mut self.ch[self.channel_indexes[i]];
                let sf = ch.saved_scale_factors[ch.scale_factors];
                for band in 0..self.num_bands {
                    let end = sfb[band + 1].min(subframe_len);
                    let exp = ch.quant_step - (ch.max_scale_factor - sf[band]) * ch.scale_factor_step;
                    let quant = 10f32.powf(exp as f32 / 20.0);
                    for (t, &c) in tmp[sfb[band]..end].iter_mut().zip(&ch.coeffs[sfb[band]..end]) {
                        *t = c * quant;
                    }
                }
                imdct_half(&tmp[..subframe_len], &mut ch.out[at..at + subframe_len], scale);
            }
        } else {
            for i in 0..self.channels_for_cur_subframe {
                self.ch[self.channel_indexes[i]].out[at..at + subframe_len].fill(0.0);
            }
        }

        // Window and overlap-add with the previous block.
        for i in 0..self.channels_for_cur_subframe {
            let ch = &mut self.ch[self.channel_indexes[i]];
            let mut winlen = ch.prev_block_len;
            let mut start = at - winlen / 2;
            if subframe_len < winlen {
                start += (winlen - subframe_len) / 2;
                winlen = subframe_len;
            }
            window_overlap(&mut ch.out[start..start + winlen], &windows()[log2(winlen) as usize - 7]);
            ch.prev_block_len = subframe_len;
        }

        for i in 0..self.channels_for_cur_subframe {
            let ch = &mut self.ch[self.channel_indexes[i]];
            if ch.cur_subframe >= ch.num_subframes {
                return Err(Error("subframe overrun"));
            }
            ch.cur_subframe += 1;
        }
        if b.overread() {
            return Err(Error("subframe overread"));
        }
        Ok(())
    }

    fn inverse_channel_transform(&mut self) {
        if !self.transform || self.channels_for_cur_subframe != 2 {
            return;
        }
        let sfb = self.sfb_offsets[self.table_idx];
        let [a, b] = &mut self.ch;
        for band in 0..self.num_bands {
            let range = sfb[band]..sfb[band + 1].min(self.subframe_len);
            if self.transform_band[band] {
                for y in range {
                    let (l, r) = (a.coeffs[y], b.coeffs[y]);
                    (a.coeffs[y], b.coeffs[y]) = (l - r, l + r);
                }
            } else {
                for y in range {
                    a.coeffs[y] *= 181.0 / 128.0;
                    b.coeffs[y] *= 181.0 / 128.0;
                }
            }
        }
    }
}

/// Sine windows for blocks of 128, 256 and 512 samples.
fn windows() -> &'static [Vec<f32>; 3] {
    static WINDOWS: OnceLock<[Vec<f32>; 3]> = OnceLock::new();
    WINDOWS.get_or_init(|| [sine_window(128), sine_window(256), sine_window(512)])
}

/// FFmpeg's `vector_fmul_window` in place: the first half of `block` holds
/// the previous block's tail, the second half the current block's head.
fn window_overlap(block: &mut [f32], window: &[f32]) {
    let len = block.len() / 2;
    for k in 0..len {
        let (i, j) = (k, 2 * len - 1 - k);
        let (s0, s1) = (block[i], block[2 * len - 1 - k]);
        let (wi, wj) = (window[i], window[j]);
        block[i] = s0 * wj - s1 * wi;
        block[j] = s0 * wi + s1 * wj;
    }
}

fn to_i16(v: f32) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `value` as `n` bits, MSB first, at bit `pos`.
    fn put_bits(data: &mut [u8], pos: usize, n: u32, value: u32) {
        for i in 0..n as usize {
            if value >> (n as usize - 1 - i) & 1 == 1 {
                data[(pos + i) / 8] |= 1 << (7 - (pos + i) % 8);
            }
        }
    }

    /// A damaged packet whose first frame says it is 0 bits long: finding
    /// the frame after a later one must give up, not walk in place forever
    /// (found by random-input testing).
    #[test]
    fn a_zero_length_frame_before_the_start_ends_the_search() {
        let mut data = vec![0u8; PACKET_BYTES];
        // Two frames, the first at payload bit 0; its 15-bit length stays 0.
        put_bits(&mut data, 0, 32, 2 << 26);
        let start = HEADER_BITS + 100;
        put_bits(&mut data, start, LENGTH_BITS, 200);
        let frame = read_frame(&data, start).expect("the frame at `start` is well formed");
        assert_eq!(frame.bits, 200);
        assert_eq!(frame.next, None);
    }

    /// The direct sum `imdct_half` stands for (FFmpeg's naive inverse MDCT:
    /// the standard formula, negated).
    fn imdct_reference(input: &[f32]) -> Vec<f32> {
        let n = input.len();
        (0..n)
            .map(|j| {
                let t = (j + n / 2) as f64 + 0.5 + n as f64 / 2.0;
                -(input
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| x as f64 * (std::f64::consts::PI / n as f64 * t * (k as f64 + 0.5)).cos())
                    .sum::<f64>() as f32)
            })
            .collect()
    }

    #[test]
    fn fast_imdct_matches_the_direct_sum() {
        for n in [128usize, 256, 512] {
            let input: Vec<f32> = (0..n).map(|i| ((i * 7919) % 97) as f32 - 48.0).collect();
            let mut fast = vec![0f32; n];
            imdct_half(&input, &mut fast, 1.0);
            let reference = imdct_reference(&input);
            for (a, b) in fast.iter().zip(&reference) {
                assert!((a - b).abs() < 1e-2 * (1.0 + b.abs()), "n {n}: {a} vs {b}");
            }
        }
    }

    /// Windowed MDCT blocks of a signal, put back together the way the
    /// decoder does (half IMDCT, then window and overlap), give the signal.
    #[test]
    fn imdct_and_window_reconstruct_a_signal() {
        let n = 256;
        let signal: Vec<f32> = (0..n * 6).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        // The 2n-sample sine window; its rising half is `sine_window(n)`.
        let window = |i: usize| ((i as f64 + 0.5) * std::f64::consts::PI / (2.0 * n as f64)).sin();
        let mut out = vec![0f32; n * 7];
        for block in 0..5 {
            let base = block * n;
            let coeffs: Vec<f32> = (0..n)
                .map(|k| {
                    (0..2 * n)
                        .map(|i| {
                            let t = i as f64 + 0.5 + n as f64 / 2.0;
                            // FFmpeg's forward MDCT pairs with its inverse
                            // by sign.
                            -signal[base + i] as f64
                                * window(i)
                                * (std::f64::consts::PI / n as f64 * t * (k as f64 + 0.5)).cos()
                        })
                        .sum::<f64>() as f32
                })
                .collect();
            // The half output is the middle of the 2n window.
            let at = base + n / 2;
            imdct_half(&coeffs, &mut out[at..at + n], 2.0 / n as f32);
            if block > 0 {
                window_overlap(&mut out[at - n / 2..at + n / 2], &sine_window(n));
            }
        }
        for i in n..5 * n {
            assert!((out[i] - signal[i]).abs() < 1e-3, "{i}: {} vs {}", out[i], signal[i]);
        }
    }

    #[test]
    fn packet_rejects_out_of_range_indices_and_partial_headers() {
        assert!(packet(&[0; 4], usize::MAX / PACKET_BYTES + 1).is_none());
        assert!(packet(&[0; 3], 0).is_none());
    }

    #[test]
    fn packet_headers_and_frame_walk() {
        // Two packets: frame A (100 bits) and B (16300 bits, crossing into
        // packet 1 by 48 bits), then C (60 bits) in packet 1; the rest is
        // padding.
        let mut data = vec![0xffu8; 2 * PACKET_BYTES];
        let header = |frames: u32, first: u32| (frames << 26) | (first << 11);
        let bits = [(0, header(2, 0), 32), (PACKET_BITS, header(1, 48), 32)];
        let put = |data: &mut Vec<u8>, at: usize, value: u32, width: u32| {
            for i in 0..width as usize {
                let bit = (value >> (width as usize - 1 - i)) & 1;
                let p = at + i;
                data[p / 8] = (data[p / 8] & !(1 << (7 - p % 8))) | ((bit as u8) << (7 - p % 8));
            }
        };
        for &(at, v, w) in &bits {
            put(&mut data, at, v, w);
        }
        put(&mut data, 32, 100, 15);
        put(&mut data, 132, 16300, 15);
        let c_start = PACKET_BITS + 32 + 48;
        put(&mut data, c_start, 60, 15);
        let a = read_frame(&data, 32).unwrap();
        assert_eq!((a.bits, a.next), (100, Some(132)));
        let b = read_frame(&data, 132).unwrap();
        assert_eq!((b.bits, b.next), (16300, Some(c_start)));
        // Its last bits come from after packet 1's header.
        assert_eq!(b.data.len(), 16300usize.div_ceil(8));
        let c = read_frame(&data, c_start).unwrap();
        assert_eq!(c.bits, 60);
        // C was packet 1's only frame: the next would be in packet 2.
        assert_eq!(c.next, None);
    }

    /// Reads bit `at` of `data`, MSB first.
    fn get_bit(data: &[u8], at: usize) -> u8 {
        (data[at / 8] >> (7 - at % 8)) & 1
    }

    /// A packet's `skip` field: a frame that runs off the end of a packet
    /// continues after the header of this stream's next packet, past the
    /// other streams' packets, and the frame after a packet's last frame is
    /// looked for there too.
    #[test]
    fn frames_skip_other_streams_packets() {
        // Packets 0, 2 and 4 are this stream's; 1 and 3 belong to another
        // stream and hold a pattern that must never show up in a frame.
        let mut data = vec![0u8; 5 * PACKET_BYTES];
        for p in [1, 3] {
            data[p * PACKET_BYTES..][..PACKET_BYTES].fill(0xa5);
        }
        let header = |frames: u32, first: u32, skip: u32| (frames << 26) | (first << 11) | skip;
        put_bits(&mut data, 0, 32, header(2, 0, 1));
        put_bits(&mut data, 2 * PACKET_BITS, 32, header(1, 48, 1));
        put_bits(&mut data, 4 * PACKET_BITS, 32, header(1, 7, 0));
        // A (100 bits), then B to 48 bits into packet 2's payload.
        let a_start = HEADER_BITS;
        put_bits(&mut data, a_start, LENGTH_BITS, 100);
        let b_start = a_start + 100;
        let b_bits = PACKET_BITS - b_start + 48;
        put_bits(&mut data, b_start, LENGTH_BITS, b_bits as u32);
        // B's tail: a pattern in packet 2's first 48 payload bits.
        let tail = 2 * PACKET_BITS + HEADER_BITS;
        put_bits(&mut data, tail, 32, 0xdead_beef);
        put_bits(&mut data, tail + 32, 16, 0x1234);
        // C, packet 2's only frame, then padding; D starts packet 4.
        let c_start = tail + 48;
        put_bits(&mut data, c_start, LENGTH_BITS, 60);
        let d_start = 4 * PACKET_BITS + HEADER_BITS + 7;
        put_bits(&mut data, d_start, LENGTH_BITS, 40);

        let a = read_frame(&data, a_start).unwrap();
        assert_eq!((a.bits, a.start, a.next), (100, a_start, Some(b_start)));
        let b = read_frame(&data, b_start).unwrap();
        assert_eq!((b.bits, b.next), (b_bits, Some(c_start)));
        // B is its bits in packet 0 followed by packet 2's first 48.
        for i in 0..b_bits {
            let at = if b_start + i < PACKET_BITS {
                b_start + i
            } else {
                tail + (b_start + i - PACKET_BITS)
            };
            assert_eq!(get_bit(&b.data, i), get_bit(&data, at), "bit {i}");
        }
        let c = read_frame(&data, c_start).unwrap();
        assert_eq!((c.bits, c.next), (60, Some(d_start)));
        let d = read_frame(&data, d_start).unwrap();
        // D was packet 4's only frame and packet 4 is the last.
        assert_eq!((d.bits, d.next), (40, None));
    }

    /// A frame that ends exactly at its packet's end: the next frame is in
    /// the stream's next packet, at that packet's first frame, even though
    /// the header counts more frames.
    #[test]
    fn a_frame_ending_at_the_packet_end_continues_in_the_next_packet() {
        let mut data = vec![0u8; 3 * PACKET_BYTES];
        let header = |frames: u32, first: u32, skip: u32| (frames << 26) | (first << 11) | skip;
        put_bits(&mut data, 0, 32, header(5, 0, 1));
        put_bits(&mut data, 2 * PACKET_BITS, 32, header(1, 3, 0));
        let (start, len) = (HEADER_BITS, PACKET_BITS - HEADER_BITS);
        put_bits(&mut data, start, LENGTH_BITS, len as u32);
        let frame = read_frame(&data, start).unwrap();
        assert_eq!(frame.bits, PACKET_BITS - HEADER_BITS);
        assert_eq!(frame.next, Some(2 * PACKET_BITS + HEADER_BITS + 3));
        // Without the stream's next packet there is no next frame.
        assert_eq!(read_frame(&data[..PACKET_BYTES], start).unwrap().next, None);
    }

    /// A frame's 15-bit length must cover the length field itself, and the
    /// all-ones length is not a frame; a frame cut off by the end of the
    /// buffer is not one either.
    #[test]
    fn frame_lengths_must_cover_the_field_and_fit() {
        for len in (0..=LENGTH_BITS).chain([0x7fff]) {
            let mut data = vec![0u8; PACKET_BYTES];
            put_bits(&mut data, 0, 32, 1 << 26);
            put_bits(&mut data, HEADER_BITS, LENGTH_BITS, len);
            assert!(read_frame(&data, HEADER_BITS).is_none(), "length {len}");
        }
        let mut data = vec![0u8; PACKET_BYTES];
        put_bits(&mut data, 0, 32, 1 << 26);
        put_bits(&mut data, HEADER_BITS, LENGTH_BITS, LENGTH_BITS + 1);
        let frame = read_frame(&data, HEADER_BITS).unwrap();
        assert_eq!((frame.bits, frame.data.len()), (16, 2));
        // One bit more than the single packet holds runs off the buffer.
        let mut data = vec![0u8; PACKET_BYTES];
        put_bits(&mut data, 0, 32, 1 << 26);
        let len = PACKET_BITS - HEADER_BITS + 1;
        put_bits(&mut data, HEADER_BITS, LENGTH_BITS, len as u32);
        assert!(read_frame(&data, HEADER_BITS).is_none());
        // So does a start past the end, or inside a cut-off header.
        assert!(read_frame(&data, PACKET_BITS + HEADER_BITS).is_none());
        assert!(read_frame(&data[..2], 0).is_none());
    }

    #[test]
    fn errors_name_the_codec() {
        assert_eq!(Error("bad tile").to_string(), "XMA: bad tile");
    }

    /// Random packets walked frame by frame and decoded, and random bits
    /// decoded directly, give frames or errors, never a panic or a loop
    /// that does not end; every decode still appends a whole frame.
    #[test]
    fn noise_never_panics() {
        let mut seed = 0xd1b5_4a32_d192_ed03u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..200 {
            let mut data: Vec<u8> = (0..PACKET_BYTES * (1 + round % 4))
                .map(|_| next() as u8)
                .collect();
            // Half the rounds have plausible headers: a few frames, the
            // first near the start, no skip, so walks get past packet 0.
            if round % 2 == 0 {
                for p in 0..data.len() / PACKET_BYTES {
                    let first = (next() % 64) as u32;
                    let header = (1 + next() as u32 % 4) << 26 | first << 11;
                    data[p * PACKET_BYTES..][..4].copy_from_slice(&header.to_be_bytes());
                    // A frame length that fits the payload.
                    let start = HEADER_BITS + first as usize;
                    let len = 16 + next() as u32 % 4000;
                    for i in 0..LENGTH_BITS as usize {
                        let bit = (len >> (LENGTH_BITS as usize - 1 - i)) & 1;
                        let at = p * PACKET_BITS + start + i;
                        data[at / 8] =
                            (data[at / 8] & !(1 << (7 - at % 8))) | ((bit as u8) << (7 - at % 8));
                    }
                }
            }
            let mut decoder = Decoder::new(
                1 + round % 2,
                [24_000, 32_000, 44_100, 48_000][round % 4] as u32,
            );
            let mut out = Vec::new();
            let mut walked = 0;
            for p in 0..data.len() / PACKET_BYTES {
                let Some(header) = packet(&data, p) else {
                    continue;
                };
                let mut at = Some(p * PACKET_BITS + HEADER_BITS + header.first_frame_bit as usize);
                while let Some(start) = at {
                    let Some(frame) = read_frame(&data, start) else {
                        break;
                    };
                    let before = out.len();
                    let _ = decoder.decode(&frame.data, frame.bits, &mut out);
                    assert_eq!(out.len() - before, FRAME_SAMPLES * decoder.channels());
                    walked += 1;
                    assert!(walked < 10_000, "the frame walk does not end");
                    at = frame.next.filter(|&n| n > start);
                }
            }
            for _ in 0..4 {
                let raw: Vec<u8> = (0..next() % 600).map(|_| next() as u8).collect();
                let bits = (next() as usize) % (raw.len() * 8 + 64);
                let before = out.len();
                let _ = decoder.decode(&raw, bits, &mut out);
                assert_eq!(out.len() - before, FRAME_SAMPLES * decoder.channels());
            }
            decoder.flush(&mut out);
        }
    }

    #[test]
    fn a_length_past_the_data_ends_at_the_data() {
        // Zeros make a stereo tile header in which no channel takes the
        // subframe, so it loops until the reader passes the frame's end.
        for (data, bits) in [(&[][..], usize::MAX), (&[0u8; 16][..], 1 << 40)] {
            for channels in [1, 2] {
                let mut decoder = Decoder::new(channels, 48_000);
                let mut out = Vec::new();
                assert!(decoder.decode(data, bits, &mut out).is_err());
                assert_eq!(out.len(), FRAME_SAMPLES * channels);
            }
        }
    }

    /// What `Decoder::new`, `reset` and `flush` promise, checked on noise
    /// frames: the channel count is clamped to 1 or 2, sample rates are
    /// rounded up to one of the four the format has, `reset` gives the
    /// output of a new decoder, and `flush` gives half a frame per channel
    /// and then leaves only silence to flush.
    #[test]
    fn reset_flush_and_construction() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let frames: Vec<Vec<u8>> = (0..8)
            .map(|_| (0..64 + next() % 512).map(|_| next() as u8).collect())
            .collect();
        let decode_all = |decoder: &mut Decoder| {
            let mut out = Vec::new();
            for frame in &frames {
                let _ = decoder.decode(frame, frame.len() * 8, &mut out);
            }
            decoder.flush(&mut out);
            out
        };

        assert_eq!(Decoder::new(0, 48_000).channels(), 1);
        assert_eq!(Decoder::new(1, 48_000).channels(), 1);
        assert_eq!(Decoder::new(2, 48_000).channels(), 2);
        assert_eq!(Decoder::new(6, 48_000).channels(), 2);

        // Rates in one bucket decode alike.
        for bucket in [
            &[0, 8_000, 24_000][..],
            &[24_001, 32_000],
            &[32_001, 44_100],
            &[44_101, 48_000, 96_000, u32::MAX],
        ] {
            for channels in [1, 2] {
                let expected = decode_all(&mut Decoder::new(channels, bucket[0]));
                assert_eq!(
                    expected.len(),
                    (frames.len() + 1) * FRAME_SAMPLES * channels - FRAME_SAMPLES / 2 * channels
                );
                for &rate in &bucket[1..] {
                    assert_eq!(
                        decode_all(&mut Decoder::new(channels, rate)),
                        expected,
                        "{rate} Hz"
                    );
                }
            }
        }

        for channels in [1, 2] {
            let fresh = Decoder::new(channels, 44_100);
            let expected = decode_all(&mut fresh.clone());
            // Some noise frames decode, so the comparisons below see sound.
            assert!(expected.iter().any(|&s| s != 0));
            // `reset` after other frames gives a new decoder's output.
            let mut used = fresh.clone();
            let mut scratch = Vec::new();
            for frame in frames.iter().rev() {
                let _ = used.decode(frame, frame.len() * 8, &mut scratch);
            }
            used.reset();
            assert_eq!(decode_all(&mut used), expected);
            // `flush` resets too, so the same decoder can run again, and a
            // second flush has nothing left to give.
            assert_eq!(decode_all(&mut used), expected);
            let mut silence = Vec::new();
            used.flush(&mut silence);
            assert_eq!(silence, vec![0; FRAME_SAMPLES / 2 * channels]);
        }
    }

    /// Walks and decodes saved input buffers (`VICTORY_XMA_STREAMS=<dir>` of
    /// `*.xma` files with their `*.ctx` context words, made outside the
    /// repository): the frames walked from the first packet must be the
    /// frames the packet headers count, and every frame must decode.
    #[test]
    #[ignore = "needs streams outside the repository"]
    fn walks_saved_streams() {
        let dir = std::env::var("VICTORY_XMA_STREAMS").unwrap();
        let mut bad = 0;
        let mut streams = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "xma") {
                continue;
            }
            streams += 1;
            let data = std::fs::read(&path).unwrap();
            let counted: u32 = (0..data.len() / PACKET_BYTES)
                .map(|i| packet(&data, i).unwrap().frames)
                .sum();
            // Context word 1: sample rate in bits 27-28, stereo in bit 29.
            let ctx = std::fs::read(path.with_extension("ctx")).unwrap();
            let word = u32::from_le_bytes(ctx[4..8].try_into().unwrap());
            let rate = [24_000, 32_000, 44_100, 48_000][(word >> 27 & 3) as usize];
            let mut decoder = Decoder::new(1 + (word >> 29 & 1) as usize, rate);
            let (mut walked, mut errors, mut peak, mut pcm) = (0, 0, 0i16, Vec::new());
            let mut at = Some(HEADER_BITS + packet(&data, 0).unwrap().first_frame_bit as usize);
            while let Some(frame) = at.and_then(|bit| read_frame(&data, bit)) {
                pcm.clear();
                errors += decoder.decode(&frame.data, frame.bits, &mut pcm).is_err() as u32;
                peak = pcm.iter().fold(peak, |p, s| p.max(s.saturating_abs()));
                walked += 1;
                at = frame.next;
            }
            println!(
                "{}: {} packets, {counted} frames counted, {walked} walked, {errors} bad, peak {peak}",
                path.display(),
                data.len() / PACKET_BYTES
            );
            bad += u32::from(walked != counted || errors > 0);
        }
        assert!(streams > 0, "VICTORY_XMA_STREAMS contains no .xma files");
        assert_eq!(bad, 0);
    }

    /// Decodes a stream's packets and compares with a reference decode of
    /// the same file (16-bit PCM): `VICTORY_XMA_REFERENCE=<dir>` holding
    /// `mono.xmadata`/`mono.ref` and `stereo.xmadata`/`stereo.ref` (made
    /// outside the repository).
    #[test]
    #[ignore = "needs reference files outside the repository"]
    fn matches_a_reference_decoder() {
        let dir = std::path::PathBuf::from(std::env::var("VICTORY_XMA_REFERENCE").unwrap());
        for (name, channels) in [("mono", 1), ("stereo", 2)] {
            let data = std::fs::read(dir.join(format!("{name}.xmadata"))).unwrap();
            let reference: Vec<i16> = std::fs::read(dir.join(format!("{name}.ref")))
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| i16::from_le_bytes(b))
                .collect();
            let mut decoder = Decoder::new(channels, 48_000);
            let mut pcm = Vec::new();
            let mut at = Some(HEADER_BITS + packet(&data, 0).unwrap().first_frame_bit as usize);
            let mut errors = 0;
            while let Some(frame) = at.and_then(|bit| read_frame(&data, bit)) {
                errors += decoder.decode(&frame.data, frame.bits, &mut pcm).is_err() as usize;
                at = frame.next;
            }
            // Find the decoder delay, then measure the error.
            let best = (0..2048 / channels)
                .map(|d| {
                    let n = reference.len().min(pcm.len() - d * channels);
                    let e: f64 = (0..n)
                        .map(|i| (pcm[d * channels + i] as f64 - reference[i] as f64).powi(2))
                        .sum();
                    (e, d, n)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap();
            let signal: f64 = reference.iter().map(|&s| (s as f64).powi(2)).sum();
            let snr = 10.0 * (signal / best.0.max(1.0)).log10();
            println!(
                "{name}: {} samples, {errors} bad frames, delay {} samples, SNR {snr:.1} dB",
                pcm.len() / channels,
                best.1
            );
            assert!(errors == 0 && snr > 40.0);
        }
    }

    /// The frame reader takes bits MSB first across byte boundaries, signs
    /// `n`-bit fields from their top bit, reads zeros past the data and
    /// reports an overread only once it has passed `end`.
    #[test]
    fn bits_read_msb_first_and_zero_past_the_data() {
        let data = [0b1010_1100, 0b0101_0011, 0xff, 0x00, 0x81];
        let mut b = Bits {
            data: &data,
            pos: 0,
            end: data.len() * 8,
        };
        assert_eq!(b.read(0), 0);
        assert_eq!(b.consumed(), 0);
        assert!(b.bit());
        assert!(!b.bit());
        assert_eq!(b.read(5), 0b10110);
        // Crosses into the second byte: 0 then 0101.
        assert_eq!(b.read(5), 0b00101);
        assert_eq!(b.consumed(), 12);
        // 0011 1111 -> as 8-bit signed, positive.
        assert_eq!(b.signed(8), 0b0011_1111);
        // 1111 0000 -> -16.
        assert_eq!(b.signed(8), -16);
        // 32 bits from a bit offset: 0000 1000 0001 then zeros past the end.
        assert_eq!(b.consumed(), 28);
        assert_eq!(b.peek32(), 0b0000_1000_0001 << 20);
        assert_eq!(b.read(32), 0b0000_1000_0001 << 20);
        assert_eq!(b.consumed(), 60);
        assert!(b.overread());
        assert_eq!(b.read(32), 0);

        // `end` is the last bit that may be read, not one past it.
        let mut b = Bits {
            data: &data,
            pos: 0,
            end: 7,
        };
        b.read(7);
        assert!(!b.overread());
        b.read(1);
        assert!(b.overread());

        // Every width reads back what `put_bits` wrote, at every bit offset.
        for n in 1..=32u32 {
            for at in 0..8 {
                let value = 0xdead_beefu32 >> (32 - n);
                let mut data = [0u8; 8];
                put_bits(&mut data, at, n, value);
                let mut b = Bits {
                    data: &data,
                    pos: at,
                    end: 64,
                };
                assert_eq!(b.read(n), value, "{n} bits at {at}");
                let mut b = Bits {
                    data: &data,
                    pos: at,
                    end: 64,
                };
                let signed = ((value << (32 - n)) as i32) >> (32 - n);
                assert_eq!(b.signed(n), signed, "{n} signed bits at {at}");
            }
        }
    }

    /// Every Huffman table is a complete prefix code, so any 32 bits decode
    /// to a symbol, and each symbol reads back from its own code. The
    /// symbol offsets match FFmpeg's (`scale` centred on 0, `vec4`/`vec2`
    /// counting the escape as -1).
    #[test]
    fn huffman_tables_are_complete_prefix_codes() {
        let c = codes();
        let tables: [(&str, &Vlc, usize); 7] = [
            ("scale", &c.scale, SCALE.len()),
            ("scale_rl", &c.scale_rl, SCALE_RL.len()),
            ("coef0", &c.coef[0], COEF0_SYMS.len()),
            ("coef1", &c.coef[1], COEF1.len()),
            ("vec4", &c.vec4, VEC4_SYMS.len()),
            ("vec2", &c.vec2, VEC2.len()),
            ("vec1", &c.vec1, VEC1.len()),
        ];
        for (name, vlc, len) in tables {
            assert_eq!(vlc.codes.len(), len, "{name}");
            let mut kraft = 0u64;
            for w in vlc.codes.windows(2) {
                assert!(w[0].0 < w[1].0, "{name}: codes in order");
            }
            for &(code, n, symbol) in &vlc.codes {
                assert!((1..=32).contains(&n), "{name}: length {n}");
                assert_eq!(
                    code as u64 & ((1u64 << (32 - n)) - 1),
                    0,
                    "{name}: code is left-aligned"
                );
                kraft += 1 << (32 - n);
                // The code itself, then all ones under it: both decode to
                // this symbol and consume exactly its length.
                for tail in [0, (1u64 << (32 - n)) - 1] {
                    let bytes = ((code as u64 | tail) as u32).to_be_bytes();
                    let mut b = Bits {
                        data: &bytes,
                        pos: 0,
                        end: 32,
                    };
                    assert_eq!(vlc.read(&mut b), symbol, "{name}");
                    assert_eq!(b.consumed(), n as usize, "{name}");
                }
            }
            assert_eq!(kraft, 1 << 32, "{name}: complete code");
        }
        let symbols = |v: &Vlc| v.codes.iter().map(|c| c.2).collect::<Vec<_>>();
        let scale = symbols(&c.scale);
        assert_eq!(*scale.iter().min().unwrap(), -60);
        assert_eq!(*scale.iter().max().unwrap(), SCALE.len() as i32 - 61);
        assert_eq!(*symbols(&c.vec4).iter().min().unwrap(), -1);
        assert_eq!(*symbols(&c.vec2).iter().min().unwrap(), -1);
    }

    /// `ff_wma_get_large_val`: 0, 10, 110 and 111 prefixes select 8, 16,
    /// 24 and 31 value bits.
    #[test]
    fn large_values_take_8_16_24_or_31_bits() {
        for (prefix, plen, n) in [
            (0u32, 1u32, 8u32),
            (0b10, 2, 16),
            (0b110, 3, 24),
            (0b111, 3, 31),
        ] {
            let value = 0x5a5a_5a5au32 >> (32 - n);
            let mut data = [0u8; 8];
            put_bits(&mut data, 0, plen, prefix);
            put_bits(&mut data, plen as usize, n, value);
            let mut b = Bits {
                data: &data,
                pos: 0,
                end: 64,
            };
            assert_eq!(large_value(&mut b), value, "{n} bits");
            assert_eq!(b.consumed(), (plen + n) as usize);
        }
    }

    /// Small helpers: `log2` floors and treats 0 as 1, `to_i16` rounds and
    /// saturates, and the sine windows rise from near 0 to near 1 with
    /// `w[i]² + w[n-1-i]² = 1` (the Princen-Bradley condition).
    #[test]
    fn log2_samples_and_windows() {
        for (v, l) in [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (511, 8),
            (512, 9),
            (usize::MAX, usize::BITS - 1),
        ] {
            assert_eq!(log2(v), l, "log2({v})");
        }
        assert_eq!(to_i16(0.0), 0);
        assert_eq!(to_i16(0.5), 16384);
        assert_eq!(to_i16(-1.0), -32768);
        assert_eq!(to_i16(1.0), 32767);
        assert_eq!(to_i16(-2.0), -32768);
        assert_eq!(to_i16(1e30), 32767);
        assert_eq!(to_i16(f32::NAN), 0);
        assert_eq!(to_i16(1.4 / 32768.0), 1);
        assert_eq!(to_i16(-1.6 / 32768.0), -2);
        for (i, w) in windows().iter().enumerate() {
            let n = 128 << i;
            assert_eq!(w.len(), n);
            assert!(w[0] > 0.0 && w[0] < 0.02 && w[n - 1] > 0.9999 && w[n - 1] <= 1.0);
            for k in 0..n {
                if k > 0 {
                    assert!(w[k] > w[k - 1]);
                }
                let sum = w[k] * w[k] + w[n - 1 - k] * w[n - 1 - k];
                assert!((sum - 1.0).abs() < 1e-6, "window {n} at {k}");
            }
        }
    }

    /// The radix-2 FFT matches a direct O(n²) DFT (`exp(-2πi nk/N)`) at
    /// every power-of-two size the decoder uses and below, and an impulse at
    /// `t` transforms to the unit-magnitude phasor `exp(-2πi kt/N)`.
    #[test]
    fn fft_matches_the_direct_dft() {
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f64 / u32::MAX as f64 * 2.0 - 1.0
        };
        for log in 0..=9 {
            let n = 1usize << log;
            let (xr, xi): (Vec<f64>, Vec<f64>) = (0..n).map(|_| (next(), next())).unzip();
            let (mut re, mut im) = (xr.clone(), xi.clone());
            fft(&mut re, &mut im);
            for k in 0..n {
                let (mut sr, mut si) = (0.0, 0.0);
                for t in 0..n {
                    let (s, c) =
                        (-2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64).sin_cos();
                    sr += xr[t] * c - xi[t] * s;
                    si += xr[t] * s + xi[t] * c;
                }
                let tol = 1e-9 * n as f64;
                assert!(
                    (re[k] - sr).abs() < tol && (im[k] - si).abs() < tol,
                    "n {n} bin {k}"
                );
            }
        }
        let n = 64;
        for t in [0, 1, 5, 63] {
            let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
            re[t] = 1.0;
            fft(&mut re, &mut im);
            for k in 0..n {
                let (s, c) =
                    (-2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64).sin_cos();
                assert!(
                    (re[k] - c).abs() < 1e-12 && (im[k] - s).abs() < 1e-12,
                    "t {t} k {k}"
                );
            }
        }
    }

    /// `window_overlap` is FFmpeg's `vector_fmul_window` with `dst` = `src0`
    /// = the block's first half (the previous tail), `src1` = its second half
    /// (the current head) and `win` = the sine window of the block's length.
    /// With a silent tail the block becomes the reversed head times the
    /// window (negated in the first half); with a silent head, the reversed
    /// tail times the window.
    #[test]
    fn window_overlap_is_vector_fmul_window() {
        for (i, window) in windows().iter().enumerate() {
            let len = 64 << i;
            let full = &window[..];
            assert_eq!(full.len(), 2 * len);
            let src: Vec<f32> = (0..2 * len)
                .map(|k| ((k * 37 + 11) % 97) as f32 / 48.0 - 1.0)
                .collect();
            let mut got = src.clone();
            window_overlap(&mut got, full);
            // The C loop, with every pointer advanced by `len`:
            // for (i = -len, j = len - 1; i < 0; i++, j--)
            //     dst[i] = src0[i]*win[j] - src1[j]*win[i];
            //     dst[j] = src0[i]*win[i] + src1[j]*win[j];
            let (tail, head) = src.split_at(len);
            let mut want = vec![0f32; 2 * len];
            let at = |i: isize| (len as isize + i) as usize;
            let (mut i, mut j) = (-(len as isize), len as isize - 1);
            while i < 0 {
                let (s0, s1) = (tail[at(i)], head[j as usize]);
                let (wi, wj) = (full[at(i)], full[at(j)]);
                want[at(i)] = s0 * wj - s1 * wi;
                want[at(j)] = s0 * wi + s1 * wj;
                (i, j) = (i + 1, j - 1);
            }
            assert_eq!(got, want, "len {len}");

            let mut silent_tail: Vec<f32> = vec![0.0; len];
            silent_tail.extend_from_slice(head);
            window_overlap(&mut silent_tail, full);
            for k in 0..len {
                assert_eq!(silent_tail[k], -head[len - 1 - k] * full[k]);
                assert_eq!(
                    silent_tail[2 * len - 1 - k],
                    head[len - 1 - k] * full[2 * len - 1 - k]
                );
            }
            let mut silent_head = tail.to_vec();
            silent_head.extend(std::iter::repeat_n(0.0, len));
            window_overlap(&mut silent_head, full);
            for k in 0..len {
                assert_eq!(silent_head[k], tail[k] * full[2 * len - 1 - k]);
                assert_eq!(silent_head[2 * len - 1 - k], tail[k] * full[k]);
            }
        }
    }

    /// `channel_transform` reads FFmpeg's WMA Pro layout for one stereo
    /// group: a 0 bit (1 is unsupported), then for two channels `1 0` for
    /// no transform, `1 1` for an unsupported one, or `0` for M/S followed by
    /// `1` (every band) or `0` and one flag per band. Mono reads nothing.
    #[test]
    fn channel_transform_reads_the_stereo_group() {
        fn read(d: &mut Decoder, bits: &str) -> (Result<(), Error>, usize) {
            let mut data = [0u8; 8];
            for (i, c) in bits.bytes().enumerate() {
                data[i / 8] |= u8::from(c == b'1') << (7 - i % 8);
            }
            let mut b = Bits {
                data: &data,
                pos: 0,
                end: bits.len(),
            };
            (d.channel_transform(&mut b), b.consumed())
        }
        let mut d = Decoder::new(2, 44_100);
        d.channels_for_cur_subframe = 2;
        d.num_bands = 4;

        assert_eq!(read(&mut d, "1").0, Err(Error("channel transform bit")));
        assert_eq!(read(&mut d, "011").0, Err(Error("channel transform type")));

        d.transform = true;
        assert_eq!(read(&mut d, "010"), (Ok(()), 3));
        assert!(!d.transform);

        assert_eq!(read(&mut d, "001"), (Ok(()), 3));
        assert!(d.transform);
        assert_eq!(d.transform_band, [true; MAX_BANDS]);

        assert_eq!(read(&mut d, "0001011"), (Ok(()), 7));
        assert!(d.transform);
        // Bands past `num_bands` keep their old flags.
        assert_eq!(d.transform_band[..5], [true, false, true, true, true]);

        assert_eq!(read(&mut d, "0000000"), (Ok(()), 7));
        assert_eq!(d.transform_band[..5], [false, false, false, false, true]);

        // A one-channel subframe of a stereo stream reads only the first bit.
        d.channels_for_cur_subframe = 1;
        assert_eq!(read(&mut d, "0111"), (Ok(()), 1));
        assert!(!d.transform);

        let mut mono = Decoder::new(1, 44_100);
        mono.transform = true;
        assert_eq!(read(&mut mono, "1111"), (Ok(()), 0));
        assert!(!mono.transform);
    }

    /// `inverse_channel_transform` turns M/S bands into `(l - r, l + r)` and
    /// scales the other bands of a transformed pair by 181/128 (about √2), up
    /// to the subframe length; untransformed subframes are left alone.
    #[test]
    fn inverse_channel_transform_undoes_mid_side() {
        let mut d = Decoder::new(2, 44_100);
        d.table_idx = 1;
        let sfb = d.sfb_offsets[1];
        assert!(d.num_sfb[1] > 4);
        d.num_bands = 3;
        // Stop inside band 2.
        d.subframe_len = sfb[2] + 2;
        assert!(d.subframe_len < sfb[3]);
        d.channels_for_cur_subframe = 2;
        d.transform_band[..3].copy_from_slice(&[true, false, true]);
        for y in 0..FRAME_SAMPLES {
            d.ch[0].coeffs[y] = y as f32;
            d.ch[1].coeffs[y] = 3.0 - (y % 7) as f32;
        }
        let fresh = (d.ch[0].coeffs, d.ch[1].coeffs);

        // No transform, or only one channel in the subframe: unchanged.
        d.transform = false;
        d.inverse_channel_transform();
        assert_eq!((d.ch[0].coeffs, d.ch[1].coeffs), fresh);
        d.transform = true;
        d.channels_for_cur_subframe = 1;
        d.inverse_channel_transform();
        assert_eq!((d.ch[0].coeffs, d.ch[1].coeffs), fresh);

        d.channels_for_cur_subframe = 2;
        d.inverse_channel_transform();
        let (l, r) = fresh;
        for y in 0..FRAME_SAMPLES {
            let band = (0..3).find(|&b| y < sfb[b + 1]);
            let want = match band {
                _ if y >= d.subframe_len => (l[y], r[y]),
                Some(1) => (l[y] * 181.0 / 128.0, r[y] * 181.0 / 128.0),
                Some(_) => (l[y] - r[y], l[y] + r[y]),
                None => (l[y], r[y]),
            };
            assert_eq!((d.ch[0].coeffs[y], d.ch[1].coeffs[y]), want, "y {y}");
        }
    }

    /// Packs a string of `0`/`1` into bytes, MSB first.
    fn bit_string(bits: &str) -> [u8; 8] {
        let mut data = [0u8; 8];
        for (i, c) in bits.bytes().enumerate() {
            data[i / 8] |= u8::from(c == b'1') << (7 - i % 8);
        }
        data
    }

    /// `subframe_length` reads `0` for a whole frame, `10` for half and `11`
    /// for a quarter, and reads nothing for the last quarter of a frame.
    /// `tile_header` reads a "fixed" bit, then per subframe one "contains"
    /// bit for each channel at the shortest length (skipped when fixed, when
    /// only one channel is that short, or for the last quarter), then the
    /// subframe length.
    #[test]
    fn tile_header_splits_the_frame_into_subframes() {
        fn read(d: &mut Decoder, bits: &str, end: usize) -> (Result<(), Error>, usize) {
            let data = bit_string(bits);
            let mut b = Bits {
                data: &data,
                pos: 0,
                end,
            };
            (d.tile_header(&mut b), b.consumed())
        }
        fn lens(d: &Decoder, c: usize) -> &[usize] {
            &d.ch[c].subframe_len[..d.ch[c].num_subframes]
        }

        let d = Decoder::new(1, 44_100);
        for (bits, offset, want, used) in [
            ("0", 0, 512, 1),
            ("10", 0, 256, 2),
            ("11", 0, 128, 2),
            ("10", 256, 256, 2),
            ("11", 384, 128, 0),
        ] {
            let data = bit_string(bits);
            let mut b = Bits {
                data: &data,
                pos: 0,
                end: 8,
            };
            assert_eq!(
                d.subframe_length(&mut b, offset),
                Ok(want),
                "{bits} at {offset}"
            );
            assert_eq!(b.consumed(), used, "{bits} at {offset}");
        }

        let mut mono = Decoder::new(1, 44_100);
        assert_eq!(read(&mut mono, "10", 2), (Ok(()), 2));
        assert_eq!(lens(&mono, 0), [512]);
        // Quarter, half, then the last quarter without a length.
        assert_eq!(read(&mut mono, "11110", 5), (Ok(()), 5));
        assert_eq!(lens(&mono, 0), [128, 256, 128]);
        assert_eq!(read(&mut mono, "1111110", 7), (Ok(()), 7));
        assert_eq!(lens(&mono, 0), [128, 128, 256]);
        // Half, then a whole frame: too long.
        assert_eq!(
            read(&mut mono, "1100", 4).0,
            Err(Error("channel longer than a frame"))
        );
        // A header cut off after its first subframe.
        assert_eq!(read(&mut mono, "11", 2).0, Err(Error("tile header")));

        let mut stereo = Decoder::new(2, 44_100);
        // Not fixed: both channels in one whole-frame subframe.
        assert_eq!(read(&mut stereo, "0110", 4), (Ok(()), 4));
        assert_eq!(
            (lens(&stereo, 0), lens(&stereo, 1)),
            (&[512][..], &[512][..])
        );
        // Fixed: no "contains" bits.
        assert_eq!(read(&mut stereo, "11011", 5), (Ok(()), 5));
        assert_eq!(lens(&stereo, 0), [256, 128, 128]);
        assert_eq!(lens(&stereo, 1), [256, 128, 128]);
        // Channel 0 takes a quarter; channel 1, now alone at 0, takes the
        // whole frame without a "contains" bit; channel 0 then takes a half
        // and the last quarter, each without one.
        assert_eq!(read(&mut stereo, "01011010", 8), (Ok(()), 8));
        assert_eq!(lens(&stereo, 0), [128, 256, 128]);
        assert_eq!(lens(&stereo, 1), [512]);
    }

    /// `Decoder::new` splits each of the three block sizes into scale factor
    /// bands: they start at 0, grow in steps that are multiples of 4 and end
    /// at the block length. Each band maps to itself at its own block size,
    /// and to a band covering its middle at the others.
    #[test]
    fn scale_factor_bands_cover_each_block() {
        for rate in [24_000, 32_000, 44_100, 48_000] {
            let d = Decoder::new(2, rate);
            for i in 0..BLOCK_SIZES {
                let len = FRAME_SAMPLES >> i;
                let n = d.num_sfb[i];
                assert!((1..MAX_BANDS).contains(&n), "{rate} Hz block {i}: {n}");
                let offsets = &d.sfb_offsets[i][..=n];
                assert_eq!(offsets[0], 0);
                assert_eq!(offsets[n], len, "{rate} Hz block {i}");
                for w in offsets.windows(2) {
                    assert!(w[0] < w[1], "{rate} Hz block {i}: {offsets:?}");
                }
                for &o in &offsets[..n] {
                    assert_eq!(o % 4, 0, "{rate} Hz block {i}: {offsets:?}");
                }
                for b in 0..n {
                    assert_eq!(d.sf_offsets[i][i][b], b, "{rate} Hz block {i}");
                    // The middle of band `b`, in whole-frame samples.
                    let mid = ((offsets[b] + offsets[b + 1] - 1) << i) >> 1;
                    for x in 0..BLOCK_SIZES {
                        let v = d.sf_offsets[i][x][b];
                        assert!(v < d.num_sfb[x], "{rate} Hz {i}->{x} band {b}");
                        assert!(
                            mid <= d.sfb_offsets[x][v + 1] << x,
                            "{rate} Hz {i}->{x} band {b}"
                        );
                        assert!(
                            v == 0 || d.sfb_offsets[x][v] << x < mid,
                            "{rate} Hz {i}->{x} band {b}"
                        );
                    }
                }
            }
            // Lower rates put the same frequencies at higher bins, so they
            // need at least as many bands.
            if rate > 24_000 {
                let lower = Decoder::new(2, rate - 1);
                assert!(lower.num_sfb[0] >= d.num_sfb[0], "{rate} Hz");
            }
        }
    }

    /// `scale_factors` reads a channel's first set as a 2-bit step and one
    /// delta per band from 45 / step; later subframes either keep the last
    /// set (a 0 bit) or send run-level changes to it (a 1 bit). Kept sets are
    /// carried to the new block size through `sf_offsets`.
    #[test]
    fn scale_factors_read_deltas_then_run_level_changes() {
        /// The bits of `sym` in `vlc`.
        fn code(vlc: &Vlc, sym: i32) -> String {
            let &(c, len, _) = vlc.codes.iter().find(|c| c.2 == sym).unwrap();
            format!("{:032b}", c)[..len as usize].to_string()
        }
        fn read(d: &mut Decoder, bits: &str) -> (Result<(), Error>, usize) {
            let mut data = vec![0u8; bits.len() / 8 + 8];
            for (i, c) in bits.bytes().enumerate() {
                data[i / 8] |= u8::from(c == b'1') << (7 - i % 8);
            }
            let mut b = Bits {
                data: &data,
                pos: 0,
                end: bits.len(),
            };
            (d.scale_factors(&mut b), b.consumed())
        }
        let c = codes();
        let (scale, rl) = (&c.scale, &c.scale_rl);
        // Run-level symbols: 1 ends, 2 is run 0 level 1, 4 is run 1 level 2.
        assert_eq!((SCALE_RL_RUN[2], SCALE_RL_LEVEL[2]), (0, 1));
        assert_eq!((SCALE_RL_RUN[4], SCALE_RL_LEVEL[4]), (1, 2));

        let mut d = Decoder::new(1, 44_100);
        d.channels_for_cur_subframe = 1;
        d.num_bands = 3;
        let sf = |d: &Decoder| d.ch[0].saved_scale_factors[d.ch[0].scale_factors][..3].to_vec();

        // First subframe: no flag bit; step 2 starts at 22.
        let bits = format!("01{}{}{}", code(scale, 1), code(scale, -2), code(scale, 0));
        assert_eq!(read(&mut d, &bits), (Ok(()), bits.len()));
        assert_eq!(sf(&d), [23, 21, 21]);
        assert_eq!(d.ch[0].scale_factor_step, 2);
        assert_eq!(d.ch[0].max_scale_factor, 23);
        assert!(d.ch[0].reuse_sf);

        // A 0 bit keeps the set.
        d.ch[0].cur_subframe = 1;
        assert_eq!(read(&mut d, "0"), (Ok(()), 1));
        assert_eq!(sf(&d), [23, 21, 21]);

        // Skip band 0, band 1 -= 2, then end.
        let bits = format!("1{}0{}", code(rl, 4), code(rl, 1));
        assert_eq!(read(&mut d, &bits), (Ok(()), bits.len()));
        assert_eq!(sf(&d), [23, 19, 21]);

        // Escape: 14 bits of level << 6 | skip << 1 | positive; band 0 += 5,
        // then +1 on bands 1 and 2, which ends the bands without symbol 1.
        let bits = format!(
            "1{}{:014b}{}1{}1",
            code(rl, 0),
            5 << 6 | 1,
            code(rl, 2),
            code(rl, 2)
        );
        assert_eq!(read(&mut d, &bits), (Ok(()), bits.len()));
        assert_eq!(sf(&d), [28, 20, 22]);
        assert_eq!(d.ch[0].max_scale_factor, 28);
        // The step is read only for the first set.
        assert_eq!(d.ch[0].scale_factor_step, 2);

        // A run past the last band is an error.
        let bits = format!("1{}{:014b}", code(rl, 0), 3 << 1);
        assert_eq!(read(&mut d, &bits).0, Err(Error("scale factor run")));

        // Kept at a half-length block, each band takes the full-frame band
        // that covers its middle.
        let mut d = Decoder::new(1, 44_100);
        d.channels_for_cur_subframe = 1;
        d.num_bands = d.num_sfb[0];
        let bits = format!("00{}", code(scale, 1).repeat(d.num_bands));
        assert_eq!(read(&mut d, &bits), (Ok(()), bits.len()));
        let full: Vec<i32> = (1..=d.num_bands as i32).map(|v| 45 + v).collect();
        assert_eq!(
            d.ch[0].saved_scale_factors[d.ch[0].scale_factors][..d.num_bands],
            full[..]
        );
        d.ch[0].cur_subframe = 1;
        d.table_idx = 1;
        d.num_bands = d.num_sfb[1];
        assert_eq!(read(&mut d, "0"), (Ok(()), 1));
        let kept = &d.ch[0].saved_scale_factors[d.ch[0].scale_factors];
        for b in 0..d.num_bands {
            assert_eq!(kept[b], full[d.sf_offsets[1][0][b]], "band {b}");
        }
        assert_eq!(d.ch[0].table_idx, 0);
        assert_eq!(
            d.ch[0].max_scale_factor,
            full[d.sf_offsets[1][0][d.num_bands - 1]]
        );
    }
    /// `coefficients` reads a table bit, then groups of four vector-coded
    /// values (vec4, escaping to two vec2 pairs, escaping to vec1 values
    /// with a large-value escape), each nonzero value followed by a sign
    /// bit where 0 is negative. Run-level codes fill the rest of the
    /// subframe: a run and level from the table, the end code, or an escape
    /// with a large value and a 0-, 2- or `esc_len`-bit run.
    #[test]
    fn coefficients_read_vectors_then_run_levels() {
        fn code(vlc: &Vlc, sym: i32) -> String {
            let &(c, len, _) = vlc.codes.iter().find(|c| c.2 == sym).unwrap();
            format!("{:032b}", c)[..len as usize].to_string()
        }
        fn read(d: &mut Decoder, bits: &str, end: usize) -> (Result<(), Error>, usize) {
            let mut data = vec![0u8; bits.len() / 8 + 8];
            for (i, c) in bits.bytes().enumerate() {
                data[i / 8] |= u8::from(c == b'1') << (7 - i % 8);
            }
            let mut b = Bits {
                data: &data,
                pos: 0,
                end,
            };
            (d.coefficients(&mut b, 0), b.consumed())
        }
        fn decoder(num_vec_coeffs: usize, transmit: bool) -> Decoder {
            let mut d = Decoder::new(1, 44_100);
            d.subframe_len = 256;
            // As `subframe` sets it: 8 bits for 256 samples.
            d.esc_len = log2(256 - 1) + 1;
            d.ch[0].num_vec_coeffs = num_vec_coeffs;
            d.transmit_num_vec_coeffs = transmit;
            // Past the subframe stays untouched.
            d.ch[0].coeffs.fill(99.0);
            d
        }
        fn nonzero(d: &Decoder) -> Vec<(usize, f32)> {
            (0..d.subframe_len)
                .filter(|&i| d.ch[0].coeffs[i] != 0.0)
                .map(|i| (i, d.ch[0].coeffs[i]))
                .collect()
        }
        let c = codes();
        let coef0 = &c.coef[0];
        // Table 0: 0 escapes, 1 ends, 3 is run 1 level 1.
        assert_eq!((COEF0_RUN[3], COEF0_LEVEL[3]), (1, 1.0));

        let bits = [
            "0".to_string(),
            // 1, 1, 1, 1 with signs + - + -.
            code(&c.vec4, 0x1111),
            "1010".into(),
            // vec4 escape: vec2 (1, 2), then a vec2 escape to vec1 3 and
            // vec1's large-value escape: 100 + 5 in 8 bits.
            code(&c.vec4, -1),
            code(&c.vec2, 0x12),
            code(&c.vec2, -1),
            code(&c.vec1, 3),
            code(&c.vec1, VEC1.len() as i32 - 1),
            format!("0{:08b}", 5),
            "0110".into(),
            // 1, 2, 0, 0.
            code(&c.vec4, 0x1200),
            "10".into(),
            // Run 1 level 1 at 13.
            code(coef0, 3),
            "1".into(),
            // Escape: value 7, a 2-bit run of 1 + 1, negative, at 16.
            code(coef0, 0),
            format!("0{:08b}", 7),
            "10".into(),
            "01".into(),
            "0".into(),
            // Escape: value 9, an 8-bit run of 10 + 4, positive, at 31.
            code(coef0, 0),
            format!("0{:08b}", 9),
            "110".into(),
            format!("{:08b}", 10),
            "1".into(),
            // Escape with no run, at 32.
            code(coef0, 0),
            format!("0{:08b}", 3),
            "0".into(),
            "1".into(),
            code(coef0, 1),
        ]
        .concat();
        let mut d = decoder(12, true);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(
            nonzero(&d),
            [
                (0, 1.0),
                (1, -1.0),
                (2, 1.0),
                (3, -1.0),
                (4, -1.0),
                (5, 2.0),
                (6, 3.0),
                (7, -105.0),
                (8, 1.0),
                (9, -2.0),
                (13, 1.0),
                (16, -7.0),
                (31, 9.0),
                (32, 3.0),
            ]
        );
        assert!(d.ch[0].coeffs[256..].iter().all(|&v| v == 99.0));

        // Without a sent count, two zeros in a row (more than 256 >> 8)
        // switch to run-level coding after that group.
        let bits = format!("0{}10{}", code(&c.vec4, 0x1200), code(coef0, 1));
        let mut d = decoder(12, false);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(nonzero(&d), [(0, 1.0), (1, -2.0)]);
        // With fewer than four vector coefficients, none are read.
        let bits = format!("0{}", code(coef0, 1));
        let mut d = decoder(3, true);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(nonzero(&d), []);

        // Table 1 has its own runs and levels: 104 is run 2 level 2.
        assert_eq!((COEF1_RUN[104], COEF1_LEVEL[104]), (2, 2.0));
        let bits = format!("1{}0{}", code(&c.coef[1], 104), code(&c.coef[1], 1));
        let mut d = decoder(0, true);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(nonzero(&d), [(2, -2.0)]);

        // A run past the subframe wraps the write, as in FFmpeg, and ends
        // the coefficients without an end code: 253 + 4 lands on 257 & 255.
        let bits = format!("0{}0{:08b}110{:08b}1", code(coef0, 0), 6, 253);
        let mut d = decoder(0, true);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(nonzero(&d), [(1, 6.0)]);

        // A broken escape (three 1 bits) ends the coefficients.
        let bits = format!("0{}0{:08b}111", code(coef0, 0), 6);
        let mut d = decoder(0, true);
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!(nonzero(&d), []);

        // Reading past the end is an error in either part.
        let bits = format!("0{}1010", code(&c.vec4, 0x1111));
        let mut d = decoder(4, true);
        assert_eq!(
            read(&mut d, &bits, bits.len() - 1).0,
            Err(Error("vector coefficients"))
        );
        let bits = format!("0{}1", code(coef0, 3));
        let mut d = decoder(0, true);
        assert_eq!(
            read(&mut d, &bits, bits.len() - 1).0,
            Err(Error("run level coefficients"))
        );
    }

    /// The bits of `sym` in `vlc`.
    fn vlc_code(vlc: &Vlc, sym: i32) -> String {
        let &(c, len, _) = vlc.codes.iter().find(|c| c.2 == sym).unwrap();
        format!("{:032b}", c)[..len as usize].to_string()
    }

    /// Bits from a string of `0`/`1`, with zero padding past the end.
    fn bits_of(bits: &str) -> Vec<u8> {
        let mut data = vec![0u8; bits.len() / 8 + 8];
        for (i, c) in bits.bytes().enumerate() {
            data[i / 8] |= u8::from(c == b'1') << (7 - i % 8);
        }
        data
    }

    /// `subframe` reads, for a subframe every channel shares: an optional
    /// fill (`1`, then 2 bits, or `00`, 4 bits of width and that many bits
    /// of length - 1), a reserved `0`, the channel transform, one "transmit"
    /// bit per channel and, if any is set, an optional vector coefficient
    /// count, a 6-bit signed quantisation step around 90 (with 5-bit
    /// extensions at -32 and 31), per-channel step modifiers for stereo, and
    /// the scale factors.
    #[test]
    fn subframe_headers_read_fill_quantisation_and_modifiers() {
        fn read(d: &mut Decoder, bits: &str, end: usize) -> (Result<(), Error>, usize) {
            for c in 0..d.channels {
                let ch = &mut d.ch[c];
                ch.num_subframes = 1;
                ch.subframe_len[0] = FRAME_SAMPLES;
                ch.decoded_samples = 0;
                ch.cur_subframe = 0;
                ch.reuse_sf = false;
            }
            d.parsed_all_subframes = false;
            let data = bits_of(bits);
            let mut b = Bits {
                data: &data,
                pos: 0,
                end,
            };
            (d.subframe(&mut b), b.consumed())
        }
        // Scale factors for one channel: step 1, then no change per band.
        let scales =
            |d: &Decoder| format!("00{}", vlc_code(&codes().scale, 0).repeat(d.num_sfb[0]));

        let mut d = Decoder::new(1, 44_100);
        assert_eq!(read(&mut d, "000", 3), (Ok(()), 3));
        assert!(d.parsed_all_subframes);
        assert_eq!((d.ch[0].cur_subframe, d.ch[0].prev_block_len), (1, 512));
        assert!(!d.ch[0].transmit_coefs);
        assert_eq!((d.table_idx, d.num_bands, d.esc_len), (0, d.num_sfb[0], 9));

        assert_eq!(read(&mut d, "01", 2).0, Err(Error("reserved bit")));
        // A 2-bit fill of 1 bit.
        assert_eq!(read(&mut d, "101100", 6), (Ok(()), 6));
        // A 3-bit length of 5 + 1 = 6 fill bits.
        assert_eq!(read(&mut d, "100001110111111100", 18), (Ok(()), 18));
        // A fill past the end of the data.
        assert_eq!(read(&mut d, "11100", 5).0, Err(Error("subframe overread")));

        // An 8-bit count of 129 * 4 is past the 512 samples.
        assert_eq!(
            read(&mut d, &format!("0011{:08b}", 129), 12).0,
            Err(Error("vector coefficient count"))
        );
        let bits = format!("0011{:08b}000000{}", 128, scales(&d));
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!((d.ch[0].num_vec_coeffs, d.ch[0].quant_step), (512, 90));

        for (step, want) in [
            ("000101", 95),
            ("111011", 85),
            // 31, then 31 and 2: 90 + 31 + 33.
            ("011111", 154),
            // -32, then 3: 90 - 32 - 3.
            ("100000", 55),
        ] {
            let ext = match step {
                "011111" => "1111100010",
                "100000" => "00011",
                _ => "",
            };
            let bits = format!("0010{step}{ext}{}", scales(&d));
            assert_eq!(
                read(&mut d, &bits, bits.len()),
                (Ok(()), bits.len()),
                "{step}"
            );
            assert_eq!(d.ch[0].quant_step, want, "{step}");
            assert_eq!(d.ch[0].num_vec_coeffs, 512);
        }

        let mut d = Decoder::new(2, 44_100);
        // No transform; both transmit; step 0; 2-bit modifiers: +3 + 1 for
        // channel 0, none for channel 1.
        let bits = format!(
            "0 0 0 10 11 0 000000 010 111 0 {}",
            scales(&d) + &scales(&d)
        )
        .replace(' ', "");
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert!(!d.transform);
        assert_eq!((d.ch[0].quant_step, d.ch[1].quant_step), (94, 90));
        // A zero-width modifier adds 1.
        let bits =
            format!("0 0 0 10 11 0 000000 000 0 1 {}", scales(&d) + &scales(&d)).replace(' ', "");
        assert_eq!(read(&mut d, &bits, bits.len()), (Ok(()), bits.len()));
        assert_eq!((d.ch[0].quant_step, d.ch[1].quant_step), (90, 91));
    }

    /// `decode` reads a 15-bit length, the tile header, the stereo
    /// post-processing matrix, an 8-bit gain and the optional trims before
    /// the subframes; a frame that reads past its length is an error but
    /// still yields a frame of samples.
    #[test]
    fn frames_read_their_header_and_check_their_length() {
        // Mono: fixed, one whole-frame subframe; gain; no trim; an empty
        // subframe.
        let body = format!("10{:08b}0000", 0xa5);
        for (len, want) in [
            (29, Ok(())),
            (40, Ok(())),
            (28, Err(Error("frame overread"))),
        ] {
            let bits = format!("{len:015b}{body}");
            let mut d = Decoder::new(1, 44_100);
            let mut out = Vec::new();
            assert_eq!(
                d.decode(&bits_of(&bits), bits.len(), &mut out),
                want,
                "{len}"
            );
            assert_eq!(out, [0; FRAME_SAMPLES]);
        }

        // Stereo: a 16-bit post-processing matrix, both trims of 10 bits,
        // then a subframe both channels share without a transform.
        let body = format!(
            "10 11{:016b} {:08b} 1 1{:010b} 1{:010b} 000 10 00",
            0xffff, 0, 3, 5
        )
        .replace(' ', "");
        let bits = format!("{:015b}{body}", 15 + body.len());
        let mut d = Decoder::new(2, 44_100);
        let mut out = Vec::new();
        assert_eq!(d.decode(&bits_of(&bits), bits.len(), &mut out), Ok(()));
        assert_eq!(out.len(), 2 * FRAME_SAMPLES);
        assert!(
            d.ch.iter()
                .all(|ch| ch.cur_subframe == 1 && ch.num_subframes == 1)
        );

        // Cut one bit short, the last subframe bit reads as a zero past the
        // end and the frame is lost.
        let mut d = Decoder::new(2, 44_100);
        assert_eq!(
            d.decode(&bits_of(&bits), bits.len() - 1, &mut out),
            Err(Error("subframe overread"))
        );
        assert_eq!(out.len(), 4 * FRAME_SAMPLES);
    }

    /// Samples in [-1, 1) scale to 16 bits, rounding half away from zero and
    /// saturating; NaN becomes silence.
    #[test]
    fn samples_round_and_saturate_to_16_bits() {
        let s = 1.0 / 32768.0;
        for (v, want) in [
            (0.0, 0),
            (0.5 * s, 1),
            (-0.5 * s, -1),
            (0.49 * s, 0),
            (-0.4 * s, 0),
            (100.0 * s, 100),
            (1.0, 32767),
            (-1.0, -32768),
            (7.5, 32767),
            (-7.5, -32768),
            (f32::INFINITY, 32767),
            (f32::NEG_INFINITY, -32768),
            (f32::NAN, 0),
        ] {
            assert_eq!(to_i16(v), want, "{v}");
        }
    }
}
