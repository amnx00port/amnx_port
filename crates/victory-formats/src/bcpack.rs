//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format,
//! kept in `bcpack-tables.bin` beside this file ([`Tables::builtin`]).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const HISTORY: usize = 256;
const DICTIONARY: usize = 255;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;

/// Bytes of the table file (see [`Tables::parse`]).
pub const TABLES_SIZE: usize = MODELS * BUCKETS * 2
    + 256
    + 64
    + 256
    + 64
    + 64
    + 512 * 2
    + DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);

/// Endpoints and indices of BC1, BC3, BC4 and BC5: small unsigned values,
/// summed per element modulo the element's width.
#[derive(Clone, Copy, Default)]
struct Plain {
    ep: [u8; 8],
    ix: [u8; 32],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ep {
    /// Endpoint 0 R G B, then endpoints 1-3, left-aligned and biased.
    e: [[u16; 3]; 4],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ix {
    mode: u8,
    ix: [i16; 16],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ep {
    ctrl: u16,
    /// Per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned.
    rows: [[u16; 8]; 3],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ix {
    bits: [u8; 2],
    ix: [[i16; 16]; 2],
}

/// The format's constant tables.
pub struct Tables {
    freq: [[u32; BUCKETS]; MODELS],
    /// Each bucket's first slot; 65536 past the model's last, and one more
    /// 65536 at the end so a bucket always has a next start.
    start: [[u32; BUCKETS + 1]; MODELS],
    /// Per model, the bucket of the first slot of each 256-slot window: the
    /// search for a slot's bucket starts there and seldom moves on.
    bucket: Box<[[u8; 256]; MODELS]>,
    /// ceil(2^32 / freq): exact division for the offsets inside a bucket.
    recip: [[u64; BUCKETS]; MODELS],
    rank1: [u8; 256],
    rank3a: [u8; 64],
    rank3c: [u8; 256],
    rank4: [u8; 64],
    rank5: [u8; 64],
    ctrl7: [u16; 512],
    bc1: Vec<Plain>,
    bc3: Vec<Plain>,
    bc4: Vec<Plain>,
    bc5: Vec<Plain>,
    bc6: Vec<(Bc6Ep, Bc6Ix)>,
    bc7: Vec<(Bc7Ep, Bc7Ix)>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        head.try_into().unwrap_or([0; N])
    }
    fn u8(&mut self) -> u8 {
        self.bytes::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes())
    }
}

impl Tables {
    /// The tables VIctory ships with.
    pub fn builtin() -> Result<Tables, Error> {
        Tables::parse(include_bytes!("bcpack-tables.bin"))
    }

    /// Reads the table file: 128 x 9 u16 model frequencies (per bucket, the
    /// frequency of each of its symbols), the rank tables (BC1 256 B, BC3
    /// alpha 64 B and colour 256 B, BC4 64 B, BC5 64 B), 512 u16 BC7 control
    /// words, then 255 dictionary entries per format in format order: BC1
    /// 6 + 16 B, BC3 8 + 32 B, BC4 2 + 16 B, BC5 4 + 32 B, BC6H (u8 mode,
    /// 12 u16, u8 mode, 16 i16), BC7 (u16 control, 24 u16, 2 u8, 32 i16).
    /// Little-endian, no padding. Every model must tile 65536.
    pub fn parse(bytes: &[u8]) -> Result<Tables, Error> {
        if bytes.len() != TABLES_SIZE {
            return Err(Error("table file has the wrong size"));
        }
        let mut r = Reader(bytes);
        let mut freq = [[0u32; BUCKETS]; MODELS];
        let mut start = [[0u32; BUCKETS + 1]; MODELS];
        let mut bucket = Box::new([[0u8; 256]; MODELS]);
        let mut recip = [[0u64; BUCKETS]; MODELS];
        for m in 0..MODELS {
            for f in &mut freq[m] {
                *f = u32::from(r.u16());
            }
            // Model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols.
            let used = m / 16 + 2;
            let mut at = 0u32;
            for b in 0..used {
                if freq[m][b] == 0 {
                    return Err(Error("a model has an empty bucket"));
                }
                start[m][b] = at;
                recip[m][b] = (1u64 << 32).div_ceil(u64::from(freq[m][b]));
                at += freq[m][b] << b.saturating_sub(1);
            }
            if at != 1 << 16 {
                return Err(Error("a model does not tile 65536"));
            }
            start[m][used..].fill(1 << 16);
            for (window, b) in bucket[m].iter_mut().enumerate() {
                let first = (window as u32) << 8;
                *b = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
            }
        }
        let rank1 = r.bytes();
        let rank3a = r.bytes();
        let rank3c = r.bytes();
        let rank4 = r.bytes();
        let rank5 = r.bytes();
        let mut ctrl7 = [0u16; 512];
        for c in &mut ctrl7 {
            *c = r.u16();
        }
        let mut plain = |ep: usize, ix: usize| {
            (0..DICTIONARY)
                .map(|_| {
                    let mut p = Plain::default();
                    for v in &mut p.ep[..ep] {
                        *v = r.u8();
                    }
                    for v in &mut p.ix[..ix] {
                        *v = r.u8();
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        let bc1 = plain(6, 16);
        let bc3 = plain(8, 32);
        let bc4 = plain(2, 16);
        let bc5 = plain(4, 32);
        let bc6 = (0..DICTIONARY)
            .map(|_| {
                // The endpoint entry's mode byte plays no part in decoding.
                r.u8();
                let mut ep = Bc6Ep::default();
                for v in ep.e.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc6Ix {
                    mode: r.u8(),
                    ..Default::default()
                };
                for v in &mut ix.ix {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        let bc7 = (0..DICTIONARY)
            .map(|_| {
                let mut ep = Bc7Ep {
                    ctrl: r.u16(),
                    ..Default::default()
                };
                for v in ep.rows.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc7Ix {
                    bits: [r.u8(), r.u8()],
                    ..Default::default()
                };
                for v in ix.ix.as_flattened_mut() {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        Ok(Tables {
            freq,
            start,
            bucket,
            recip,
            rank1,
            rank3a,
            rank3c,
            rank4,
            rank5,
            ctrl7,
            bc1,
            bc3,
            bc4,
            bc5,
            bc6,
            bc7,
        })
    }
}

/// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error. On an error
/// `dst` is left as it was, even when the bad block comes after good ones.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let (block, widths, per) = layout(bc)?;
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let blocks = dst.len() / block;
    let mut records = vec![0u8; blocks * widths.len() * per];
    let mut next = src.iter().copied();
    lanes(tables, widths, per, blocks, |_| next.next(), &mut records)?;
    // BC6H modes and BC7 control codes are checked block by block while
    // unpacking, so unpack aside and copy only a whole success.
    let mut out = vec![0u8; dst.len()];
    unpack(bc, &records, &mut out, tables)?;
    dst.copy_from_slice(&out);
    Ok(())
}

/// Block bytes, lane widths and symbols per lane per block of BC `bc`.
fn layout(bc: u8) -> Result<(usize, &'static [u8], usize), Error> {
    Ok(match bc {
        1 => (8, &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (16, &[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (8, &[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (16, &[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        6 => (16, &[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
        7 => (16, &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
        _ => return Err(Error("unsupported BC format")),
    })
}

/// Turns the lanes' records (one per block of `dst`) into BC blocks.
fn unpack(bc: u8, records: &[u8], dst: &mut [u8], t: &Tables) -> Result<(), Error> {
    let (block, widths, per) = layout(bc)?;
    let records = records.chunks_exact(widths.len() * per);
    let out = dst.chunks_exact_mut(block);
    match bc {
        1 => plain(records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
        3 => plain(records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
        4 => plain(records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
        5 => plain(records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
        6 => bc6(records, out, &t.bc6),
        _ => bc7(records, out, t),
    }
}

// Lanes.

#[derive(Clone, Copy, Default)]
struct Lane {
    /// Symbol bits - 1 at the full alphabet.
    width: u8,
    /// 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    phase: u8,
    long: bool,
    reset: bool,
    model: u8,
    need: bool,
    latch: Option<u8>,
    out: Option<u8>,
    stalled: bool,
    x: u32,
    left: u32,
}

impl Lane {
    /// One step: take the latched byte (header, length or payload) and
    /// decode at most one symbol.
    fn step(&mut self, t: &Tables) -> Result<(), Error> {
        if self.phase != 3 {
            let Some(b) = self.latch.take() else {
                return Ok(());
            };
            self.need = true;
            match self.phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = self.width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.reset = b & 0x10 != 0;
                    self.long = b & 0x20 != 0;
                    self.left = 0;
                    self.phase = 1;
                }
                1 if self.long => {
                    self.left = b.into();
                    self.phase = 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.phase = 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.phase = 3;
                }
            }
            return Ok(());
        }
        let mut x = self.x;
        if x < LOW
            && let Some(b) = self.latch.take()
        {
            self.left = self.left.wrapping_sub(1);
            x = x << 8 | u32::from(b);
        }
        self.need = self.left != 0 && self.latch.is_none();
        if x > LOW {
            let freq = &t.freq[usize::from(self.model)];
            let start = &t.start[usize::from(self.model)];
            let slot = x & 0xffff;
            // The bucket: starts are ascending, and 65536 past the model's last.
            let b = bucket_of(t, self.model, slot);
            // Bucket 0 holds symbol 0, which owns slots 1 .. f[0] - 1 (slot 0
            // is never used). Bucket b > 0 holds 2^(b-1) symbols of frequency
            // f[b] from start[b] on; offset and frequency are below 2^16, where
            // the reciprocal divides exactly.
            let (sym, f, rem) = if b == 0 {
                (0, freq[0] - 1, slot.wrapping_sub(1))
            } else {
                let offset = slot - start[b];
                let q = ((u64::from(offset) * t.recip[usize::from(self.model)][b]) >> 32) as u32;
                ((1u32 << (b - 1)) + q, freq[b], offset - q * freq[b])
            };
            self.x = (x >> 16).wrapping_mul(f).wrapping_add(rem);
            self.out = Some(sym as u8);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.phase = 0;
            self.need = self.latch.is_none();
            if self.reset {
                self.x = 0;
            }
        }
        Ok(())
    }
}

/// The bucket holding `slot` (below 65536): the last whose start is at or
/// below it. The window table gives the bucket of the window's first slot;
/// later buckets that start inside the window are stepped over.
#[inline]
fn bucket_of(t: &Tables, model: u8, slot: u32) -> usize {
    let start = &t.start[usize::from(model)];
    let mut b = usize::from(t.bucket[usize::from(model)][(slot >> 8) as usize]);
    while start[b + 1] <= slot {
        b += 1;
    }
    b
}

/// Decodes `per` symbols per block on every lane into `records` (one record
/// of `widths.len() * per` symbols per block), taking input from `fetch`
/// (given the lane that takes the byte) in the order of the schedule: each
/// round the lanes step in order and take at most 8 bytes between them.
// shortcut: one lane step at a time (BC1 roughly 30-60 MB/s, BC7 15-20 MB/s
// on one core); if texture streaming needs more, run the schedule only to
// split the input per lane and decode the lanes apart.
fn lanes(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    mut fetch: impl FnMut(usize) -> Option<u8>,
    records: &mut [u8],
) -> Result<(), Error> {
    let mut lanes = [Lane::default(); 16];
    let lanes = &mut lanes[..widths.len()];
    for (lane, &width) in lanes.iter_mut().zip(widths) {
        lane.width = width;
        lane.need = true;
    }
    let stride = widths.len() * per;
    let full = (1u32 << widths.len()) - 1;
    let (mut pos, mut done, mut idle) = (0, 0u32, 0);
    // Where symbol `pos` of lane 0 goes; lane j's is `per * j` further on.
    let mut at = 0;
    while pos < blocks * per {
        let (mut taken, mut moved) = (0, false);
        for (j, lane) in lanes.iter_mut().enumerate() {
            let bit = 1 << j;
            let before = taken;
            if !lane.stalled {
                lane.step(t)?;
            }
            let waiting = lane.out.is_some() && done & bit != 0;
            if lane.need && !waiting && taken < ROUND_BYTES {
                // Past the end nothing arrives; in a valid stream only finished lanes ask there.
                if let Some(b) = fetch(j) {
                    lane.latch = Some(b);
                    taken += 1;
                }
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled
                && let Some(sym) = lane.out.take()
            {
                records[at + j * per] = sym;
                done |= bit;
                moved = true;
            }
        }
        if done == full {
            pos += 1;
            at += if pos.is_multiple_of(per) {
                stride - per + 1
            } else {
                1
            };
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}

// Blocks.

/// The record's endpoint and index references.
fn references(s: &[u8], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (interleave(s[0], s[1]), interleave(s[2], s[3]))
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Two 4-bit symbols, the first on the even bits.
fn interleave(a: u8, b: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
    spread(a) | spread(b) << 1
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
fn pick<'a, T>(i: usize, r: usize, history: &'a [T], dictionary: &'a [T], zero: &'a T) -> &'a T {
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
fn zz(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Element widths of the plain formats' endpoints and indices.
struct Widths {
    ep: [u8; 8],
    ix: [u8; 32],
    paired: bool,
}

const fn widths(ep: [u8; 8], ix3: usize, ix2: usize, paired: bool) -> Widths {
    let mut ix = [0; 32];
    let mut k = 0;
    while k < ix3 + ix2 {
        ix[k] = if k < ix3 { 3 } else { 2 };
        k += 1;
    }
    Widths { ep, ix, paired }
}

const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16, false);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16, true);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0, false);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0, true);

fn plain<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[Plain],
    w: &Widths,
    residual: impl Fn(&[u8]) -> Plain,
    pack: fn(&Plain, &mut [u8]),
) -> Result<(), Error> {
    let mut history = vec![Plain::default(); HISTORY];
    let zero = Plain::default();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, w.paired);
        let ep_ref = &pick(i, ep_ref, &history, dictionary, &zero).ep;
        let ix_ref = &pick(i, ix_ref, &history, dictionary, &zero).ix;
        let mut v = residual(s);
        for ((v, r), bits) in v.ep.iter_mut().zip(ep_ref).zip(w.ep) {
            *v = v.wrapping_add(*r) & ((1u16 << bits) - 1) as u8;
        }
        for ((v, r), bits) in v.ix.iter_mut().zip(ix_ref).zip(w.ix) {
            *v = v.wrapping_add(*r) & ((1u8 << bits) - 1);
        }
        history[i % HISTORY] = v;
        pack(&v, block);
    }
    Ok(())
}

/// 2-bit index residuals: one symbol per 2x2 quad.
fn quads(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            ix[base + p] = v >> (2 * k) & 3;
        }
    }
}

/// 3-bit index residuals: one symbol per two pixels in raster order.
fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = v >> 4 & 7;
    }
}

fn bc1_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
        v.ep[k] = zz(s[2 + k], bits);
    }
    quads(&t.rank1, &s[8..12], &mut v.ix);
    v
}

fn bc3_residual(t: &Tables, s: &[u8]) -> Plain {
    // Colour lanes carry each channel's two endpoints together.
    let mut v = Plain {
        ep: [
            zz(s[14], 5),
            zz(s[16], 6),
            zz(s[18], 5),
            zz(s[15], 5),
            zz(s[17], 6),
            zz(s[19], 5),
            zz(s[4], 8),
            zz(s[5], 8),
        ],
        ..Default::default()
    };
    pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
    quads(&t.rank3c, &s[20..24], &mut v.ix[16..]);
    v
}

fn bc4_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    v.ep[..2].copy_from_slice(&[zz(s[2], 8), zz(s[3], 8)]);
    pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
    v
}

fn bc5_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for k in 0..4 {
        v.ep[k] = zz(s[4 + k], 8);
    }
    pairs(&t.rank5, &s[8..24], &mut v.ix);
    v
}

/// Indices along the endpoint line -> BC1 and BC4 codes.
const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

fn bc1_color(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let bits = (0..16).fold(0, |acc, p| {
        acc | BC1_CODE[usize::from(ix[p] & 3)] << (2 * p)
    });
    out[..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
}

/// BC4 half: endpoints are kept 128 below their stored value.
fn bc4_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let bits = (0..16).fold(0u64, |acc, p| {
        acc | BC4_CODE[usize::from(ix[p] & 7)] << (16 + 3 * p)
    });
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
    out[..8].copy_from_slice(&v.to_le_bytes());
}

fn bc1_pack(v: &Plain, out: &mut [u8]) {
    bc1_color(&v.ep, &v.ix, out);
}

fn bc3_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
    bc1_color(&v.ep, &v.ix[16..], &mut out[8..]);
}

fn bc4_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix, out);
}

fn bc5_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
    bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
}

/// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
fn nibbles(b: u8, odd_first: bool) -> [i16; 2] {
    let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
    let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (z(half(b)), z(half(b >> 1)));
    if odd_first { [odd, even] } else { [even, odd] }
}

/// Index residual plus reference, both left-aligned to 16 bits at their own
/// widths, back to `bits` bits (signed).
fn add_index(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
    let up = |v: i16, b: u8| {
        if b == 0 || b > 16 {
            0
        } else {
            (v as u16) << (16 - b)
        }
    };
    let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

/// Little-endian bit writer for one 128-bit block.
#[derive(Default)]
struct Bits {
    v: u128,
    at: u32,
}

impl Bits {
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 && self.at < 128 {
            self.v |= (u128::from(value) & ((1u128 << n) - 1)) << self.at;
        }
        self.at += n;
    }
}

// BC7.

/// Public BC7 anchor indices: second subset of two; second and third of three.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
const BC7_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the primary and secondary sets per mode.
const BC7_INDEX: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];

fn pbits(mode: u16) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// The channel swapped with alpha (0 none). Only modes 4 and 5 rotate; in
/// a control word for another mode the bits mean nothing, and acting on them
/// would leave red without precision.
fn rotation(ctrl: u16) -> usize {
    if matches!(ctrl & 7, 4 | 5) {
        usize::from(ctrl >> 3 & 3)
    } else {
        0
    }
}

/// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
/// with the rotated channel.
fn bc7_precision(ctrl: u16) -> [u8; 4] {
    let (mode, rot) = (ctrl & 7, rotation(ctrl));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let mut history = vec![(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if ctrl == 0 {
            return Err(Error("unused BC7 control code"));
        }
        let mode = usize::from(ctrl & 7);
        let subsets = usize::from((ctrl & 0xff) >> 6);
        let prec = bc7_precision(ctrl);
        let three = subsets == 3;
        let or0 = |v: u8| if three { 0 } else { v };
        let if3 = |v: u8| if three { v } else { 0 };
        let stored = [
            [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
            [
                s[12],
                s[13],
                s[14],
                s[15],
                s[16],
                s[17],
                or0(s[18]),
                or0(s[19]),
            ],
            [
                s[20],
                s[21],
                if3(s[10]),
                if3(s[11]),
                if3(s[18]),
                if3(s[19]),
                0,
                0,
            ],
        ];
        let r = &pick(i, ep_ref, &history, &t.bc7, &zero).0;
        let mut ep = Bc7Ep {
            ctrl,
            ..Default::default()
        };
        // Rows the reference lacks build on its first row.
        let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
        let columns = if prec[3] > 0 { 8 } else { 6 };
        for c in 0..columns {
            let p = prec[c / 2];
            if p == 0 {
                continue;
            }
            let mask = 0xffffu16 << (16 - p);
            for (row, st) in stored.iter().enumerate().take(subsets) {
                let v = st[c];
                let residual =
                    (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << (16 - p);
                let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                ep.rows[row][c] = base.wrapping_add(residual);
            }
        }
        let bits = BC7_INDEX[mode];
        let rx = &pick(i, ix_ref, &history, &t.bc7, &zero).1;
        let mut ix = Bc7Ix {
            bits,
            ..Default::default()
        };
        for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
            if bits[set] == 0 {
                continue;
            }
            for (k, o) in offsets.into_iter().enumerate() {
                for (b, &byte) in s[o..o + 2].iter().enumerate() {
                    let pair = nibbles(byte, k < 3);
                    for (h, &v) in pair.iter().enumerate() {
                        let p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(v, bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
            }
        }
        history[i % HISTORY] = (ep, ix);
        bc7_pack(&ep, &ix, prec, block);
    }
    Ok(())
}

fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, prec: [u8; 4], out: &mut [u8]) {
    let ctrl = ep.ctrl;
    let mode = ctrl & 7;
    let subsets = usize::from((ctrl & 0xff) >> 6);
    let part = usize::from(ctrl >> 8 & 63);
    let rot = rotation(ctrl);
    let mut w = Bits::default();
    w.put(1 << mode, u32::from(mode) + 1);
    match mode {
        0 => w.put(part as u64, 4),
        1 | 2 | 3 | 7 => w.put(part as u64, 6),
        4 => {
            w.put(rot as u64, 2);
            w.put(u64::from(ctrl >> 5 & 1), 1);
        }
        5 => w.put(rot as u64, 2),
        _ => {}
    }
    let ends = 2 * subsets;
    let field = |row: usize, col: usize, p: u8| u64::from(ep.rows[row][col] >> (16 - p));
    let mut order = [0, 1, 2, 3];
    if rot != 0 {
        order.swap(3, rot - 1);
    }
    for ch in order {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        let drop = u32::from(ch == 0 && pbits(mode));
        for e in 0..ends {
            w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32::from(p) - drop);
        }
    }
    match mode {
        1 => (0..subsets).for_each(|row| w.put(field(row, 0, prec[0]) & 1, 1)),
        m if pbits(m) => (0..ends).for_each(|e| w.put(field(e / 2, e % 2, prec[0]) & 1, 1)),
        _ => {}
    }
    let anchors: &[u8] = match subsets {
        2 => &[0, ANCHOR2[part]],
        3 => &[0, ANCHOR3A[part], ANCHOR3B[part]],
        _ => &[0],
    };
    let mut indices = |set: usize| {
        let bits = ix.bits[set];
        for (p, &v) in ix.ix[set].iter().enumerate() {
            let n = u32::from(bits) - u32::from(anchors.contains(&(p as u8)));
            w.put(u64::from(v as u16), n);
        }
    };
    match mode {
        4 => {
            indices(1);
            indices(0);
        }
        5 => {
            indices(0);
            indices(1);
        }
        _ => indices(0),
    }
    out.copy_from_slice(&w.v.to_le_bytes());
}

// BC6H.

/// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
const BC6_BITS: [[u8; 4]; 15] = [
    [0; 4],
    [10, 5, 5, 5],
    [7, 6, 6, 6],
    [11, 5, 4, 4],
    [11, 4, 5, 4],
    [11, 4, 4, 5],
    [9, 5, 5, 5],
    [8, 6, 5, 5],
    [8, 5, 6, 5],
    [8, 5, 5, 6],
    [6, 6, 6, 6],
    [10, 10, 10, 10],
    [11, 9, 9, 9],
    [12, 8, 8, 8],
    [16, 4, 4, 4],
];
/// Mode code and its width, modes 1-14.
const BC6_CODE: [(u64, u32); 15] = [
    (0, 0),
    (0, 2),
    (1, 2),
    (2, 5),
    (6, 5),
    (10, 5),
    (14, 5),
    (18, 5),
    (22, 5),
    (26, 5),
    (30, 5),
    (3, 5),
    (7, 5),
    (11, 5),
    (15, 5),
];

/// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is
/// the partition) of endpoint `ep`, from bit `first` up.
#[derive(Clone, Copy)]
struct Run {
    ch: u8,
    ep: u8,
    first: u8,
    len: u8,
}

const fn r(ch: u8, ep: u8, first: u8, len: u8) -> Run {
    Run { ch, ep, first, len }
}

const R: u8 = 0;
const G: u8 = 1;
const B: u8 = 2;
const D: Run = r(3, 0, 0, 5);

/// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
/// high base bits in ascending order (what the console's packer does).
#[rustfmt::skip]
const BC6_LAYOUT: [&[Run]; 15] = [
    &[],
    &[
        r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
        r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(G, 2, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
        r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(B, 3, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
        r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
        r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
        r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
        r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
        r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
        r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
    ],
];

fn bc6_index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

fn bc6<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[(Bc6Ep, Bc6Ix)],
) -> Result<(), Error> {
    let mut history = vec![(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ..Default::default()
        },
    );
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    let wide = |a: u8, b: u8| {
        (0..8).fold(0u16, |acc, k| {
            acc | u16::from(a >> k & 1) << (2 * k) | u16::from(b >> k & 1) << (2 * k + 1)
        })
    };
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error("BC6H mode out of range"));
        }
        let two = mode < 11;
        let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
        let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
        let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
        let (e2, e3) = if two {
            (
                [s[16], s[18], s[20]].map(|v| z(v.into())),
                [s[17], s[19], s[21]].map(|v| z(v.into())),
            )
        } else {
            (e0, e1)
        };
        let r = &pick(i, ep_ref, &history, dictionary, &zero).0;
        let mut ep = Bc6Ep::default();
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        let mask = |sh: u32| 0xffffu16 << sh;
        for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
            *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
        }
        for (k, res) in [e1, e2, e3]
            .iter()
            .enumerate()
            .take(if two { 3 } else { 1 })
        {
            for c in 0..3 {
                ep.e[k + 1][c] =
                    r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
            }
        }
        if !two {
            for c in 0..3 {
                ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
            }
            ep.e[3] = ep.e[1];
        }
        let rx = &pick(i, ix_ref, &history, dictionary, &zero).1;
        let bits = bc6_index_bits(mode);
        let mut ix = Bc6Ix {
            mode,
            ..Default::default()
        };
        for k in 0..8 {
            for (h, v) in nibbles(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                ix.ix[p] = add_index(v, bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = (ep, ix);
        // Fields: the working value has its top bit flipped.
        let field = |k: usize, c: usize| {
            let sh = if k == 0 { shift[0] } else { shift[1 + c] };
            u64::from((ep.e[k][c] ^ 0x8000) >> sh)
        };
        let mut w = Bits::default();
        let (code, n) = BC6_CODE[usize::from(mode)];
        w.put(code, n);
        let part = s[31] & 31;
        for run in BC6_LAYOUT[usize::from(mode)] {
            if run.ch == 3 {
                w.put(part.into(), 5);
            } else {
                w.put(
                    field(run.ep.into(), run.ch.into()) >> run.first,
                    run.len.into(),
                );
            }
        }
        let anchor = if two { ANCHOR2[usize::from(part)] } else { 0 };
        for (p, &v) in ix.ix.iter().enumerate() {
            let short = p == 0 || p == usize::from(anchor);
            let v = (v as u16).wrapping_add(1 << (bits - 1));
            w.put(v.into(), u32::from(bits) - u32::from(short));
        }
        block.copy_from_slice(&w.v.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shipped_tables_parse() {
        assert!(super::Tables::builtin().is_ok());
    }

    use super::*;

    /// Tables we made: uniform models, identity ranks, a BC7 code per mode,
    /// empty dictionaries.
    fn tables() -> Vec<u8> {
        let mut t = Vec::new();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend((0..=255u8).map(|v| v & 0x77)); // rank1 (unused by the tests beyond 2-bit fields)
        t.extend(0..64u8); // rank3a
        t.extend(0..=255u8); // rank3c
        t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4)); // rank4: low 3 bits, next 3 bits
        t.extend(0..64u8); // rank5
        for code in 0..512u16 {
            // Code c: mode 6 (one subset), the only one the tests use.
            let ctrl: u16 = if code == 0 { 0x46 } else { 0 };
            t.extend_from_slice(&ctrl.to_le_bytes());
        }
        t.resize(TABLES_SIZE, 0);
        t
    }

    /// A lane's symbols as one fragment: uniform model, rANS from 2^24, no
    /// reset, so the lane keeps 2^24 when it ends.
    fn fragment(width: u8, symbols: &[u8]) -> Vec<u8> {
        let f = 1u32 << (15 - width);
        let mut x = LOW;
        let mut bytes = Vec::new();
        for &s in symbols.iter().rev() {
            let (freq, start) = if s == 0 {
                (f - 1, 1)
            } else {
                (f, u32::from(s) * f)
            };
            while x >= (freq << 16) {
                bytes.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << 16) + x % freq + start;
        }
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.reverse();
        let mut out = vec![0];
        if bytes.len() > 256 {
            out[0] |= 0x20;
            let n = bytes.len() - 257;
            out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
        } else {
            out.push((bytes.len() - 1) as u8);
        }
        out.extend(bytes);
        out
    }

    /// Interleaves per-lane fragments the way the decoder takes bytes. A lane
    /// that has finished still asks: a filler fragment (two payload bytes)
    /// brings its state to 2^24 with a byte latched, which parks it for good.
    fn stream(t: &Tables, widths: &[u8], per: usize, records: &[Vec<u8>]) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = fragment(widths[j], &syms);
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    const BC4_LANES: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];

    fn zigzag(v: i32) -> u8 {
        if v >= 0 {
            (2 * v) as u8
        } else {
            (-2 * v - 1) as u8
        }
    }

    #[test]
    fn bc4_blocks_with_and_without_references() {
        let t = Tables::parse(&tables()).unwrap();
        // Block 0: endpoints 10, -3 (+128 stored), indices p -> p & 7.
        let mut b0 = vec![0, 0, zigzag(10), zigzag(-3)];
        b0.extend((0..8).map(|k| ((2 * k) & 7) | ((2 * k + 1) & 7) << 3));
        // Block 1 refers to block 0 for both: residual endpoints +1, +1; indices +1.
        let mut b1 = vec![0, 1, zigzag(1), zigzag(1)];
        b1.extend((0..8).map(|_| 1 | 1 << 3));
        let src = stream(&t, &BC4_LANES, 1, &[b0, b1]);
        let mut dst = [0u8; 16];
        decode(4, &src, &mut dst, &t).unwrap();
        let block = |e0: u8, e1: u8, ix: &dyn Fn(usize) -> usize| {
            let bits = (0..16).fold(0u64, |a, p| a | BC4_CODE[ix(p)] << (16 + 3 * p));
            (u64::from(e0) | u64::from(e1) << 8 | bits).to_le_bytes()
        };
        assert_eq!(dst[..8], block(138, 125, &|p| p & 7));
        assert_eq!(dst[8..], block(139, 126, &|p| (p + 1) & 7));
    }

    #[test]
    fn bc5_endpoint_and_index_references_cross_the_history_boundary() {
        let mut t = Tables::parse(&tables()).unwrap();
        t.bc5[0].ep[..4].copy_from_slice(&[1, 2, 3, 4]);
        t.bc5[254].ix[..16].fill(7);
        t.bc5[254].ix[16..].fill(3);

        let mut records = vec![vec![0; 24]; 258];
        // Index distance 255 plus endpoint delta +2 wraps to distance 1.
        // At block zero these select opposite ends of the dictionary.
        records[0][..4].copy_from_slice(&[2, 0, 15, 15]);
        records[1][4..8].copy_from_slice(&[20, 40, 60, 80]);
        // Only the first red pixel has a nonzero index in block 2.
        records[2][8] = 1;
        // Block 256 copies block 1. Block 257 takes its endpoints from
        // block 256 (now in slot zero) and its indices from block 2.
        records[256][..4].copy_from_slice(&[0, 0, 15, 15]);
        records[257][..4].copy_from_slice(&[2, 0, 15, 15]);

        let widths = [3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
        let src = stream(&t, &widths, 2, &records);
        let mut dst = vec![0; 16 * records.len()];
        decode(5, &src, &mut dst, &t).unwrap();

        // BC5 stores two BC4 blocks. Six bytes per channel hold sixteen
        // little-endian 3-bit codes, here all 1 for red and 4 for green.
        let dictionary = [
            129, 130, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24, // red
            131, 132, 0x24, 0x49, 0x92, 0x24, 0x49, 0x92, // green
        ];
        let endpoints = [138, 148, 0, 0, 0, 0, 0, 0, 158, 168, 0, 0, 0, 0, 0, 0];
        let empty = [128, 128, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0];
        for (i, block) in dst.as_chunks::<16>().0.iter().enumerate() {
            let mut expected = match i {
                0 => dictionary,
                1 | 256 | 257 => endpoints,
                _ => empty,
            };
            if i == 2 || i == 257 {
                expected[2] = 2;
            }
            assert_eq!(*block, expected, "BC5 block {i}");
        }
    }

    #[test]
    fn lanes_take_bytes_in_schedule_order_and_long_fragments() {
        let t = Tables::parse(&tables()).unwrap();
        // Many blocks: long fragments (over 256 payload bytes) on the wide lanes.
        let records: Vec<Vec<u8>> = (0..400u32)
            .map(|i| {
                let mut r = vec![0, 0, (i * 7) as u8, (i * 13) as u8];
                r.extend((0..8).map(|k| ((i + k) * 5 % 64) as u8));
                r
            })
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut got = vec![0u8; records.len() * 12];
        let mut next = src.iter().copied();
        lanes(&t, &BC4_LANES, 1, records.len(), |_| next.next(), &mut got).unwrap();
        assert_eq!(got, records.concat());
    }

    #[test]
    fn bc7_mode6_block() {
        let t = Tables::parse(&tables()).unwrap();
        // Control code 0 -> mode 6; endpoints R 8 bits (p-bit low), G B A 7 bits.
        let mut s = vec![0u8; 32];
        s[4] = zigzag(0x55); // R0 = 0x55: field 0x2a, p-bit 1
        s[5] = zigzag(-1); // R1 = 0xff (8 bits)
        s[6] = zigzag(3);
        s[10] = zigzag(0x7f >> 1);
        let src = stream(
            &t,
            &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4],
            2,
            &[s],
        );
        let mut dst = [0u8; 16];
        decode(7, &src, &mut dst, &t).unwrap();
        let mut w = Bits::default();
        w.put(1 << 6, 7);
        for (v, n) in [
            (0x2a, 7),
            (0x7f, 7),
            (3, 7),
            (0, 7),
            (0, 7),
            (0, 7),
            (0x3f, 7),
            (0, 7),
        ] {
            w.put(v, n);
        }
        w.put(1, 1);
        w.put(1, 1);
        assert_eq!(dst, w.v.to_le_bytes());
    }

    #[test]
    fn malformed_input_is_an_error() {
        let t = Tables::parse(&tables()).unwrap();
        let records: Vec<Vec<u8>> = (0..64u8)
            .map(|i| vec![0, 0, i, i, 1, 2, 3, 4, 5, 6, 7, 8])
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut dst = vec![0u8; 64 * 8];
        decode(4, &src, &mut dst, &t).unwrap();
        // Truncated.
        assert!(decode(4, &src[..src.len() / 2], &mut dst, &t).is_err());
        // Headers promising more payload than the stream holds. (Other corrupt
        // headers decode to garbage: the format has no redundancy.)
        let headers = [[0x30; 8], [0xff; 8]].concat();
        assert!(decode(4, &headers, &mut dst[..8], &t).is_err());
        assert!(decode(4, &[], &mut dst, &t).is_err());
        // Destination not a whole number of blocks, empty, more blocks than coded.
        assert!(decode(4, &src, &mut dst[..12], &t).is_err());
        assert!(decode(4, &src, &mut [], &t).is_err());
        assert!(decode(4, &src, &mut vec![0u8; 65 * 8], &t).is_err());
        // Unsupported formats, bad tables.
        assert!(decode(2, &src, &mut dst, &t).is_err());
        assert!(Tables::parse(&[0; 16]).is_err());
        let mut no_tiling = tables();
        no_tiling[0] = 1;
        assert!(Tables::parse(&no_tiling).is_err());
    }

    /// `Tables::parse` takes exactly `TABLES_SIZE` bytes, and every model's
    /// used buckets must be non-empty and tile 65536 exactly, in the first,
    /// a middle and the last model of each width group. Buckets past a
    /// model's used ones are never read.
    /// Every BC number but 1 and 3-7 is refused, and no error writes to
    /// `dst`: not a bad format or size, not a stream that runs out, and not
    /// a bad BC6H mode or BC7 control code in a block after good ones.
    #[test]
    fn errors_leave_the_destination_alone() {
        let t = Tables::parse(&tables()).unwrap();
        let mut dst = vec![0xa5u8; 4 * 16];
        for bc in (0..=255u8).filter(|bc| !matches!(bc, 1 | 3..=7)) {
            assert!(decode(bc, &[0; 64], &mut dst, &t).is_err(), "BC{bc}");
        }
        assert!(decode(4, &[0; 64], &mut dst[..12], &t).is_err());
        assert!(decode(4, &[], &mut dst, &t).is_err());
        assert!(dst.iter().all(|&b| b == 0xa5));

        const BC6_LANES: [u8; 16] = [3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4];
        const BC7_LANES: [u8; 16] = [3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4];
        // Records of 32 symbols: byte 30 picks the BC6H mode (0-13 valid)
        // and, with byte 31, the BC7 control code (only code 0 is used by
        // these tables).
        let record = |b30: u8| {
            let mut r = vec![0u8; 32];
            r[30] = b30;
            r
        };
        for (bc, widths, bad) in [(6, &BC6_LANES, 14), (7, &BC7_LANES, 1)] {
            let good: Vec<Vec<u8>> = (0..4).map(|_| record(0)).collect();
            let src = stream(&t, widths, 2, &good);
            let mut want = vec![0u8; 4 * 16];
            decode(bc, &src, &mut want, &t).unwrap();
            for at in 0..4 {
                let mut records = good.clone();
                records[at] = record(bad);
                let src = stream(&t, widths, 2, &records);
                let mut dst = vec![0xa5u8; 4 * 16];
                assert!(decode(bc, &src, &mut dst, &t).is_err(), "BC{bc} {at}");
                assert!(dst.iter().all(|&b| b == 0xa5), "BC{bc} bad block {at}");
                // The good blocks before it still decode on their own.
                if at > 0 {
                    let mut part = vec![0u8; at * 16];
                    decode(bc, &src, &mut part, &t).unwrap();
                    assert_eq!(part, want[..at * 16], "BC{bc} {at}");
                }
            }
        }
    }

    #[test]
    fn tables_parse_checks_size_and_every_model() {
        let base = tables();
        let wrong = Some(Error("table file has the wrong size"));
        assert_eq!(Tables::parse(&base[..TABLES_SIZE - 1]).err(), wrong);
        assert_eq!(
            Tables::parse(&[base.as_slice(), &[0]].concat()).err(),
            wrong
        );
        assert_eq!(Tables::parse(&[]).err(), wrong);
        assert_eq!(
            Error("table file has the wrong size").to_string(),
            "BCPack: table file has the wrong size"
        );
        let freq_at = |m: usize, b: usize| (m * BUCKETS + b) * 2;
        let with = |edits: &[(usize, usize, u16)]| {
            let mut t = base.clone();
            for &(m, b, f) in edits {
                t[freq_at(m, b)..][..2].copy_from_slice(&f.to_le_bytes());
            }
            Tables::parse(&t).err()
        };
        for group in 0..MODELS / 16 {
            let used = group + 2;
            let f = 1u16 << (15 - group);
            for m in [16 * group, 16 * group + 7, 16 * group + 15] {
                for b in 0..used {
                    assert_eq!(
                        with(&[(m, b, 0)]),
                        Some(Error("a model has an empty bucket")),
                        "model {m} bucket {b}"
                    );
                }
                // One unit more or less in bucket 0 breaks the tiling.
                for f0 in [f - 1, f + 1] {
                    assert_eq!(
                        with(&[(m, 0, f0)]),
                        Some(Error("a model does not tile 65536")),
                        "model {m}"
                    );
                }
                // Moving weight between buckets keeps it: the last bucket
                // holds 2^(used - 2) symbols per unit of frequency.
                let wide = 1u16 << (used - 2);
                assert_eq!(
                    with(&[(m, 0, f + wide), (m, used - 1, f - 1)]),
                    None,
                    "model {m}"
                );
                // Unused buckets are free.
                for b in used..BUCKETS {
                    assert_eq!(with(&[(m, b, 0xffff)]), None, "model {m} bucket {b}");
                }
            }
        }
    }

    /// The stream carries no length: bytes after the ones the lanes need
    /// change nothing, and asking for fewer blocks than were coded gives
    /// the first ones unchanged.
    #[test]
    fn trailing_bytes_and_fewer_blocks_are_ignored() {
        let t = Tables::parse(&tables()).unwrap();
        let records: Vec<Vec<u8>> = (0..48u8)
            .map(|i| {
                let mut r = vec![0, u8::from(i > 0), i.wrapping_mul(3), i ^ 0x15];
                r.extend((0..8).map(|k| (i + k) % 64));
                r
            })
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut want = vec![0u8; records.len() * 8];
        decode(4, &src, &mut want, &t).unwrap();
        for tail in [&[0u8][..], &[0xff; 3], &[0x5a; 1000]] {
            let mut got = vec![0u8; want.len()];
            decode(4, &[src.as_slice(), tail].concat(), &mut got, &t).unwrap();
            assert_eq!(got, want, "tail of {} bytes", tail.len());
        }
        for blocks in [1, 2, 17, 47] {
            let mut got = vec![0u8; blocks * 8];
            decode(4, &src, &mut got, &t).unwrap();
            assert_eq!(got, want[..blocks * 8], "{blocks} blocks");
        }
    }

    #[test]
    fn noise_never_panics() {
        let t = Tables::parse(&tables()).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..300 {
            let len = round * 7 % 900;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            for bc in [1, 3, 4, 5, 6, 7] {
                let mut dst = vec![0u8; 16 * (1 + round % 40)];
                let _ = decode(bc, &src, &mut dst, &t);
            }
        }
    }

    /// Records the lanes can deliver (each lane's symbols below 2^(width + 1))
    /// through the block stage with the shipped tables, so real dictionary
    /// entries and BC7 control codes are the references; then the same with
    /// those tables' ranks, control codes and dictionaries overwritten by
    /// noise, which `Tables::parse` accepts. Noise streams also go through
    /// `decode` with the shipped tables. Some record sets must decode and some
    /// must be errors (BC6H modes past 14, unused BC7 control codes).
    #[test]
    fn shipped_tables_never_panic() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let shipped = include_bytes!("bcpack-tables.bin");
        let mut noisy = shipped.to_vec();
        for b in &mut noisy[MODELS * BUCKETS * 2..] {
            *b = next() as u8;
        }
        let (mut ok, mut err) = (0, 0);
        for (pass, bytes) in [&shipped[..], &noisy].into_iter().enumerate() {
            let t = Tables::parse(bytes).unwrap();
            for round in 0..64 {
                for bc in [1, 3, 4, 5, 6, 7] {
                    let (block, widths, per) = layout(bc).unwrap();
                    // Up to 300 blocks: references reach back past the 256-block history.
                    let blocks = 1 + next() as usize % 300;
                    let records: Vec<u8> = (0..blocks)
                        .flat_map(|_| widths.iter().flat_map(|&w| [w; 2][..per].to_vec()))
                        .map(|w| (next() % (2 << w)) as u8)
                        .collect();
                    let mut dst = vec![0xa5u8; blocks * block];
                    match unpack(bc, &records, &mut dst, &t) {
                        Ok(()) => ok += 1,
                        Err(_) => err += 1,
                    }
                    if pass == 0 && round % 8 == 0 {
                        let src: Vec<u8> = (0..next() % 2000).map(|_| next() as u8).collect();
                        let _ = decode(bc, &src, &mut dst, &t);
                    }
                }
            }
        }
        assert!(ok > 0 && err > 0);
    }

    /// Tables whose models tile 65536 in shapes the shipped file never uses:
    /// one bucket holding nearly all of the range and every other bucket a
    /// frequency of 1, or a random split. `Tables::parse` accepts them, so
    /// noise streams decoded with them must give blocks or errors, never a
    /// panic or an arithmetic overflow in the range decoder.
    #[test]
    fn extreme_models_never_panic() {
        let mut seed = 0x6a09_e667_f3bc_c908u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let shipped = include_bytes!("bcpack-tables.bin");
        let (mut ok, mut err) = (0, 0);
        for round in 0..96 {
            let mut bytes = shipped.to_vec();
            for m in 0..MODELS {
                // Bucket b > 0 holds 2^(b-1) symbols; model m uses m / 16 + 2.
                let used = m / 16 + 2;
                let weight = |b: usize| 1u32 << b.saturating_sub(1);
                let mut freq = [1u32; BUCKETS];
                let mut rest = (1 << 16) - (0..used).map(weight).sum::<u32>();
                let big = (next() as usize + round) % used;
                for b in (1..used).rev() {
                    let most = (rest / weight(b)).min(u32::from(u16::MAX) - 1);
                    let extra = match round % 3 {
                        0 if b == big => most,
                        0 => 0,
                        1 => most,
                        _ => next() as u32 % (most + 1),
                    };
                    freq[b] += extra;
                    rest -= extra * weight(b);
                }
                // Bucket 0 takes what is left when it fits in a u16, else
                // bucket 1 (also one symbol wide) takes the overflow.
                let spill = rest.saturating_sub(u32::from(u16::MAX) - 1);
                freq[1] += spill;
                rest -= spill;
                freq[0] += rest;
                for (b, f) in freq.iter().enumerate() {
                    // Buckets past `used` are never read.
                    let v = if b < used { *f as u16 } else { next() as u16 };
                    bytes[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&v.to_le_bytes());
                }
            }
            if round % 2 == 1 {
                for b in &mut bytes[MODELS * BUCKETS * 2..] {
                    *b = next() as u8;
                }
            }
            let t = Tables::parse(&bytes).unwrap();
            for bc in [1, 3, 4, 5, 6, 7] {
                let src: Vec<u8> = (0..next() % 3000).map(|_| next() as u8).collect();
                let mut dst = vec![0u8; 16 * (1 + next() as usize % 300)];
                match decode(bc, &src, &mut dst, &t) {
                    Ok(()) => ok += 1,
                    Err(_) => err += 1,
                }
            }
        }
        assert!(ok > 0 && err > 0);
    }

    #[test]
    fn zigzag_and_interleave_helpers() {
        // zz: 0, -1, 1, -2, ... modulo 2^bits, one-to-one on each width.
        for bits in [5u32, 6, 8] {
            let mask = (1u32 << bits) - 1;
            for (v, want) in [(0, 0), (1, mask), (2, 1), (3, mask - 1), (4, 2)] {
                assert_eq!(u32::from(zz(v, bits)), want, "zz({v}, {bits})");
            }
            let mut seen = vec![false; 1 << bits];
            for v in 0..=mask {
                let z = usize::from(zz(v as u8, bits));
                assert!(!seen[z], "zz({v}, {bits}) repeats");
                seen[z] = true;
            }
        }

        // interleave puts the first symbol on the even bits and is
        // one-to-one; nibbles takes the halves back out, zigzag-decoded.
        assert_eq!(interleave(0xf, 0), 0x55);
        assert_eq!(interleave(0, 0xf), 0xaa);
        assert_eq!(interleave(5, 0), 0x11);
        let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
        let mut seen = [false; 256];
        for a in 0..16u8 {
            for b in 0..16u8 {
                let v = interleave(a, b);
                assert!(!seen[usize::from(v)]);
                seen[usize::from(v)] = true;
                assert_eq!(nibbles(v, false), [z(a), z(b)]);
                assert_eq!(nibbles(v, true), [z(b), z(a)]);
            }
        }
        assert_eq!(nibbles(0x01, false), [-1, 0]);
        assert_eq!(nibbles(0x0a, false), [0, -2]);
    }

    #[test]
    fn references_and_picks() {
        // The endpoint reference is the index reference plus a zigzag delta,
        // wrapping in a byte.
        assert_eq!(references(&[0, 5], false), (5, 5));
        assert_eq!(references(&[1, 5], false), (4, 5));
        assert_eq!(references(&[2, 5], false), (6, 5));
        assert_eq!(references(&[3, 5], false), (3, 5));
        assert_eq!(references(&[1, 0], false), (255, 0));
        assert_eq!(references(&[0xfe, 0x80], false), (0xff, 0x80));
        // Paired records interleave two 4-bit symbols for each reference.
        assert_eq!(references(&[2, 0, 5, 0], true), (19, 17));
        assert_eq!(references(&[0, 0, 0, 0xf], true), (0xaa, 0xaa));

        let history: Vec<usize> = (0..HISTORY).map(|k| 1000 + k).collect();
        let dictionary = [7, 8, 9];
        let pick = |i, r| *pick(i, r, &history, &dictionary, &0);
        assert_eq!(pick(5, 0), 0);
        assert_eq!(pick(5, 1), 1004);
        assert_eq!(pick(5, 5), 1000);
        assert_eq!(pick(5, 6), 7);
        assert_eq!(pick(5, 8), 9);
        assert_eq!(pick(0, 1), 7);
        // History is a ring of HISTORY blocks.
        assert_eq!(pick(HISTORY + 3, 1), 1002);
        assert_eq!(pick(HISTORY + 3, HISTORY), 1003);
    }

    #[test]
    fn index_sums_and_block_bits() {
        // Same widths: the sum wraps at the width and reads back signed.
        assert_eq!(add_index(1, 3, 2, 3), 3);
        assert_eq!(add_index(3, 3, 1, 3), -4);
        assert_eq!(add_index(-1, 3, 1, 3), 0);
        // A wider reference is cut to the residual's width, rounding down.
        assert_eq!(add_index(1, 3, 3, 4), 2);
        assert_eq!(add_index(0, 2, 15, 4), -1);
        // A narrower reference is scaled up to the residual's width.
        assert_eq!(add_index(0, 4, 1, 2), 4);
        // No residual bits gives 0; a reference with no (or too many) bits
        // adds nothing.
        assert_eq!(add_index(5, 0, 5, 3), 0);
        assert_eq!(add_index(7, 3, 5, 0), -1);
        assert_eq!(add_index(2, 3, 5, 17), 2);

        let mut w = Bits::default();
        w.put(0xff, 4);
        w.put(u64::MAX, 0);
        w.put(0b10, 2);
        assert_eq!((w.v, w.at), (0x2f, 6));
        w.at = 127;
        w.put(3, 2);
        assert_eq!((w.v >> 127, w.at), (1, 129));
        w.put(u64::MAX, 64);
        assert_eq!((w.v, w.at), (1 << 127 | 0x2f, 193));
    }

    #[test]
    fn bc7_and_bc6_mode_details() {
        let pbit_modes: Vec<u16> = (0..8).filter(|&m| pbits(m)).collect();
        assert_eq!(pbit_modes, [0, 1, 3, 6, 7]);

        // Only modes 4 and 5 rotate.
        for mode in 0..8u16 {
            for rot in 0..4u16 {
                let want = if matches!(mode, 4 | 5) {
                    usize::from(rot)
                } else {
                    0
                };
                assert_eq!(
                    rotation(mode | rot << 3),
                    want,
                    "mode {mode} rotation {rot}"
                );
            }
        }

        // The p-bit adds to red; rotation swaps alpha with a colour channel.
        assert_eq!(bc7_precision(0), [5, 4, 4, 0]);
        assert_eq!(bc7_precision(2), [5, 5, 5, 0]);
        assert_eq!(bc7_precision(6), [8, 7, 7, 7]);
        assert_eq!(bc7_precision(6 | 1 << 3), [8, 7, 7, 7]);
        assert_eq!(bc7_precision(4), [5, 5, 5, 6]);
        assert_eq!(bc7_precision(4 | 1 << 3), [6, 5, 5, 5]);
        assert_eq!(bc7_precision(5 | 3 << 3), [7, 7, 8, 7]);

        let index_bits: Vec<u8> = (0..=16).map(bc6_index_bits).collect();
        assert_eq!(
            index_bits,
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0]
        );
    }

    #[test]
    fn bc1_and_bc4_halves_pack_their_codes() {
        // Indices run along the endpoint line; BC1 codes are 0, 2, 3, 1.
        let mut out = [0u8; 8];
        bc1_color(&[31, 63, 31, 0, 0, 0], &[0; 16], &mut out);
        assert_eq!(out, [0xff, 0xff, 0, 0, 0, 0, 0, 0]);
        bc1_color(&[1, 2, 3, 4, 5, 6], &[1; 16], &mut out);
        assert_eq!(&out[..4], &[0x43, 0x08, 0xa6, 0x20]);
        assert_eq!(&out[4..], &0xaaaa_aaaau32.to_le_bytes());
        let mut ix = [0u8; 16];
        ix[0] = 3;
        ix[15] = 2;
        bc1_color(&[0; 6], &ix, &mut out);
        assert_eq!(&out[4..], &(1u32 | 3 << 30).to_le_bytes());

        // BC4 endpoints are stored 128 above; codes are 0, 2..7, 1.
        bc4_alpha(0x80, 0x7f, &[0; 16], &mut out);
        assert_eq!(out, [0, 0xff, 0, 0, 0, 0, 0, 0]);
        let mut ix = [0u8; 16];
        ix[0] = 7;
        ix[1] = 1;
        ix[15] = 6;
        bc4_alpha(0, 0, &ix, &mut out);
        let want = 0x8080u64 | 1 << 16 | 2 << 19 | 7 << 61;
        assert_eq!(out, want.to_le_bytes());
    }

    /// The plain formats' element widths, which bytes of a record each
    /// residual reads its endpoints and indices from, how BC3 and BC5 stack
    /// their halves, and `plain` summing residuals onto the dictionary and
    /// then onto earlier blocks, wrapping at each element's width.
    #[test]
    fn plain_widths_residuals_and_stacked_packs() {
        let ix_of = |w: &Widths| w.ix;
        let mut want = [0u8; 32];
        want[..16].fill(2);
        assert_eq!(
            (BC1.ep, ix_of(&BC1), BC1.paired),
            ([5, 6, 5, 5, 6, 5, 0, 0], want, false)
        );
        want[..16].fill(3);
        want[16..].fill(2);
        assert_eq!(
            (BC3.ep, ix_of(&BC3), BC3.paired),
            ([5, 6, 5, 5, 6, 5, 8, 8], want, true)
        );
        want[16..].fill(0);
        assert_eq!(
            (BC4.ep, ix_of(&BC4), BC4.paired),
            ([8, 8, 0, 0, 0, 0, 0, 0], want, false)
        );
        want.fill(3);
        assert_eq!(
            (BC5.ep, ix_of(&BC5), BC5.paired),
            ([8, 8, 8, 8, 0, 0, 0, 0], want, true)
        );

        let mut t = Tables::parse(&tables()).unwrap();
        for (i, r) in t.rank1.iter_mut().enumerate() {
            *r = i as u8;
        }
        for (i, r) in t.rank3c.iter_mut().enumerate() {
            *r = i as u8;
        }
        for (i, r) in t.rank4.iter_mut().enumerate() {
            *r = i as u8;
        }

        // BC1: six zigzag endpoints from byte 2, quads from byte 8.
        let mut s = [0u8; 12];
        s[2..8].copy_from_slice(&[1, 1, 1, 2, 2, 2]);
        s[8] = 0b1110_0100;
        let v = bc1_residual(&t, &s);
        assert_eq!(v.ep, [31, 63, 31, 1, 1, 1, 0, 0]);
        assert_eq!(
            [v.ix[0], v.ix[1], v.ix[4], v.ix[5], v.ix[2]],
            [0, 1, 2, 3, 0]
        );

        // BC3: alpha endpoints in bytes 4-5, colour lanes in 14-19 carry each
        // channel's two endpoints together, alpha pairs from 6, colour quads
        // from 20.
        let mut s = [0u8; 24];
        for (at, k) in [
            (14, 1),
            (15, 2),
            (16, 3),
            (17, 4),
            (18, 5),
            (19, 6),
            (4, 7),
            (5, 8),
        ] {
            s[at] = 2 * k;
        }
        s[6] = 0x21;
        s[20] = 0b1110_0100;
        let v = bc3_residual(&t, &s);
        assert_eq!(v.ep, [1, 3, 5, 2, 4, 6, 7, 8]);
        assert_eq!(&v.ix[..3], &[1, 2, 0]);
        assert_eq!([v.ix[16], v.ix[17], v.ix[20], v.ix[21]], [0, 1, 2, 3]);

        // BC4: endpoints in bytes 2-3, pairs from 4.
        let mut s = [0u8; 12];
        s[2..5].copy_from_slice(&[3, 4, 0x35]);
        let v = bc4_residual(&t, &s);
        assert_eq!(&v.ep[..3], &[0xfe, 2, 0]);
        assert_eq!(&v.ix[..3], &[5, 3, 0]);

        // BC5: four endpoints from byte 4, sixteen pairs from 8.
        let mut s = [0u8; 24];
        s[4..8].copy_from_slice(&[0, 1, 2, 3]);
        s[8] = 0x17;
        s[23] = 0x30;
        let v = bc5_residual(&t, &s);
        assert_eq!(&v.ep[..5], &[0, 0xff, 1, 0xfe, 0]);
        assert_eq!((&v.ix[..2], &v.ix[30..]), (&[7, 1][..], &[0, 3][..]));

        // BC3 is a BC4 alpha half (endpoints 6-7, indices 0-15) then a BC1
        // colour half (indices 16-31); BC5 is two BC4 halves.
        let mut v = Plain::default();
        for (k, e) in v.ep.iter_mut().enumerate() {
            *e = 3 * k as u8 + 1;
        }
        for (k, x) in v.ix.iter_mut().enumerate() {
            *x = (k * 5 % 8) as u8;
        }
        let (mut got, mut want) = ([0u8; 16], [0u8; 16]);
        bc3_pack(&v, &mut got);
        bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut want[..8]);
        bc1_color(&v.ep, &v.ix[16..], &mut want[8..]);
        assert_eq!(got, want);
        bc5_pack(&v, &mut got);
        bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut want[..8]);
        bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut want[8..]);
        assert_eq!(got, want);

        // Two BC1 records, each referring one back: the first lands on the
        // dictionary and wraps every endpoint to 0, the second on the first.
        let dictionary = [Plain {
            ep: [31, 63, 31, 0, 0, 0, 0, 0],
            ix: [1; 32],
        }];
        let mut records = [[0u8; 12]; 2];
        records[0][1] = 1;
        records[0][2..5].copy_from_slice(&[2, 2, 2]);
        records[1][1] = 1;
        records[1][2] = 1;
        records[1][8] = 1;
        let mut out = [[0u8; 8]; 2];
        plain(
            records.iter().map(|r| &r[..]),
            out.iter_mut().map(|o| &mut o[..]),
            &dictionary,
            &BC1,
            |s| bc1_residual(&t, s),
            bc1_pack,
        )
        .unwrap();
        assert_eq!(out[0], [0, 0, 0, 0, 0xaa, 0xaa, 0xaa, 0xaa]);
        assert_eq!(out[1], [0x00, 0xf8, 0, 0, 0xab, 0xaa, 0xaa, 0xaa]);
    }

    /// `quads` writes each 2-bit residual symbol to one 2x2 quad of the 4x4
    /// block, low bits first in the order top-left, top-right, bottom-left,
    /// bottom-right; `pairs` writes a 3-bit residual to each of two raster
    /// neighbours from bits 0-2 and 4-6. Both look the symbol up in the rank
    /// table first, and `pairs` ignores the symbol's top two bits.
    #[test]
    fn quads_and_pairs_place_index_residuals() {
        let mut rank = [0u8; 256];
        for (i, r) in rank.iter_mut().enumerate() {
            *r = !(i as u8);
        }
        // Symbol !v ranks to v: quad q gets v = 0b11_10_01_00 rotated by q.
        let syms: Vec<u8> = (0..4).map(|q| !0b1110_0100u8.rotate_left(2 * q)).collect();
        let mut ix = [9u8; 16];
        quads(&rank, &syms, &mut ix);
        assert_eq!(
            ix,
            [
                0, 1, 3, 0, //
                2, 3, 1, 2, //
                2, 3, 1, 2, //
                0, 1, 3, 0,
            ]
        );
        // Fewer symbols leave the other quads alone.
        let mut ix = [9u8; 16];
        quads(&rank, &syms[..1], &mut ix);
        assert_eq!(ix, [0, 1, 9, 9, 2, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]);

        let mut rank = [0u8; 64];
        for (i, r) in rank.iter_mut().enumerate() {
            // Bit 3 and bit 7 must be dropped.
            *r = (i as u8 & 7) | 8 | ((7 - (i as u8 & 7)) << 4) | 0x80;
        }
        let syms: Vec<u8> = (0..8).map(|k| k | k << 6).collect();
        let mut ix = [9u8; 16];
        pairs(&rank, &syms, &mut ix);
        assert_eq!(ix, [0, 7, 1, 6, 2, 5, 3, 4, 4, 3, 5, 2, 6, 1, 7, 0]);
        let mut ix = [9u8; 16];
        pairs(&rank, &syms[..3], &mut ix);
        assert_eq!(ix, [0, 7, 1, 6, 2, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]);
    }

    /// `bc7_pack` read back by a BC7 reader written from the public mode
    /// table: mode prefix, partition, rotation and index-selection bits,
    /// colour and alpha widths, per-endpoint and shared p-bits, and index
    /// widths with one bit fewer at each anchor. Every mode, every partition,
    /// each rotation and both index selections give back the endpoints and
    /// indices that went in, in exactly 128 bits.
    #[test]
    fn bc7_blocks_follow_the_public_layout() {
        // Subsets, partition bits, rotation bits, index-selection bits,
        // colour bits, alpha bits, endpoint p-bits, shared p-bits, index
        // bits, secondary index bits.
        const MODES: [[u32; 10]; 8] = [
            [3, 4, 0, 0, 4, 0, 1, 0, 3, 0],
            [2, 6, 0, 0, 6, 0, 0, 1, 3, 0],
            [3, 6, 0, 0, 5, 0, 0, 0, 2, 0],
            [2, 6, 0, 0, 7, 0, 1, 0, 2, 0],
            [1, 0, 2, 1, 5, 6, 0, 0, 2, 3],
            [1, 0, 2, 0, 7, 8, 0, 0, 2, 2],
            [1, 0, 0, 0, 7, 7, 1, 0, 4, 0],
            [2, 6, 0, 0, 5, 5, 1, 0, 2, 0],
        ];
        struct Reader(u128, u32);
        impl Reader {
            fn take(&mut self, n: u32) -> u16 {
                let v = (self.0 >> self.1) as u16 & ((1u32 << n) - 1) as u16;
                self.1 += n;
                v
            }
        }
        let mut seed = 0x6a09_e667_f3bc_c908u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for mode in 0..8u16 {
            let [ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2] = MODES[usize::from(mode)];
            let pbit = epb + spb;
            for round in 0..64u16 {
                let part = round & ((1 << pb) - 1);
                let rot = if rb > 0 { round / 2 % 4 } else { 0 };
                let sel = if isb > 0 { round % 2 } else { 0 };
                let ctrl = mode | rot << 3 | sel << 5 | (ns as u16) << 6 | part << 8;
                let prec = bc7_precision(ctrl);
                let mut ep = Bc7Ep {
                    ctrl,
                    ..Default::default()
                };
                for row in ep.rows.iter_mut().take(ns as usize) {
                    for (c, v) in row.iter_mut().enumerate() {
                        let p = prec[c / 2];
                        *v = if p == 0 {
                            0
                        } else {
                            next() as u16 & 0xffff << (16 - p)
                        };
                    }
                }
                let anchors: Vec<usize> = match ns {
                    2 => vec![0, usize::from(ANCHOR2[usize::from(part)])],
                    3 => vec![
                        0,
                        usize::from(ANCHOR3A[usize::from(part)]),
                        usize::from(ANCHOR3B[usize::from(part)]),
                    ],
                    _ => vec![0],
                };
                // The set written first: mode 4 leads with its 2-bit set.
                let order = if mode == 4 { [1, 0] } else { [0, 1] };
                let mut ix = Bc7Ix {
                    bits: BC7_INDEX[usize::from(mode)],
                    ..Default::default()
                };
                assert_eq!(
                    [ix.bits[order[0]], ix.bits[order[1]]].map(u32::from),
                    [ib, ib2],
                    "mode {mode} index bits"
                );
                for set in 0..2 {
                    let bits = u32::from(ix.bits[set]);
                    if bits == 0 {
                        continue;
                    }
                    for (p, v) in ix.ix[set].iter_mut().enumerate() {
                        let first = set == order[0] && anchors.contains(&p) || p == 0;
                        let n = bits - u32::from(first);
                        *v = (next() & ((1 << n) - 1)) as i16;
                    }
                }
                let mut out = [0u8; 16];
                bc7_pack(&ep, &ix, prec, &mut out);

                let mut r = Reader(u128::from_le_bytes(out), 0);
                let what = format!("mode {mode} part {part} rot {rot} sel {sel}");
                assert_eq!(r.take(u32::from(mode) + 1), 1 << mode, "{what}");
                assert_eq!(r.take(pb), part, "{what}");
                assert_eq!(r.take(rb), rot, "{what}");
                assert_eq!(r.take(isb), sel, "{what}");
                let ends = 2 * ns as usize;
                let mut got = [[0u16; 6]; 4];
                for (ch, g) in got.iter_mut().enumerate() {
                    let width = if ch < 3 { cb } else { ab };
                    for v in g.iter_mut().take(if width > 0 { ends } else { 0 }) {
                        *v = r.take(width);
                    }
                }
                let mut p = [0u16; 6];
                for e in 0..ends {
                    if epb > 0 {
                        p[e] = r.take(1);
                    } else if spb > 0 && e % 2 == 0 {
                        p[e] = r.take(1);
                        p[e + 1] = p[e];
                    }
                }
                if pbit > 0 {
                    for e in 0..ends {
                        got[0][e] = got[0][e] << 1 | p[e];
                    }
                }
                if rot != 0 {
                    got.swap(3, usize::from(rot) - 1);
                }
                for (ch, g) in got.iter().enumerate() {
                    let w = prec[ch];
                    for (e, &v) in g.iter().enumerate().take(if w > 0 { ends } else { 0 }) {
                        let mut want = ep.rows[e / 2][2 * ch + e % 2] >> (16 - w);
                        if ch == 0 && spb > 0 {
                            // The shared p-bit is the subset's first endpoint's.
                            want = want & !1 | ep.rows[e / 2][0] >> (16 - w) & 1;
                        }
                        assert_eq!(v, want, "{what} channel {ch} endpoint {e}");
                    }
                }
                for (k, set) in order.into_iter().enumerate() {
                    let bits = u32::from(ix.bits[set]);
                    if bits == 0 {
                        continue;
                    }
                    for (q, &want) in ix.ix[set].iter().enumerate() {
                        let anchor = if k == 0 { anchors.contains(&q) } else { q == 0 };
                        let v = r.take(bits - u32::from(anchor));
                        assert_eq!(v as i16, want, "{what} set {set} pixel {q}");
                    }
                }
                assert_eq!(r.1, 128, "{what}");
            }
        }
    }

    /// BC6H blocks read back by a reader written from the public header
    /// table (Microsoft's "BC6H format" page): mode bits, then each mode's
    /// endpoint fields in their listed order, the 5-bit partition for the two
    /// region modes, and 3- or 4-bit indices with one bit fewer at pixel 0
    /// and the second region's anchor. Every mode and partition gives back
    /// the endpoint fields and indices put in a record, in exactly 128 bits,
    /// each field bit written once; a following record that references the
    /// first with zero residuals packs the same block.
    #[test]
    fn bc6_blocks_follow_the_public_layout() {
        // Mode bits, then R G B widths of endpoints w, x, y, z as the page's
        // mode table lists them (y and z are unused by modes 11-14).
        const MODES: [(u16, u32, [[u32; 4]; 3]); 14] = [
            (0b00, 2, [[10, 5, 5, 5]; 3]),
            (0b01, 2, [[7, 6, 6, 6]; 3]),
            (0b00010, 5, [[11, 5, 5, 5], [11, 4, 4, 4], [11, 4, 4, 4]]),
            (0b00110, 5, [[11, 4, 4, 4], [11, 5, 5, 5], [11, 4, 4, 4]]),
            (0b01010, 5, [[11, 4, 4, 4], [11, 4, 4, 4], [11, 5, 5, 5]]),
            (0b01110, 5, [[9, 5, 5, 5]; 3]),
            (0b10010, 5, [[8, 6, 6, 6], [8, 5, 5, 5], [8, 5, 5, 5]]),
            (0b10110, 5, [[8, 5, 5, 5], [8, 6, 6, 6], [8, 5, 5, 5]]),
            (0b11010, 5, [[8, 5, 5, 5], [8, 5, 5, 5], [8, 6, 6, 6]]),
            (0b11110, 5, [[6, 6, 6, 6]; 3]),
            (0b00011, 5, [[10, 10, 0, 0]; 3]),
            (0b00111, 5, [[11, 9, 0, 0]; 3]),
            (0b01011, 5, [[12, 8, 0, 0]; 3]),
            (0b01111, 5, [[16, 4, 0, 0]; 3]),
        ];
        // The header after the mode bits; `hi:lo` fills bits lo..=hi from
        // the lowest stream bit up. The page spells the base's high bits of
        // modes 13 and 14 as [10:11] and [10:15]; the console writes them
        // ascending, as here.
        const HEADERS: [&str; 14] = [
            "gy4 by4 bz4 rw9:0 gw9:0 bw9:0 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 by3:0 \
             ry4:0 bz2 rz4:0 bz3 d4:0",
            "gy5 gz4 gz5 rw6:0 bz0 bz1 by4 gw6:0 by5 bz2 gy4 bw6:0 bz3 bz5 bz4 rx5:0 gy3:0 \
             gx5:0 gz3:0 bx5:0 by3:0 ry5:0 rz5:0 d4:0",
            "rw9:0 gw9:0 bw9:0 rx4:0 rw10 gy3:0 gx3:0 gw10 bz0 gz3:0 bx3:0 bw10 bz1 by3:0 \
             ry4:0 bz2 rz4:0 bz3 d4:0",
            "rw9:0 gw9:0 bw9:0 rx3:0 rw10 gz4 gy3:0 gx4:0 gw10 gz3:0 bx3:0 bw10 bz1 by3:0 \
             ry3:0 bz0 bz2 rz3:0 gy4 bz3 d4:0",
            "rw9:0 gw9:0 bw9:0 rx3:0 rw10 by4 gy3:0 gx3:0 gw10 bz0 gz3:0 bx4:0 bw10 by3:0 \
             ry3:0 bz1 bz2 rz3:0 bz4 bz3 d4:0",
            "rw8:0 by4 gw8:0 gy4 bw8:0 bz4 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 by3:0 \
             ry4:0 bz2 rz4:0 bz3 d4:0",
            "rw7:0 gz4 by4 gw7:0 bz2 gy4 bw7:0 bz3 bz4 rx5:0 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 \
             by3:0 ry5:0 rz5:0 d4:0",
            "rw7:0 bz0 by4 gw7:0 gy5 gy4 bw7:0 gz5 bz4 rx4:0 gz4 gy3:0 gx5:0 gz3:0 bx4:0 bz1 \
             by3:0 ry4:0 bz2 rz4:0 bz3 d4:0",
            "rw7:0 bz1 by4 gw7:0 by5 gy4 bw7:0 bz5 bz4 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx5:0 \
             by3:0 ry4:0 bz2 rz4:0 bz3 d4:0",
            "rw5:0 gz4 bz0 bz1 by4 gw5:0 gy5 by5 bz2 gy4 bw5:0 gz5 bz3 bz5 bz4 rx5:0 gy3:0 \
             gx5:0 gz3:0 bx5:0 by3:0 ry5:0 rz5:0 d4:0",
            "rw9:0 gw9:0 bw9:0 rx9:0 gx9:0 bx9:0",
            "rw9:0 gw9:0 bw9:0 rx8:0 rw10 gx8:0 gw10 bx8:0 bw10",
            "rw9:0 gw9:0 bw9:0 rx7:0 rw11:10 gx7:0 gw11:10 bx7:0 bw11:10",
            "rw9:0 gw9:0 bw9:0 rx3:0 rw15:10 gx3:0 gw15:10 bx3:0 bw15:10",
        ];
        let mut seed = 0xbb67_ae85_84ca_a73bu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Zigzag (0, -1, 1, ...) of a signed value, and a 16-bit value split
        // into the even and odd bits `bc6` interleaves back.
        let zig = |v: i16| ((v << 1) ^ (v >> 15)) as u16;
        let unweave = |v: u16| {
            let half = |v: u16| (0..8).fold(0u8, |acc, k| acc | ((v >> (2 * k) & 1) as u8) << k);
            [half(v), half(v >> 1)]
        };
        let spread = |v: u16| (0..4).fold(0u8, |acc, k| acc | ((v >> k & 1) as u8) << (2 * k));
        for (m, &(code, code_bits, widths)) in MODES.iter().enumerate() {
            let mode = m + 1;
            let two = mode < 11;
            let bits = if two { 3 } else { 4 };
            let ends = if two { 4 } else { 2 };
            let endpoint_bits: u32 = widths.iter().flat_map(|w| &w[..ends]).sum();
            assert_eq!(
                endpoint_bits,
                if two {
                    75 - 3 * u32::from(mode > 2)
                } else {
                    60
                }
            );
            for part in 0..if two { 32u8 } else { 4 } {
                let what = format!("mode {mode} part {part}");
                let anchor = if two {
                    usize::from(ANCHOR2[usize::from(part)])
                } else {
                    0
                };
                let mut s = [0u8; 32];
                s[30] = m as u8;
                s[31] = part | (next() as u8 & 0xe0);
                // Field values, then the residuals against the zero reference
                // that give them: the stored field is the endpoint with its
                // top bit flipped.
                let mut fields = [[0u16; 3]; 4];
                for (e, row) in fields.iter_mut().enumerate().take(ends) {
                    for (c, f) in row.iter_mut().enumerate() {
                        let w = widths[c][e];
                        *f = (next() & ((1 << w) - 1)) as u16;
                        let res = u32::from(*f) ^ 1 << (w - 1);
                        let signed = ((res << (32 - w)) as i32 >> (32 - w)) as i16;
                        let z = zig(signed);
                        if e < 2 {
                            let [a, b] = unweave(z);
                            s[4 + 6 * e + 2 * c] = a;
                            s[5 + 6 * e + 2 * c] = b;
                        } else {
                            s[16 + 2 * c + (e - 2)] = u8::try_from(z).unwrap();
                        }
                    }
                }
                let mut want_ix = [0u16; 16];
                let mut sym = [0u16; 16];
                for (p, (w, z)) in want_ix.iter_mut().zip(&mut sym).enumerate() {
                    let n = bits - u32::from(p == 0 || p == anchor);
                    *w = (next() & ((1 << n) - 1)) as u16;
                    *z = zig(*w as i16 - (1 << (bits - 1)));
                }
                for k in 0..8 {
                    let (first, second) = (sym[2 * k], sym[2 * k + 1]);
                    let (even, odd) = if k < 6 {
                        (second, first)
                    } else {
                        (first, second)
                    };
                    s[22 + k] = spread(even) | spread(odd) << 1;
                }
                // A second record: the first as reference, zero residuals.
                let mut again = [0u8; 32];
                again[2] = 1;
                again[30] = s[30];
                again[31] = s[31];
                let mut out = [[0u8; 16]; 2];
                bc6(
                    [&s[..], &again[..]].into_iter(),
                    out.iter_mut().map(|o| &mut o[..]),
                    &[],
                )
                .unwrap();
                assert_eq!(out[1], out[0], "{what} referenced again");

                let v = u128::from_le_bytes(out[0]);
                let mut at = 0;
                let mut take = |n: u32| {
                    let r = (v >> at) as u16 & ((1u32 << n) - 1) as u16;
                    at += n;
                    r
                };
                assert_eq!(take(code_bits), code, "{what}");
                let mut got = [[0u16; 3]; 4];
                let mut seen = [[0u16; 3]; 4];
                let mut got_part = None;
                for f in HEADERS[m].split_whitespace() {
                    let (name, range) = f.split_at(if f.starts_with('d') { 1 } else { 2 });
                    let (hi, lo) = match range.split_once(':') {
                        Some((hi, lo)) => (hi.parse::<u32>().unwrap(), lo.parse::<u32>().unwrap()),
                        None => (range.parse().unwrap(), range.parse().unwrap()),
                    };
                    let value = take(hi - lo + 1);
                    if name == "d" {
                        assert_eq!((hi, lo, got_part), (4, 0, None), "{what}");
                        got_part = Some(value);
                        continue;
                    }
                    let c = "rgb".find(&name[..1]).unwrap();
                    let e = "wxyz".find(&name[1..]).unwrap();
                    let mask = (((1u32 << (hi - lo + 1)) - 1) << lo) as u16;
                    assert_eq!(seen[e][c] & mask, 0, "{what} {f} written twice");
                    seen[e][c] |= mask;
                    got[e][c] |= value << lo;
                }
                for e in 0..4 {
                    for c in 0..3 {
                        let w = if e < ends { widths[c][e] } else { 0 };
                        let full = ((1u32 << w) - 1) as u16;
                        assert_eq!(seen[e][c], full, "{what} endpoint {e} channel {c} bits");
                    }
                }
                assert_eq!(got, fields, "{what} endpoints");
                assert_eq!(got_part, two.then_some(u16::from(part)), "{what}");
                for (p, &w) in want_ix.iter().enumerate() {
                    let n = bits - u32::from(p == 0 || p == anchor);
                    assert_eq!(take(n), w, "{what} pixel {p}");
                }
                assert_eq!(at, 128, "{what}");
            }
        }
    }

    /// The window table finds the bucket a scan of the starts finds, for
    /// every slot of every model, in the shipped tables and in ours (whose
    /// buckets are wide) and in tables whose buckets are one slot wide.
    #[test]
    fn bucket_lookup_matches_a_scan_of_the_starts() {
        let mut narrow = tables();
        for m in 0..MODELS {
            // Buckets 1.. of one slot each where they fit; bucket 0 takes the rest.
            let used = m / 16 + 2;
            let rest = (1u32 << 16) - (1..used).map(|b| 1u32 << (b - 1)).sum::<u32>();
            for b in 0..used {
                let f = if b == 0 { rest } else { 1 };
                narrow[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&(f as u16).to_le_bytes());
            }
        }
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
            Tables::parse(&narrow).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                let start = &t.start[usize::from(model)];
                for slot in 0..1u32 << 16 {
                    let scan = start[1..].iter().filter(|&&s| s <= slot).count();
                    assert_eq!(
                        bucket_of(&t, model, slot),
                        scan,
                        "model {model} slot {slot}"
                    );
                }
            }
        }
    }

    /// One rANS decode as first written: scan for the bucket, divide for the
    /// symbol, rebuild its frequency and start from the symbol.
    fn reference_decode(t: &Tables, model: u8, x: u32) -> (u8, u32) {
        let (freq, start) = (&t.freq[usize::from(model)], &t.start[usize::from(model)]);
        let slot = x & 0xffff;
        let b = start[1..BUCKETS].iter().filter(|&&s| s <= slot).count();
        let sym = match b {
            0 => 0,
            1 => 1,
            _ => (1u32 << (b - 1)) + (slot - start[b]) / freq[b],
        } as u8;
        let (f, c) = if sym == 0 {
            (freq[0] - 1, 1)
        } else {
            let k = (8 - sym.leading_zeros()) as usize;
            let f = freq[k];
            (
                f,
                ((u32::from(sym) - (1 << (k - 1))) * f + start[k]) & 0xffff,
            )
        };
        (
            sym,
            (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot),
        )
    }

    /// A payload step decodes the symbol and state the first version did, for
    /// every model and slot of the shipped tables and ours, at assorted states.
    #[test]
    fn payload_step_matches_the_reference_decode() {
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                for high in [0x100u32, 0x1234, 0xffff] {
                    for slot in 0..1u32 << 16 {
                        let x = high << 16 | slot;
                        if x <= LOW {
                            continue;
                        }
                        let mut lane = Lane {
                            phase: 3,
                            model,
                            x,
                            left: 1,
                            ..Default::default()
                        };
                        lane.step(&t).unwrap();
                        let (sym, next) = reference_decode(&t, model, x);
                        assert_eq!(
                            (lane.out, lane.x),
                            (Some(sym), next),
                            "model {model} x {x:#x}"
                        );
                    }
                }
            }
        }
    }

    /// Pseudo-random records that fit each lane's alphabet; BC7 keeps control
    /// code 0 (mode 6), the only one the synthetic tables define.
    fn random_records(widths: &[u8], per: usize, blocks: usize, bc7: bool) -> Vec<Vec<u8>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..blocks)
            .map(|_| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let alphabet = 1u32 << (widths[k / per] + 1);
                        (seed as u32 % alphabet) as u8
                    })
                    .collect();
                if bc7 {
                    r[30] = 0;
                    r[31] = 0;
                }
                r
            })
            .collect()
    }

    /// The fast paths decode exactly what one lane step at a time does, on
    /// every format's lane layout, with short and long fragments.
    #[test]
    fn every_layout_round_trips_through_the_lanes() {
        let t = Tables::parse(&tables()).unwrap();
        let layouts: [(&[u8], usize); 6] = [
            (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
            (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
            (&BC4_LANES, 1),
            (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
            (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
            (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
        ];
        for (widths, per) in layouts {
            for blocks in [1, 2, 17, 300, 1000] {
                let records = random_records(widths, per, blocks, false);
                let src = stream(&t, widths, per, &records);
                let mut got = vec![0u8; records.concat().len()];
                let mut next = src.iter().copied();
                lanes(&t, widths, per, blocks, |_| next.next(), &mut got).unwrap();
                assert_eq!(got, records.concat(), "{widths:?} x{per}, {blocks} blocks");
            }
        }
    }

    /// Decode throughput on synthetic streams (no game data):
    /// `cargo test --release -p victory-formats bcpack_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn bcpack_throughput() {
        let t = Tables::parse(&tables()).unwrap();
        let cases: [(u8, &[u8], usize, usize); 3] = [
            (1, &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1, 8),
            (4, &BC4_LANES, 1, 8),
            (7, &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2, 16),
        ];
        for (bc, widths, per, block) in cases {
            // One fragment per lane holds at most 65792 bytes.
            let blocks = 1 << 14;
            let records = random_records(widths, per, blocks, bc == 7);
            let src = stream(&t, widths, per, &records);
            let mut dst = vec![0u8; blocks * block];
            decode(bc, &src, &mut dst, &t).unwrap();
            let runs = 40;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                decode(bc, std::hint::black_box(&src), &mut dst, &t).unwrap();
            }
            let secs = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "BC{bc}: {blocks} blocks, {:.1} MB out/s, {:.1} ns/block",
                (blocks * block) as f64 / secs / 1e6,
                secs * 1e9 / blocks as f64
            );
        }
    }

    /// Compares with reference outputs made outside the repository:
    /// `VICTORY_BCPACK_CASES`, a directory of `<name>.bc<n>` packed chunks
    /// with `<name>.out` beside them. The tables are the built-in ones unless
    /// `VICTORY_BCPACK_TABLES` names another table file.
    #[test]
    #[ignore = "needs reference chunks outside the repository"]
    fn reference_chunks() {
        let cases = std::env::var_os("VICTORY_BCPACK_CASES").expect("VICTORY_BCPACK_CASES");
        let t = match std::env::var_os("VICTORY_BCPACK_TABLES") {
            Some(tables) => Tables::parse(&std::fs::read(tables).unwrap()).unwrap(),
            None => Tables::builtin().unwrap(),
        };
        let mut n = 0;
        for entry in std::fs::read_dir(cases).unwrap() {
            let path = entry.unwrap().path();
            let Some(bc) = path
                .extension()
                .and_then(|e| e.to_str()?.strip_prefix("bc")?.parse().ok())
            else {
                continue;
            };
            let want = std::fs::read(path.with_extension("out")).unwrap();
            let mut got = vec![0u8; want.len()];
            decode(bc, &std::fs::read(&path).unwrap(), &mut got, &t).unwrap();
            assert!(got == want, "{}", path.display());
            n += 1;
        }
        assert!(n > 0);
    }
}
