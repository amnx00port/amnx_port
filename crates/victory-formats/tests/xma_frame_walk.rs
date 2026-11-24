//! Walking an XMA2 stream frame by frame through `read_frame` and `next`.
//!
//! The buffers are built here: frames of known bits laid end to end across
//! the packets of one stream, with packets of other streams (all ones)
//! between them, and packet headers computed from where each frame lands.

use victory_formats::xma::{self, PACKET_BYTES};

const PACKET_BITS: usize = PACKET_BYTES * 8;
const HEADER_BITS: usize = 32;

fn header(frames: u32, first_frame_bit: u32, skip: u32) -> [u8; 4] {
    (frames << 26 | first_frame_bit << 11 | skip).to_be_bytes()
}

fn set_header(data: &mut [u8], index: usize, h: [u8; 4]) {
    data[index * PACKET_BYTES..][..4].copy_from_slice(&h);
}

fn put_bit(data: &mut [u8], pos: usize, bit: bool) {
    let mask = 0x80 >> (pos % 8);
    if bit {
        data[pos / 8] |= mask;
    } else {
        data[pos / 8] &= !mask;
    }
}

/// A frame's bits: its 15-bit length, then a deterministic pattern.
fn frame_bits(len: usize, seed: u32) -> Vec<bool> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            if i < 15 {
                (len >> (14 - i)) & 1 == 1
            } else {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state & 1 == 1
            }
        })
        .collect()
}

fn pack(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        put_bit(&mut out, i, b);
    }
    out
}

/// The packets of one stream (`own`, ascending) inside `total` packets; the
/// rest belong to other streams and are filled with ones.
struct Stream {
    data: Vec<u8>,
    own: Vec<usize>,
    /// For each own packet: how many frames begin in it, and the first one.
    starts: Vec<(u32, Option<usize>)>,
    pos: usize,
}

impl Stream {
    fn new(total: usize, own: &[usize]) -> Stream {
        let mut data = vec![0xffu8; total * PACKET_BYTES];
        for &p in own {
            data[p * PACKET_BYTES..][..PACKET_BYTES].fill(0);
        }
        Stream {
            data,
            own: own.to_vec(),
            starts: vec![(0, None); own.len()],
            pos: own[0] * PACKET_BITS + HEADER_BITS,
        }
    }

    /// Steps to the next payload bit of this stream when at a packet end.
    fn align(&mut self) {
        if self.pos.is_multiple_of(PACKET_BITS) {
            let previous = self.pos / PACKET_BITS - 1;
            let i = self.own.iter().position(|&p| p == previous).unwrap();
            self.pos = self.own[i + 1] * PACKET_BITS + HEADER_BITS;
        }
    }

    /// Appends a frame and returns where it starts.
    fn push(&mut self, bits: &[bool]) -> usize {
        self.align();
        let start = self.pos;
        let i = self
            .own
            .iter()
            .position(|&p| p == start / PACKET_BITS)
            .unwrap();
        self.starts[i].0 += 1;
        self.starts[i].1.get_or_insert(start);
        for &bit in bits {
            self.align();
            put_bit(&mut self.data, self.pos, bit);
            self.pos += 1;
        }
        start
    }

    /// Writes every own packet's header. A packet where no frame begins
    /// points past its payload.
    fn finish(mut self) -> Vec<u8> {
        for (i, &p) in self.own.iter().enumerate() {
            let skip = self.own.get(i + 1).map_or(0, |&n| n - p - 1);
            let (frames, first) = self.starts[i];
            let first = first.map_or(0x7fff, |s| s - p * PACKET_BITS - HEADER_BITS);
            set_header(&mut self.data, p, header(frames, first as u32, skip as u32));
        }
        self.data
    }
}

/// Follows `next` from `start` until it runs out.
fn walk(data: &[u8], start: usize) -> Vec<xma::Frame> {
    let mut out = vec![];
    let mut at = Some(start);
    while let Some(pos) = at {
        let f = xma::read_frame(data, pos).expect("every `next` is a frame");
        at = f.next;
        out.push(f);
    }
    out
}

#[test]
fn walking_a_stream_returns_every_frame_in_order() {
    // Frames shorter than a packet, ones spanning a packet boundary, one
    // longer than a whole packet, with other streams' packets in between.
    let lens = [
        900,
        6000,
        12000,
        16,
        PACKET_BITS + 3000,
        5000,
        777,
        9001,
        40,
    ];
    let mut s = Stream::new(9, &[0, 1, 3, 4, 7, 8]);
    let mut starts = vec![];
    let mut frames = vec![];
    for (i, &len) in lens.iter().enumerate() {
        let bits = frame_bits(len, 100 + i as u32);
        starts.push(s.push(&bits));
        frames.push(bits);
    }
    let data = s.finish();

    let got = walk(&data, starts[0]);
    assert_eq!(got.len(), lens.len());
    for (i, f) in got.iter().enumerate() {
        assert_eq!(f.start, starts[i], "frame {i}");
        assert_eq!(f.bits, lens[i], "frame {i}");
        assert_eq!(f.data, pack(&frames[i]), "frame {i}");
    }
    // Starting in the middle sees the same tail.
    let tail = walk(&data, starts[4]);
    assert_eq!(tail.len(), lens.len() - 4);
    assert!(tail.iter().zip(&got[4..]).all(|(a, b)| a.data == b.data));
}

#[test]
fn length_field_split_by_a_packet_header() {
    // Seven bits of the length field end packet 0; the other eight follow
    // packet 1's header.
    let mut s = Stream::new(2, &[0, 1]);
    s.pos = PACKET_BITS - 7;
    let first = frame_bits(500, 1);
    let second = frame_bits(60, 2);
    let a = s.push(&first);
    let b = s.push(&second);
    let data = s.finish();
    assert_eq!(a, PACKET_BITS - 7);
    assert_eq!(b, PACKET_BITS + HEADER_BITS + 500 - 7);

    let f = xma::read_frame(&data, a).unwrap();
    assert_eq!(f.bits, 500);
    assert_eq!(f.data, pack(&first));
    assert_eq!(f.next, Some(b));
    let g = xma::read_frame(&data, b).unwrap();
    assert_eq!(g.data, pack(&second));
    assert_eq!(g.next, None);
}

#[test]
fn padding_after_the_last_frame_skips_other_streams() {
    // Packet 0's only frame ends mid-packet; the rest is padding, and two
    // packets of another stream sit before this stream's next packet.
    let mut data = vec![0u8; PACKET_BYTES * 4];
    data[PACKET_BYTES..PACKET_BYTES * 3].fill(0xff);
    set_header(&mut data, 0, header(1, 10, 2));
    set_header(&mut data, 3, header(1, 33, 0));
    let start = HEADER_BITS + 10;
    let bits = frame_bits(200, 7);
    for (i, &b) in bits.iter().enumerate() {
        put_bit(&mut data, start + i, b);
    }
    // Garbage where a frame would follow without the padding rule.
    put_bit(&mut data, start + 200, true);

    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.data, pack(&bits));
    assert_eq!(f.next, Some(3 * PACKET_BITS + HEADER_BITS + 33));
}

#[test]
fn next_must_land_inside_the_buffer() {
    // The next packet's header is all that is left of the buffer: its first
    // frame would start right after it.
    let mut full = vec![0u8; PACKET_BYTES * 2];
    set_header(&mut full, 0, header(1, 0, 0));
    set_header(&mut full, 1, header(1, 0, 0));
    let bits = frame_bits(100, 3);
    for (i, &b) in bits.iter().enumerate() {
        put_bit(&mut full, HEADER_BITS + i, b);
    }
    let next = PACKET_BITS + HEADER_BITS;
    assert_eq!(
        xma::read_frame(&full, HEADER_BITS).unwrap().next,
        Some(next)
    );
    let five = &full[..PACKET_BYTES + 5];
    assert_eq!(xma::read_frame(five, HEADER_BITS).unwrap().next, Some(next));
    let four = &full[..PACKET_BYTES + 4];
    assert_eq!(xma::read_frame(four, HEADER_BITS).unwrap().next, None);
    // Without even the header there is no next packet to look at.
    let none = &full[..PACKET_BYTES + 3];
    assert_eq!(xma::read_frame(none, HEADER_BITS).unwrap().next, None);
}

#[test]
fn frames_count_decides_whether_the_rest_is_padding() {
    // Two frames end to end in packet 0. With a count of 2 the second is
    // the last and its `next` jumps to packet 1; with 3 it is followed
    // directly, where a third frame begins.
    for (count, expect_jump) in [(2, true), (3, false)] {
        let mut data = vec![0u8; PACKET_BYTES * 2];
        set_header(&mut data, 0, header(count, 0, 0));
        set_header(&mut data, 1, header(1, 4, 0));
        let mut at = HEADER_BITS;
        let mut starts = vec![];
        for (i, len) in [80usize, 90, 30].into_iter().enumerate() {
            starts.push(at);
            for (j, &b) in frame_bits(len, 50 + i as u32).iter().enumerate() {
                put_bit(&mut data, at + j, b);
            }
            at += len;
        }
        let f = xma::read_frame(&data, starts[1]).unwrap();
        let expect = if expect_jump {
            PACKET_BITS + HEADER_BITS + 4
        } else {
            starts[2]
        };
        assert_eq!(f.next, Some(expect), "count {count}");
    }
}
