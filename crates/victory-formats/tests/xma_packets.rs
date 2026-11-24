//! XMA2 packet headers and frame gathering on hand-built packets.
//!
//! Every buffer is synthetic: packets of zeros with headers and frame bits
//! written at known bit positions, so the expected frame bits, boundaries
//! and next-frame positions are computed here rather than taken from a
//! recorded stream.

use victory_formats::xma::{self, Decoder, FRAME_SAMPLES, PACKET_BYTES};

const PACKET_BITS: usize = PACKET_BYTES * 8;
const HEADER_BITS: usize = 32;

fn header(frames: u32, first_frame_bit: u32, metadata: u32, skip: u32) -> [u8; 4] {
    (frames << 26 | first_frame_bit << 11 | metadata << 8 | skip).to_be_bytes()
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

/// Writes `bits` starting at absolute bit `start`, stepping over packet
/// headers and the `skip` packets of other streams the way the stream
/// cursor reads them (missing `skips` entries are 0). Returns the bit position after the last one.
fn write_stream(data: &mut [u8], start: usize, bits: &[bool], skips: &[usize]) -> usize {
    let mut pos = start;
    for &bit in bits {
        if pos.is_multiple_of(PACKET_BITS) {
            let previous = pos / PACKET_BITS;
            if previous > 0 {
                pos = (previous + skips.get(previous - 1).copied().unwrap_or(0)) * PACKET_BITS;
            }
            pos += HEADER_BITS;
        }
        put_bit(data, pos, bit);
        pos += 1;
    }
    pos
}

fn pack(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        put_bit(&mut out, i, b);
    }
    out
}

#[test]
fn packet_header_fields_ignore_metadata() {
    let mut data = vec![0u8; PACKET_BYTES * 2];
    set_header(&mut data, 0, header(63, 0x7fff, 0b111, 0xff));
    set_header(&mut data, 1, header(1, 0, 0b101, 0));
    let p = xma::packet(&data, 0).unwrap();
    assert_eq!((p.frames, p.first_frame_bit, p.skip), (63, 0x7fff, 0xff));
    let p = xma::packet(&data, 1).unwrap();
    assert_eq!((p.frames, p.first_frame_bit, p.skip), (1, 0, 0));
}

#[test]
fn packet_past_the_buffer_is_none() {
    let data = vec![0u8; PACKET_BYTES];
    assert!(xma::packet(&data, 0).is_some());
    assert!(xma::packet(&data, 1).is_none());
    assert!(xma::packet(&data, usize::MAX).is_none());
    // Only the first four bytes of a packet are needed for its header.
    let short = vec![0u8; PACKET_BYTES + 4];
    assert!(xma::packet(&short, 1).is_some());
    assert!(xma::packet(&short[..PACKET_BYTES + 3], 1).is_none());
    assert!(xma::packet(&[], 0).is_none());
}

#[test]
fn frames_in_one_packet_chain_to_each_other() {
    let mut data = vec![0u8; PACKET_BYTES];
    set_header(&mut data, 0, header(3, 0, 0, 0));
    let lens = [40, 300, 77];
    let mut at = HEADER_BITS;
    let mut starts = vec![];
    let mut frames = vec![];
    for (i, &len) in lens.iter().enumerate() {
        let bits = frame_bits(len, i as u32 + 7);
        starts.push(at);
        at = write_stream(&mut data, at, &bits, &[]);
        frames.push(bits);
    }
    for i in 0..lens.len() {
        let f = xma::read_frame(&data, starts[i]).unwrap();
        assert_eq!(f.bits, lens[i]);
        assert_eq!(f.start, starts[i]);
        assert_eq!(f.data, pack(&frames[i]));
        let expect = (i + 1 < lens.len()).then(|| starts[i + 1]);
        // The last frame to begin in the only packet has nowhere to go.
        assert_eq!(f.next, expect, "frame {i}");
    }
}

#[test]
fn last_frame_in_a_packet_jumps_to_the_next_packets_first_frame() {
    let mut data = vec![0u8; PACKET_BYTES * 2];
    // Packet 1's first frame begins after 24 bits that would finish an
    // earlier frame.
    set_header(&mut data, 0, header(1, 5, 0, 0));
    set_header(&mut data, 1, header(1, 24, 0, 0));
    let start = HEADER_BITS + 5;
    write_stream(&mut data, start, &frame_bits(64, 3), &[]);
    let second = PACKET_BITS + HEADER_BITS + 24;
    write_stream(&mut data, second, &frame_bits(90, 4), &[]);

    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.bits, 64);
    assert_eq!(f.next, Some(second));
    let g = xma::read_frame(&data, second).unwrap();
    assert_eq!(g.bits, 90);
    assert_eq!(g.data, pack(&frame_bits(90, 4)));
    assert_eq!(g.next, None);
}

#[test]
fn frame_spanning_two_packets_skips_the_second_header() {
    let mut data = vec![0u8; PACKET_BYTES * 2];
    let len = 1000;
    let start = PACKET_BITS - 400;
    let first_frame_bit = (start - HEADER_BITS) as u32;
    let spill = len - 400;
    set_header(&mut data, 0, header(1, first_frame_bit, 0, 0));
    set_header(&mut data, 1, header(1, spill as u32, 0, 0));
    let bits = frame_bits(len, 11);
    let end = write_stream(&mut data, start, &bits, &[]);
    assert_eq!(end, PACKET_BITS + HEADER_BITS + spill);
    // A frame begins right where the spilled one ends.
    write_stream(&mut data, end, &frame_bits(50, 12), &[]);

    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.bits, len);
    assert_eq!(f.data, pack(&bits));
    assert_eq!(f.next, Some(end));
    assert_eq!(xma::read_frame(&data, end).unwrap().bits, 50);
}

#[test]
fn frame_spanning_packets_steps_over_other_streams() {
    // Packet 0 says two packets of other streams follow; packet 3 holds the
    // rest of the frame.
    let mut data = vec![0u8; PACKET_BYTES * 4];
    let len = 700;
    let start = PACKET_BITS - 200;
    let spill = len - 200;
    set_header(&mut data, 0, header(1, (start - HEADER_BITS) as u32, 0, 2));
    // The other streams' packets are filled with ones: any read of them
    // would show up in the frame bits.
    data[PACKET_BYTES..PACKET_BYTES * 3].fill(0xff);
    set_header(&mut data, 3, header(0, spill as u32, 0, 0));
    let bits = frame_bits(len, 21);
    let end = write_stream(&mut data, start, &bits, &[2, 0, 0]);
    assert_eq!(end, 3 * PACKET_BITS + HEADER_BITS + spill);

    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.bits, len);
    assert_eq!(f.data, pack(&bits));
    assert_eq!(f.next, Some(end));
}

#[test]
fn frame_ending_exactly_at_a_packet_end_continues_after_skips() {
    let mut data = vec![0u8; PACKET_BYTES * 3];
    let len = 300;
    let start = PACKET_BITS - len;
    // More frames than are read: the boundary alone moves to the next
    // packet of this stream, past one packet of another stream.
    set_header(&mut data, 0, header(5, (start - HEADER_BITS) as u32, 0, 1));
    set_header(&mut data, 2, header(1, 8, 0, 0));
    write_stream(&mut data, start, &frame_bits(len, 31), &[1, 0]);
    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.bits, len);
    assert_eq!(f.next, Some(2 * PACKET_BITS + HEADER_BITS + 8));
}

#[test]
fn lengths_that_are_not_frames_are_rejected() {
    let mut data = vec![0u8; PACKET_BYTES];
    set_header(&mut data, 0, header(1, 0, 0, 0));
    for len in [0usize, 1, 15, 0x7fff] {
        let mut bits = vec![false; 15];
        for (i, b) in bits.iter_mut().enumerate() {
            *b = (len >> (14 - i)) & 1 == 1;
        }
        write_stream(&mut data, HEADER_BITS, &bits, &[]);
        assert!(
            xma::read_frame(&data, HEADER_BITS).is_none(),
            "length {len}"
        );
    }
    // The smallest real frame: one bit past its own length field.
    write_stream(&mut data, HEADER_BITS, &frame_bits(16, 1), &[]);
    let f = xma::read_frame(&data, HEADER_BITS).unwrap();
    assert_eq!((f.bits, f.data.len()), (16, 2));
}

#[test]
fn frame_running_off_the_buffer_is_none() {
    let mut data = vec![0u8; PACKET_BYTES];
    let start = PACKET_BITS - 100;
    set_header(&mut data, 0, header(1, (start - HEADER_BITS) as u32, 0, 0));
    // 200 bits but only 100 left and no next packet.
    let mut bits = frame_bits(200, 5);
    bits.truncate(100);
    write_stream(&mut data, start, &bits, &[]);
    assert!(xma::read_frame(&data, start).is_none());
    // A start past the buffer reads nothing.
    assert!(xma::read_frame(&data, PACKET_BITS * 4).is_none());
}

#[test]
fn damaged_earlier_frame_length_stops_the_next_frame_search() {
    // The header claims a frame at bit 0 of the payload whose length field
    // is zero; asking for the frame after it cannot walk there.
    let mut data = vec![0u8; PACKET_BYTES];
    set_header(&mut data, 0, header(2, 0, 0, 0));
    let start = HEADER_BITS + 100;
    write_stream(&mut data, start, &frame_bits(40, 9), &[]);
    let f = xma::read_frame(&data, start).unwrap();
    assert_eq!(f.bits, 40);
    assert_eq!(f.next, None);
}

#[test]
fn decoder_clamps_channels() {
    assert_eq!(Decoder::new(0, 48_000).channels(), 1);
    assert_eq!(Decoder::new(1, 48_000).channels(), 1);
    assert_eq!(Decoder::new(2, 44_100).channels(), 2);
    assert_eq!(Decoder::new(6, 22_050).channels(), 2);
}

#[test]
fn fresh_decoder_flushes_half_a_frame_of_silence() {
    for channels in [1, 2] {
        let mut d = Decoder::new(channels, 48_000);
        let mut out = vec![];
        d.flush(&mut out);
        assert_eq!(out.len(), FRAME_SAMPLES / 2 * channels);
        assert!(out.iter().all(|&s| s == 0));
    }
}

#[test]
fn undecodable_frames_still_keep_the_stream_timing() {
    for channels in [1, 2] {
        let mut d = Decoder::new(channels, 48_000);
        let mut out = vec![7i16];
        // An empty frame and a frame of ones: both fail, both emit a frame
        // of silence after what was already in `out`.
        assert!(d.decode(&[], 0, &mut out).is_err());
        assert_eq!(out.len(), 1 + FRAME_SAMPLES * channels);
        let ones = vec![0xffu8; 64];
        assert!(d.decode(&ones, ones.len() * 8, &mut out).is_err());
        assert_eq!(out.len(), 1 + 2 * FRAME_SAMPLES * channels);
        assert_eq!(out[0], 7);
        assert!(out[1..].iter().all(|&s| s == 0));
        d.flush(&mut out);
        assert_eq!(
            out.len(),
            1 + (2 * FRAME_SAMPLES + FRAME_SAMPLES / 2) * channels
        );
    }
}
