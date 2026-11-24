//! XMA2 flush semantics and gathered-frame padding on synthetic input.
//!
//! Hand-built packets and xorshift noise frames only: no recorded audio,
//! so these pin down what `flush` and `read_frame` promise for any input.

use victory_formats::xma::{self, Decoder, FRAME_SAMPLES, PACKET_BYTES};

const HEADER_BITS: usize = 32;

/// `len` bytes of xorshift noise whose 15-bit length field is `bits`.
fn noise_frame(len: usize, bits: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    let mut data: Vec<u8> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let field = (bits.min(0x7fff) as u16) << 1;
    data[0] = (field >> 8) as u8;
    data[1] = (data[1] & 1) | field as u8;
    data
}

fn frames() -> Vec<(Vec<u8>, usize)> {
    (1..=24u32)
        .map(|seed| {
            let len = 16 + (seed as usize * 53) % 300;
            let bits = len * 8 - (seed as usize % 8);
            (noise_frame(len, bits, seed), bits)
        })
        .collect()
}

#[test]
fn flush_leaves_the_decoder_as_good_as_new() {
    let frames = frames();
    for channels in [1, 2] {
        let mut used = Decoder::new(channels, 48_000);
        let mut sink = vec![];
        for (data, bits) in &frames {
            let _ = used.decode(data, *bits, &mut sink);
        }
        used.flush(&mut sink);
        // A second flush has no overlap left to give.
        let mut again = vec![];
        used.flush(&mut again);
        assert_eq!(again, vec![0; FRAME_SAMPLES / 2 * channels]);
        // And the stream after it decodes like a brand new one.
        let mut fresh = Decoder::new(channels, 48_000);
        let (mut a, mut b) = (vec![], vec![]);
        for (data, bits) in frames.iter().rev() {
            let ra = used.decode(data, *bits, &mut a);
            let rb = fresh.decode(data, *bits, &mut b);
            assert_eq!(ra, rb);
        }
        used.flush(&mut a);
        fresh.flush(&mut b);
        assert_eq!(a, b);
    }
}

#[test]
fn flush_appends_after_existing_samples() {
    let frames = frames();
    for channels in [1, 2] {
        let mut plain = Decoder::new(channels, 44_100);
        let mut prefixed = plain.clone();
        let mut a = vec![];
        let mut b = vec![-1i16, 2, -3];
        for (data, bits) in &frames[..5] {
            let _ = plain.decode(data, *bits, &mut a);
            let _ = prefixed.decode(data, *bits, &mut b);
        }
        let before = a.len();
        plain.flush(&mut a);
        prefixed.flush(&mut b);
        assert_eq!(a.len() - before, FRAME_SAMPLES / 2 * channels);
        assert_eq!(b[..3], [-1, 2, -3]);
        assert_eq!(b[3..], a[..]);
    }
}

#[test]
fn gathered_frames_pad_their_last_byte_with_zeros_and_keep_their_length() {
    for bits in [16usize, 17, 20, 23, 31, 33, 100, 1001] {
        // One packet whose payload is all ones after a frame of `bits`
        // bits: the frame's bits are its length field then ones.
        let mut data = vec![0xffu8; PACKET_BYTES];
        // frames = 1, first_frame_bit = 0, skip = 0.
        data[..4].copy_from_slice(&(1u32 << 26).to_be_bytes());
        for i in 0..15 {
            let bit = (bits >> (14 - i)) & 1 == 1;
            let pos = HEADER_BITS + i;
            let mask = 0x80u8 >> (pos % 8);
            if bit {
                data[pos / 8] |= mask;
            } else {
                data[pos / 8] &= !mask;
            }
        }
        let f = xma::read_frame(&data, HEADER_BITS).unwrap();
        assert_eq!((f.bits, f.start), (bits, HEADER_BITS));
        assert_eq!(f.data.len(), bits.div_ceil(8));
        // The length field reads back, so the frame feeds the decoder as is.
        let field = (u16::from_be_bytes([f.data[0], f.data[1]]) >> 1) as usize;
        assert_eq!(field, bits);
        // Bits past the frame are zero even though the packet held ones.
        let spare = f.data.len() * 8 - bits;
        let last = *f.data.last().unwrap();
        assert_eq!(last & ((1u16 << spare) - 1) as u8, 0, "bits {bits}");
        // And every frame bit after the length field is a one.
        for i in 15..bits {
            assert_eq!((f.data[i / 8] >> (7 - i % 8)) & 1, 1, "bits {bits} bit {i}");
        }
        let mut out = vec![];
        let _ = Decoder::new(1, 48_000).decode(&f.data, f.bits, &mut out);
        assert_eq!(out.len(), FRAME_SAMPLES);
    }
}
