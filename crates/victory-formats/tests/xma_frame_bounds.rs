//! XMA2 decoder: what a frame's bit count bounds, and that an undecodable
//! frame drops the overlap it would have carried into the next one.
//!
//! Synthetic input only: xorshift noise with a chosen length field.

use victory_formats::xma::{Decoder, FRAME_SAMPLES};

fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

/// `len` bytes of noise whose first 15 bits are `bits`.
fn noise_frame(len: usize, bits: usize, seed: u32) -> Vec<u8> {
    let mut data = noise(len, seed);
    let field = (bits.min(0x7fff) as u16) << 1;
    data[0] = (field >> 8) as u8;
    data[1] = (data[1] & 1) | field as u8;
    data
}

fn frames() -> Vec<(Vec<u8>, usize)> {
    (1..=64u32)
        .map(|seed| {
            let len = 16 + (seed as usize * 53) % 500;
            let bits = len * 8 - 3 - (seed as usize % 40);
            (noise_frame(len, bits, seed), bits)
        })
        .collect()
}

/// `data` with every bit from `bits` on replaced by `fill`'s bits.
fn with_tail(data: &[u8], bits: usize, fill: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for i in bits..data.len() * 8 {
        let mask = 0x80u8 >> (i % 8);
        out[i / 8] = (out[i / 8] & !mask) | (fill[i / 8] & mask);
    }
    out
}

fn decode_one(channels: usize, data: &[u8], bits: usize) -> (Vec<i16>, bool, Vec<i16>) {
    let mut d = Decoder::new(channels, 48_000);
    let mut out = vec![];
    let ok = d.decode(data, bits, &mut out).is_ok();
    let mut tail = vec![];
    d.flush(&mut tail);
    (out, ok, tail)
}

#[test]
fn bits_after_the_bit_count_play_no_part_in_a_decoded_frame() {
    let mut decoded = 0;
    for (i, (data, bits)) in frames().iter().enumerate() {
        for channels in [1, 2] {
            let zeros = with_tail(data, *bits, &vec![0; data.len()]);
            let ones = with_tail(data, *bits, &vec![0xff; data.len()]);
            let other = with_tail(data, *bits, &noise(data.len(), 977 + i as u32));
            let reference = decode_one(channels, &zeros, *bits);
            if !reference.1 {
                continue;
            }
            decoded += 1;
            assert_eq!(decode_one(channels, &ones, *bits), reference, "frame {i}");
            assert_eq!(decode_one(channels, &other, *bits), reference, "frame {i}");
            assert_eq!(decode_one(channels, data, *bits), reference, "frame {i}");
        }
    }
    assert!(decoded > 0, "no frame decoded");
}

#[test]
fn bytes_past_the_bit_count_can_be_cut_off() {
    let mut decoded = 0;
    for (i, (data, bits)) in frames().iter().enumerate() {
        for channels in [1, 2] {
            let whole = decode_one(channels, data, *bits);
            if !whole.1 {
                continue;
            }
            decoded += 1;
            let cut = &data[..bits.div_ceil(8)];
            assert_eq!(decode_one(channels, cut, *bits), whole, "frame {i}");
        }
    }
    assert!(decoded > 0, "no frame decoded");
}

#[test]
fn an_undecodable_frame_drops_the_overlap_it_inherited() {
    let all = frames();
    let bad = vec![0xffu8; 64];
    let mut checked = 0;
    for channels in [1, 2] {
        for (data, bits) in &all {
            let mut d = Decoder::new(channels, 48_000);
            let mut out = vec![];
            if d.decode(data, *bits, &mut out).is_err() {
                continue;
            }
            let mut probe = d.clone();
            let mut tail = vec![];
            probe.flush(&mut tail);
            if tail.iter().all(|&s| s == 0) {
                continue;
            }
            checked += 1;
            let mut out = vec![];
            assert!(d.decode(&bad, bad.len() * 8, &mut out).is_err());
            assert_eq!(out, vec![0; FRAME_SAMPLES * channels]);
            let mut tail = vec![];
            d.flush(&mut tail);
            assert_eq!(tail, vec![0; FRAME_SAMPLES / 2 * channels]);
        }
    }
    assert!(checked > 0, "no frame left an audible overlap");
}
