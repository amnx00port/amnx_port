//! XMA2 frame decoder invariants on synthetic, deterministic input.
//!
//! No recorded audio: every frame is a pseudo-random byte pattern with a
//! chosen length field, so these tests pin down what the decoder promises
//! for any input (output size, determinism, state handling) rather than
//! particular sample values.

use victory_formats::xma::{Decoder, Error, FRAME_SAMPLES};

/// `len` bytes of xorshift noise whose first 15 bits are `bits`, the
/// frame's length field.
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
    if len >= 2 {
        let field = (bits.min(0x7fff) as u16) << 1;
        data[0] = (field >> 8) as u8;
        data[1] = (data[1] & 1) | field as u8;
    }
    data
}

fn frames() -> Vec<(Vec<u8>, usize)> {
    let mut frames = vec![];
    for seed in 1..=48u32 {
        let len = 8 + (seed as usize * 37) % 400;
        let bits = len * 8 - (seed as usize % 8);
        frames.push((noise_frame(len, bits, seed), bits));
    }
    frames
}

fn run(d: &mut Decoder, frames: &[(Vec<u8>, usize)]) -> (Vec<i16>, Vec<Result<(), Error>>) {
    let mut out = vec![];
    let results = frames
        .iter()
        .map(|(data, bits)| d.decode(data, *bits, &mut out))
        .collect();
    d.flush(&mut out);
    (out, results)
}

#[test]
fn any_frame_appends_exactly_one_frame_of_samples() {
    for channels in [1, 2] {
        for rate in [22_050, 32_000, 44_100, 48_000] {
            let mut d = Decoder::new(channels, rate);
            let mut out = vec![];
            for (i, (data, bits)) in frames().iter().enumerate() {
                let _ = d.decode(data, *bits, &mut out);
                assert_eq!(out.len(), (i + 1) * FRAME_SAMPLES * channels);
            }
        }
    }
}

#[test]
fn decoding_is_deterministic_and_clones_share_no_state() {
    let frames = frames();
    for channels in [1, 2] {
        let mut a = Decoder::new(channels, 48_000);
        let mut b = a.clone();
        let first = run(&mut a, &frames);
        // `a` was flushed (and so reset) and `b` never ran: both must match a
        // fresh decoder.
        assert_eq!(run(&mut b, &frames), first);
        assert_eq!(run(&mut a, &frames), first);
    }
}

#[test]
fn clone_mid_stream_continues_identically() {
    let frames = frames();
    let (head, tail) = frames.split_at(frames.len() / 2);
    for channels in [1, 2] {
        let mut a = Decoder::new(channels, 44_100);
        let mut sink = vec![];
        for (data, bits) in head {
            let _ = a.decode(data, *bits, &mut sink);
        }
        let mut b = a.clone();
        assert_eq!(run(&mut a, tail), run(&mut b, tail));
    }
}

#[test]
fn reset_matches_a_fresh_decoder() {
    let frames = frames();
    for channels in [1, 2] {
        let mut used = Decoder::new(channels, 32_000);
        let mut sink = vec![];
        for (data, bits) in &frames {
            let _ = used.decode(data, *bits, &mut sink);
        }
        used.reset();
        let mut fresh = Decoder::new(channels, 32_000);
        assert_eq!(run(&mut used, &frames), run(&mut fresh, &frames));
    }
}

#[test]
fn flush_after_reset_is_silence() {
    let frames = frames();
    for channels in [1, 2] {
        let mut d = Decoder::new(channels, 48_000);
        let mut sink = vec![];
        for (data, bits) in &frames {
            let _ = d.decode(data, *bits, &mut sink);
        }
        d.reset();
        let mut out = vec![];
        d.flush(&mut out);
        assert_eq!(out, vec![0; FRAME_SAMPLES / 2 * channels]);
    }
}

#[test]
fn bit_count_past_the_data_is_the_end_of_the_data() {
    for (data, _) in frames() {
        for channels in [1, 2] {
            let exact = {
                let mut d = Decoder::new(channels, 48_000);
                let mut out = vec![];
                let r = d.decode(&data, data.len() * 8, &mut out);
                (out, r)
            };
            for bits in [data.len() * 8 + 1, data.len() * 64, usize::MAX] {
                let mut d = Decoder::new(channels, 48_000);
                let mut out = vec![];
                let r = d.decode(&data, bits, &mut out);
                assert_eq!((out, r), exact);
            }
        }
    }
}

#[test]
fn sample_rates_in_one_bucket_decode_alike() {
    // Rates round up to the next of 24, 32, 44.1 and 48 kHz.
    let frames = frames();
    for channels in [1, 2] {
        for bucket in [
            &[8_000, 16_000, 22_050, 24_000][..],
            &[24_001, 30_000, 32_000],
            &[32_001, 40_000, 44_100],
            &[44_101, 48_000, 96_000, u32::MAX],
        ] {
            let expected = run(&mut Decoder::new(channels, bucket[0]), &frames);
            for &rate in &bucket[1..] {
                assert_eq!(
                    run(&mut Decoder::new(channels, rate), &frames),
                    expected,
                    "{rate} Hz"
                );
            }
        }
    }
}

#[test]
fn channel_counts_above_two_decode_as_stereo() {
    let frames = frames();
    let stereo = run(&mut Decoder::new(2, 48_000), &frames);
    for channels in [3, 6, usize::MAX] {
        assert_eq!(run(&mut Decoder::new(channels, 48_000), &frames), stereo);
    }
    let mono = run(&mut Decoder::new(1, 48_000), &frames);
    assert_eq!(run(&mut Decoder::new(0, 48_000), &frames), mono);
}

#[test]
fn decoding_appends_without_touching_earlier_samples() {
    let frames = frames();
    let mut d = Decoder::new(2, 48_000);
    let prefix = [i16::MIN, -1, 0, 1, i16::MAX];
    let mut out = prefix.to_vec();
    for (data, bits) in &frames {
        let _ = d.decode(data, *bits, &mut out);
    }
    d.flush(&mut out);
    assert_eq!(out[..prefix.len()], prefix);
    let (plain, _) = run(&mut Decoder::new(2, 48_000), &frames);
    assert_eq!(out[prefix.len()..], plain[..]);
}

#[test]
fn errors_name_the_codec() {
    let e = Error("frame overread");
    assert_eq!(e.to_string(), "XMA: frame overread");
    let boxed: Box<dyn std::error::Error> = Box::new(e);
    assert_eq!(boxed.to_string(), "XMA: frame overread");
}
