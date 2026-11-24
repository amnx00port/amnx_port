//! BCPack streams and table files beyond the basic contract, on synthetic,
//! deterministic input.
//!
//! Like `bcpack_contract.rs` there is no packed title data: streams are
//! xorshift noise and table files are the shipped one with fields changed.
//! These pin down that a stream's blocks come out in order (a shorter
//! destination gets a prefix of a longer one), that what `dst` held before
//! plays no part, that the table fields `Tables::parse` documents as unused
//! (a model's buckets past its shape, the BC6H endpoint mode byte) change
//! nothing, that only the model frequencies are validated, and that one
//! `Tables` can be shared across threads.

use victory_formats::bcpack::{decode, Error, Tables, TABLES_SIZE};

const TABLE_FILE: &[u8] = include_bytes!("../src/bcpack-tables.bin");
const FORMATS: [(u8, usize); 6] = [(1, 8), (3, 16), (4, 8), (5, 16), (6, 16), (7, 16)];
const MODELS: usize = 128;
const BUCKETS: usize = 9;
/// Bytes of the model frequencies at the start of the file.
const MODEL_BYTES: usize = MODELS * BUCKETS * 2;
/// Bytes of the rank tables and BC7 control words after the models.
const RANK_AND_CONTROL_BYTES: usize = 256 + 64 + 256 + 64 + 64 + 512 * 2;
/// Bytes of one format's 255 dictionary entries, in format order.
const DICTIONARY_BYTES: [usize; 6] = [
    255 * (6 + 16),
    255 * (8 + 32),
    255 * (2 + 16),
    255 * (4 + 32),
    255 * (1 + 24 + 1 + 32),
    255 * (2 + 48 + 2 + 64),
];

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

fn tables() -> Tables {
    Tables::builtin().expect("the shipped tables parse")
}

/// Offset in the file of model `m`'s bucket `b` frequency.
fn freq_at(m: usize, b: usize) -> usize {
    (m * BUCKETS + b) * 2
}

/// Buckets model `m` uses: model 16 g + shape has g + 2.
fn used_buckets(m: usize) -> usize {
    m / 16 + 2
}

/// Every (format, block bytes, stream) among the first seeds whose noise
/// decodes `blocks` blocks, with the decoded bytes.
fn decodable(t: &Tables, blocks: usize) -> Vec<(u8, usize, Vec<u8>, Vec<u8>)> {
    let mut found = Vec::new();
    for (bc, block) in FORMATS {
        for seed in 1..64u32 {
            let src = noise(16384, seed);
            let mut dst = vec![0u8; block * blocks];
            if decode(bc, &src, &mut dst, t).is_ok() {
                found.push((bc, block, src, dst));
            }
        }
    }
    // Whether noise decodes is up to the models; the properties below are
    // only meaningful if some stream did.
    assert!(!found.is_empty(), "no noise stream decoded {blocks} blocks");
    found
}

#[test]
fn the_dictionaries_fill_the_file_after_the_models_ranks_and_control_words() {
    let dictionaries: usize = DICTIONARY_BYTES.iter().sum();
    assert_eq!(
        MODEL_BYTES + RANK_AND_CONTROL_BYTES + dictionaries,
        TABLES_SIZE
    );
}

#[test]
fn every_shipped_model_tiles_65536_with_its_used_buckets() {
    for m in 0..MODELS {
        let total: u32 = (0..used_buckets(m))
            .map(|b| {
                let at = freq_at(m, b);
                let f = u32::from(u16::from_le_bytes([TABLE_FILE[at], TABLE_FILE[at + 1]]));
                assert_ne!(f, 0, "model {m} bucket {b} is empty");
                // Bucket b > 0 holds 2^(b-1) symbols of this frequency.
                f << b.saturating_sub(1)
            })
            .sum();
        assert_eq!(total, 1 << 16, "model {m}");
    }
}

#[test]
fn an_empty_used_bucket_in_any_model_is_refused() {
    for m in [0, 15, 16, 63, 64, 127] {
        for b in [0, used_buckets(m) - 1] {
            let mut bytes = TABLE_FILE.to_vec();
            bytes[freq_at(m, b)..][..2].copy_from_slice(&0u16.to_le_bytes());
            assert_eq!(
                Tables::parse(&bytes).err(),
                Some(Error("a model has an empty bucket")),
                "model {m} bucket {b}"
            );
        }
    }
}

#[test]
fn a_model_off_by_one_in_its_last_used_bucket_does_not_tile() {
    for m in [0, 31, 112, 127] {
        let at = freq_at(m, used_buckets(m) - 1);
        let first = u16::from_le_bytes([TABLE_FILE[at], TABLE_FILE[at + 1]]);
        for value in [first + 1, first - 1] {
            if value == 0 {
                continue;
            }
            let mut bytes = TABLE_FILE.to_vec();
            bytes[at..][..2].copy_from_slice(&value.to_le_bytes());
            assert_eq!(
                Tables::parse(&bytes).err(),
                Some(Error("a model does not tile 65536")),
                "model {m}: {first} -> {value}"
            );
        }
    }
}

#[test]
fn buckets_past_a_models_shape_are_ignored() {
    // Models 0-111 leave buckets unused; anything there, zero included,
    // parses and decodes exactly like the shipped file.
    let shipped = tables();
    let streams = decodable(&shipped, 4);
    for fill in [0u16, 1, 0x7fff, u16::MAX] {
        let mut bytes = TABLE_FILE.to_vec();
        for m in 0..MODELS {
            for b in used_buckets(m)..BUCKETS {
                bytes[freq_at(m, b)..][..2].copy_from_slice(&fill.to_le_bytes());
            }
        }
        let changed = Tables::parse(&bytes).expect("unused buckets are not checked");
        for (bc, _, src, want) in &streams {
            let mut got = vec![0u8; want.len()];
            decode(*bc, src, &mut got, &changed).expect("same stream, same models");
            assert_eq!(&got, want, "BC{bc}, unused buckets = {fill:#x}");
        }
    }
}

#[test]
fn only_the_model_frequencies_are_validated() {
    // Ranks, control words and dictionaries are constants the decoder
    // trusts: any bytes there parse.
    for (seed, fill) in [(3u32, None), (5, Some(0u8)), (9, Some(0xff))] {
        let mut bytes = TABLE_FILE.to_vec();
        let rest = &mut bytes[MODEL_BYTES..];
        match fill {
            Some(b) => rest.fill(b),
            None => rest.copy_from_slice(&noise(rest.len(), seed)),
        }
        assert!(Tables::parse(&bytes).is_ok(), "rest filled {fill:?}");
    }
}

#[test]
fn bc6h_endpoint_mode_bytes_play_no_part() {
    let shipped = tables();
    let bc6_start =
        MODEL_BYTES + RANK_AND_CONTROL_BYTES + DICTIONARY_BYTES[..4].iter().sum::<usize>();
    let entry = DICTIONARY_BYTES[4] / 255;
    let mut bytes = TABLE_FILE.to_vec();
    for e in 0..255 {
        bytes[bc6_start + e * entry] ^= 0xff;
    }
    let changed = Tables::parse(&bytes).expect("still a table file");
    let mut checked = 0;
    for seed in 1..256u32 {
        let src = noise(16384, seed);
        let mut want = vec![0u8; 16 * 8];
        let shipped_result = decode(6, &src, &mut want, &shipped);
        let mut got = vec![0u8; 16 * 8];
        let changed_result = decode(6, &src, &mut got, &changed);
        assert_eq!(shipped_result, changed_result, "seed {seed}");
        if shipped_result.is_ok() {
            assert_eq!(got, want, "seed {seed}");
            checked += 1;
        }
    }
    assert!(checked > 0, "no BC6H noise stream decoded");
}

#[test]
fn builtin_tables_decode_like_the_parsed_file() {
    let builtin = tables();
    let parsed = Tables::parse(TABLE_FILE).expect("the shipped file parses");
    for (bc, _, src, want) in decodable(&builtin, 4) {
        let mut got = vec![0u8; want.len()];
        decode(bc, &src, &mut got, &parsed).expect("same tables");
        assert_eq!(got, want, "BC{bc}");
    }
}

#[test]
fn a_shorter_destination_gets_the_first_blocks_of_a_longer_one() {
    // Blocks come out in stream order and only refer back: decoding fewer
    // blocks from the same stream gives a prefix of the longer result.
    let t = tables();
    for (bc, block, src, long) in decodable(&t, 12) {
        for blocks in [1usize, 2, 5, 11] {
            let mut short = vec![0u8; block * blocks];
            decode(bc, &src, &mut short, &t)
                .unwrap_or_else(|e| panic!("BC{bc}: 12 blocks decode but {blocks} fail: {e}"));
            assert_eq!(short, long[..short.len()], "BC{bc}, {blocks} blocks");
        }
    }
}

#[test]
fn what_dst_held_before_plays_no_part() {
    let t = tables();
    for (bc, _, src, want) in decodable(&t, 6) {
        for fill in [0x00u8, 0x55, 0xff] {
            let mut got = vec![fill; want.len()];
            decode(bc, &src, &mut got, &t).expect("same stream");
            assert_eq!(got, want, "BC{bc}, dst filled {fill:#x}");
        }
        let mut got = noise(want.len(), u32::from(bc) * 977);
        decode(bc, &src, &mut got, &t).expect("same stream");
        assert_eq!(got, want, "BC{bc}, dst filled with noise");
    }
}

#[test]
fn one_table_set_decodes_the_same_on_many_threads() {
    let t = tables();
    let streams = decodable(&t, 8);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for (bc, _, src, want) in &streams {
                        let mut got = vec![0u8; want.len()];
                        decode(*bc, src, &mut got, &t).expect("same stream, same tables");
                        assert_eq!(&got, want, "BC{bc}");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("a decoding thread panicked");
        }
    });
}
