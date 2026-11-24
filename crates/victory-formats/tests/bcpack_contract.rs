//! BCPack's public contract on synthetic, deterministic input.
//!
//! No packed title data: the streams are xorshift noise and the table files
//! are the shipped one with single fields changed. These pin down what
//! `Tables::parse` and `decode` promise for any input (which errors, that a
//! failed decode leaves `dst` alone, that trailing bytes are ignored, that
//! decoding is deterministic) rather than particular block values.

use victory_formats::bcpack::{decode, Error, Tables, TABLES_SIZE};

const TABLE_FILE: &[u8] = include_bytes!("../src/bcpack-tables.bin");
const FORMATS: [(u8, usize); 6] = [(1, 8), (3, 16), (4, 8), (5, 16), (6, 16), (7, 16)];

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

/// Model 0's bucket 0 frequency, the first u16 of the file.
fn with_first_freq(value: u16) -> Vec<u8> {
    let mut bytes = TABLE_FILE.to_vec();
    bytes[..2].copy_from_slice(&value.to_le_bytes());
    bytes
}

#[test]
fn shipped_table_file_is_exactly_tables_size() {
    assert_eq!(TABLE_FILE.len(), TABLES_SIZE);
    assert!(Tables::parse(TABLE_FILE).is_ok());
}

#[test]
fn table_files_of_any_other_size_are_refused() {
    let wrong = Error("table file has the wrong size");
    for len in [0, 1, TABLES_SIZE - 1, TABLES_SIZE + 1, TABLES_SIZE * 2] {
        let mut bytes = TABLE_FILE.to_vec();
        bytes.resize(len, 0);
        assert_eq!(Tables::parse(&bytes).err(), Some(wrong), "{len} bytes");
    }
}

#[test]
fn an_all_zero_table_file_is_refused() {
    let err = Tables::parse(&vec![0; TABLES_SIZE]).err();
    assert!(
        matches!(
            err,
            Some(Error(
                "a model has an empty bucket" | "a model does not tile 65536"
            ))
        ),
        "{err:?}"
    );
}

#[test]
fn a_model_that_does_not_sum_to_65536_is_refused() {
    let first = u16::from_le_bytes([TABLE_FILE[0], TABLE_FILE[1]]);
    for value in [first.wrapping_add(1), first.wrapping_sub(1), u16::MAX] {
        if value == first || value == 0 {
            continue;
        }
        assert!(
            Tables::parse(&with_first_freq(value)).is_err(),
            "first frequency {first} -> {value}"
        );
    }
}

#[test]
fn a_model_with_an_empty_bucket_is_refused() {
    assert!(Tables::parse(&with_first_freq(0)).is_err());
}

#[test]
fn errors_display_with_the_format_name() {
    let text = Error("unsupported BC format").to_string();
    assert_eq!(text, "BCPack: unsupported BC format");
    let boxed: Box<dyn std::error::Error> = Box::new(Error("x"));
    assert_eq!(boxed.to_string(), "BCPack: x");
}

#[test]
fn formats_other_than_bc1_and_bc3_to_bc7_are_refused() {
    let t = tables();
    let src = noise(4096, 7);
    for bc in [0u8, 2, 8, 9, 16, 255] {
        let mut dst = [0xa5u8; 64];
        assert_eq!(
            decode(bc, &src, &mut dst, &t),
            Err(Error("unsupported BC format")),
            "BC{bc}"
        );
        assert!(dst.iter().all(|&b| b == 0xa5), "BC{bc} touched dst");
    }
}

#[test]
fn destinations_that_are_not_whole_blocks_are_refused_untouched() {
    let t = tables();
    let src = noise(4096, 11);
    for (bc, block) in FORMATS {
        for len in [0, 1, block - 1, block + 1, block * 3 + block / 2] {
            let mut dst = vec![0x5au8; len];
            assert_eq!(
                decode(bc, &src, &mut dst, &t),
                Err(Error("destination is not a whole number of blocks")),
                "BC{bc}, {len} bytes"
            );
            assert!(dst.iter().all(|&b| b == 0x5a), "BC{bc}, {len} bytes");
        }
    }
}

#[test]
fn an_empty_stream_is_an_error_and_leaves_dst_alone() {
    let t = tables();
    for (bc, block) in FORMATS {
        let mut dst = vec![0x3cu8; block * 4];
        assert!(decode(bc, &[], &mut dst, &t).is_err(), "BC{bc}");
        assert!(dst.iter().all(|&b| b == 0x3c), "BC{bc}");
    }
}

#[test]
fn noise_never_panics_and_failures_leave_dst_alone() {
    let t = tables();
    for (bc, block) in FORMATS {
        for seed in 1..24u32 {
            for blocks in [1usize, 2, 17] {
                let src = noise(64 + seed as usize * 37, seed.wrapping_mul(0x9e37_79b9));
                let mut dst = vec![0xc3u8; block * blocks];
                if decode(bc, &src, &mut dst, &t).is_err() {
                    assert!(
                        dst.iter().all(|&b| b == 0xc3),
                        "BC{bc} seed {seed} {blocks} blocks"
                    );
                }
            }
        }
    }
}

#[test]
fn decoding_is_deterministic_and_ignores_trailing_bytes() {
    let t = tables();
    let mut checked = 0;
    for (bc, block) in FORMATS {
        for seed in 1..64u32 {
            let src = noise(8192, seed);
            let mut first = vec![0u8; block * 4];
            if decode(bc, &src, &mut first, &t).is_err() {
                continue;
            }
            let mut again = vec![0xffu8; block * 4];
            decode(bc, &src, &mut again, &t).expect("same input, same result");
            assert_eq!(first, again, "BC{bc} seed {seed}");

            let mut longer = src.clone();
            longer.extend(noise(512, seed ^ 0xdead_beef));
            let mut tail = vec![0u8; block * 4];
            decode(bc, &longer, &mut tail, &t).expect("trailing bytes are ignored");
            assert_eq!(first, tail, "BC{bc} seed {seed} with trailing bytes");
            checked += 1;
        }
    }
    // Whether noise decodes is up to the models; the property is only
    // meaningful if at least one stream did.
    assert!(checked > 0, "no noise stream decoded for any format");
}

#[test]
fn a_stream_cut_short_fails_and_leaves_dst_alone() {
    // Cutting a stream short of what the lanes need must fail rather than
    // decode garbage: find a stream that decodes, then halve it until it
    // stops decoding, and check the error is the stream error.
    let t = tables();
    for (bc, block) in FORMATS {
        for seed in 1..64u32 {
            let src = noise(8192, seed);
            let mut dst = vec![0u8; block * 8];
            if decode(bc, &src, &mut dst, &t).is_err() {
                continue;
            }
            let mut len = src.len();
            while len > 0 && decode(bc, &src[..len], &mut dst, &t).is_ok() {
                len /= 2;
            }
            let mut fresh = vec![0x77u8; block * 8];
            let err = decode(bc, &src[..len], &mut fresh, &t).unwrap_err();
            assert!(
                fresh.iter().all(|&b| b == 0x77),
                "BC{bc} seed {seed} cut to {len}: {err}"
            );
            break;
        }
    }
}
