//! Concurrent use of the formats API on synthetic, deterministic input.
//!
//! Nothing here reads title data: textures are filled with an index
//! pattern and XMA frames are xorshift noise with a chosen length field.
//! These pin down that the public types can cross and be shared between
//! threads, that one `Texture` detiles every subresource of one buffer on
//! many threads exactly as it does on one, that subresources tile into
//! disjoint bytes so the order (or thread) that writes them does not
//! matter, and that an XMA decoder moved to another thread mid-stream
//! carries its overlap with it.

use std::thread;

use victory_formats::bcpack;
use victory_formats::texture::{self, Dimension, Layout, Swizzle, Texture, TextureDesc};
use victory_formats::xma::{self, Decoder};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_and_sync() {
    assert_send_sync::<Layout>();
    assert_send_sync::<Texture>();
    assert_send_sync::<texture::Error>();
    assert_send_sync::<Decoder>();
    assert_send_sync::<xma::Error>();
    assert_send_sync::<bcpack::Tables>();
    assert_send_sync::<bcpack::Error>();
    // Errors box into the usual thread-crossing error type.
    let boxed: Vec<Box<dyn std::error::Error + Send + Sync>> = vec![
        Box::new(texture::Error::Invalid),
        Box::new(xma::Error("x")),
        Box::new(bcpack::Error("x")),
    ];
    let moved = thread::spawn(move || boxed.iter().map(ToString::to_string).collect::<Vec<_>>())
        .join()
        .expect("the thread panicked");
    assert_eq!(moved.len(), 3);
    assert!(moved.iter().all(|s| !s.is_empty()));
}

/// Textures with several planes, mips and slices across tiled and linear
/// modes with addressing. Formats: 28 RGBA8, 45 D24S8 (a stencil plane), 71 BC1.
fn textures() -> Vec<Texture> {
    let mut out = vec![];
    for (dimension, w, h, d) in [
        (Dimension::Tex2d, 70, 33, 3),
        (Dimension::Tex2d, 256, 256, 2),
        (Dimension::Tex3d, 40, 24, 9),
    ] {
        for format in [28, 45, 71] {
            if dimension == Dimension::Tex3d && format == 45 {
                continue;
            }
            for swizzle in [
                Swizzle::LINEAR,
                Swizzle::SW_4KB_S,
                Swizzle::SW_64KB_D,
                Swizzle::SW_64KB_S_X,
            ] {
                let desc = TextureDesc {
                    dimension,
                    width: w,
                    height: h,
                    depth_or_array: d,
                    mips: 0,
                    format,
                    samples: 1,
                    swizzle,
                };
                // XOR modes build but have no addressing: copies skip them.
                match Texture::new(&desc) {
                    Ok(t) if t.planes.iter().all(|p| p.layout.addressable()) => out.push(t),
                    _ => {}
                }
            }
        }
    }
    assert!(out.len() >= 12, "too few textures built: {}", out.len());
    out
}

/// Every subresource's tightly packed rows, each byte tagged by its
/// subresource and position.
fn linear_images(t: &Texture) -> Vec<(u32, Vec<u8>, usize, usize)> {
    (0..t.subresource_count())
        .map(|i| {
            let f = t.footprint(i).expect("every index has a footprint");
            let bytes = (0..f.size)
                .map(|n| (n as u32).wrapping_mul(31).wrapping_add(i * 97) as u8 | 1)
                .collect();
            (i, bytes, f.row_pitch, f.depth_pitch)
        })
        .collect()
}

fn tile_all(t: &Texture, images: &[(u32, Vec<u8>, usize, usize)], order: &[usize]) -> Vec<u8> {
    let mut tiled = vec![0u8; t.size as usize];
    for &k in order {
        let (i, ref bytes, rp, dp) = images[k];
        t.tile(&mut tiled, i, bytes, rp, dp)
            .unwrap_or_else(|e| panic!("tile {i}: {e}"));
    }
    tiled
}

#[test]
fn subresources_tile_the_same_in_any_order() {
    for t in textures() {
        let images = linear_images(&t);
        let forward: Vec<usize> = (0..images.len()).collect();
        let reverse: Vec<usize> = forward.iter().rev().copied().collect();
        // Odd indices first, then even: interleaves planes, mips and slices.
        let shuffled: Vec<usize> = forward
            .iter()
            .filter(|k| *k % 2 == 1)
            .chain(forward.iter().filter(|k| *k % 2 == 0))
            .copied()
            .collect();
        let a = tile_all(&t, &images, &forward);
        assert_eq!(a, tile_all(&t, &images, &reverse), "{t:?}");
        assert_eq!(a, tile_all(&t, &images, &shuffled), "{t:?}");
    }
}

#[test]
fn one_texture_detiles_every_subresource_on_many_threads() {
    for t in textures() {
        let images = linear_images(&t);
        let order: Vec<usize> = (0..images.len()).collect();
        let tiled = tile_all(&t, &images, &order);
        let (t, tiled) = (&t, &tiled);
        let back: Vec<(u32, Vec<u8>)> = thread::scope(|scope| {
            let handles: Vec<_> = images
                .iter()
                .map(|&(i, ref bytes, rp, dp)| {
                    scope.spawn(move || {
                        let mut out = vec![0u8; bytes.len()];
                        t.detile(tiled, i, &mut out, rp, dp)
                            .unwrap_or_else(|e| panic!("detile {i}: {e}"));
                        (i, out)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a detiling thread panicked"))
                .collect()
        });
        for ((i, bytes, ..), (j, out)) in images.iter().zip(&back) {
            assert_eq!(i, j);
            assert_eq!(bytes, out, "subresource {i} of {t:?}");
        }
    }
}

#[test]
fn subresources_tiled_on_separate_threads_match_one_thread() {
    for t in textures() {
        let images = linear_images(&t);
        let order: Vec<usize> = (0..images.len()).collect();
        let expected = tile_all(&t, &images, &order);
        let t = &t;
        // Each thread tiles one subresource into its own buffer; since
        // subresources own disjoint bytes, OR-ing the buffers rebuilds the
        // whole texture (every tagged byte is non-zero).
        let parts: Vec<Vec<u8>> = thread::scope(|scope| {
            let handles: Vec<_> = images
                .iter()
                .map(|&(i, ref bytes, rp, dp)| {
                    scope.spawn(move || {
                        let mut tiled = vec![0u8; t.size as usize];
                        t.tile(&mut tiled, i, bytes, rp, dp)
                            .unwrap_or_else(|e| panic!("tile {i}: {e}"));
                        tiled
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a tiling thread panicked"))
                .collect()
        });
        let mut merged = vec![0u8; t.size as usize];
        let mut owner = vec![usize::MAX; t.size as usize];
        for (k, part) in parts.iter().enumerate() {
            for (n, &byte) in part.iter().enumerate() {
                if byte != 0 {
                    assert_eq!(
                        owner[n],
                        usize::MAX,
                        "byte {n} written by {} and {k}",
                        owner[n]
                    );
                    owner[n] = k;
                    merged[n] = byte;
                }
            }
        }
        assert_eq!(merged, expected, "{t:?}");
    }
}

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
    (1..=32u32)
        .map(|seed| {
            let len = 12 + (seed as usize * 41) % 360;
            let bits = len * 8 - (seed as usize % 8);
            (noise_frame(len, bits, seed), bits)
        })
        .collect()
}

fn run(d: &mut Decoder, frames: &[(Vec<u8>, usize)]) -> (Vec<i16>, Vec<Result<(), xma::Error>>) {
    let mut out = vec![];
    let results = frames
        .iter()
        .map(|(data, bits)| d.decode(data, *bits, &mut out))
        .collect();
    d.flush(&mut out);
    (out, results)
}

#[test]
fn decoders_on_separate_threads_match_one_thread() {
    let frames = frames();
    let configs = [
        (1, 24_000),
        (1, 48_000),
        (2, 32_000),
        (2, 44_100),
        (2, 48_000),
    ];
    let sequential: Vec<_> = configs
        .iter()
        .map(|&(c, r)| run(&mut Decoder::new(c, r), &frames))
        .collect();
    let frames = &frames;
    let parallel: Vec<_> = thread::scope(|scope| {
        let handles: Vec<_> = configs
            .iter()
            .map(|&(c, r)| scope.spawn(move || run(&mut Decoder::new(c, r), frames)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a decoding thread panicked"))
            .collect()
    });
    assert_eq!(sequential, parallel);
}

#[test]
fn a_decoder_moved_to_another_thread_mid_stream_keeps_its_overlap() {
    let frames = frames();
    let (head, tail) = frames.split_at(frames.len() / 3);
    for channels in [1, 2] {
        let mut whole = Decoder::new(channels, 48_000);
        let expected = run(&mut whole, &frames);

        let mut d = Decoder::new(channels, 48_000);
        let mut out = vec![];
        let mut results: Vec<_> = head
            .iter()
            .map(|(data, bits)| d.decode(data, *bits, &mut out))
            .collect();
        let tail = tail.to_vec();
        let (rest, rest_results) = thread::spawn(move || run(&mut d, &tail))
            .join()
            .expect("the decoding thread panicked");
        out.extend(rest);
        results.extend(rest_results);
        assert_eq!((out, results), expected);
    }
}
