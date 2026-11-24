//! Display (`_D`) swizzle modes through the public API: their micro tile
//! patterns, where they agree with and differ from the standard (`_S`)
//! ones, and whole chains in them. Every expected number is worked out by
//! hand from the micro tile bit patterns, so nothing here needs data from
//! outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8: u32 = 61;
const R16: u32 = 56;
const R8G8B8A8: u32 = 28;
const R16G16B16A16: u32 = 10;
const R32G32B32A32: u32 = 2;

fn texture(w: u32, h: u32, array: u32, mips: u32, format: u32, swizzle: Swizzle) -> Texture {
    Texture::new(&TextureDesc {
        dimension: Dimension::Tex2d,
        width: w,
        height: h,
        depth_or_array: array,
        mips,
        format,
        samples: 1,
        swizzle,
    })
    .unwrap()
}

/// Offsets of element (x, y) of mip 0, slice 0.
fn at(t: &Texture, x: u32, y: u32) -> u64 {
    t.planes[0].layout.offset(0, x, y, 0).unwrap()
}

/// 1-byte display micro tiles take `x0 x1 x2 y1 y0 y2 x3 y3` on address
/// bits 0-7: y0 and y1 trade places against the standard
/// `x0 x1 x2 x3 y0 y1 y2 y3`, and x3 moves up past y2.
#[test]
fn one_byte_display_tiles_swap_the_low_row_bits() {
    let d = texture(16, 16, 1, 1, R8, Swizzle::SW_256B_D);
    let s = texture(16, 16, 1, 1, R8, Swizzle::SW_256B_S);
    assert_eq!(d.size, 256);
    // (x, y, display, standard)
    for (x, y, display, standard) in [
        (1, 0, 1, 1),
        (4, 0, 4, 4),
        (8, 0, 64, 8),
        (0, 1, 16, 16),
        (0, 2, 8, 32),
        (0, 4, 32, 64),
        (0, 8, 128, 128),
        (15, 15, 255, 255),
    ] {
        assert_eq!(at(&d, x, y), display, "display ({x}, {y})");
        assert_eq!(at(&s, x, y), standard, "standard ({x}, {y})");
    }
}

/// 2-byte display and standard micro tiles share one pattern
/// (`-- x0 x1 x2 y0 y1 y2 x3`), so every address of a 16×8 tile agrees.
#[test]
fn two_byte_display_and_standard_tiles_coincide() {
    let d = texture(16, 8, 1, 1, R16, Swizzle::SW_256B_D);
    let s = texture(16, 8, 1, 1, R16, Swizzle::SW_256B_S);
    let mut seen = HashSet::new();
    for y in 0..8 {
        for x in 0..16 {
            let a = at(&d, x, y);
            assert_eq!(a, at(&s, x, y), "({x}, {y})");
            assert_eq!(a % 2, 0);
            assert!(seen.insert(a));
        }
    }
    assert_eq!(seen.len(), 128);
    assert_eq!(at(&d, 8, 0), 128);
    assert_eq!(at(&d, 0, 4), 64);
}

/// Wider elements: RGBA8 display takes `x0 x1 y0 y1 x2 y2` on bits 2-7
/// (standard: `x0 x1 y0 y1 y2 x2`), RGBA16 `x0 y0 x1 x2 y1` on bits 3-7
/// (standard: `x0 y0 y1 x1 x2`), RGBA32 `x0 y0 x1 y1` on bits 4-7
/// (standard: `y0 y1 x0 x1`).
#[test]
fn wide_element_display_tiles_follow_their_patterns() {
    // (x, y, display offset, standard offset)
    type Points = &'static [(u32, u32, u64, u64)];
    let cases: [(u32, u32, u32, Points); 3] = [
        (
            R8G8B8A8,
            8,
            8,
            &[(4, 0, 64, 128), (0, 4, 128, 64), (2, 2, 40, 40)],
        ),
        (
            R16G16B16A16,
            8,
            4,
            &[
                (1, 0, 8, 8),
                (2, 0, 32, 64),
                (4, 0, 64, 128),
                (0, 2, 128, 32),
            ],
        ),
        (
            R32G32B32A32,
            4,
            4,
            &[
                (1, 0, 16, 64),
                (0, 1, 32, 16),
                (2, 0, 64, 128),
                (0, 2, 128, 32),
            ],
        ),
    ];
    for (format, w, h, points) in cases {
        let d = texture(w, h, 1, 1, format, Swizzle::SW_256B_D);
        let s = texture(w, h, 1, 1, format, Swizzle::SW_256B_S);
        assert_eq!((d.size, s.size), (256, 256), "format {format}");
        for &(x, y, display, standard) in points {
            assert_eq!(at(&d, x, y), display, "format {format} display ({x}, {y})");
            assert_eq!(
                at(&s, x, y),
                standard,
                "format {format} standard ({x}, {y})"
            );
        }
    }
}

/// Larger display blocks add the same bits above the micro tile as the
/// standard ones (y, x, y, x...): a 4 KiB RGBA8 block is 32×32, bit 8 is
/// y3 and bit 9 x3, so only the bits below 8 tell the modes apart.
#[test]
fn four_kib_display_blocks_stack_micro_tiles_like_standard_ones() {
    let d = texture(32, 32, 1, 1, R8G8B8A8, Swizzle::SW_4KB_D);
    let s = texture(32, 32, 1, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    assert_eq!(d.planes[0].layout.block, [32, 32, 1]);
    assert_eq!(d.size, 4096);
    for (x, y, both) in [(0, 8, 256), (8, 0, 512), (0, 16, 1024), (16, 0, 2048)] {
        assert_eq!((at(&d, x, y), at(&s, x, y)), (both, both), "({x}, {y})");
    }
    // Inside each micro tile the bits below 8 still differ.
    assert_eq!(at(&d, 12, 0), 512 + 64);
    assert_eq!(at(&s, 12, 0), 512 + 128);
}

/// 256-byte display blocks have no tail: a 16×16 1-byte chain of three
/// mips takes one block each, smallest first, and every mip round-trips
/// through tile and detile without touching the others.
#[test]
fn two_fifty_six_byte_display_chain_round_trips() {
    let t = texture(16, 16, 1, 0, R8, Swizzle::SW_256B_D);
    let l = &t.planes[0].layout;
    assert_eq!((t.mips(), l.first_tail_mip), (5, 5));
    let offsets: Vec<_> = l.mips.iter().map(|m| m.offset).collect();
    assert_eq!(offsets, [1024, 768, 512, 256, 0]);
    assert_eq!(t.size, 1280);

    let mut tiled = vec![0u8; t.size as usize];
    for mip in 0..t.mips() {
        let f = t.footprint(mip).unwrap();
        let src: Vec<u8> = (0..f.size).map(|i| (i as u8) ^ (mip as u8 * 37)).collect();
        t.tile(&mut tiled, mip, &src, f.row_pitch, f.depth_pitch)
            .unwrap();
    }
    for mip in 0..t.mips() {
        let f = t.footprint(mip).unwrap();
        let mut out = vec![0u8; f.size];
        t.detile(&tiled, mip, &mut out, f.row_pitch, f.depth_pitch)
            .unwrap();
        let src: Vec<u8> = (0..f.size).map(|i| (i as u8) ^ (mip as u8 * 37)).collect();
        assert_eq!(out, src, "mip {mip}");
    }
}

/// A 64 KiB display chain with a tail: tail slots match the standard
/// mode's (they depend only on the block), every element of every mip and
/// array slice lands on its own byte inside the texture, and the buffer
/// checks refuse a tiled buffer one byte short or an index past the end.
#[test]
fn sixty_four_kib_display_chain_with_a_tail_is_a_bijection() {
    let d = texture(64, 64, 2, 0, R8G8B8A8, Swizzle::SW_64KB_D);
    let s = texture(64, 64, 2, 0, R8G8B8A8, Swizzle::SW_64KB_S);
    let (dl, sl) = (&d.planes[0].layout, &s.planes[0].layout);
    assert_eq!(dl.first_tail_mip, sl.first_tail_mip);
    assert_eq!((dl.slice_size, d.size), (sl.slice_size, s.size));
    let tails: Vec<_> = dl.mips.iter().map(|m| m.tail).collect();
    assert_eq!(tails, sl.mips.iter().map(|m| m.tail).collect::<Vec<_>>());

    let mut seen = HashSet::new();
    for slice in 0..2 {
        for (mip, m) in dl.mips.iter().enumerate() {
            for y in 0..m.height {
                for x in 0..m.width {
                    let a = dl.offset(mip as u32, x, y, slice).unwrap();
                    assert!(a + 4 <= d.size, "mip {mip} ({x}, {y}) slice {slice}");
                    assert!(seen.insert(a), "mip {mip} ({x}, {y}) slice {slice}");
                }
            }
        }
    }
    let elements: u32 = dl.mips.iter().map(|m| m.width * m.height).sum();
    assert_eq!(seen.len() as u32, 2 * elements);

    let f = d.footprint(0).unwrap();
    let linear = vec![0u8; f.size];
    let mut short = vec![0u8; d.size as usize - 1];
    assert_eq!(
        d.tile(&mut short, 0, &linear, f.row_pitch, f.depth_pitch),
        Err(Error::TooSmall)
    );
    let mut whole = vec![0u8; d.size as usize];
    let past = d.subresource_count();
    assert_eq!(
        d.tile(&mut whole, past, &linear, f.row_pitch, f.depth_pitch),
        Err(Error::Invalid)
    );
}
