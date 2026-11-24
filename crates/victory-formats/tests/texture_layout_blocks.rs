//! Block sizes at both ends of the range: 64 KiB tails and thick slabs,
//! 256-byte blocks that have no tail, and block-compressed elements in a
//! tiled block. Every expected number is worked out by hand from the
//! layout rules, so nothing here needs data from outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const BC1: u32 = 71;
const D24S8: u32 = 45;

fn texture(
    dimension: Dimension,
    w: u32,
    h: u32,
    d: u32,
    mips: u32,
    format: u32,
    swizzle: Swizzle,
) -> Texture {
    Texture::new(&TextureDesc {
        dimension,
        width: w,
        height: h,
        depth_or_array: d,
        mips,
        format,
        samples: 1,
        swizzle,
    })
    .unwrap()
}

/// A 64 KiB RGBA8 block is 128×128 elements and its tail holds up to 12
/// mips, the largest 64×128. A 64×64 full chain (7 mips) fits in the
/// tail: mip `i` takes slot `11 - i`, at byte `16 << m` above slot 6.
/// Address bits 8, 10, ... are y bits 3, 4, ...; bits 9, 11, ... x bits
/// 3, 4, ..., so each slot names where its mip starts.
#[test]
fn sixty_four_kib_tail_places_the_whole_chain_in_one_block() {
    let t = texture(Dimension::Tex2d, 64, 64, 1, 0, R8G8B8A8, Swizzle::SW_64KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [128, 128, 1]);
    assert_eq!(t.mips(), 7);
    assert_eq!(l.first_tail_mip, 0);
    assert_eq!((l.slice_size, t.size, t.alignment), (65536, 65536, 65536));

    let expected = [
        (32768, 64, 0),
        (16384, 0, 64),
        (8192, 32, 0),
        (4096, 0, 32),
        (2048, 16, 0),
        (1536, 8, 16),
        (1280, 0, 24),
    ];
    for (i, &(offset, x, y)) in expected.iter().enumerate() {
        let tail = l.mips[i].tail.expect("in the tail");
        assert_eq!((tail.offset, tail.x, tail.y), (offset, x, y), "mip {i}");
        assert_eq!(
            l.offset(i as u32, 0, 0, 0),
            Ok(u64::from(offset)),
            "mip {i}"
        );
    }

    // The whole chain stays inside the block without sharing a byte.
    let mut seen = HashSet::new();
    for (i, m) in l.mips.iter().enumerate() {
        for y in 0..m.height {
            for x in 0..m.width {
                let at = l.offset(i as u32, x, y, 0).unwrap();
                assert!(at + 4 <= 65536 && seen.insert(at), "mip {i} ({x}, {y})");
            }
        }
    }
    assert_eq!(seen.len(), 4096 + 1024 + 256 + 64 + 16 + 4 + 1);
}

/// One more mip than the 64 KiB tail holds: a 128×128 chain's mip 0 is a
/// whole standard block after the tail block, and mip 1 (64×64) starts
/// the tail at slot 11, as mip 0 of a 64×64 chain does.
#[test]
fn sixty_four_kib_chain_too_wide_for_the_tail_puts_mip_zero_after_it() {
    let t = texture(
        Dimension::Tex2d,
        128,
        128,
        1,
        0,
        R8G8B8A8,
        Swizzle::SW_64KB_D,
    );
    let l = &t.planes[0].layout;
    assert_eq!(t.mips(), 8);
    assert_eq!(l.first_tail_mip, 1);
    assert_eq!(
        (l.mips[0].offset, l.mips[0].pitch, l.mips[0].rows),
        (65536, 128, 128)
    );
    assert_eq!(t.size, 2 * 65536);
    assert_eq!(l.mips[1].tail.map(|t| t.offset), Some(32768));
    assert_eq!(l.offset(0, 0, 0, 0), Ok(65536));
    assert_eq!(l.offset(0, 127, 127, 0), Ok(2 * 65536 - 4));
}

/// 256-byte blocks have no tail: every mip of a 16×16 RGBA8 chain is
/// whole 8×8 blocks, placed smallest first, so the 1×1, 2×2 and 4×4 mips
/// each take a full 256-byte block.
#[test]
fn two_fifty_six_byte_blocks_have_no_tail() {
    let t = texture(Dimension::Tex2d, 16, 16, 2, 0, R8G8B8A8, Swizzle::SW_256B_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [8, 8, 1]);
    assert_eq!(t.mips(), 5);
    assert_eq!(l.first_tail_mip, 5);
    assert!(l.mips.iter().all(|m| m.tail.is_none()));
    let placed: Vec<_> = l.mips.iter().map(|m| (m.pitch, m.rows, m.offset)).collect();
    assert_eq!(
        placed,
        [
            (16, 16, 1024),
            (8, 8, 768),
            (8, 8, 512),
            (8, 8, 256),
            (8, 8, 0)
        ]
    );
    assert_eq!((l.slice_size, t.size, t.alignment), (2048, 4096, 256));
    // Second array slice, mip 0, next block across.
    assert_eq!(l.offset(0, 8, 0, 1), Ok(2048 + 1024 + 256));
    assert_eq!(l.offset(4, 0, 0, 1), Ok(2048));
}

/// A 64 KiB thick RGBA8 block is 32×32×16 elements. x bits 0-4 land on
/// address bits 2, 6, 9, 12, 15; y bits on 3, 5, 8, 11, 14; z bits on 4,
/// 7, 10, 13.
#[test]
fn sixty_four_kib_thick_block_is_a_32_by_32_by_16_slab() {
    let t = texture(
        Dimension::Tex3d,
        32,
        32,
        16,
        1,
        R8G8B8A8,
        Swizzle::SW_64KB_S,
    );
    let l = &t.planes[0].layout;
    assert!(l.thick);
    assert_eq!(l.block, [32, 32, 16]);
    assert_eq!((l.slices, t.size), (1, 65536));
    assert_eq!(l.offset(0, 16, 0, 0), Ok(32768));
    assert_eq!(l.offset(0, 0, 16, 0), Ok(16384));
    assert_eq!(l.offset(0, 0, 0, 8), Ok(8192));
    assert_eq!(l.offset(0, 0, 0, 1), Ok(16));
    assert_eq!(l.offset(0, 31, 31, 15), Ok(65532));
    assert_eq!(l.offset(0, 0, 0, 16), Err(Error::Invalid));

    // A 1×1×1 volume still takes the whole slab.
    let tiny = texture(Dimension::Tex3d, 1, 1, 1, 1, R8G8B8A8, Swizzle::SW_64KB_S);
    assert_eq!(tiny.size, 65536);
}

/// BC1 elements are 4×4 texels of 8 bytes, so a 4 KiB block holds 32×16
/// of them. Inside it x bits 0-4 land on address bits 3, 6, 7, 9, 11 and
/// y bits 0-3 on 4, 5, 8, 10. Coordinates and footprints count elements.
#[test]
fn block_compressed_elements_tile_like_eight_byte_texels() {
    let t = texture(Dimension::Tex2d, 128, 64, 1, 1, BC1, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [32, 16, 1]);
    let m = &l.mips[0];
    assert_eq!((m.width, m.height, m.pitch, m.rows), (32, 16, 32, 16));
    assert_eq!(t.size, 4096);
    assert_eq!(l.offset(0, 1, 0, 0), Ok(8));
    assert_eq!(l.offset(0, 0, 1, 0), Ok(16));
    assert_eq!(l.offset(0, 7, 7, 0), Ok(8 + 64 + 128 + 16 + 32 + 256));
    assert_eq!(l.offset(0, 31, 15, 0), Ok(4088));
    assert_eq!(l.offset(0, 32, 0, 0), Err(Error::Invalid));

    let f = t.footprint(0).unwrap();
    assert_eq!((f.row_pitch, f.size), (32 * 8, 4096));
    let linear: Vec<u8> = (0..f.size).map(|i| (i * 7 + 1) as u8).collect();
    let mut tiled = vec![0; 4096];
    t.tile(&mut tiled, 0, &linear, f.row_pitch, 0).unwrap();
    // Element (1, 0) is linear bytes 8..16 and tiled bytes 8..16; element
    // (0, 1) is linear bytes 256.. and tiled bytes 16..
    assert_eq!(&tiled[8..16], &linear[8..16]);
    assert_eq!(&tiled[16..24], &linear[256..264]);
    let mut back = vec![0; f.size];
    t.detile(&tiled, 0, &mut back, f.row_pitch, 0).unwrap();
    assert_eq!(back, linear);
}

/// D24S8 with mips and array slices: a subresource index is
/// `mip + slice * mips + plane * mips * slices`, and the stencil plane's
/// subresources follow every depth one.
#[test]
fn depth_stencil_array_subresources_count_planes_last() {
    let t = texture(Dimension::Tex2d, 8, 8, 3, 2, D24S8, Swizzle::SW_4KB_S);
    assert_eq!(t.planes.len(), 2);
    assert_eq!(t.subresource_count(), 12);
    let got: Vec<_> = [0, 1, 2, 5, 6, 7, 11]
        .iter()
        .map(|&i| t.subresource(i).map(|s| (s.plane, s.mip, s.slice)))
        .collect();
    assert_eq!(
        got,
        [
            Some((0, 0, 0)),
            Some((0, 1, 0)),
            Some((0, 0, 1)),
            Some((0, 1, 2)),
            Some((1, 0, 0)),
            Some((1, 1, 0)),
            Some((1, 1, 2)),
        ]
    );
    assert_eq!(t.subresource(12), None);
    // Depth rows are 4 bytes an element, stencil rows 1.
    assert_eq!(t.footprint(0).map(|f| f.row_pitch), Some(32));
    assert_eq!(t.footprint(6).map(|f| f.row_pitch), Some(8));
    assert_eq!(t.footprint(7).map(|f| f.row_pitch), Some(4));
    assert_eq!(t.planes[1].offset % t.planes[1].layout.alignment, 0);
}
