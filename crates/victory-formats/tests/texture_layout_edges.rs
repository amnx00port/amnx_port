//! Edge cases of linear and tiled texture layouts through the public API:
//! row and depth padding, the smallest buffers copies accept, and padding
//! the copies must never touch. Every expected offset is worked out by
//! hand from the layout rules, so nothing here needs data from outside the
//! repository.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const BC1: u32 = 71;

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

/// A linear 3×2 RGBA8 texture pads each row to 256 bytes (64 elements).
/// Tiling writes exactly the 24 bytes of its texels at `y * 256 + x * 4`
/// and leaves the row padding alone.
#[test]
fn linear_rows_pad_to_256_bytes_and_padding_is_untouched() {
    let t = texture(Dimension::Tex2d, 3, 2, 1, 1, R8G8B8A8, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    assert_eq!((l.mips[0].pitch, l.mips[0].rows), (64, 2));
    assert_eq!((t.size, t.alignment), (512, 256));

    let linear: Vec<u8> = (1..=24).collect();
    let mut tiled = vec![0xcc; 512];
    t.tile(&mut tiled, 0, &linear, 12, 0).unwrap();
    let mut expected = vec![0xcc; 512];
    expected[0..12].copy_from_slice(&linear[0..12]);
    expected[256..268].copy_from_slice(&linear[12..24]);
    assert_eq!(tiled, expected);

    let mut back = vec![0; 24];
    t.detile(&tiled, 0, &mut back, 12, 0).unwrap();
    assert_eq!(back, linear);
}

/// A linear 2×2×3 RGBA8 volume keeps each depth slice in its own 512-byte
/// slice. A caller's padded rows (12 → 16 bytes) and padded depth slices
/// (40 bytes) are honoured, the gaps between them are left alone, and the
/// smallest buffer is the last row's end: `2 * 40 + 16 + 8 = 104` bytes.
#[test]
fn linear_volume_honours_row_and_depth_pitch() {
    let t = texture(Dimension::Tex3d, 2, 2, 3, 1, R8G8B8A8, Swizzle::LINEAR);
    assert_eq!(t.size, 1536);
    let mut tiled = vec![0u8; 1536];
    for z in 0..3 {
        for y in 0..2 {
            for x in 0..2 {
                let at = z * 512 + y * 256 + x * 4;
                for b in 0..4 {
                    tiled[at + b] = (z * 16 + y * 8 + x * 4 + b + 1) as u8;
                }
            }
        }
    }
    let (row_pitch, depth_pitch) = (16, 40);
    let mut linear = vec![0xee; 104];
    t.detile(&tiled, 0, &mut linear, row_pitch, depth_pitch)
        .unwrap();
    for (i, &v) in linear.iter().enumerate() {
        let (z, rest) = (i / depth_pitch, i % depth_pitch);
        let (y, x_byte) = (rest / row_pitch, rest % row_pitch);
        if y < 2 && x_byte < 8 {
            assert_eq!(v as usize, z * 16 + y * 8 + x_byte + 1, "byte {i}");
        } else {
            assert_eq!(v, 0xee, "padding byte {i} was written");
        }
    }

    let mut short = vec![0; 103];
    assert_eq!(
        t.detile(&tiled, 0, &mut short, row_pitch, depth_pitch),
        Err(Error::TooSmall)
    );
}

/// A 5×3 RGBA8 texture in a 4 KiB standard block (32×32 elements) fills
/// only its own 60 bytes; the 32-element padding of every row and the
/// unused rows stay as they were. Inside the first 8×8 micro tile, x bits
/// 0-2 land on address bits 2, 3, 7 and y bits 0-2 on 4, 5, 6.
#[test]
fn tiled_padding_is_untouched_and_micro_tile_bits_place_elements() {
    let t = texture(Dimension::Tex2d, 5, 3, 1, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [32, 32, 1]);
    assert_eq!((l.mips[0].pitch, l.mips[0].rows), (32, 32));
    assert_eq!(t.size, 4096);

    let spread = |x: u64, y: u64| {
        (x & 1) << 2
            | (x >> 1 & 1) << 3
            | (x >> 2 & 1) << 7
            | (y & 1) << 4
            | (y >> 1 & 1) << 5
            | (y >> 2 & 1) << 6
    };
    for y in 0..3 {
        for x in 0..5 {
            assert_eq!(
                l.offset(0, x, y, 0),
                Ok(spread(x.into(), y.into())),
                "({x}, {y})"
            );
        }
    }

    let linear: Vec<u8> = (1..=60).collect();
    let mut tiled = vec![0; 4096];
    t.tile(&mut tiled, 0, &linear, 20, 0).unwrap();
    let mut expected = vec![0; 4096];
    for y in 0..3u64 {
        for x in 0..5u64 {
            let at = spread(x, y) as usize;
            let from = (y * 20 + x * 4) as usize;
            expected[at..at + 4].copy_from_slice(&linear[from..from + 4]);
        }
    }
    assert_eq!(tiled, expected);
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), 60);
}

/// BC1 5×5 texels are 2×2 blocks of 8 bytes. A linear chain places mips
/// smallest first, each row padded to 32 blocks (256 bytes): mip 2 (one
/// block) at 0, mip 1 (one block) at 256, mip 0 (two rows) at 512.
#[test]
fn linear_block_compressed_chain_rounds_texels_to_blocks() {
    let t = texture(Dimension::Tex2d, 5, 5, 1, 0, BC1, Swizzle::LINEAR);
    assert_eq!(t.mips(), 3);
    let l = &t.planes[0].layout;
    let got: Vec<_> = l
        .mips
        .iter()
        .map(|m| (m.width, m.height, m.pitch, m.rows, m.offset))
        .collect();
    assert_eq!(
        got,
        [(2, 2, 32, 2, 512), (1, 1, 32, 1, 256), (1, 1, 32, 1, 0)]
    );
    assert_eq!(t.size, 1024);
    let rows: Vec<_> = (0..3).map(|i| t.footprint(i).unwrap().row_pitch).collect();
    assert_eq!(rows, [16, 8, 8]);

    // Mip 0's second block row starts one padded row (256 bytes) later.
    let mut tiled = vec![0; 1024];
    let linear: Vec<u8> = (1..=32).collect();
    t.tile(&mut tiled, 0, &linear, 16, 0).unwrap();
    assert_eq!(&tiled[512..528], &linear[0..16]);
    assert_eq!(&tiled[768..784], &linear[16..32]);
    assert!(tiled[..512].iter().all(|&b| b == 0));
}

/// Copies refuse a row pitch shorter than one row, a linear buffer one
/// byte short of the last row, and a texture buffer one byte short of the
/// texture, in both directions; the exact sizes are accepted.
#[test]
fn copies_accept_exact_buffers_and_refuse_one_byte_less() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_256B_S, Swizzle::SW_64KB_D] {
        let t = texture(Dimension::Tex2d, 7, 3, 1, 1, R8G8B8A8, swizzle);
        let size = t.size as usize;
        let (row, need) = (28, 2 * 32 + 28);
        let mut tiled = vec![0; size];
        let mut linear = vec![0; need];

        assert_eq!(
            t.detile(&tiled, 0, &mut linear, 32, 0),
            Ok(()),
            "{swizzle:?}"
        );
        assert_eq!(t.tile(&mut tiled, 0, &linear, 32, 0), Ok(()), "{swizzle:?}");
        assert_eq!(
            t.detile(&tiled, 0, &mut linear, row - 1, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
        assert_eq!(
            t.tile(&mut tiled, 0, &linear[..need - 1], 32, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
        assert_eq!(
            t.detile(&tiled[..size - 1], 0, &mut linear, 32, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
        assert_eq!(
            t.tile(&mut tiled[..size - 1], 0, &linear, 32, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
    }
}

/// A linear 3×2 RGBA8 array of three slices stores each slice's 512 bytes
/// one after the other. Subresource `mip + slice * mips` copies only into
/// its own slice; the other slices stay as they were.
#[test]
fn linear_array_slices_are_disjoint() {
    let t = texture(Dimension::Tex2d, 3, 2, 3, 1, R8G8B8A8, Swizzle::LINEAR);
    assert_eq!(t.array_size(), 3);
    assert_eq!(t.planes[0].layout.slice_size, 512);
    assert_eq!(t.size, 1536);

    let linear: Vec<u8> = (1..=24).collect();
    let mut tiled = vec![0xcc; 1536];
    t.tile(&mut tiled, 1, &linear, 12, 0).unwrap();
    let mut expected = vec![0xcc; 1536];
    expected[512..524].copy_from_slice(&linear[0..12]);
    expected[768..780].copy_from_slice(&linear[12..24]);
    assert_eq!(tiled, expected);
    assert_eq!(t.planes[0].layout.offset(0, 2, 1, 2), Ok(1024 + 256 + 8));

    // Past the last slice there is no subresource.
    assert_eq!(t.subresource(3), None);
    assert_eq!(t.tile(&mut tiled, 3, &linear, 12, 0), Err(Error::Invalid));
}

/// RGBA8 in 256-byte blocks is 8×8 elements, so a 9×9 texture takes 2×2
/// blocks: element (8, 0) starts the second block, (0, 8) the third (two
/// blocks per row of blocks) and (8, 8) the fourth. A copy writes exactly
/// its 81 texels and none of the padding.
#[test]
fn tiny_blocks_step_to_the_next_block_at_eight_elements() {
    let t = texture(Dimension::Tex2d, 9, 9, 1, 1, R8G8B8A8, Swizzle::SW_256B_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [8, 8, 1]);
    assert_eq!((l.mips[0].pitch, l.mips[0].rows), (16, 16));
    assert_eq!(t.size, 1024);
    assert_eq!(l.offset(0, 8, 0, 0), Ok(256));
    assert_eq!(l.offset(0, 0, 8, 0), Ok(512));
    assert_eq!(l.offset(0, 8, 8, 0), Ok(768));
    // x bit 2 lands on address bit 7, y bit 2 on bit 6.
    assert_eq!(l.offset(0, 4, 4, 0), Ok(128 | 64));
    assert_eq!(l.offset(0, 9, 0, 0), Err(Error::Invalid));

    let linear = vec![0xab; 9 * 36];
    let mut tiled = vec![0; 1024];
    t.tile(&mut tiled, 0, &linear, 36, 0).unwrap();
    assert_eq!(tiled.iter().filter(|&&b| b == 0xab).count(), 81 * 4);
    // Block 1 holds column 8 only: x bits 0-2 are 0, so only y varies
    // (bits 4-6), one element per 16 bytes.
    for y in 0..8 {
        let at = 256 + y * 16;
        assert_eq!(&tiled[at..at + 4], &[0xab; 4], "row {y}");
        assert_eq!(&tiled[at + 4..at + 16], &[0; 12], "row {y}");
    }
}

/// A linear 3×2 D24S8 texture is a 512-byte depth plane followed by a
/// one-byte-per-element stencil plane (rows still padded to 256 bytes).
/// Subresource 1 is the stencil plane's mip 0: copies land at
/// `512 + y * 256 + x` and never touch the depth plane.
#[test]
fn stencil_plane_copies_stay_in_their_plane() {
    const D24S8: u32 = 45;
    let t = texture(Dimension::Tex2d, 3, 2, 1, 1, D24S8, Swizzle::LINEAR);
    assert_eq!(t.planes.len(), 2);
    assert_eq!((t.planes[1].offset, t.planes[1].stencil), (512, true));
    assert_eq!(t.size, 1024);
    assert_eq!(t.subresource_count(), 2);
    assert_eq!(t.footprint(1).unwrap().row_pitch, 3);

    let mut tiled = vec![0; 1024];
    t.tile(&mut tiled, 1, &[1, 2, 3, 4, 5, 6], 3, 0).unwrap();
    let mut expected = vec![0; 1024];
    expected[512..515].copy_from_slice(&[1, 2, 3]);
    expected[768..771].copy_from_slice(&[4, 5, 6]);
    assert_eq!(tiled, expected);

    let mut back = [0; 6];
    t.detile(&tiled, 1, &mut back, 3, 0).unwrap();
    assert_eq!(back, [1, 2, 3, 4, 5, 6]);
    // The whole texture must be there, even to copy only the depth plane.
    assert_eq!(
        t.detile(&tiled[..1023], 0, &mut [0; 12 * 2], 12, 0),
        Err(Error::TooSmall)
    );
}
