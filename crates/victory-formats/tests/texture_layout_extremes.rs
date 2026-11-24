//! Addressing at the largest descriptions the layouts accept (16384 texels
//! per side, 16384 array slices or depth slices). Sizes, chain lengths and
//! the offsets of the last elements are worked out by hand from the layout
//! rules; only small buffers are allocated, so nothing here needs data from
//! outside the repository or much memory.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8: u32 = 61;
const R8G8B8A8: u32 = 28;
const R32G32B32A32: u32 = 2;

fn texture(
    dimension: Dimension,
    (w, h, d): (u32, u32, u32),
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

/// A 16384² RGBA8 texture in 64 KiB `S` blocks of 128×128 elements is
/// 128×128 blocks, 1 GiB. The last element sits at the same place inside
/// the last block as element (127, 127) does inside the first, so its
/// offset is that one plus `size - 64 KiB`; the first element of each
/// block row and column steps by whole blocks.
#[test]
fn largest_tiled_texture_places_its_last_element_in_its_last_block() {
    let t = texture(
        Dimension::Tex2d,
        (16384, 16384, 1),
        1,
        R8G8B8A8,
        Swizzle::SW_64KB_S,
    );
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [128, 128, 1]);
    assert_eq!(t.size, 1 << 30);
    let first = l.offset(0, 127, 127, 0).unwrap();
    assert!(first < 65536);
    assert_eq!(
        l.offset(0, 16383, 16383, 0).unwrap(),
        first + (1 << 30) - 65536
    );
    assert_eq!(l.offset(0, 128, 0, 0).unwrap(), 65536);
    assert_eq!(l.offset(0, 0, 128, 0).unwrap(), 128 * 65536);
    assert_eq!(l.offset(0, 16384, 0, 0), Err(Error::Invalid));
    assert_eq!(l.offset(0, 0, 16384, 0), Err(Error::Invalid));
}

/// The full chain of a 16384-wide texture has 15 mips, down to 1×1. In a
/// linear layout of RGBA8 the last element of mip 0 is at
/// `16383 * 16384 * 4 + 16383 * 4`, and the chain packs smallest first,
/// so mip 0 starts after the 14 smaller ones.
#[test]
fn largest_linear_chain_has_fifteen_mips_and_mip_zero_last() {
    let t = texture(
        Dimension::Tex2d,
        (16384, 16384, 1),
        0,
        R8G8B8A8,
        Swizzle::LINEAR,
    );
    assert_eq!(t.mips(), 15);
    let l = &t.planes[0].layout;
    // Rows pad to 64 elements: mips 0..=8 have pitch 16384 >> i, the
    // rest 64, and each has (16384 >> i) rows.
    let bytes = |i: u32| {
        let pitch = (16384u64 >> i).max(64);
        pitch * (16384u64 >> i) * 4
    };
    let smaller: u64 = (1..15).map(bytes).sum();
    assert_eq!(l.mips[0].offset, smaller);
    assert_eq!(t.size, smaller + bytes(0));
    let last = l.offset(0, 16383, 16383, 0).unwrap();
    assert_eq!(last, smaller + 16383 * 16384 * 4 + 16383 * 4);
    assert_eq!(last + 4, t.size);
    assert_eq!(l.offset(14, 0, 0, 0).unwrap(), 0);
}

/// A 16384-texel 1D chain of 16-byte texels pads rows to 16 elements:
/// widths 16384..16 keep their size, the last four mips (8, 4, 2, 1)
/// round up to 16. `16 * (16384 + ... + 16) + 4 * 16 * 16 = 525056`.
#[test]
fn largest_one_dimensional_chain_pads_its_small_mips_to_sixteen() {
    let t = texture(
        Dimension::Tex1d,
        (16384, 1, 1),
        0,
        R32G32B32A32,
        Swizzle::LINEAR,
    );
    assert_eq!(t.mips(), 15);
    assert_eq!(t.size, 525056);
    let l = &t.planes[0].layout;
    let pitches: Vec<u32> = l.mips.iter().map(|m| m.pitch).collect();
    let expected: Vec<u32> = (0..15).map(|i| (16384u32 >> i).max(16)).collect();
    assert_eq!(pitches, expected);
    assert_eq!(l.offset(0, 16383, 0, 0).unwrap() + 16, t.size);
}

/// 16384 array slices of a 1×1 R8 texture in 256-byte blocks each take one
/// block: 4 MiB, the last slice at `16383 * 256`. Copies reach the last
/// subresource and stay inside its block; the index after it is refused.
#[test]
fn largest_array_copies_its_last_slice_into_its_last_block() {
    let t = texture(Dimension::Tex2d, (1, 1, 16384), 1, R8, Swizzle::SW_256B_S);
    assert_eq!(t.array_size(), 16384);
    assert_eq!(t.subresource_count(), 16384);
    assert_eq!(t.size, 16384 * 256);
    let l = &t.planes[0].layout;
    assert_eq!(l.offset(0, 0, 0, 16383).unwrap(), 16383 * 256);
    assert_eq!(l.offset(0, 0, 0, 16384), Err(Error::Invalid));

    let mut tiled = vec![0u8; t.size as usize];
    t.tile(&mut tiled, 16383, &[0x5a], 1, 0).unwrap();
    let mut expected = vec![0u8; t.size as usize];
    expected[16383 * 256] = 0x5a;
    assert_eq!(tiled, expected);
    let mut back = [0u8];
    t.detile(&tiled, 16383, &mut back, 1, 0).unwrap();
    assert_eq!(back, [0x5a]);
    assert_eq!(t.tile(&mut tiled, 16384, &[0], 1, 0), Err(Error::Invalid));
}

/// A 1×1×16384 R8 volume in thick 64 KiB `S` blocks (64×32×32 elements)
/// is 512 slabs of 32 depth slices, 32 MiB. Depth slice 16383 is slice 31
/// of the last slab, so its offset is slice 31's plus `511 * 64 KiB`, and
/// every slab boundary steps by one block.
#[test]
fn deepest_thick_volume_puts_its_last_slice_in_its_last_slab() {
    let t = texture(Dimension::Tex3d, (1, 1, 16384), 1, R8, Swizzle::SW_64KB_S);
    let l = &t.planes[0].layout;
    assert!(l.thick);
    assert_eq!(l.block, [64, 32, 32]);
    assert_eq!((l.slices, t.size), (512, 512 * 65536));
    let in_slab = l.offset(0, 0, 0, 31).unwrap();
    assert!(in_slab < 65536);
    assert_eq!(l.offset(0, 0, 0, 16383).unwrap(), in_slab + 511 * 65536);
    for slab in [1u64, 2, 255, 511] {
        assert_eq!(l.offset(0, 0, 0, (slab * 32) as u32).unwrap(), slab * 65536);
    }
    assert_eq!(l.offset(0, 0, 0, 16384), Err(Error::Invalid));
}

/// A linear 1×1×16384 volume keeps each depth slice in its own 256-byte
/// row: 4 MiB, depth slice `z` at `z * 256`. A tight copy of the whole
/// subresource is 16384 bytes, one per slice, and lands one byte per row.
#[test]
fn deepest_linear_volume_copies_one_byte_per_depth_slice() {
    let t = texture(Dimension::Tex3d, (1, 1, 16384), 1, R8, Swizzle::LINEAR);
    assert_eq!(t.size, 16384 * 256);
    let f = t.footprint(0).unwrap();
    assert_eq!((f.row_pitch, f.depth_pitch, f.size), (1, 1, 16384));
    let linear: Vec<u8> = (0..16384u32).map(|z| (z * 7 + 1) as u8).collect();
    let mut tiled = vec![0u8; t.size as usize];
    t.tile(&mut tiled, 0, &linear, 1, 1).unwrap();
    for (i, &b) in tiled.iter().enumerate() {
        let want = if i % 256 == 0 { linear[i / 256] } else { 0 };
        assert_eq!(b, want, "byte {i}");
    }
    let mut back = vec![0u8; 16384];
    t.detile(&tiled, 0, &mut back, 1, 1).unwrap();
    assert_eq!(back, linear);
    assert_eq!(
        t.detile(&tiled, 0, &mut back[..16383], 1, 1),
        Err(Error::TooSmall)
    );
}
