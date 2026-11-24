//! Thick (3D standard) mip chains with a tail, through the public API:
//! where the tail starts, where each mip and depth slab lands, that every
//! element of every mip gets its own byte, and that each mip copies in and
//! out without touching the others. Every expected number is worked out by
//! hand from addrlib's thick block rules, so nothing here needs data from
//! outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Layout, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;

fn volume(w: u32, h: u32, d: u32, mips: u32, swizzle: Swizzle) -> Texture {
    Texture::new(&TextureDesc {
        dimension: Dimension::Tex3d,
        width: w,
        height: h,
        depth_or_array: d,
        mips,
        format: R8G8B8A8,
        samples: 1,
        swizzle,
    })
    .unwrap()
}

/// Every element of every mip and depth slice, as (mip, x, y, z, offset).
fn every_element(l: &Layout) -> Vec<(u32, u32, u32, u32, u64)> {
    let mut all = vec![];
    for (mip, m) in l.mips.iter().enumerate() {
        let mip = mip as u32;
        for z in 0..m.depth {
            for y in 0..m.height {
                for x in 0..m.width {
                    all.push((mip, x, y, z, l.offset(mip, x, y, z).unwrap()));
                }
            }
        }
    }
    all
}

/// Elements land on distinct, element-aligned bytes inside the texture.
fn assert_bijection(t: &Texture) {
    let l = &t.planes[0].layout;
    let mut seen = HashSet::new();
    for (mip, x, y, z, at) in every_element(l) {
        assert_eq!(at % 4, 0, "mip {mip} ({x}, {y}, {z})");
        assert!(at + 4 <= t.size, "mip {mip} ({x}, {y}, {z}) at {at}");
        assert!(seen.insert(at), "mip {mip} ({x}, {y}, {z}) at {at}");
    }
}

/// Tiles a distinct pattern into every mip, then reads each one back.
fn assert_mips_round_trip(t: &Texture) {
    let pattern = |mip: u32, size: usize| -> Vec<u8> {
        (0..size)
            .map(|i| ((i as u8).wrapping_mul(7) ^ (mip as u8).wrapping_mul(53)) | 1)
            .collect()
    };
    let mut tiled = vec![0u8; t.size as usize];
    for mip in 0..t.mips() {
        let f = t.footprint(mip).unwrap();
        t.tile(
            &mut tiled,
            mip,
            &pattern(mip, f.size),
            f.row_pitch,
            f.depth_pitch,
        )
        .unwrap();
    }
    let written: usize = (0..t.mips()).map(|m| t.footprint(m).unwrap().size).sum();
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), written);
    for mip in 0..t.mips() {
        let f = t.footprint(mip).unwrap();
        let mut out = vec![0u8; f.size];
        t.detile(&tiled, mip, &mut out, f.row_pitch, f.depth_pitch)
            .unwrap();
        assert_eq!(out, pattern(mip, f.size), "mip {mip}");
    }
}

/// A 16×16×16 RGBA8 volume in 4 KiB standard blocks: blocks are 8×16×8
/// and the tail takes mips up to 8×8 (the thick tail halves y for a
/// 4 KiB block), at most five of them. Mip 0 (16×16, two blocks across)
/// is the only one outside it; mips 1-4 sit in the tail at 1024, 768,
/// 512 and 256 (x from odd address bits, y from even ones, in 4×4×4
/// micro tiles). Each 8-deep slab holds a 512-byte share of the tail and
/// 1024 of mip 0 per depth slice: 12 KiB a slab, two slabs.
#[test]
fn four_kib_thick_chain_keeps_four_mips_in_the_tail() {
    let t = volume(16, 16, 16, 0, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert!(l.thick);
    assert_eq!(l.block, [8, 16, 8]);
    assert_eq!((t.mips(), l.first_tail_mip), (5, 1));
    assert_eq!((l.slices, l.slice_size, t.size), (2, 12288, 24576));
    assert_eq!(l.mips[0].offset, 4096);
    assert_eq!(
        (l.mips[0].pitch, l.mips[0].rows, l.mips[0].depth_pitch),
        (16, 16, 16)
    );
    let tails: Vec<_> = l.mips[1..]
        .iter()
        .map(|m| {
            let t = m.tail.unwrap();
            (t.offset, t.x, t.y)
        })
        .collect();
    assert_eq!(tails, [(1024, 0, 8), (768, 4, 4), (512, 4, 0), (256, 0, 4)]);
    let depths: Vec<_> = l.mips.iter().map(|m| m.depth).collect();
    assert_eq!(depths, [16, 8, 4, 2, 1]);

    // The tail block opens each slab's chain; mip 0 follows it, each
    // slab's eight depth slices in their own 12 KiB.
    for (mip, x, y, z, at) in every_element(l) {
        let range = match (mip, z / 8) {
            (0, 0) => 4096..12288,
            (0, _) => 16384..24576,
            _ => 0..4096,
        };
        assert!(range.contains(&at), "mip {mip} ({x}, {y}, {z}) at {at}");
    }
    assert_eq!(l.offset(0, 0, 0, 8), Ok(12288 + 4096));
    assert_eq!(l.offset(1, 0, 0, 8), Err(Error::Invalid));

    assert_bijection(&t);
    assert_mips_round_trip(&t);
}

/// A 64×64×32 RGBA8 volume in 64 KiB standard blocks: blocks are
/// 32×32×16 and the tail takes mips up to 16×32 (the thick tail halves x
/// for a 64 KiB block), at most ten. Mips 0 (64×64) and 1 (32×32) stay
/// outside it; mip 2 onwards sit at 8192, 4096, 2048, 1536 and 1280.
/// Each 16-deep slab holds the tail block, then mip 1's 64 KiB, then
/// mip 0's 256 KiB.
#[test]
fn sixty_four_kib_thick_chain_puts_two_mips_after_the_tail() {
    let t = volume(64, 64, 32, 0, Swizzle::SW_64KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.block, [32, 32, 16]);
    assert_eq!((t.mips(), l.first_tail_mip), (7, 2));
    assert_eq!((l.slices, l.slice_size), (2, 393_216));
    assert_eq!(t.size, 786_432);
    assert_eq!((l.mips[0].offset, l.mips[1].offset), (131_072, 65_536));
    let tails: Vec<_> = l.mips[2..].iter().map(|m| m.tail.unwrap().offset).collect();
    assert_eq!(tails, [8192, 4096, 2048, 1536, 1280]);

    for (mip, x, y, z, at) in every_element(l) {
        let range = match (mip, z / 16) {
            (0, 0) => 131_072..393_216,
            (0, _) => 393_216 + 131_072..786_432,
            (1, _) => 65_536..131_072,
            _ => 0..65_536,
        };
        assert!(range.contains(&at), "mip {mip} ({x}, {y}, {z}) at {at}");
    }

    assert_bijection(&t);
    assert_mips_round_trip(&t);
}

/// A depth that is not a multiple of the slab still rounds up to whole
/// slabs, and a chain whose depth outlives its width and height (4×4×64)
/// keeps every mip in the tail of the first slab.
#[test]
fn deep_thin_thick_chain_lives_in_the_first_slab_tail() {
    let t = volume(4, 4, 64, 0, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    // Seven mips, more than the five a 4 KiB thick tail holds: mips 0 and
    // 1 are placed as blocks, the last five in the tail.
    assert_eq!((t.mips(), l.first_tail_mip), (7, 2));
    assert_eq!(l.slices, 8);
    assert_eq!(t.size, u64::from(l.slices) * l.slice_size);
    assert_bijection(&t);
    assert_mips_round_trip(&t);

    let t = volume(8, 8, 9, 0, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(l.slices, 2);
    assert_eq!(
        l.mips.iter().map(|m| m.depth).collect::<Vec<_>>(),
        [9, 4, 2, 1]
    );
    assert_bijection(&t);
    assert_mips_round_trip(&t);
}

/// The XOR variants of the thick modes keep the standard layout's blocks,
/// tail and size, but their addresses need the GPU's pipe configuration:
/// offsets and copies are Unsupported, while out-of-range requests are
/// still Invalid. The Z-order XOR mode on a volume is thin instead: one
/// depth slice per slice.
#[test]
fn xor_volumes_size_like_their_base_mode_but_do_not_address() {
    let s = volume(64, 64, 32, 0, Swizzle::SW_64KB_S);
    let x = volume(64, 64, 32, 0, Swizzle::SW_64KB_S_X);
    let (sl, xl) = (&s.planes[0].layout, &x.planes[0].layout);
    assert!(xl.thick && !xl.addressable());
    assert_eq!((xl.block, xl.first_tail_mip), (sl.block, sl.first_tail_mip));
    assert_eq!((xl.slice_size, x.size), (sl.slice_size, s.size));
    assert_eq!(xl.offset(0, 0, 0, 0), Err(Error::Unsupported));
    assert_eq!(xl.offset(0, 64, 0, 0), Err(Error::Invalid));
    let f = x.footprint(0).unwrap();
    let mut tiled = vec![0u8; x.size as usize];
    assert_eq!(
        x.tile(&mut tiled, 0, &vec![0; f.size], f.row_pitch, f.depth_pitch),
        Err(Error::Unsupported)
    );

    let z = volume(64, 64, 32, 1, Swizzle::SW_64KB_Z_X);
    let zl = &z.planes[0].layout;
    assert!(!zl.thick && !zl.addressable());
    assert_eq!((zl.block[2], zl.slices), (1, 32));
}
