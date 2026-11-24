//! Mip tails, thick 3D blocks, linear volume chains and the description
//! bounds, through the public API. Every expected number is worked out by
//! hand from the layout rules (block sizes, micro tile bit patterns, tail
//! offsets), so nothing here needs data from outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const R8: u32 = 61;
const R32G32B32A32: u32 = 2;

fn desc(
    dimension: Dimension,
    w: u32,
    h: u32,
    d: u32,
    mips: u32,
    format: u32,
    swizzle: Swizzle,
) -> TextureDesc {
    TextureDesc {
        dimension,
        width: w,
        height: h,
        depth_or_array: d,
        mips,
        format,
        samples: 1,
        swizzle,
    }
}

fn texture(
    dimension: Dimension,
    w: u32,
    h: u32,
    d: u32,
    mips: u32,
    format: u32,
    swizzle: Swizzle,
) -> Texture {
    Texture::new(&desc(dimension, w, h, d, mips, format, swizzle)).unwrap()
}

/// A 16×16 RGBA8 chain in 4 KiB standard blocks (32×32 elements) fits
/// entirely in the mip tail: one 4 KiB block, every mip at offset 0. The
/// tail holds at most 8 mips; mip `i` takes tail slot `m = 7 - i`, at byte
/// `16 << m` for `m > 6`, else `m << 8`. Those offsets name the element
/// position each mip starts at, and element (0, 0) of each mip addresses
/// exactly its slot.
#[test]
fn whole_chain_in_the_tail_starts_each_mip_at_its_slot() {
    let t = texture(Dimension::Tex2d, 16, 16, 1, 0, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(t.mips(), 5);
    assert_eq!(l.first_tail_mip, 0);
    assert_eq!((l.slice_size, t.size), (4096, 4096));

    // (slot offset, start x, start y). Address bits 8, 10, ... are y
    // bits 3, 4, ...; bits 9, 11, ... are x bits 3, 4, ...
    let expected = [
        (2048, 16, 0),
        (1536, 8, 16),
        (1280, 0, 24),
        (1024, 0, 16),
        (768, 8, 8),
    ];
    for (i, &(offset, x, y)) in expected.iter().enumerate() {
        let m = &l.mips[i];
        assert_eq!(m.offset, 0, "mip {i}");
        let tail = m.tail.expect("in the tail");
        assert_eq!((tail.offset, tail.x, tail.y), (offset, x, y), "mip {i}");
        assert_eq!(
            l.offset(i as u32, 0, 0, 0),
            Ok(u64::from(offset)),
            "mip {i}"
        );
    }

    // The last element of mip 0 is at (31, 15) in the block: x bits 0-4
    // land on 2, 3, 7, 9, 11 and y bits 0-3 on 4, 5, 6, 8.
    assert_eq!(l.offset(0, 15, 15, 0), Ok(2700 + 368));
}

/// No two elements of a tail chain share a byte, every one lies inside
/// the tail block, and a copy of every mip round-trips without touching
/// another mip.
#[test]
fn tail_mips_never_overlap_and_round_trip() {
    let t = texture(Dimension::Tex2d, 16, 16, 1, 0, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    let mut seen = HashSet::new();
    for (i, m) in l.mips.iter().enumerate() {
        for y in 0..m.height {
            for x in 0..m.width {
                let at = l.offset(i as u32, x, y, 0).unwrap();
                assert!(at + 4 <= 4096, "mip {i} ({x}, {y}) at {at}");
                assert!(seen.insert(at), "mip {i} ({x}, {y}) reuses {at}");
            }
        }
    }
    // 256 + 64 + 16 + 4 + 1 elements.
    assert_eq!(seen.len(), 341);

    let mut tiled = vec![0; 4096];
    for i in 0..t.mips() {
        let f = t.footprint(i).unwrap();
        let linear = vec![i as u8 + 1; f.size];
        t.tile(&mut tiled, i, &linear, f.row_pitch, 0).unwrap();
    }
    for i in 0..t.mips() {
        let f = t.footprint(i).unwrap();
        let mut back = vec![0; f.size];
        t.detile(&tiled, i, &mut back, f.row_pitch, 0).unwrap();
        assert!(back.iter().all(|&b| b == i as u8 + 1), "mip {i}");
    }
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), 341 * 4);
}

/// A 64×64 RGBA8 chain in 4 KiB blocks: mips 0 (64×64, 16 KiB) and 1
/// (32×32, one block) are too wide for the tail, whose largest mip is
/// 16×32. The tail block comes first, then the standard mips from the
/// smallest up: mip 1 at 4096, mip 0 at 8192.
#[test]
fn mips_wider_than_the_tail_follow_the_tail_block() {
    let t = texture(Dimension::Tex2d, 64, 64, 1, 0, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!(t.mips(), 7);
    assert_eq!(l.first_tail_mip, 2);
    let placed: Vec<_> = l.mips[..2]
        .iter()
        .map(|m| (m.pitch, m.rows, m.offset))
        .collect();
    assert_eq!(placed, [(64, 64, 8192), (32, 32, 4096)]);
    assert!(l.mips[..2].iter().all(|m| m.tail.is_none()));
    assert!(l.mips[2..].iter().all(|m| m.tail.is_some()));
    assert_eq!(t.size, 4096 + 4096 + 16384);

    // Mip 0 is 2×2 blocks of 32×32: the next block across is 4 KiB on,
    // the next row of blocks two blocks on.
    assert_eq!(l.offset(0, 32, 0, 0), Ok(8192 + 4096));
    assert_eq!(l.offset(0, 0, 32, 0), Ok(8192 + 8192));
    assert_eq!(l.offset(1, 31, 31, 0), Ok(4096 + 4092));
    // The first tail mip (16×16) starts at slot 7, like a whole-tail chain.
    assert_eq!(l.offset(2, 0, 0, 0), Ok(2048));

    // Each array slice repeats the whole chain.
    let a = texture(Dimension::Tex2d, 64, 64, 3, 0, R8G8B8A8, Swizzle::SW_4KB_S);
    let la = &a.planes[0].layout;
    assert_eq!((la.slice_size, a.size), (24576, 3 * 24576));
    assert_eq!(la.offset(2, 0, 0, 2), Ok(2 * 24576 + 2048));
}

/// A 3D RGBA8 texture in 4 KiB standard blocks is thick: each block is
/// 8×16×8 elements. Inside it x bits 0-2 land on address bits 2, 6, 9,
/// y bits 0-3 on 3, 5, 8, 11 and z bits 0-2 on 4, 7, 10, so the last
/// element of a block is its last 4 bytes.
#[test]
fn thick_blocks_interleave_depth_into_the_address() {
    let t = texture(Dimension::Tex3d, 8, 16, 8, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert!(l.thick);
    assert_eq!(l.block, [8, 16, 8]);
    assert_eq!((l.slices, t.size), (1, 4096));
    assert_eq!(l.offset(0, 1, 0, 0), Ok(4));
    assert_eq!(l.offset(0, 0, 1, 0), Ok(8));
    assert_eq!(l.offset(0, 0, 0, 1), Ok(16));
    assert_eq!(l.offset(0, 0, 0, 4), Ok(1024));
    assert_eq!(l.offset(0, 0, 8, 0), Ok(2048));
    assert_eq!(l.offset(0, 7, 15, 7), Ok(4092));

    // Nine slices need a second slab of eight.
    let t = texture(Dimension::Tex3d, 8, 16, 9, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!((l.slices, t.size), (2, 8192));
    assert_eq!(l.offset(0, 0, 0, 8), Ok(4096));
    assert_eq!(l.offset(0, 0, 0, 9), Err(Error::Invalid));
}

/// A thick 3×2×3 volume round-trips through padded linear rows and depth
/// slices; only its 18 elements are written, and the 3D subresource is
/// mip 0 of slice 0 only.
#[test]
fn thick_volume_round_trips_with_depth_pitch() {
    let t = texture(Dimension::Tex3d, 3, 2, 3, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    assert_eq!((t.array_size(), t.subresource_count()), (1, 1));
    let (row_pitch, depth_pitch) = (16, 40);
    let mut linear = vec![0; 2 * depth_pitch + row_pitch + 12];
    for z in 0..3 {
        for y in 0..2 {
            for b in 0..12 {
                linear[z * depth_pitch + y * row_pitch + b] = (z * 24 + y * 12 + b + 1) as u8;
            }
        }
    }
    let mut tiled = vec![0; t.size as usize];
    t.tile(&mut tiled, 0, &linear, row_pitch, depth_pitch)
        .unwrap();
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), 18 * 4);
    let l = &t.planes[0].layout;
    let at = l.offset(0, 2, 1, 2).unwrap() as usize;
    assert_eq!(
        &tiled[at..at + 4],
        &[
            2 * 24 + 12 + 9,
            2 * 24 + 12 + 10,
            2 * 24 + 12 + 11,
            2 * 24 + 12 + 12
        ]
    );

    let mut back = vec![0; linear.len()];
    t.detile(&tiled, 0, &mut back, row_pitch, depth_pitch)
        .unwrap();
    assert_eq!(back, linear);
    assert_eq!(
        t.tile(&mut tiled, 1, &linear, row_pitch, depth_pitch),
        Err(Error::Invalid)
    );
}

/// A linear 4×4×4 RGBA8 volume with a full chain: rows pad to 64
/// elements, mips sit smallest first (mip 2 at 0, mip 1 at 256, mip 0 at
/// 768) and every depth slice holds its own 1792-byte chain. Mip 1 is
/// 2×2×2, so its subresource spans depth slices 0 and 1.
#[test]
fn linear_volume_chain_repeats_per_depth_slice() {
    let t = texture(Dimension::Tex3d, 4, 4, 4, 0, R8G8B8A8, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    assert_eq!(t.mips(), 3);
    let got: Vec<_> = l
        .mips
        .iter()
        .map(|m| (m.width, m.depth, m.rows, m.offset))
        .collect();
    assert_eq!(got, [(4, 4, 4, 768), (2, 2, 2, 256), (1, 1, 1, 0)]);
    assert_eq!((l.slice_size, t.size), (1792, 4 * 1792));
    assert_eq!(l.offset(1, 1, 1, 1), Ok(1792 + 256 + 256 + 4));

    let f = t.footprint(1).unwrap();
    assert_eq!((f.row_pitch, f.depth_pitch, f.size), (8, 16, 32));
    let linear: Vec<u8> = (1..=32).collect();
    let mut tiled = vec![0; t.size as usize];
    t.tile(&mut tiled, 1, &linear, 8, 16).unwrap();
    let mut expected = vec![0; t.size as usize];
    for z in 0..2 {
        for y in 0..2 {
            let at = z * 1792 + 256 + y * 256;
            let from = z * 16 + y * 8;
            expected[at..at + 8].copy_from_slice(&linear[from..from + 8]);
        }
    }
    assert_eq!(tiled, expected);
}

/// Linear rows pad to 256 bytes whatever the element size: 256 one-byte
/// elements, or 16 sixteen-byte ones. A 1D texture is one row.
#[test]
fn linear_pitch_alignment_follows_element_size() {
    let t = texture(Dimension::Tex1d, 70, 1, 1, 1, R8, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    assert_eq!((l.block, l.mips[0].pitch, t.size), ([256, 1, 1], 256, 256));

    let t = texture(Dimension::Tex2d, 17, 2, 1, 1, R32G32B32A32, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    assert_eq!((l.block, l.mips[0].pitch), ([16, 1, 1], 32));
    assert_eq!(t.size, 2 * 32 * 16);
    assert_eq!(l.offset(0, 16, 1, 0), Ok(512 + 256));
}

/// Descriptions are refused at the edges: zero or over 16384 texels per
/// side, more than 16 mips, a 1D texture with height, a thin-only mode in
/// 3D, MSAA with mips, an unknown swizzle mode or format. 16384 is fine.
#[test]
fn descriptions_outside_the_bounds_are_invalid() {
    use Dimension::*;
    let ok = desc(Tex2d, 16384, 1, 1, 1, R8G8B8A8, Swizzle::LINEAR);
    assert!(Texture::new(&ok).is_ok());
    let bad = [
        desc(Tex2d, 16385, 1, 1, 1, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex2d, 0, 1, 1, 1, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex2d, 1, 1, 0, 1, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex2d, 1, 1, 16385, 1, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex2d, 4, 4, 1, 17, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex1d, 4, 2, 1, 1, R8G8B8A8, Swizzle::LINEAR),
        desc(Tex3d, 4, 4, 4, 1, R8G8B8A8, Swizzle::SW_4KB_D),
        desc(Tex3d, 4, 4, 4, 1, R8G8B8A8, Swizzle::SW_256B_S),
        desc(Tex2d, 4, 4, 1, 1, R8G8B8A8, Swizzle(3)),
        desc(Tex2d, 4, 4, 1, 1, 6, Swizzle::LINEAR),
        desc(Tex2d, 4, 4, 1, 1, 0, Swizzle::LINEAR),
    ];
    for d in bad {
        assert_eq!(Texture::new(&d), Err(Error::Invalid), "{d:?}");
    }
    let msaa = TextureDesc {
        samples: 4,
        mips: 2,
        ..desc(Tex2d, 8, 8, 1, 2, R8G8B8A8, Swizzle::SW_64KB_D)
    };
    assert_eq!(Texture::new(&msaa), Err(Error::Invalid));
}

/// XOR modes and MSAA have layouts but no addressing: copies say
/// `Unsupported` for a valid subresource, while coordinates outside the
/// mip are still `Invalid`.
#[test]
fn xor_and_msaa_layouts_are_sized_but_not_addressable() {
    let t = texture(Dimension::Tex2d, 8, 8, 1, 1, R8G8B8A8, Swizzle::SW_64KB_S_X);
    let l = &t.planes[0].layout;
    assert!(!l.addressable());
    assert_eq!(t.size, 65536);
    assert_eq!(l.offset(0, 0, 0, 0), Err(Error::Unsupported));
    assert_eq!(l.offset(0, 8, 0, 0), Err(Error::Invalid));
    let mut tiled = vec![0; 65536];
    assert_eq!(
        t.tile(&mut tiled, 0, &[0; 256], 32, 0),
        Err(Error::Unsupported)
    );
    assert!(tiled.iter().all(|&b| b == 0));

    let msaa = Texture::new(&TextureDesc {
        samples: 4,
        ..desc(Dimension::Tex2d, 8, 8, 1, 1, R8G8B8A8, Swizzle::SW_64KB_D)
    })
    .unwrap();
    assert!(!msaa.planes[0].layout.addressable());
    assert_eq!(
        msaa.detile(&vec![0; msaa.size as usize], 0, &mut [0; 256], 32, 0),
        Err(Error::Unsupported)
    );
}
