//! Swizzle mode classification, description rules per dimension, and the
//! argument checks and byte placement of the tiled copies, through the
//! public API. Every texture is synthetic and every expected value comes
//! from the GFX10 swizzle mode table or from `Layout::offset`, so nothing
//! here needs data from outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Layout, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const R16: u32 = 56;

fn desc(
    dimension: Dimension,
    w: u32,
    h: u32,
    d: u32,
    mips: u32,
    samples: u32,
    swizzle: Swizzle,
) -> TextureDesc {
    TextureDesc {
        dimension,
        width: w,
        height: h,
        depth_or_array: d,
        mips,
        format: R8G8B8A8,
        samples,
        swizzle,
    }
}

/// The GFX10 mode numbers: 0 linear, 1-2 256 B, 5-6 4 KiB, 9-10 64 KiB,
/// 17-18 64 KiB `_T`, 21-23 4 KiB `_X`, 24-27 64 KiB `_X`. Every other
/// number below 32 is a GFX9 mode or reserved, and has no block size.
#[test]
fn every_mode_number_has_its_block_size_and_xor_flag() {
    for n in 0..32u32 {
        let s = Swizzle(n);
        let expected = match n {
            0 => Some(0),
            1 | 2 => Some(8),
            5 | 6 | 21..=23 => Some(12),
            9 | 10 | 17 | 18 | 24..=27 => Some(16),
            _ => None,
        };
        assert_eq!(s.block_log2(), expected, "mode {n}");
        assert_eq!(s.xor(), n >= 16, "mode {n}");
        if expected.is_none() {
            assert_eq!(
                Texture::new(&desc(Dimension::Tex2d, 8, 8, 1, 1, 1, s)),
                Err(Error::Invalid),
                "mode {n}"
            );
        }
    }
    // The named constants carry those numbers.
    let named = [
        (Swizzle::LINEAR, 0),
        (Swizzle::SW_256B_S, 1),
        (Swizzle::SW_256B_D, 2),
        (Swizzle::SW_4KB_S, 5),
        (Swizzle::SW_4KB_D, 6),
        (Swizzle::SW_64KB_S, 9),
        (Swizzle::SW_64KB_D, 10),
        (Swizzle::SW_64KB_S_T, 17),
        (Swizzle::SW_64KB_D_T, 18),
        (Swizzle::SW_4KB_S_X, 21),
        (Swizzle::SW_4KB_D_X, 22),
        (Swizzle::SW_4KB_R_X, 23),
        (Swizzle::SW_64KB_Z_X, 24),
        (Swizzle::SW_64KB_S_X, 25),
        (Swizzle::SW_64KB_D_X, 26),
        (Swizzle::SW_64KB_R_X, 27),
    ];
    for (s, n) in named {
        assert_eq!(s, Swizzle(n));
    }
}

/// 3D textures take linear, the standard modes and the 64 KiB `_X` modes
/// (0, 5, 9, 17, 21, 24-27); the 256 B modes, the display modes below 24
/// and 4 KiB `R_X` are 2D only. The same modes all build a 2D texture.
#[test]
fn volumes_accept_only_the_modes_with_a_3d_layout() {
    let volume_modes = [0, 5, 9, 17, 21, 24, 25, 26, 27];
    for n in [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27] {
        let s = Swizzle(n);
        let v = Texture::new(&desc(Dimension::Tex3d, 8, 8, 4, 1, 1, s));
        assert_eq!(v.is_ok(), volume_modes.contains(&n), "mode {n}");
        if v.is_err() {
            assert_eq!(v, Err(Error::Invalid), "mode {n}");
        }
        assert!(
            Texture::new(&desc(Dimension::Tex2d, 8, 8, 4, 1, 1, s)).is_ok(),
            "mode {n}"
        );
    }
}

/// A 1D texture is one row; multisampling is 2D, one mip, and a power of
/// two up to 8 samples. Samples 0 counts as 1.
#[test]
fn one_dimensional_and_multisampled_descriptions_follow_their_rules() {
    use Dimension::*;
    let s = Swizzle::SW_64KB_S;
    let ok = |d: TextureDesc| Texture::new(&d).is_ok();

    assert!(ok(desc(Tex1d, 64, 1, 4, 0, 1, s)));
    assert!(!ok(desc(Tex1d, 64, 2, 1, 1, 1, s)));

    for samples in [0, 1, 2, 4, 8] {
        assert!(ok(desc(Tex2d, 64, 64, 1, 1, samples, s)), "{samples}");
    }
    for samples in [3, 6, 16] {
        assert!(!ok(desc(Tex2d, 64, 64, 1, 1, samples, s)), "{samples}");
    }
    assert!(!ok(desc(Tex2d, 64, 64, 1, 2, 4, s)));
    assert!(!ok(desc(Tex3d, 64, 64, 4, 1, 4, s)));
    assert!(!ok(desc(Tex1d, 64, 1, 1, 1, 4, s)));

    // Samples 0 builds the same texture as samples 1.
    assert_eq!(
        Texture::new(&desc(Tex2d, 64, 64, 1, 1, 0, s)),
        Texture::new(&desc(Tex2d, 64, 64, 1, 1, 1, s))
    );
}

/// The copies check their arguments in a fixed order: a mip past the
/// chain or a slice past the array is `Invalid` (a 3D texture only has
/// slice 0); then a mode without addressing is `Unsupported`; then short
/// buffers or a row pitch narrower than a row are `TooSmall`.
#[test]
fn copies_check_the_subresource_before_the_buffers() {
    let t = Texture::new(&desc(Dimension::Tex2d, 16, 8, 3, 2, 1, Swizzle::SW_4KB_S)).unwrap();
    let l = &t.planes[0].layout;
    let tiled = vec![0u8; l.size as usize];
    let f = l.footprint(0).unwrap();
    assert_eq!((f.row_pitch, f.depth_pitch, f.size), (64, 512, 512));
    let mut out = vec![0u8; f.size];

    let detile = |tiled: &[u8], mip, slice, out: &mut [u8], row_pitch| {
        l.detile(tiled, mip, slice, out, row_pitch, f.depth_pitch)
    };
    assert_eq!(detile(&tiled, 0, 0, &mut out, 64), Ok(()));
    assert_eq!(detile(&tiled, 0, 2, &mut out, 64), Ok(()));
    assert_eq!(detile(&tiled, 2, 0, &mut out, 64), Err(Error::Invalid));
    assert_eq!(detile(&tiled, 0, 3, &mut out, 64), Err(Error::Invalid));
    // Out of range wins over short buffers.
    assert_eq!(detile(&[], 2, 0, &mut [], 0), Err(Error::Invalid));
    assert_eq!(detile(&tiled, 0, 0, &mut out, 63), Err(Error::TooSmall));
    assert_eq!(
        detile(&tiled[1..], 0, 0, &mut out, 64),
        Err(Error::TooSmall)
    );
    assert_eq!(
        detile(&tiled, 0, 0, &mut out[1..], 64),
        Err(Error::TooSmall)
    );

    let v = Texture::new(&desc(Dimension::Tex3d, 8, 8, 4, 1, 1, Swizzle::SW_4KB_S)).unwrap();
    let vl = &v.planes[0].layout;
    let vf = vl.footprint(0).unwrap();
    let vt = vec![0u8; vl.size as usize];
    let mut vo = vec![0u8; vf.size];
    assert_eq!(vf.size, 8 * 8 * 4 * 4);
    assert_eq!(
        vl.detile(&vt, 0, 0, &mut vo, vf.row_pitch, vf.depth_pitch),
        Ok(())
    );
    assert_eq!(
        vl.detile(&vt, 0, 1, &mut vo, vf.row_pitch, vf.depth_pitch),
        Err(Error::Invalid)
    );

    // An `_X` mode sizes but does not copy, even with buffers far too
    // small; a bad subresource is still `Invalid`.
    let x = Texture::new(&desc(Dimension::Tex2d, 16, 8, 1, 1, 1, Swizzle::SW_4KB_S_X)).unwrap();
    let xl = &x.planes[0].layout;
    assert_eq!(xl.detile(&[], 0, 0, &mut [], 0, 0), Err(Error::Unsupported));
    assert_eq!(xl.tile(&mut [], 0, 0, &[], 0, 0), Err(Error::Unsupported));
    assert_eq!(xl.detile(&[], 1, 0, &mut [], 0, 0), Err(Error::Invalid));
}

/// The bytes `tile` writes for one subresource are exactly the bytes
/// `offset` names for its elements, whatever the slice and mip; nothing
/// else in the texture changes.
fn assert_tile_writes_only_its_elements(l: &Layout, mip: u32, slice: u32) {
    let f = l.footprint(mip).unwrap();
    let n = l.desc.element_bytes as usize;
    let m = &l.mips[mip as usize];
    let linear: Vec<u8> = (0..f.size).map(|i| (i % 251) as u8 + 1).collect();
    let mut tiled = vec![0u8; l.size as usize];
    l.tile(&mut tiled, mip, slice, &linear, f.row_pitch, f.depth_pitch)
        .unwrap();

    let depth = l.subresource_depth(mip);
    let mut expected = HashSet::new();
    for z in 0..depth {
        let zz = if l.desc.dimension == Dimension::Tex3d {
            z
        } else {
            slice
        };
        for y in 0..m.height {
            for x in 0..m.width {
                let at = l.offset(mip, x, y, zz).unwrap() as usize;
                let from = z as usize * f.depth_pitch + y as usize * f.row_pitch + x as usize * n;
                assert_eq!(&tiled[at..at + n], &linear[from..from + n]);
                expected.extend(at..at + n);
            }
        }
    }
    assert_eq!(expected.len(), f.size, "mip {mip} slice {slice}");
    for (i, &b) in tiled.iter().enumerate() {
        if !expected.contains(&i) {
            assert_eq!(b, 0, "byte {i} outside mip {mip} slice {slice}");
        }
    }
}

#[test]
fn tile_touches_only_the_subresource_it_names() {
    // A 2D array whose later mips sit in the 4 KiB tail, a thick volume
    // and a 256 B chain without a tail.
    let cases = [
        desc(Dimension::Tex2d, 40, 24, 3, 0, 1, Swizzle::SW_4KB_S),
        desc(Dimension::Tex2d, 40, 24, 3, 0, 1, Swizzle::SW_4KB_D),
        desc(Dimension::Tex3d, 12, 10, 6, 0, 1, Swizzle::SW_64KB_S),
        desc(Dimension::Tex2d, 9, 7, 2, 0, 1, Swizzle::SW_256B_S),
        desc(Dimension::Tex2d, 9, 7, 2, 0, 1, Swizzle::LINEAR),
    ];
    for d in cases {
        let t = Texture::new(&d).unwrap();
        let l = &t.planes[0].layout;
        for mip in 0..t.mips() {
            for slice in 0..t.array_size() {
                assert_tile_writes_only_its_elements(l, mip, slice);
            }
        }
    }
}

/// With a row pitch and depth pitch wider than the footprint, `detile`
/// fills each row's elements and leaves the padding after them alone;
/// `tile` reads back from the same places.
#[test]
fn padded_linear_rows_keep_their_gaps() {
    let t = Texture::new(&desc(Dimension::Tex3d, 5, 3, 2, 1, 1, Swizzle::SW_64KB_S)).unwrap();
    let l = &t.planes[0].layout;
    let (row_pitch, depth_pitch) = (32, 128);
    let bytes = 5 * 4;
    let mut tiled: Vec<u8> = (0..l.size).map(|i| (i % 253) as u8 + 1).collect();
    let mut linear = vec![0xAAu8; depth_pitch + 2 * row_pitch + bytes];
    l.detile(&tiled, 0, 0, &mut linear, row_pitch, depth_pitch)
        .unwrap();

    for (i, &b) in linear.iter().enumerate() {
        let (z, in_slice) = (i / depth_pitch, i % depth_pitch);
        let (y, in_row) = (in_slice / row_pitch, in_slice % row_pitch);
        if y < 3 && in_row < bytes {
            let x = (in_row / 4) as u32;
            let at = l.offset(0, x, y as u32, z as u32).unwrap() as usize + in_row % 4;
            assert_eq!(b, tiled[at], "linear byte {i}");
        } else {
            assert_eq!(b, 0xAA, "padding byte {i}");
        }
    }

    let before = tiled.clone();
    tiled.fill(0);
    l.tile(&mut tiled, 0, 0, &linear, row_pitch, depth_pitch)
        .unwrap();
    for (i, (&now, &was)) in tiled.iter().zip(&before).enumerate() {
        assert!(now == 0 || now == was, "tiled byte {i}");
    }
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), 5 * 3 * 2 * 4);
}

/// Every subresource of a tiled array with a full chain round trips
/// through the texture's own copies, and together they write each element
/// byte once: the subresources never share memory.
#[test]
fn every_subresource_of_a_tiled_array_owns_its_bytes() {
    let mut d = desc(Dimension::Tex2d, 33, 17, 4, 0, 1, Swizzle::SW_64KB_D);
    d.format = R16;
    let t = Texture::new(&d).unwrap();
    assert_eq!(t.mips(), 6);
    assert_eq!(t.subresource_count(), 24);

    let mut tiled = vec![0u8; t.size as usize];
    let mut owner = vec![u32::MAX; t.size as usize];
    let mut total = 0;
    for index in 0..t.subresource_count() {
        let f = t.footprint(index).unwrap();
        let linear: Vec<u8> = (0..f.size)
            .map(|i| (i as u32 * 7 + index * 13) as u8 | 1)
            .collect();
        let before = tiled.clone();
        t.tile(&mut tiled, index, &linear, f.row_pitch, f.depth_pitch)
            .unwrap();
        for (i, (&now, &was)) in tiled.iter().zip(&before).enumerate() {
            if now != was {
                assert_eq!(owner[i], u32::MAX, "byte {i} reused by {index}");
                owner[i] = index;
            }
        }
        total += f.size;

        let mut back = vec![0u8; f.size];
        t.detile(&tiled, index, &mut back, f.row_pitch, f.depth_pitch)
            .unwrap();
        assert_eq!(back, linear, "subresource {index}");
    }
    assert_eq!(owner.iter().filter(|&&o| o != u32::MAX).count(), total);
    assert_eq!(t.footprint(24), None);
    assert_eq!(t.detile(&tiled, 24, &mut [], 0, 0), Err(Error::Invalid));
}
