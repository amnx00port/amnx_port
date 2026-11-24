//! Row and depth pitches wider than the footprint, one-row and one-slice
//! subresources whose pitches never matter, element offsets across every
//! addressable mode, and subresource indices past the chain, through the
//! public API. Every texture is synthetic and every expected value comes
//! from the footprint or from `Layout::offset`, so nothing here needs data
//! from outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Layout, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const R8: u32 = 61;
const R16G16B16A16: u32 = 10;
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
    .unwrap_or_else(|e| panic!("{w}x{h}x{d} format {format} {swizzle:?}: {e:?}"))
}

/// Tiled memory whose every byte differs from its neighbours.
fn pattern(len: u64) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
}

/// A subresource one row tall and one slice deep only needs its row: any
/// row pitch at least a row wide (even `usize::MAX`) and any depth pitch
/// are accepted with a buffer of exactly one row, and give the same bytes.
#[test]
fn one_row_subresources_ignore_their_pitches() {
    for swizzle in [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_4KB_D,
        Swizzle::SW_64KB_S,
    ] {
        for (dimension, h) in [(Dimension::Tex1d, 1), (Dimension::Tex2d, 1)] {
            let t = texture(dimension, 24, h, 2, 1, R8G8B8A8, swizzle);
            let l = &t.planes[0].layout;
            let f = l.footprint(0).unwrap();
            assert_eq!((f.row_pitch, f.size), (96, 96), "{dimension:?} {swizzle:?}");
            let tiled = pattern(l.size);
            let mut tight = vec![0u8; f.size];
            l.detile(&tiled, 0, 1, &mut tight, f.row_pitch, f.depth_pitch)
                .unwrap();
            for (row_pitch, depth_pitch) in [
                (96, 0),
                (97, usize::MAX),
                (usize::MAX, 0),
                (usize::MAX, usize::MAX),
            ] {
                let mut out = vec![0u8; f.size];
                assert_eq!(
                    l.detile(&tiled, 0, 1, &mut out, row_pitch, depth_pitch),
                    Ok(()),
                    "{dimension:?} {swizzle:?} {row_pitch} {depth_pitch}"
                );
                assert_eq!(out, tight, "{dimension:?} {swizzle:?} {row_pitch}");
                let mut back = vec![0u8; l.size as usize];
                l.tile(&mut back, 0, 1, &out, row_pitch, depth_pitch)
                    .unwrap();
                for x in 0..24 {
                    let at = l.offset(0, x, 0, 1).unwrap() as usize;
                    assert_eq!(back[at..at + 4], tiled[at..at + 4], "{swizzle:?} x {x}");
                }
            }
            // One byte short of the row is still too small.
            assert_eq!(
                l.detile(&tiled, 0, 1, &mut tight[1..], usize::MAX, 0),
                Err(Error::TooSmall)
            );
            assert_eq!(
                l.detile(&tiled, 0, 1, &mut tight, 95, 0),
                Err(Error::TooSmall)
            );
        }
    }
}

/// Rows and depth slices land `row_pitch` and `depth_pitch` bytes apart
/// when those are wider than the footprint; the gaps between them keep
/// what the buffer held, and the smallest buffer is the last row's end.
#[test]
fn wide_pitches_place_rows_and_slices_and_keep_the_gaps() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
        let t = texture(Dimension::Tex3d, 6, 5, 3, 1, R8G8B8A8, swizzle);
        let l = &t.planes[0].layout;
        let f = l.footprint(0).unwrap();
        assert_eq!(
            (f.row_pitch, f.depth_pitch, f.size),
            (24, 120, 360),
            "{swizzle:?}"
        );
        let tiled = pattern(l.size);
        let mut tight = vec![0u8; f.size];
        l.detile(&tiled, 0, 0, &mut tight, f.row_pitch, f.depth_pitch)
            .unwrap();

        let (row_pitch, depth_pitch) = (40, 7 * 40 + 3);
        let need = 2 * depth_pitch + 4 * row_pitch + 24;
        let mut wide = vec![0xEEu8; need];
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut wide[..need - 1], row_pitch, depth_pitch),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
        l.detile(&tiled, 0, 0, &mut wide, row_pitch, depth_pitch)
            .unwrap();
        let mut copied = vec![false; need];
        for z in 0..3 {
            for y in 0..5 {
                let from = z * f.depth_pitch + y * f.row_pitch;
                let to = z * depth_pitch + y * row_pitch;
                assert_eq!(
                    wide[to..to + 24],
                    tight[from..from + 24],
                    "{swizzle:?} z {z} y {y}"
                );
                copied[to..to + 24].fill(true);
            }
        }
        for (i, &c) in copied.iter().enumerate() {
            if !c {
                assert_eq!(wide[i], 0xEE, "{swizzle:?} gap byte {i}");
            }
        }

        // Tiling the wide rows back writes the same texture.
        let mut back = vec![0u8; l.size as usize];
        l.tile(&mut back, 0, 0, &wide, row_pitch, depth_pitch)
            .unwrap();
        let mut again = vec![0u8; f.size];
        l.detile(&back, 0, 0, &mut again, f.row_pitch, f.depth_pitch)
            .unwrap();
        assert_eq!(again, tight, "{swizzle:?}");
    }
}

/// Every element of every mip and slice, in every mode that addresses,
/// starts on a multiple of its element size, lies inside the layout and
/// owns its bytes: no two elements share an offset.
#[test]
fn element_offsets_are_aligned_inside_and_distinct_in_every_mode() {
    let modes = [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_256B_D,
        Swizzle::SW_4KB_S,
        Swizzle::SW_4KB_D,
        Swizzle::SW_64KB_S,
        Swizzle::SW_64KB_D,
    ];
    for swizzle in modes {
        for format in [R8, R8G8B8A8, R16G16B16A16] {
            for (dimension, w, h, d) in [
                (Dimension::Tex2d, 1, 1, 1),
                (Dimension::Tex2d, 19, 7, 2),
                (Dimension::Tex1d, 33, 1, 3),
                (Dimension::Tex3d, 9, 6, 5),
            ] {
                // 3D textures take only linear and the thick `S` modes.
                let thick = matches!(swizzle.0, 0 | 5 | 9);
                if dimension == Dimension::Tex3d && !thick {
                    continue;
                }
                let t = texture(dimension, w, h, d, 0, format, swizzle);
                let l = &t.planes[0].layout;
                assert!(l.addressable());
                let n = u64::from(l.desc.element_bytes);
                let what = format!("{swizzle:?} format {format} {dimension:?} {w}x{h}x{d}");
                let mut seen = HashSet::new();
                for (mip, m) in l.mips.iter().enumerate() {
                    let depth = if dimension == Dimension::Tex3d {
                        m.depth
                    } else {
                        d
                    };
                    for z in 0..depth {
                        for y in 0..m.height {
                            for x in 0..m.width {
                                let at = l.offset(mip as u32, x, y, z).unwrap();
                                assert_eq!(at % n, 0, "{what} mip {mip} ({x}, {y}, {z})");
                                assert!(at + n <= l.size, "{what} mip {mip} ({x}, {y}, {z})");
                                assert!(seen.insert(at), "{what} mip {mip} ({x}, {y}, {z})");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A mip past the chain is `Invalid` for every request that names one,
/// however far past, and even in a mode that does not address; its
/// footprint is `None` and it covers no depth slices.
#[test]
fn mips_past_the_chain_are_refused_everywhere() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_64KB_S, Swizzle::SW_64KB_R_X] {
        let t = texture(Dimension::Tex3d, 8, 8, 4, 0, R8G8B8A8, swizzle);
        let l: &Layout = &t.planes[0].layout;
        let count = l.mips.len() as u32;
        assert_eq!(count, 4, "{swizzle:?}");
        assert_eq!(l.subresource_depth(count - 1), 1, "{swizzle:?}");
        for mip in [count, count + 1, u32::MAX] {
            assert_eq!(
                l.offset(mip, 0, 0, 0),
                Err(Error::Invalid),
                "{swizzle:?} {mip}"
            );
            assert_eq!(l.footprint(mip), None, "{swizzle:?} {mip}");
            assert_eq!(l.subresource_depth(mip), 0, "{swizzle:?} {mip}");
            let tiled = vec![0u8; l.size as usize];
            let mut out = vec![0u8; 1024];
            assert_eq!(
                l.detile(&tiled, mip, 0, &mut out, 64, 512),
                Err(Error::Invalid)
            );
            let mut tiled = tiled;
            assert_eq!(
                l.tile(&mut tiled, mip, 0, &out, 64, 512),
                Err(Error::Invalid)
            );
        }
    }
}

/// D3D12 subresource indices of a depth-stencil array run mip fastest,
/// then array slice, then plane; every index below the count decodes to
/// that triple and back, and the footprint follows the plane's element
/// size. Past the count there is no subresource and copies are `Invalid`
/// before the buffer sizes are checked.
#[test]
fn depth_stencil_array_indices_decode_mip_slice_plane() {
    let t = texture(Dimension::Tex2d, 12, 8, 3, 0, D24S8, Swizzle::SW_4KB_S);
    assert_eq!((t.mips(), t.array_size(), t.planes.len()), (4, 3, 2));
    let count = t.subresource_count();
    assert_eq!(count, 24);
    for index in 0..count {
        let s = t.subresource(index).unwrap();
        assert_eq!(s.mip + s.slice * 4 + s.plane * 12, index);
        assert!(s.mip < 4 && s.slice < 3 && s.plane < 2, "{index}");
        let bytes = if s.plane == 0 { 4 } else { 1 };
        let width = (12usize >> s.mip).max(1);
        assert_eq!(
            t.footprint(index).unwrap().row_pitch,
            width * bytes,
            "{index}"
        );
    }
    for index in [count, count + 1, u32::MAX] {
        assert_eq!(t.subresource(index), None);
        assert_eq!(t.footprint(index), None);
        assert_eq!(t.detile(&[], index, &mut [], 0, 0), Err(Error::Invalid));
        assert_eq!(t.tile(&mut [], index, &[], 0, 0), Err(Error::Invalid));
    }
}
