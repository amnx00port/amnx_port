//! Argument checks of `Layout` through the public API: which error each bad
//! coordinate, index or pitch gives, which check wins when several fail,
//! and that absurd pitches are refused instead of overflowing. Every layout
//! is built from a synthetic `Desc`, so nothing here needs data from
//! outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Desc, Dimension, Error, Layout, Swizzle};

fn desc(dimension: Dimension, w: u32, h: u32, d: u32, mips: u32, swizzle: Swizzle) -> Desc {
    Desc {
        dimension,
        width: w,
        height: h,
        depth: d,
        mips,
        element_bytes: 4,
        block: 1,
        samples: 1,
        swizzle,
    }
}

fn layout(dimension: Dimension, w: u32, h: u32, d: u32, mips: u32, swizzle: Swizzle) -> Layout {
    Layout::new(&desc(dimension, w, h, d, mips, swizzle)).unwrap()
}

/// The addressable 2D modes: linear and every non-XOR GFX10 mode.
const MODES: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// Each error has its own message, names the texture code, and works as a
/// boxed `std::error::Error`.
#[test]
fn errors_have_distinct_messages_naming_textures() {
    let all = [Error::Invalid, Error::Unsupported, Error::TooSmall];
    let messages: HashSet<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(messages.len(), all.len());
    for e in all {
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        let text = boxed.to_string();
        assert!(text.starts_with("texture: "), "{text}");
        assert!(boxed.source().is_none());
    }
}

/// Building a layout twice from one description gives the same layout, and
/// the layout keeps the description it was built from.
#[test]
fn layouts_are_deterministic_and_keep_their_description() {
    for swizzle in MODES {
        let d = desc(Dimension::Tex2d, 37, 19, 3, 4, swizzle);
        let a = Layout::new(&d).unwrap();
        assert_eq!(a, Layout::new(&d).unwrap(), "{swizzle:?}");
        assert_eq!(a.desc, d, "{swizzle:?}");
        assert_eq!(a.mips.len(), 4, "{swizzle:?}");
        assert_eq!(a.slices, 3, "{swizzle:?}");
    }
}

/// The last element of every mip and slice has an offset; one step past it
/// on any axis, or a mip past the chain, is `Invalid`.
#[test]
fn offsets_stop_exactly_at_each_mip_edge() {
    for swizzle in MODES {
        let l = layout(Dimension::Tex2d, 37, 19, 3, 4, swizzle);
        for (i, m) in l.mips.iter().enumerate() {
            let i = i as u32;
            let (w, h) = (m.width, m.height);
            let last = l.offset(i, w - 1, h - 1, 2).unwrap();
            assert!(last + 4 <= l.size, "{swizzle:?} mip {i}");
            assert_eq!(
                l.offset(i, w, 0, 0),
                Err(Error::Invalid),
                "{swizzle:?} mip {i}"
            );
            assert_eq!(
                l.offset(i, 0, h, 0),
                Err(Error::Invalid),
                "{swizzle:?} mip {i}"
            );
            assert_eq!(
                l.offset(i, 0, 0, 3),
                Err(Error::Invalid),
                "{swizzle:?} mip {i}"
            );
            assert_eq!(
                l.offset(i, u32::MAX, u32::MAX, u32::MAX),
                Err(Error::Invalid),
                "{swizzle:?} mip {i}"
            );
        }
        assert_eq!(l.offset(4, 0, 0, 0), Err(Error::Invalid), "{swizzle:?}");
        assert_eq!(
            l.offset(u32::MAX, 0, 0, 0),
            Err(Error::Invalid),
            "{swizzle:?}"
        );
        assert_eq!(l.subresource_depth(4), 0, "{swizzle:?}");
        assert_eq!(l.footprint(4), None, "{swizzle:?}");
    }
}

/// In a 3D texture `z` is bounded by the mip's own depth, which halves
/// down the chain, not by the depth of mip 0.
#[test]
fn volume_offsets_bound_z_by_the_mips_depth() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
        let l = layout(Dimension::Tex3d, 16, 16, 8, 4, swizzle);
        for (i, depth) in [8u32, 4, 2, 1].into_iter().enumerate() {
            let i = i as u32;
            assert_eq!(l.subresource_depth(i), depth, "{swizzle:?} mip {i}");
            assert!(l.offset(i, 0, 0, depth - 1).is_ok(), "{swizzle:?} mip {i}");
            assert_eq!(
                l.offset(i, 0, 0, depth),
                Err(Error::Invalid),
                "{swizzle:?} mip {i}"
            );
        }
    }
}

/// An XOR layout cannot address anything, but a coordinate outside the
/// mip is still `Invalid`, not `Unsupported`: the bounds check comes first.
#[test]
fn xor_offsets_report_bad_coordinates_before_missing_addressing() {
    let l = layout(Dimension::Tex2d, 64, 64, 2, 2, Swizzle::SW_64KB_S_X);
    assert!(!l.addressable());
    assert_eq!(l.offset(0, 0, 0, 0), Err(Error::Unsupported));
    assert_eq!(l.offset(1, 31, 31, 1), Err(Error::Unsupported));
    assert_eq!(l.offset(0, 64, 0, 0), Err(Error::Invalid));
    assert_eq!(l.offset(1, 0, 32, 0), Err(Error::Invalid));
    assert_eq!(l.offset(0, 0, 0, 2), Err(Error::Invalid));
    assert_eq!(l.offset(2, 0, 0, 0), Err(Error::Invalid));
}

/// Copies of an XOR layout also check the mip and slice before refusing
/// the mode, whatever the buffers.
#[test]
fn xor_copies_report_bad_indices_before_missing_addressing() {
    let l = layout(Dimension::Tex2d, 64, 64, 2, 2, Swizzle::SW_64KB_S_X);
    let mut tiled = vec![0u8; l.size as usize];
    let mut linear = vec![0u8; 64 * 64 * 4];
    assert_eq!(
        l.detile(&tiled, 0, 0, &mut linear, 256, 0),
        Err(Error::Unsupported)
    );
    assert_eq!(
        l.detile(&tiled, 2, 0, &mut linear, 256, 0),
        Err(Error::Invalid)
    );
    assert_eq!(
        l.detile(&tiled, 0, 2, &mut linear, 256, 0),
        Err(Error::Invalid)
    );
    assert_eq!(
        l.tile(&mut tiled, 2, 0, &linear, 256, 0),
        Err(Error::Invalid)
    );
    assert_eq!(
        l.tile(&mut tiled, 0, 2, &linear, 256, 0),
        Err(Error::Invalid)
    );
    assert!(tiled.iter().all(|&b| b == 0));
    assert!(linear.iter().all(|&b| b == 0));
}

/// A 3D copy takes every depth slice of the mip at once, so only slice 0
/// names a subresource; an array takes slices below its size.
#[test]
fn copies_take_slice_zero_of_volumes_and_real_slices_of_arrays() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
        let v = layout(Dimension::Tex3d, 8, 8, 4, 1, swizzle);
        let f = v.footprint(0).unwrap();
        let tiled = vec![0u8; v.size as usize];
        let mut linear = vec![0u8; f.size];
        assert_eq!(
            v.detile(&tiled, 0, 0, &mut linear, f.row_pitch, f.depth_pitch),
            Ok(())
        );
        for slice in [1, 3, 4, u32::MAX] {
            assert_eq!(
                v.detile(&tiled, 0, slice, &mut linear, f.row_pitch, f.depth_pitch),
                Err(Error::Invalid),
                "{swizzle:?} slice {slice}"
            );
        }

        let a = layout(Dimension::Tex2d, 8, 8, 4, 1, swizzle);
        let f = a.footprint(0).unwrap();
        let tiled = vec![0u8; a.size as usize];
        let mut linear = vec![0u8; f.size];
        for slice in 0..4 {
            assert_eq!(
                a.detile(&tiled, 0, slice, &mut linear, f.row_pitch, 0),
                Ok(())
            );
        }
        for slice in [4, u32::MAX] {
            assert_eq!(
                a.detile(&tiled, 0, slice, &mut linear, f.row_pitch, 0),
                Err(Error::Invalid),
                "{swizzle:?} slice {slice}"
            );
        }
    }
}

/// A row pitch shorter than one row of the mip is refused even when the
/// linear buffer is large enough for the whole texture.
#[test]
fn row_pitch_shorter_than_a_row_is_too_small() {
    for swizzle in MODES {
        let l = layout(Dimension::Tex2d, 16, 4, 1, 1, swizzle);
        let tiled = vec![0u8; l.size as usize];
        let mut linear = vec![0u8; 1 << 16];
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut linear, 64, 0),
            Ok(()),
            "{swizzle:?}"
        );
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut linear, 63, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut linear, 0, 0),
            Err(Error::TooSmall),
            "{swizzle:?}"
        );
    }
}

/// Pitches so large that the last row's offset does not fit in `usize`
/// are refused as `TooSmall`, never overflow, and copy nothing.
#[test]
fn pitches_that_overflow_are_too_small() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
        let l = layout(Dimension::Tex3d, 8, 8, 4, 1, swizzle);
        let f = l.footprint(0).unwrap();
        let mut tiled = vec![0u8; l.size as usize];
        let mut linear = vec![0u8; f.size];
        for (row, depth) in [
            (usize::MAX, f.depth_pitch),
            (f.row_pitch, usize::MAX),
            (usize::MAX, usize::MAX),
            (usize::MAX / 7 + 1, f.depth_pitch),
            (f.row_pitch, usize::MAX / 3 + 1),
        ] {
            assert_eq!(
                l.detile(&tiled, 0, 0, &mut linear, row, depth),
                Err(Error::TooSmall),
                "{swizzle:?} {row} {depth}"
            );
            assert_eq!(
                l.tile(&mut tiled, 0, 0, &linear, row, depth),
                Err(Error::TooSmall),
                "{swizzle:?} {row} {depth}"
            );
        }
        assert!(tiled.iter().all(|&b| b == 0));
        assert!(linear.iter().all(|&b| b == 0));
    }
}

/// One-row and one-slice subresources never step by their pitches, so huge
/// pitches there are fine: only the bytes actually reached must fit.
#[test]
fn pitches_that_are_never_stepped_may_be_huge() {
    for swizzle in MODES {
        let l = layout(Dimension::Tex2d, 16, 1, 1, 1, swizzle);
        let tiled: Vec<u8> = (0..l.size).map(|i| i as u8).collect();
        let mut wide = vec![0u8; 64];
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut wide, usize::MAX, usize::MAX),
            Ok(()),
            "{swizzle:?}"
        );
        let mut packed = vec![0u8; 64];
        assert_eq!(
            l.detile(&tiled, 0, 0, &mut packed, 64, 0),
            Ok(()),
            "{swizzle:?}"
        );
        assert_eq!(wide, packed, "{swizzle:?}");
    }
}

/// Every element's `offset` is where `detile` reads it from: filling the
/// tiled bytes with their own offsets and detiling gives each element the
/// bytes of its offset, in every mode, mip and slice.
#[test]
fn detile_reads_each_element_from_its_offset() {
    for swizzle in MODES {
        let l = layout(Dimension::Tex2d, 23, 11, 2, 3, swizzle);
        let tiled: Vec<u8> = (0..l.size)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[3])
            .collect();
        for mip in 0..3 {
            let f = l.footprint(mip).unwrap();
            let m = l.mips[mip as usize];
            for slice in 0..2 {
                let mut linear = vec![0u8; f.size];
                l.detile(&tiled, mip, slice, &mut linear, f.row_pitch, f.depth_pitch)
                    .unwrap();
                for y in 0..m.height {
                    for x in 0..m.width {
                        let at = l.offset(mip, x, y, slice).unwrap() as usize;
                        let l_at = y as usize * f.row_pitch + x as usize * 4;
                        assert_eq!(
                            linear[l_at..][..4],
                            tiled[at..][..4],
                            "{swizzle:?} mip {mip} slice {slice} ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}
