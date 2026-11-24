//! Every DXGI format number the layouts know, through [`Texture`]: the
//! planes a format gets, where they sit, and whole-texture copies of every
//! subresource on small synthetic textures in each addressable 2D mode.
//! The expected element sizes come from [`texture::format`]; the layout
//! rules are checked as invariants, so nothing here needs data from outside
//! the repository.

use victory_formats::texture::{self, Dimension, Error, Swizzle, Texture, TextureDesc};

const MODES: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// Every format number with a layout, low to high.
fn formats() -> Vec<u32> {
    (0..=255)
        .filter(|&f| texture::format(f).is_some())
        .collect()
}

fn desc(w: u32, h: u32, array: u32, mips: u32, format: u32, swizzle: Swizzle) -> TextureDesc {
    TextureDesc {
        dimension: Dimension::Tex2d,
        width: w,
        height: h,
        depth_or_array: array,
        mips,
        format,
        samples: 1,
        swizzle,
    }
}

/// A byte pattern that differs between subresources and between the
/// bytes of one subresource.
fn pattern(index: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            (i as u32)
                .wrapping_mul(31)
                .wrapping_add(index.wrapping_mul(101))
                .wrapping_add(7) as u8
        })
        .collect()
}

/// Every format number with a layout makes a texture; every other number
/// is refused as invalid rather than given a guessed size.
#[test]
fn format_numbers_without_a_layout_make_no_texture() {
    let known = formats();
    assert!(known.len() > 80, "{} formats", known.len());
    for f in 0..=255 {
        let made = Texture::new(&desc(8, 8, 1, 1, f, Swizzle::SW_4KB_S));
        if known.contains(&f) {
            assert!(made.is_ok(), "format {f}: {made:?}");
        } else {
            assert_eq!(made.unwrap_err(), Error::Invalid, "format {f}");
        }
    }
}

/// Each format's planes match its table entry: one plane of its element
/// size and block, plus a one-byte stencil plane for depth-stencil
/// formats. The stencil plane starts at the first multiple of its own
/// alignment past the depth plane, and the texture ends with its last
/// plane and is aligned like its most aligned plane.
#[test]
fn planes_follow_the_format_table() {
    for f in formats() {
        let want = texture::format(f).unwrap();
        for swizzle in MODES {
            let what = format!("format {f} {swizzle:?}");
            let t = Texture::new(&desc(19, 11, 3, 0, f, swizzle)).unwrap();
            assert_eq!(t.planes.len(), if want.stencil { 2 } else { 1 }, "{what}");

            let depth = &t.planes[0];
            assert_eq!((depth.offset, depth.stencil), (0, false), "{what}");
            assert_eq!(depth.layout.desc.element_bytes, want.bytes, "{what}");
            assert_eq!(depth.layout.desc.block, want.block, "{what}");

            if let Some(stencil) = t.planes.get(1) {
                assert!(stencil.stencil, "{what}");
                assert_eq!(
                    (stencil.layout.desc.element_bytes, stencil.layout.desc.block),
                    (1, 1),
                    "{what}"
                );
                assert_eq!(
                    stencil.offset,
                    depth.layout.size.next_multiple_of(stencil.layout.alignment),
                    "{what}"
                );
                assert_eq!(stencil.layout.mips.len(), depth.layout.mips.len(), "{what}");
            }

            let last = t.planes.last().unwrap();
            assert_eq!(t.size, last.offset + last.layout.size, "{what}");
            let alignment = t.planes.iter().map(|p| p.layout.alignment).max().unwrap();
            assert_eq!(t.alignment, alignment, "{what}");
            assert_eq!(t.size % depth.layout.alignment, 0, "{what}");
        }
    }
}

/// Formats with the same element size and block lay out the same way:
/// the layout depends on bytes per element, not on what the bytes mean.
#[test]
fn formats_of_one_element_size_share_their_layout() {
    for swizzle in MODES {
        let mut seen: Vec<((u32, u32), texture::Layout)> = Vec::new();
        for f in formats() {
            let t = Texture::new(&desc(37, 21, 2, 0, f, swizzle)).unwrap();
            let l = &t.planes[0].layout;
            let key = (l.desc.element_bytes, l.desc.block);
            match seen.iter().find(|(k, _)| *k == key) {
                Some((_, first)) => {
                    assert_eq!(l.mips, first.mips, "format {f} {swizzle:?}");
                    assert_eq!(
                        (l.size, l.alignment),
                        (first.size, first.alignment),
                        "format {f} {swizzle:?}"
                    );
                }
                None => seen.push((key, l.clone())),
            }
        }
        // 16, 8, 4, 2 and 1 bytes per texel; 8 and 16 bytes per 4×4 block.
        assert_eq!(seen.len(), 7, "{swizzle:?}");
    }
}

/// Writing every subresource of every plane (each with its own pattern,
/// at its packed footprint) and then reading each back gives the data
/// written: subresources of one texture never share bytes, whatever the
/// format, plane or mode. An odd-sized array with its full chain covers
/// partial blocks, mip tails and the stencil plane's subresource indices.
#[test]
fn every_subresource_of_every_format_round_trips() {
    for f in formats() {
        for swizzle in MODES {
            let what = format!("format {f} {swizzle:?}");
            let t = Texture::new(&desc(13, 6, 2, 0, f, swizzle)).unwrap();
            let count = t.subresource_count();
            assert_eq!(count, t.mips() * 2 * t.planes.len() as u32, "{what}");

            let mut tiled = vec![0u8; t.size as usize];
            for i in 0..count {
                let fp = t.footprint(i).unwrap();
                t.tile(
                    &mut tiled,
                    i,
                    &pattern(i, fp.size),
                    fp.row_pitch,
                    fp.depth_pitch,
                )
                .unwrap_or_else(|e| panic!("{what} tile {i}: {e}"));
            }
            for i in 0..count {
                let fp = t.footprint(i).unwrap();
                let mut back = vec![0u8; fp.size];
                t.detile(&tiled, i, &mut back, fp.row_pitch, fp.depth_pitch)
                    .unwrap_or_else(|e| panic!("{what} detile {i}: {e}"));
                assert_eq!(back, pattern(i, fp.size), "{what} subresource {i}");
            }
            assert_eq!(t.footprint(count), None, "{what}");
        }
    }
}

/// A stencil subresource's footprint is one byte per texel, and the depth
/// footprint of the same mip and slice is the format's element size per
/// texel: the two planes copy at their own sizes.
#[test]
fn stencil_footprints_are_one_byte_per_texel() {
    for f in formats()
        .into_iter()
        .filter(|&f| texture::format(f).unwrap().stencil)
    {
        let bytes = texture::format(f).unwrap().bytes as usize;
        let t = Texture::new(&desc(10, 5, 3, 0, f, Swizzle::SW_64KB_D)).unwrap();
        let per_plane = t.mips() * t.array_size();
        for i in 0..per_plane {
            let s = t.subresource(i).unwrap();
            let stencil = t.subresource(i + per_plane).unwrap();
            assert_eq!(
                (stencil.plane, stencil.mip, stencil.slice),
                (1, s.mip, s.slice),
                "format {f}"
            );
            let w = (10usize >> s.mip).max(1);
            let h = (5usize >> s.mip).max(1);
            let depth = t.footprint(i).unwrap();
            let stencil = t.footprint(i + per_plane).unwrap();
            assert_eq!(
                (depth.row_pitch, depth.size),
                (w * bytes, w * h * bytes),
                "format {f} {i}"
            );
            assert_eq!(
                (stencil.row_pitch, stencil.size),
                (w, w * h),
                "format {f} {i}"
            );
        }
    }
}

/// Block-compressed formats copy whole 4×4 blocks: a 1×1 texture still
/// has one block of the format's size, and a 5×5 texture has 2×2 blocks.
#[test]
fn block_compressed_footprints_round_up_to_whole_blocks() {
    for f in formats() {
        let want = texture::format(f).unwrap();
        if want.block != 4 {
            continue;
        }
        let bytes = want.bytes as usize;
        for ((w, h), blocks) in [
            ((1, 1), (1, 1)),
            ((4, 4), (1, 1)),
            ((5, 5), (2, 2)),
            ((9, 3), (3, 1)),
        ] {
            let t = Texture::new(&desc(w, h, 1, 1, f, Swizzle::SW_4KB_S)).unwrap();
            let fp = t.footprint(0).unwrap();
            assert_eq!(fp.row_pitch, blocks.0 * bytes, "format {f} {w}x{h}");
            assert_eq!(fp.size, blocks.0 * blocks.1 * bytes, "format {f} {w}x{h}");
        }
    }
}
