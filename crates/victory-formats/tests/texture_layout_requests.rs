//! How [`Texture::new`] reads a request and how the whole-texture copies
//! check theirs: a sample count of 0 means 1, a mip count of 0 means the
//! full chain (counted from width, height and 3D depth, never array size),
//! longer chains end in 1x1 mips up to 16, and a copy checks its subresource
//! index before the buffer size and never writes past [`Texture::size`].
//! Every expectation is worked out by hand, so nothing here needs data from
//! outside the repository.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const D24S8: u32 = 45;
const BC1: u32 = 71;

const MODES: [Swizzle; 4] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
];

#[allow(clippy::too_many_arguments)]
fn desc(
    dimension: Dimension,
    w: u32,
    h: u32,
    d: u32,
    mips: u32,
    format: u32,
    samples: u32,
    swizzle: Swizzle,
) -> TextureDesc {
    TextureDesc {
        dimension,
        width: w,
        height: h,
        depth_or_array: d,
        mips,
        format,
        samples,
        swizzle,
    }
}

/// A sample count of 0 asks for a single-sampled texture: the same planes,
/// sizes and alignment as asking for 1, with 1 in each plane's description.
#[test]
fn zero_samples_means_one() {
    for swizzle in MODES {
        for format in [R8G8B8A8, D24S8, BC1] {
            let zero = Texture::new(&desc(Dimension::Tex2d, 13, 7, 2, 0, format, 0, swizzle));
            let one = Texture::new(&desc(Dimension::Tex2d, 13, 7, 2, 0, format, 1, swizzle));
            assert_eq!(zero, one, "format {format} {swizzle:?}");
            for p in &zero.unwrap().planes {
                assert_eq!(p.layout.desc.samples, 1, "format {format} {swizzle:?}");
            }
        }
    }
}

/// Mip count 0 is the full chain: one mip per halving of the largest of
/// width, height and (3D only) depth, down to 1. It equals asking for that
/// count by number.
#[test]
fn zero_mips_is_the_full_chain() {
    use Dimension::*;
    // (dimension, width, height, depth or array, full chain)
    let cases = [
        (Tex1d, 1, 1, 1, 1),
        (Tex1d, 2, 1, 1, 2),
        (Tex1d, 3, 1, 4, 2),
        (Tex1d, 1024, 1, 1, 11),
        (Tex2d, 1, 1, 1, 1),
        (Tex2d, 7, 1, 1, 3),
        (Tex2d, 8, 1, 1, 4),
        (Tex2d, 5, 17, 1, 5),
        // An array's size never adds mips.
        (Tex2d, 4, 4, 64, 3),
        (Tex2d, 1, 1, 256, 1),
        // A 3D texture's depth does.
        (Tex3d, 4, 4, 64, 7),
        (Tex3d, 1, 1, 9, 4),
        (Tex3d, 16, 2, 2, 5),
    ];
    for (dimension, w, h, d, chain) in cases {
        let what = format!("{dimension:?} {w}x{h}x{d}");
        let full =
            Texture::new(&desc(dimension, w, h, d, 0, R8G8B8A8, 1, Swizzle::LINEAR)).unwrap();
        assert_eq!(full.mips(), chain, "{what}");
        let named = Texture::new(&desc(
            dimension,
            w,
            h,
            d,
            chain,
            R8G8B8A8,
            1,
            Swizzle::LINEAR,
        ))
        .unwrap();
        assert_eq!(full, named, "{what}");
    }
}

/// Asking for more mips than the full chain is allowed up to 16: the extra
/// mips are 1x1 (one element) and copy like any other; 17 makes no texture.
#[test]
fn chains_past_the_full_chain_end_in_single_elements() {
    for (dimension, w, h, d, chain, modes) in [
        (Dimension::Tex2d, 9, 3, 2, 4, &MODES[..]),
        (
            Dimension::Tex3d,
            2,
            2,
            8,
            4,
            &[Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S][..],
        ),
    ] {
        for &swizzle in modes {
            let what = format!("{dimension:?} {w}x{h}x{d} {swizzle:?}");
            for mips in 1..=16 {
                let t =
                    Texture::new(&desc(dimension, w, h, d, mips, R8G8B8A8, 1, swizzle)).unwrap();
                assert_eq!(t.mips(), mips, "{what} mips {mips}");
                assert_eq!(t.subresource_count(), mips * t.array_size(), "{what}");
                let l = &t.planes[0].layout;
                for m in chain..mips {
                    let m = &l.mips[m as usize];
                    assert_eq!(
                        (m.width, m.height, m.depth),
                        (1, 1, 1),
                        "{what} mips {mips}"
                    );
                }
                let mut tiled = vec![0; t.size as usize];
                for index in 0..t.subresource_count() {
                    let fp = t.footprint(index).unwrap();
                    let linear = vec![index as u8 + 1; fp.size];
                    t.tile(&mut tiled, index, &linear, fp.row_pitch, fp.depth_pitch)
                        .unwrap();
                    let mut out = vec![0; fp.size];
                    t.detile(&tiled, index, &mut out, fp.row_pitch, fp.depth_pitch)
                        .unwrap();
                    assert_eq!(out, linear, "{what} mips {mips} index {index}");
                }
            }
            assert_eq!(
                Texture::new(&desc(dimension, w, h, d, 17, R8G8B8A8, 1, swizzle)),
                Err(Error::Invalid),
                "{what}"
            );
        }
    }
}

/// A copy checks the subresource index first: past the last subresource it
/// is `Invalid` whatever the buffer, and only a real subresource with a
/// buffer shorter than the texture is `TooSmall`. Neither failure writes.
#[test]
fn copies_check_the_index_before_the_buffer() {
    for swizzle in MODES {
        let t = Texture::new(&desc(Dimension::Tex2d, 6, 5, 2, 0, D24S8, 1, swizzle)).unwrap();
        let what = format!("{swizzle:?}");
        let past = t.subresource_count();
        let size = t.size as usize;
        let fp = t.footprint(0).unwrap();
        let linear = vec![0x5a; fp.size];

        for len in [0, size - 1, size] {
            let mut tiled = vec![0xcc; len];
            assert_eq!(
                t.tile(&mut tiled, past, &linear, fp.row_pitch, fp.depth_pitch),
                Err(Error::Invalid),
                "{what} len {len}"
            );
            assert!(tiled.iter().all(|&b| b == 0xcc), "{what} len {len}");
            let mut out = vec![0xcc; fp.size];
            assert_eq!(
                t.detile(&tiled, past, &mut out, fp.row_pitch, fp.depth_pitch),
                Err(Error::Invalid),
                "{what} len {len}"
            );
            assert!(out.iter().all(|&b| b == 0xcc), "{what} len {len}");
        }

        // The stencil plane's last subresource needs the whole texture too,
        // and so does plane 0's first.
        for index in [0, past - 1] {
            let fp = t.footprint(index).unwrap();
            let linear = vec![0x5a; fp.size];
            let mut tiled = vec![0xcc; size - 1];
            assert_eq!(
                t.tile(&mut tiled, index, &linear, fp.row_pitch, fp.depth_pitch),
                Err(Error::TooSmall),
                "{what} index {index}"
            );
            assert!(tiled.iter().all(|&b| b == 0xcc), "{what} index {index}");
            let mut out = vec![0xcc; fp.size];
            assert_eq!(
                t.detile(&tiled, index, &mut out, fp.row_pitch, fp.depth_pitch),
                Err(Error::TooSmall),
                "{what} index {index}"
            );
            assert!(out.iter().all(|&b| b == 0xcc), "{what} index {index}");
        }
    }
}

/// A buffer longer than the texture is accepted, every subresource lands
/// inside the first [`Texture::size`] bytes, and the bytes after it are
/// never written; copying each subresource back out returns what went in.
#[test]
fn copies_never_write_past_the_texture() {
    const SLACK: usize = 300;
    for swizzle in MODES {
        for format in [R8G8B8A8, D24S8, BC1] {
            let t = Texture::new(&desc(Dimension::Tex2d, 11, 9, 3, 0, format, 1, swizzle)).unwrap();
            let what = format!("format {format} {swizzle:?}");
            let size = t.size as usize;
            let mut tiled = vec![0xcc; size + SLACK];
            for index in 0..t.subresource_count() {
                let fp = t.footprint(index).unwrap();
                let linear: Vec<u8> = (0..fp.size)
                    .map(|i| (i as u32 ^ index.wrapping_mul(97)) as u8 | 1)
                    .collect();
                t.tile(&mut tiled, index, &linear, fp.row_pitch, fp.depth_pitch)
                    .unwrap();
                let mut out = vec![0; fp.size];
                t.detile(&tiled, index, &mut out, fp.row_pitch, fp.depth_pitch)
                    .unwrap();
                assert_eq!(out, linear, "{what} index {index}");
            }
            assert!(tiled[size..].iter().all(|&b| b == 0xcc), "{what}");
        }
    }
}
