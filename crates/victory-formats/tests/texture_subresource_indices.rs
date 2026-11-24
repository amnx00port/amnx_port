//! D3D12 subresource indices through [`Texture`]: every index of a sweep of
//! shapes maps back to `mip + slice * mips + plane * mips * array_size`, a
//! full chain (`mips: 0`) counts the levels of the largest side, and the
//! whole-texture copies check the index before the buffer. Synthetic
//! textures only; nothing here reads data from outside the repository.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

/// R8G8B8A8_UNORM: one plane of 4-byte texels.
const RGBA8: u32 = 28;
/// D24_UNORM_S8_UINT: a depth plane and a stencil plane.
const D24S8: u32 = 45;
/// BC1_UNORM: 8-byte 4x4 blocks.
const BC1: u32 = 71;

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
    .unwrap_or_else(|e| panic!("{dimension:?} {w}x{h}x{d} mips {mips} format {format}: {e}"))
}

/// Levels of a full chain: down to 1 along the largest side.
fn full_chain(largest: u32) -> u32 {
    32 - largest.max(1).leading_zeros()
}

/// Every index below the count lands on a distinct (plane, mip, slice)
/// that rebuilds the index, and the first index past the end has none.
#[test]
fn every_index_inverts_the_d3d12_formula() {
    let shapes = [
        (Dimension::Tex2d, (16, 16, 1), 1),
        (Dimension::Tex2d, (16, 8, 6), 3),
        (Dimension::Tex2d, (33, 17, 4), 0),
        (Dimension::Tex2d, (1, 1, 7), 1),
        (Dimension::Tex3d, (16, 16, 8), 0),
        (Dimension::Tex3d, (8, 4, 2), 2),
    ];
    for (dimension, size, mips) in shapes {
        for format in [RGBA8, D24S8, BC1] {
            let t = texture(dimension, size, mips, format, Swizzle::SW_4KB_S);
            let (m, a, p) = (t.mips(), t.array_size(), t.planes.len() as u32);
            let what = format!("{dimension:?} {size:?} mips {mips} format {format}");
            assert_eq!(t.subresource_count(), m * a * p, "{what}");
            let mut seen = std::collections::HashSet::new();
            for i in 0..t.subresource_count() {
                let s = t
                    .subresource(i)
                    .unwrap_or_else(|| panic!("{what}: index {i}"));
                assert!(s.mip < m && s.slice < a && s.plane < p, "{what}: {s:?}");
                assert_eq!(s.mip + s.slice * m + s.plane * m * a, i, "{what}: {s:?}");
                assert!(
                    seen.insert((s.plane, s.mip, s.slice)),
                    "{what}: {s:?} twice"
                );
                assert_eq!(
                    t.footprint(i),
                    t.planes[s.plane as usize].layout.footprint(s.mip),
                    "{what}: index {i}"
                );
            }
            let end = t.subresource_count();
            assert_eq!(t.subresource(end), None, "{what}");
            assert_eq!(t.subresource(u32::MAX), None, "{what}");
            assert_eq!(t.footprint(end), None, "{what}");
        }
    }
}

/// `mips: 0` asks for the full chain: an array's size does not lengthen
/// it, a 3D texture's depth does, and every plane gets the same count.
#[test]
fn a_full_chain_follows_the_largest_side() {
    for (w, h, d) in [
        (1, 1, 1),
        (2, 1, 1),
        (17, 3, 1),
        (64, 64, 1),
        (5, 300, 1),
        (4, 4, 64),
    ] {
        let arr = texture(Dimension::Tex2d, (w, h, d), 0, D24S8, Swizzle::SW_64KB_S);
        assert_eq!(arr.mips(), full_chain(w.max(h)), "2D {w}x{h} array {d}");
        assert_eq!(arr.array_size(), d, "2D {w}x{h} array {d}");
        for plane in &arr.planes {
            assert_eq!(plane.layout.mips.len() as u32, arr.mips(), "2D {w}x{h}x{d}");
        }

        let vol = texture(Dimension::Tex3d, (w, h, d), 0, RGBA8, Swizzle::SW_64KB_S);
        assert_eq!(vol.mips(), full_chain(w.max(h).max(d)), "3D {w}x{h}x{d}");
        assert_eq!(vol.array_size(), 1, "3D {w}x{h}x{d}");
        assert_eq!(vol.subresource_count(), vol.mips(), "3D {w}x{h}x{d}");
    }
}

/// A 3D texture's subresource is a whole mip: its footprint spans every
/// depth slice of that mip, halving with the mip like the other sides.
#[test]
fn volume_footprints_span_the_mips_depth() {
    let t = texture(Dimension::Tex3d, (8, 8, 8), 0, RGBA8, Swizzle::SW_4KB_S);
    assert_eq!(t.mips(), 4);
    for mip in 0..4 {
        let side = 8 >> mip;
        let f = t.footprint(mip).unwrap();
        assert_eq!(f.row_pitch, side * 4, "mip {mip}");
        assert_eq!(f.depth_pitch, side * side * 4, "mip {mip}");
        assert_eq!(f.size, side * side * side * 4, "mip {mip}");
    }
}

/// The whole-texture copies refuse an index past the last subresource as
/// invalid before they look at the buffer, refuse a buffer shorter than
/// the texture as too small, and leave the destination alone either way.
#[test]
fn copies_check_the_index_before_the_buffer() {
    let t = texture(Dimension::Tex2d, (8, 8, 2), 2, D24S8, Swizzle::SW_4KB_S);
    let end = t.subresource_count();
    let size = t.size as usize;
    let f = t.footprint(0).unwrap();

    let mut linear = vec![0xa5u8; f.size];
    let short = vec![0u8; size - 1];
    assert_eq!(
        t.detile(&short, end, &mut linear, f.row_pitch, 0),
        Err(Error::Invalid)
    );
    assert_eq!(
        t.detile(&short, 0, &mut linear, f.row_pitch, 0),
        Err(Error::TooSmall)
    );
    assert!(linear.iter().all(|&b| b == 0xa5));

    let mut tiled = vec![0x5au8; size];
    assert_eq!(
        t.tile(&mut tiled, end, &linear, f.row_pitch, 0),
        Err(Error::Invalid)
    );
    let mut short = vec![0x5au8; size - 1];
    assert_eq!(
        t.tile(&mut short, end, &linear, f.row_pitch, 0),
        Err(Error::Invalid)
    );
    assert_eq!(
        t.tile(&mut short, 0, &linear, f.row_pitch, 0),
        Err(Error::TooSmall)
    );
    assert!(tiled.iter().chain(&short).all(|&b| b == 0x5a));

    // The last index, the stencil plane's last mip of its last slice, copies.
    let last = t.footprint(end - 1).unwrap();
    let rows: Vec<u8> = (0..last.size).map(|i| i as u8 ^ 0x3c).collect();
    t.tile(&mut tiled, end - 1, &rows, last.row_pitch, 0)
        .unwrap();
    let mut back = vec![0; last.size];
    t.detile(&tiled, end - 1, &mut back, last.row_pitch, 0)
        .unwrap();
    assert_eq!(back, rows);
}
