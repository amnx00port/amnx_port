//! Moving a whole texture from one swizzle mode to another through
//! [`Texture`]: detiling every subresource out of one mode and tiling it
//! into another keeps every element, and going back gives the first
//! texture's bytes exactly. Modes whose micro tiles share a pattern (2-byte
//! standard and display) give byte-identical textures. Every texture is
//! synthetic and filled with a deterministic pattern, so nothing here needs
//! data from outside the repository.

use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

/// R32G32B32A32_FLOAT: 16-byte texels.
const RGBA32F: u32 = 2;
/// R8G8B8A8_UNORM: 4-byte texels.
const RGBA8: u32 = 28;
/// D24_UNORM_S8_UINT: a depth plane and a stencil plane.
const D24S8: u32 = 45;
/// R16_UNORM: 2-byte texels.
const R16: u32 = 56;
/// R8_UNORM: 1-byte texels.
const R8: u32 = 61;
/// BC1_UNORM: 8-byte 4x4 blocks.
const BC1: u32 = 71;
/// BC7_UNORM: 16-byte 4x4 blocks.
const BC7: u32 = 98;

/// The addressable 2D modes: linear and every non-XOR GFX10 mode.
const MODES_2D: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// The addressable 3D modes: linear and the thick standard ones.
const MODES_3D: [Swizzle; 3] = [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S];

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
    .unwrap_or_else(|e| panic!("{dimension:?} {w}x{h}x{d} format {format} {swizzle:?}: {e}"))
}

/// Packed linear rows for subresource `index`, distinct per subresource
/// and per byte.
fn pattern(t: &Texture, index: u32) -> Vec<u8> {
    let f = t.footprint(index).unwrap();
    (0..f.size)
        .map(|i| {
            (i as u32)
                .wrapping_mul(97)
                .wrapping_add(index.wrapping_mul(31)) as u8
                | 1
        })
        .collect()
}

/// A zeroed texture with every subresource tiled from [`pattern`].
fn fill(t: &Texture) -> Vec<u8> {
    let mut tiled = vec![0u8; t.size as usize];
    for i in 0..t.subresource_count() {
        let f = t.footprint(i).unwrap();
        t.tile(&mut tiled, i, &pattern(t, i), f.row_pitch, f.depth_pitch)
            .unwrap();
    }
    tiled
}

/// Every subresource of `from` (holding `data`) detiled and tiled into a
/// zeroed texture laid out as `to`.
fn transcode(from: &Texture, data: &[u8], to: &Texture) -> Vec<u8> {
    assert_eq!(from.subresource_count(), to.subresource_count());
    let mut out = vec![0u8; to.size as usize];
    for i in 0..from.subresource_count() {
        let f = from.footprint(i).unwrap();
        assert_eq!(Some(f), to.footprint(i), "subresource {i}");
        let mut linear = vec![0u8; f.size];
        from.detile(data, i, &mut linear, f.row_pitch, f.depth_pitch)
            .unwrap();
        to.tile(&mut out, i, &linear, f.row_pitch, f.depth_pitch)
            .unwrap();
    }
    out
}

/// Detiles every subresource of `t` and checks it against [`pattern`].
fn check(t: &Texture, data: &[u8], what: &str) {
    for i in 0..t.subresource_count() {
        let f = t.footprint(i).unwrap();
        let mut linear = vec![0u8; f.size];
        t.detile(data, i, &mut linear, f.row_pitch, f.depth_pitch)
            .unwrap();
        assert!(linear == pattern(t, i), "{what} subresource {i}");
    }
}

/// Every pair of addressable 2D modes, on odd-sized arrays with full
/// chains (partial blocks, mip tails, several slices) and on formats of
/// every element size, block compression and a stencil plane: A to B keeps
/// every subresource, and B back to A is byte-identical to A.
#[test]
fn two_d_textures_survive_every_mode_pair() {
    let shapes = [((19, 7, 3), 0), ((64, 64, 1), 0), ((5, 33, 2), 3)];
    for (size, mips) in shapes {
        for format in [R8, R16, RGBA8, RGBA32F, BC1, BC7, D24S8] {
            for a in MODES_2D {
                let ta = texture(Dimension::Tex2d, size, mips, format, a);
                let da = fill(&ta);
                for b in MODES_2D {
                    let what = format!("{size:?} mips {mips} format {format} {a:?} -> {b:?}");
                    let tb = texture(Dimension::Tex2d, size, mips, format, b);
                    let db = transcode(&ta, &da, &tb);
                    check(&tb, &db, &what);
                    assert!(transcode(&tb, &db, &ta) == da, "{what} and back");
                }
            }
        }
    }
}

/// The same for volumes: linear depth slices and thick slabs move into
/// each other without losing or moving an element, including a depth that
/// is not a whole number of slabs.
#[test]
fn volumes_survive_every_mode_pair() {
    for size in [(16, 16, 16), (9, 5, 11), (4, 4, 64)] {
        for format in [R8, RGBA8, BC1] {
            for a in MODES_3D {
                let ta = texture(Dimension::Tex3d, size, 0, format, a);
                let da = fill(&ta);
                for b in MODES_3D {
                    let what = format!("{size:?} format {format} {a:?} -> {b:?}");
                    let tb = texture(Dimension::Tex3d, size, 0, format, b);
                    let db = transcode(&ta, &da, &tb);
                    check(&tb, &db, &what);
                    assert!(transcode(&tb, &db, &ta) == da, "{what} and back");
                }
            }
        }
    }
}

/// 1D textures and arrays of them move between linear and tiled modes the
/// same way.
#[test]
fn one_d_arrays_survive_every_mode_pair() {
    for size in [(300, 1, 1), (17, 1, 5)] {
        for a in MODES_2D {
            let ta = texture(Dimension::Tex1d, size, 0, RGBA8, a);
            let da = fill(&ta);
            for b in MODES_2D {
                let what = format!("1D {size:?} {a:?} -> {b:?}");
                let tb = texture(Dimension::Tex1d, size, 0, RGBA8, b);
                let db = transcode(&ta, &da, &tb);
                check(&tb, &db, &what);
                assert!(transcode(&tb, &db, &ta) == da, "{what} and back");
            }
        }
    }
}

/// 2-byte standard and display micro tiles share one pattern and larger
/// blocks stack them the same way, so a whole 2-byte texture (tail
/// included) is byte-for-byte the same in either mode of each block size,
/// while 4-byte textures, whose patterns differ, are not.
#[test]
fn two_byte_standard_and_display_textures_are_identical() {
    let pairs = [
        (Swizzle::SW_256B_S, Swizzle::SW_256B_D),
        (Swizzle::SW_4KB_S, Swizzle::SW_4KB_D),
        (Swizzle::SW_64KB_S, Swizzle::SW_64KB_D),
    ];
    for (s, d) in pairs {
        for (size, mips) in [((64, 64, 2), 0), ((23, 9, 1), 0), ((256, 128, 1), 4)] {
            let ts = texture(Dimension::Tex2d, size, mips, R16, s);
            let td = texture(Dimension::Tex2d, size, mips, R16, d);
            assert_eq!(ts.size, td.size, "{size:?} {s:?}");
            assert!(fill(&ts) == fill(&td), "{size:?} {s:?} / {d:?}");

            let ts = texture(Dimension::Tex2d, size, mips, RGBA8, s);
            let td = texture(Dimension::Tex2d, size, mips, RGBA8, d);
            assert!(fill(&ts) != fill(&td), "{size:?} {s:?} / {d:?} RGBA8");
        }
    }
}
