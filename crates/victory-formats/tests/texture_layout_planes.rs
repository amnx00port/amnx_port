//! A sweep of synthetic depth-stencil textures, whose stencil is a second
//! plane, through every addressable swizzle mode: odd and power-of-two
//! sizes, arrays and full mip chains. Each plane is laid out exactly as a
//! one-plane texture of its element size would be, the stencil plane starts
//! at the first multiple of its alignment past the depth plane, and every
//! element of every subresource of both planes owns its own bytes. Nothing
//! here needs data from outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

/// D32_FLOAT_S8X24_UINT, D24_UNORM_S8_UINT and the console's 16-bit depth
/// with 8-bit stencil, with the depth element size and a one-plane format
/// of that size (R32_FLOAT, R32_FLOAT, R16_FLOAT).
const DEPTH_STENCIL: [(u32, u32, u32); 3] = [(20, 4, 41), (45, 4, 41), (118, 2, 54)];

/// R8_UNORM: a one-plane format with the stencil's element size.
const R8: u32 = 61;

const MODES: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// Width, height and array size.
const SHAPES: [(u32, u32, u32); 6] = [
    (1, 1, 1),
    (3, 2, 1),
    (17, 9, 3),
    (64, 64, 1),
    (100, 1, 2),
    (1, 75, 1),
];

fn texture((w, h, a): (u32, u32, u32), format: u32, swizzle: Swizzle) -> Texture {
    Texture::new(&TextureDesc {
        dimension: Dimension::Tex2d,
        width: w,
        height: h,
        depth_or_array: a,
        mips: 0,
        format,
        samples: 1,
        swizzle,
    })
    .unwrap_or_else(|e| panic!("{w}x{h}x{a} format {format} {swizzle:?}: {e}"))
}

#[test]
fn each_plane_is_laid_out_like_a_one_plane_texture_of_its_size() {
    for &(format, depth_bytes, depth_like) in &DEPTH_STENCIL {
        for &swizzle in &MODES {
            for &shape in &SHAPES {
                let what = format!("{shape:?} format {format} {swizzle:?}");
                let t = texture(shape, format, swizzle);
                assert_eq!(t.planes.len(), 2, "{what}");
                let (depth, stencil) = (&t.planes[0], &t.planes[1]);
                assert_eq!((depth.offset, depth.stencil), (0, false), "{what}");
                assert!(stencil.stencil, "{what}");
                assert_eq!(depth.layout.desc.element_bytes, depth_bytes, "{what}");
                assert_eq!(stencil.layout.desc.element_bytes, 1, "{what}");

                let alone = texture(shape, depth_like, swizzle);
                assert_eq!(depth.layout, alone.planes[0].layout, "{what}: depth");
                let alone = texture(shape, R8, swizzle);
                assert_eq!(stencil.layout, alone.planes[0].layout, "{what}: stencil");

                // The stencil starts at the first multiple of its own
                // alignment at or past the end of the depth plane, and the
                // texture ends with it.
                let a = stencil.layout.alignment;
                assert_eq!(stencil.offset % a, 0, "{what}");
                assert!(stencil.offset >= depth.layout.size, "{what}");
                assert!(stencil.offset - depth.layout.size < a, "{what}");
                assert_eq!(t.size, stencil.offset + stencil.layout.size, "{what}");
                assert_eq!(
                    t.alignment,
                    depth.layout.alignment.max(stencil.layout.alignment),
                    "{what}"
                );
                assert_eq!(t.mips(), depth.layout.mips.len() as u32, "{what}");
                assert_eq!(t.mips(), stencil.layout.mips.len() as u32, "{what}");
            }
        }
    }
}

/// Every element of every subresource of both planes, offset by its plane,
/// lies inside the texture at a multiple of its size, and no two share a
/// byte: the planes never overlap.
#[test]
fn both_planes_own_their_bytes_in_every_mode() {
    for &(format, _, _) in &DEPTH_STENCIL {
        for &swizzle in &MODES {
            for &shape in &SHAPES {
                let what = format!("{shape:?} format {format} {swizzle:?}");
                let t = texture(shape, format, swizzle);
                let mut seen = HashSet::new();
                for p in &t.planes {
                    let l = &p.layout;
                    let n = u64::from(l.desc.element_bytes);
                    for (mip, m) in l.mips.iter().enumerate() {
                        for z in 0..l.slices {
                            for y in 0..m.height {
                                for x in 0..m.width {
                                    let at = p.offset + l.offset(mip as u32, x, y, z).unwrap();
                                    let here = format!(
                                        "{what}: plane {} mip {mip} ({x}, {y}, {z})",
                                        p.offset
                                    );
                                    assert_eq!(at % n, 0, "{here}");
                                    assert!(at + n <= t.size, "{here}");
                                    for byte in at..at + n {
                                        assert!(seen.insert(byte), "{here} shares {byte}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Writes a distinct odd pattern into every subresource of both planes,
/// then reads each one back: nothing written to one plane reaches the
/// other, and only padding stays zero.
#[test]
fn every_subresource_of_both_planes_round_trips() {
    for &(format, depth_bytes, _) in &DEPTH_STENCIL {
        for &swizzle in &MODES {
            for &(w, h, a) in &SHAPES {
                let what = format!("{w}x{h}x{a} format {format} {swizzle:?}");
                let t = texture((w, h, a), format, swizzle);
                assert_eq!(t.subresource_count(), t.mips() * a * 2, "{what}");
                let mut tiled = vec![0u8; t.size as usize];
                let mut written = Vec::new();
                for index in 0..t.subresource_count() {
                    let s = t.subresource(index).unwrap();
                    assert_eq!(s.plane, index / (t.mips() * a), "{what}: {index}");
                    let f = t.footprint(index).unwrap();
                    let bytes = if s.plane == 0 { depth_bytes } else { 1 };
                    let width = (w >> s.mip).max(1) as usize;
                    let height = (h >> s.mip).max(1) as usize;
                    assert_eq!(f.row_pitch, width * bytes as usize, "{what}: {index}");
                    assert_eq!(f.size, f.row_pitch * height, "{what}: {index}");
                    let data: Vec<u8> = (0..f.size)
                        .map(|i| (i as u32).wrapping_mul(13).wrapping_add(index * 41) as u8 | 1)
                        .collect();
                    t.tile(&mut tiled, index, &data, f.row_pitch, f.depth_pitch)
                        .unwrap_or_else(|e| panic!("{what}: tile {index}: {e}"));
                    written.push(data);
                }
                for (index, data) in written.iter().enumerate() {
                    let f = t.footprint(index as u32).unwrap();
                    let mut back = vec![0u8; f.size];
                    t.detile(&tiled, index as u32, &mut back, f.row_pitch, f.depth_pitch)
                        .unwrap_or_else(|e| panic!("{what}: detile {index}: {e}"));
                    assert_eq!(&back, data, "{what}: subresource {index}");
                }
                let owned: usize = written.iter().map(Vec::len).sum();
                let nonzero = tiled.iter().filter(|&&b| b != 0).count();
                assert_eq!(nonzero, owned, "{what}");
                // The gap between the planes is padding nothing writes.
                let gap = t.planes[0].layout.size as usize..t.planes[1].offset as usize;
                assert!(tiled[gap].iter().all(|&b| b == 0), "{what}");
            }
        }
    }
}

/// Copying a stencil subresource through the texture gives the same bytes,
/// at the stencil plane's offset, as copying it into an R8 texture alone.
#[test]
fn stencil_copies_match_a_lone_eight_bit_texture() {
    for &swizzle in &MODES {
        let shape = (19, 7, 2);
        let what = format!("{swizzle:?}");
        let t = texture(shape, 45, swizzle);
        let alone = texture(shape, R8, swizzle);
        let per_plane = t.mips() * shape.2;
        let mut tiled = vec![0u8; t.size as usize];
        let mut lone = vec![0u8; alone.size as usize];
        for index in 0..per_plane {
            let f = t.footprint(per_plane + index).unwrap();
            assert_eq!(Some(f), alone.footprint(index), "{what}: {index}");
            let data: Vec<u8> = (0..f.size)
                .map(|i| (i as u8) ^ (index as u8 * 7) | 1)
                .collect();
            t.tile(&mut tiled, per_plane + index, &data, f.row_pitch, 0)
                .unwrap();
            alone.tile(&mut lone, index, &data, f.row_pitch, 0).unwrap();
        }
        let at = t.planes[1].offset as usize;
        assert_eq!(&tiled[at..], &lone[..], "{what}");
        assert!(tiled[..at].iter().all(|&b| b == 0), "{what}");
    }
}

/// The XOR modes size both planes like their base mode but address
/// neither: every copy, of either plane, is unsupported, and a bad index
/// or short buffer is still reported first.
#[test]
fn xor_depth_stencil_is_sized_but_neither_plane_copies() {
    let pairs = [
        (Swizzle::SW_4KB_S_X, Swizzle::SW_4KB_S),
        (Swizzle::SW_4KB_D_X, Swizzle::SW_4KB_D),
        (Swizzle::SW_64KB_S_X, Swizzle::SW_64KB_S),
        (Swizzle::SW_64KB_D_X, Swizzle::SW_64KB_D),
        (Swizzle::SW_64KB_S_T, Swizzle::SW_64KB_S),
        (Swizzle::SW_64KB_D_T, Swizzle::SW_64KB_D),
    ];
    for (xor, base) in pairs {
        let what = format!("{xor:?}");
        let t = texture((33, 17, 2), 45, xor);
        let b = texture((33, 17, 2), 45, base);
        assert_eq!(t.size, b.size, "{what}");
        assert_eq!(t.alignment, b.alignment, "{what}");
        assert_eq!(t.planes[1].offset, b.planes[1].offset, "{what}");
        assert!(!t.planes[0].layout.addressable(), "{what}");
        assert!(!t.planes[1].layout.addressable(), "{what}");

        let mut tiled = vec![0u8; t.size as usize];
        for index in 0..t.subresource_count() {
            let f = t.footprint(index).unwrap();
            let data = vec![1u8; f.size];
            assert_eq!(
                t.tile(&mut tiled, index, &data, f.row_pitch, f.depth_pitch),
                Err(Error::Unsupported),
                "{what}: {index}"
            );
        }
        assert!(tiled.iter().all(|&b| b == 0), "{what}");
        let count = t.subresource_count();
        assert_eq!(
            t.detile(&tiled, count, &mut [0; 64], 64, 0),
            Err(Error::Invalid),
            "{what}"
        );
        assert_eq!(
            t.detile(&tiled[..tiled.len() - 1], 0, &mut [0; 64], 64, 0),
            Err(Error::TooSmall),
            "{what}"
        );
    }
}
