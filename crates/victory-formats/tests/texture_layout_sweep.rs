//! A sweep of synthetic textures through every addressable swizzle mode:
//! power-of-two and odd sizes, arrays, full mip chains, every element size
//! and block compression. For each texture the tests check invariants that
//! hold whatever the mode: every element of every subresource has its own
//! aligned bytes inside the texture, the packed footprint is the mip's
//! elements times their size, and writing every subresource then reading
//! each one back gives the data written. Nothing here needs data from
//! outside the repository.

use std::collections::HashSet;

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

/// R32G32B32A32_FLOAT, R32G32_FLOAT, R8G8B8A8_UNORM, R16_FLOAT, R8_UNORM,
/// BC1_UNORM (8-byte 4×4 blocks) and BC3_UNORM (16-byte 4×4 blocks).
const FORMATS: [(u32, u32, u32); 7] = [
    (2, 16, 1),
    (16, 8, 1),
    (28, 4, 1),
    (54, 2, 1),
    (61, 1, 1),
    (71, 8, 4),
    (77, 16, 4),
];

const MODES_2D: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// Width, height and array size: powers of two, odd sizes, one-texel and
/// one-row extremes, and sizes just past a block edge.
const SHAPES_2D: [(u32, u32, u32); 7] = [
    (1, 1, 1),
    (7, 3, 2),
    (16, 16, 1),
    (17, 9, 3),
    (33, 1, 1),
    (1, 40, 2),
    (64, 20, 1),
];

fn texture(
    dimension: Dimension,
    (w, h, d): (u32, u32, u32),
    format: u32,
    swizzle: Swizzle,
) -> Texture {
    Texture::new(&TextureDesc {
        dimension,
        width: w,
        height: h,
        depth_or_array: d,
        mips: 0,
        format,
        samples: 1,
        swizzle,
    })
    .unwrap_or_else(|e| panic!("{dimension:?} {w}x{h}x{d} format {format} {swizzle:?}: {e}"))
}

/// Elements along one side of mip `mip` as copies see them.
fn elements(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// Every element of every subresource lies inside the texture, starts at
/// a multiple of its size, and no two elements share a byte.
fn assert_elements_own_their_bytes(t: &Texture, what: &str) {
    let l = &t.planes[0].layout;
    let n = u64::from(l.desc.element_bytes);
    let depth = |mip: u32| {
        if l.desc.dimension == Dimension::Tex3d {
            l.mips[mip as usize].depth
        } else {
            l.slices
        }
    };
    let mut seen = HashSet::new();
    for mip in 0..t.mips() {
        let m = l.mips[mip as usize];
        for z in 0..depth(mip) {
            for y in 0..m.height {
                for x in 0..m.width {
                    let at = l.offset(mip, x, y, z).unwrap();
                    assert_eq!(at % n, 0, "{what}: mip {mip} ({x}, {y}, {z}) at {at}");
                    assert!(
                        at + n <= l.size,
                        "{what}: mip {mip} ({x}, {y}, {z}) at {at}"
                    );
                    assert!(
                        seen.insert(at),
                        "{what}: mip {mip} ({x}, {y}, {z}) shares {at}"
                    );
                }
            }
        }
    }
}

/// The packed footprint of each subresource is its mip's elements times
/// the element size, with every depth slice of a volume one after another.
fn assert_footprints(t: &Texture, (w, h, d): (u32, u32, u32), bytes: u32, block: u32, what: &str) {
    let volume = t.planes[0].layout.desc.dimension == Dimension::Tex3d;
    for index in 0..t.subresource_count() {
        let s = t.subresource(index).unwrap();
        let f = t.footprint(index).unwrap();
        let row = (elements(w, s.mip, block) * bytes) as usize;
        let rows = elements(h, s.mip, block) as usize;
        let slices = if volume {
            (d >> s.mip).max(1) as usize
        } else {
            1
        };
        assert_eq!(f.row_pitch, row, "{what}: subresource {index}");
        assert_eq!(f.depth_pitch, row * rows, "{what}: subresource {index}");
        assert_eq!(f.size, row * rows * slices, "{what}: subresource {index}");
    }
    assert_eq!(t.footprint(t.subresource_count()), None, "{what}");
}

/// Writes a distinct pattern into every subresource, then reads each one
/// back: subresources never overwrite each other, and the bytes no
/// subresource owns stay as they were.
fn assert_subresources_round_trip(t: &Texture, what: &str) {
    let mut tiled = vec![0u8; t.size as usize];
    let mut written = Vec::new();
    for index in 0..t.subresource_count() {
        let f = t.footprint(index).unwrap();
        let data: Vec<u8> = (0..f.size)
            .map(|i| (i as u32).wrapping_mul(31).wrapping_add(index * 97 + 1) as u8 | 1)
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
    // Every byte written is odd, so the zero bytes left are exactly the
    // padding: the element count tells how many bytes were written.
    let l = &t.planes[0].layout;
    let owned: usize = written.iter().map(Vec::len).sum();
    let nonzero = tiled.iter().filter(|&&b| b != 0).count();
    assert_eq!(nonzero, owned, "{what}: bytes written ({})", l.size);
}

#[test]
fn every_two_dimensional_shape_owns_its_bytes_in_every_mode() {
    for &(format, bytes, block) in &FORMATS {
        for &swizzle in &MODES_2D {
            for &shape in &SHAPES_2D {
                let what = format!("{shape:?} format {format} {swizzle:?}");
                let t = texture(Dimension::Tex2d, shape, format, swizzle);
                assert_eq!(t.planes.len(), 1, "{what}");
                assert_eq!(t.size % t.alignment, 0, "{what}");
                assert_elements_own_their_bytes(&t, &what);
                assert_footprints(&t, shape, bytes, block, &what);
                assert_subresources_round_trip(&t, &what);
            }
        }
    }
}

/// Volumes in the addressable modes with a 3D layout: thin (linear) and
/// thick (the 4 KiB and 64 KiB standard blocks), with depths that are and
/// are not a multiple of the thick block's depth.
#[test]
fn every_volume_shape_owns_its_bytes_in_every_mode() {
    let modes = [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S];
    let shapes = [(1, 1, 1), (4, 4, 4), (5, 3, 9), (8, 2, 17), (2, 9, 3)];
    for &(format, bytes, block) in &FORMATS[..5] {
        for &swizzle in &modes {
            for &shape in &shapes {
                let what = format!("volume {shape:?} format {format} {swizzle:?}");
                let t = texture(Dimension::Tex3d, shape, format, swizzle);
                assert_eq!(t.array_size(), 1, "{what}");
                assert_eq!(t.size % t.alignment, 0, "{what}");
                assert_elements_own_their_bytes(&t, &what);
                assert_footprints(&t, shape, bytes, block, &what);
                assert_subresources_round_trip(&t, &what);
            }
        }
    }
}

/// 1D textures and arrays of them: one row per mip, laid out like a 2D
/// texture of height 1 in every mode.
#[test]
fn every_one_dimensional_shape_owns_its_bytes_in_every_mode() {
    let shapes = [(1, 1, 1), (5, 1, 3), (64, 1, 1), (100, 1, 2)];
    for &(format, bytes, block) in &FORMATS[..5] {
        for &swizzle in &MODES_2D {
            for &shape in &shapes {
                let what = format!("1D {shape:?} format {format} {swizzle:?}");
                let t = texture(Dimension::Tex1d, shape, format, swizzle);
                assert_eq!(t.array_size(), shape.2, "{what}");
                assert_elements_own_their_bytes(&t, &what);
                assert_footprints(&t, shape, bytes, block, &what);
                assert_subresources_round_trip(&t, &what);
            }
        }
    }
}

/// A packed footprint is also the smallest buffer a copy takes: one byte
/// less is refused for every subresource of every mode, and the tiled
/// memory must be the whole texture.
#[test]
fn footprints_are_the_smallest_buffers_in_every_mode() {
    for &swizzle in &MODES_2D {
        let what = format!("{swizzle:?}");
        let t = texture(Dimension::Tex2d, (13, 6, 2), 28, swizzle);
        let mut tiled = vec![0u8; t.size as usize];
        for index in 0..t.subresource_count() {
            let f = t.footprint(index).unwrap();
            let data = vec![0x5a; f.size];
            assert_eq!(
                t.tile(
                    &mut tiled,
                    index,
                    &data[..f.size - 1],
                    f.row_pitch,
                    f.depth_pitch
                ),
                Err(Error::TooSmall),
                "{what}: subresource {index}"
            );
            assert_eq!(
                t.tile(
                    &mut tiled[..t.size as usize - 1],
                    index,
                    &data,
                    f.row_pitch,
                    f.depth_pitch
                ),
                Err(Error::TooSmall),
                "{what}: subresource {index}"
            );
            if f.row_pitch > 4 {
                assert_eq!(
                    t.tile(&mut tiled, index, &data, f.row_pitch - 4, f.depth_pitch),
                    Err(Error::TooSmall),
                    "{what}: subresource {index} row pitch"
                );
            }
            t.tile(&mut tiled, index, &data, f.row_pitch, f.depth_pitch)
                .unwrap();
        }
    }
}
