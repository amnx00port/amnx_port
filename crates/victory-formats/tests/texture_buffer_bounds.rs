//! Whole-texture buffers through [`Texture::tile`] and [`Texture::detile`]:
//! a tiled buffer longer than [`Texture::size`] copies like one of exactly
//! that size and its extra bytes are never written or read, tiling one
//! subresource writes exactly its footprint's worth of bytes inside its own
//! plane, and the size check covers the whole texture even for a depth
//! subresource that ends before the stencil plane. Synthetic textures only;
//! nothing here reads data from outside the repository.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

/// R8_UNORM, R8G8B8A8_UNORM, D24_UNORM_S8_UINT (two planes), BC1_UNORM
/// (8-byte blocks) and BC7_UNORM (16-byte blocks).
const FORMATS: [u32; 5] = [61, 28, 45, 71, 98];

/// Every mode that addresses a 2D array: linear and the non-XOR tiled
/// modes.
const MODES: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// Odd sides, three array slices and the full chain: partial blocks, mip
/// tails and several slices per plane.
fn texture(format: u32, swizzle: Swizzle) -> Texture {
    Texture::new(&TextureDesc {
        dimension: Dimension::Tex2d,
        width: 37,
        height: 19,
        depth_or_array: 3,
        mips: 0,
        format,
        samples: 1,
        swizzle,
    })
    .unwrap_or_else(|e| panic!("format {format} {swizzle:?}: {e}"))
}

/// Bytes that are never zero and differ between subresources.
fn pattern(index: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(31) ^ index.wrapping_mul(97)) as u8 | 1)
        .collect()
}

/// A tiled buffer with bytes past the texture: tiling every subresource
/// into it never writes those bytes, the texture's own bytes come out
/// identical to tiling into an exact-size buffer, and detiling ignores
/// whatever the extra bytes hold.
#[test]
fn bytes_past_the_texture_are_never_touched() {
    const EXTRA: usize = 4099;
    for format in FORMATS {
        for swizzle in MODES {
            let t = texture(format, swizzle);
            let size = t.size as usize;
            let what = format!("format {format} {swizzle:?}");

            let mut exact = vec![0u8; size];
            let mut longer = vec![0u8; size + EXTRA];
            longer[size..].fill(0xe7);
            for index in 0..t.subresource_count() {
                let f = t.footprint(index).unwrap();
                let rows = pattern(index, f.size);
                t.tile(&mut exact, index, &rows, f.row_pitch, f.depth_pitch)
                    .unwrap_or_else(|e| panic!("{what} index {index}: {e}"));
                t.tile(&mut longer, index, &rows, f.row_pitch, f.depth_pitch)
                    .unwrap_or_else(|e| panic!("{what} index {index}: {e}"));
            }
            assert_eq!(longer[..size], exact[..], "{what}");
            assert!(longer[size..].iter().all(|&b| b == 0xe7), "{what}");

            // Different noise past the end reads back the same rows.
            for (i, b) in longer[size..].iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(13) ^ 0x5c;
            }
            for index in 0..t.subresource_count() {
                let f = t.footprint(index).unwrap();
                let mut from_exact = vec![0u8; f.size];
                let mut from_longer = vec![0xffu8; f.size];
                t.detile(&exact, index, &mut from_exact, f.row_pitch, f.depth_pitch)
                    .unwrap();
                t.detile(&longer, index, &mut from_longer, f.row_pitch, f.depth_pitch)
                    .unwrap();
                assert_eq!(from_longer, from_exact, "{what} index {index}");
                assert_eq!(from_exact, pattern(index, f.size), "{what} index {index}");
            }
        }
    }
}

/// Tiling one subresource of nonzero bytes into a zeroed texture changes
/// exactly as many bytes as its packed footprint holds, every one of them
/// inside its own plane: no element is written twice and none spills into
/// the other plane, the gap between planes or past the texture.
#[test]
fn one_subresource_writes_its_footprint_inside_its_plane() {
    for format in FORMATS {
        for swizzle in MODES {
            let t = texture(format, swizzle);
            let size = t.size as usize;
            for index in 0..t.subresource_count() {
                let what = format!("format {format} {swizzle:?} index {index}");
                let s = t.subresource(index).unwrap();
                let plane = &t.planes[s.plane as usize];
                let owned = plane.offset as usize..(plane.offset + plane.layout.size) as usize;

                let f = t.footprint(index).unwrap();
                let rows = pattern(index, f.size);
                let mut tiled = vec![0u8; size];
                t.tile(&mut tiled, index, &rows, f.row_pitch, f.depth_pitch)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));

                let mut written = 0;
                for (at, &b) in tiled.iter().enumerate() {
                    if b != 0 {
                        assert!(owned.contains(&at), "{what}: byte {at} outside {owned:?}");
                        written += 1;
                    }
                }
                assert_eq!(written, f.size, "{what}");
            }
        }
    }
}

/// A depth subresource lies wholly before the stencil plane, yet a buffer
/// that holds the depth plane but not the whole texture is still too
/// small for it, in both directions, and is left as it was.
#[test]
fn the_size_check_covers_the_whole_texture_not_one_plane() {
    for swizzle in MODES {
        let t = texture(45, swizzle);
        let depth = &t.planes[0];
        let what = format!("{swizzle:?}");
        assert!(depth.offset + depth.layout.size < t.size, "{what}");

        let f = t.footprint(0).unwrap();
        let rows = pattern(0, f.size);
        let mut short = vec![0x33u8; depth.layout.size as usize];
        assert_eq!(
            t.tile(&mut short, 0, &rows, f.row_pitch, f.depth_pitch),
            Err(Error::TooSmall),
            "{what}"
        );
        assert!(short.iter().all(|&b| b == 0x33), "{what}");

        let mut linear = vec![0x44u8; f.size];
        assert_eq!(
            t.detile(&short, 0, &mut linear, f.row_pitch, f.depth_pitch),
            Err(Error::TooSmall),
            "{what}"
        );
        assert!(linear.iter().all(|&b| b == 0x44), "{what}");

        // One byte more than the texture is enough for the same copy.
        let mut enough = vec![0u8; t.size as usize + 1];
        t.tile(&mut enough, 0, &rows, f.row_pitch, f.depth_pitch)
            .unwrap();
        t.detile(&enough, 0, &mut linear, f.row_pitch, f.depth_pitch)
            .unwrap();
        assert_eq!(linear, rows, "{what}");
    }
}
