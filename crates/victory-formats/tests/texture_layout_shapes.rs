//! Odd shapes through the public API: chain lengths of non-square and
//! volume textures, odd sizes whose placement rounds up while their data
//! rounds down, 1D textures in 2D blocks, the 16-bit depth tail fix and
//! whole-texture buffer checks. Every expected number is worked out by hand
//! from the layout rules, so nothing here needs data from outside the
//! repository.

use victory_formats::texture::{Dimension, Error, Swizzle, Texture, TextureDesc};

const R8G8B8A8: u32 = 28;
const R8: u32 = 61;
const BC1: u32 = 71;
const D16: u32 = 55;

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
    .unwrap()
}

/// A full chain (`mips: 0`) runs until the largest extent reaches 1: the
/// width or height of a 2D texture, also the depth of a 3D one, but never
/// the array size.
#[test]
fn full_chain_length_follows_the_largest_extent() {
    use Dimension::*;
    let cases = [
        (Tex2d, 256, 1, 1, 9),
        (Tex2d, 1, 300, 1, 9),
        (Tex2d, 1, 1, 1, 1),
        (Tex1d, 1024, 1, 1, 11),
        (Tex3d, 4, 4, 64, 7),
        (Tex2d, 4, 4, 64, 3),
    ];
    for (dimension, w, h, d, mips) in cases {
        let t = texture(dimension, w, h, d, 0, R8G8B8A8, Swizzle::LINEAR);
        assert_eq!(t.mips(), mips, "{dimension:?} {w}×{h}×{d}");
        // Every mip of the chain is at least one element on each side.
        let last = &t.planes[0].layout.mips[mips as usize - 1];
        assert_eq!((last.width, last.height, last.depth), (1, 1, 1));
    }
    // A 2D array of 64 has 64 slices of 3 mips each; a 3D texture of
    // depth 64 is one slice whose mips shrink in depth too.
    assert_eq!(
        texture(Tex2d, 4, 4, 64, 0, R8, Swizzle::LINEAR).subresource_count(),
        192
    );
    assert_eq!(
        texture(Tex3d, 4, 4, 64, 0, R8, Swizzle::LINEAR).subresource_count(),
        7
    );
}

/// An R8 64×5 linear chain: data heights halve rounding down (5, 2, 1)
/// while the placed rows halve rounding up (5, 3, 2). Rows pad to 256
/// bytes, so mip 2 takes 2 × 256 bytes at 0, mip 1 3 × 256 at 512 and mip
/// 0 5 × 256 at 1280.
#[test]
fn odd_linear_heights_place_rounded_up_rows() {
    let t = texture(Dimension::Tex2d, 64, 5, 1, 3, R8, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    let placed: Vec<_> = l
        .mips
        .iter()
        .map(|m| (m.width, m.height, m.pitch, m.rows, m.offset))
        .collect();
    assert_eq!(
        placed,
        [
            (64, 5, 256, 5, 1280),
            (32, 2, 256, 3, 512),
            (16, 1, 256, 2, 0)
        ]
    );
    assert_eq!((l.slice_size, t.size), (2560, 2560));
    // Footprints follow the data, not the placement.
    let sizes: Vec<_> = (0..3).map(|i| t.footprint(i).unwrap().size).collect();
    assert_eq!(sizes, [320, 64, 16]);
    // The last row of mip 1's data is its second, not its third.
    assert_eq!(l.offset(1, 31, 1, 0), Ok(512 + 256 + 31));
    assert_eq!(l.offset(1, 0, 2, 0), Err(Error::Invalid));
}

/// BC1 9×9: mip 0 is 3×3 blocks, mip 1 4×4 texels so one block, but
/// placement halves mip 0's three blocks to two. Linear BC1 rows pad to 32
/// blocks (256 bytes), so mip 1 takes 2 rows (512 bytes) at 0 and mip 0
/// 3 rows (768 bytes) at 512.
#[test]
fn odd_block_compressed_sizes_place_more_blocks_than_they_copy() {
    let t = texture(Dimension::Tex2d, 9, 9, 1, 2, BC1, Swizzle::LINEAR);
    let l = &t.planes[0].layout;
    let placed: Vec<_> = l
        .mips
        .iter()
        .map(|m| (m.width, m.height, m.pitch, m.rows, m.offset))
        .collect();
    assert_eq!(placed, [(3, 3, 32, 3, 512), (1, 1, 32, 2, 0)]);
    assert_eq!(t.size, 1280);
    let f = t.footprint(1).unwrap();
    assert_eq!((f.row_pitch, f.size), (8, 8));

    // Mip 1's one block round-trips and lands at the start of the slice.
    let mut tiled = vec![0u8; t.size as usize];
    let block = [1, 2, 3, 4, 5, 6, 7, 8];
    t.tile(&mut tiled, 1, &block, 8, 8).unwrap();
    assert_eq!(&tiled[..8], &block);
    assert!(tiled[8..].iter().all(|&b| b == 0));
    let mut back = [0u8; 8];
    t.detile(&tiled, 1, &mut back, 8, 8).unwrap();
    assert_eq!(back, block);
}

/// A 1D texture in 4 KiB standard blocks still takes whole 32×32 RGBA8
/// blocks: 64 texels are two blocks (8 KiB), of which only the first row
/// holds data. In the `S` pattern for 4-byte elements x bits 0, 1, 2 take
/// address bits 2, 3, 7 and the 4 KiB block adds x3 at bit 9 and x4 at bit
/// 11; texel 32 starts the second block.
#[test]
fn one_dimensional_textures_fill_one_row_of_two_dimensional_blocks() {
    let t = texture(Dimension::Tex1d, 64, 1, 2, 1, R8G8B8A8, Swizzle::SW_4KB_S);
    let l = &t.planes[0].layout;
    assert_eq!((l.mips[0].pitch, l.mips[0].rows), (64, 32));
    assert_eq!((l.slice_size, t.size), (8192, 16384));
    let offsets: Vec<_> = [0, 1, 2, 4, 8, 16, 31, 32, 63]
        .iter()
        .map(|&x| l.offset(0, x, 0, 0).unwrap())
        .collect();
    assert_eq!(offsets, [0, 4, 8, 128, 512, 2048, 2700, 4096, 4096 + 2700]);
    assert_eq!(l.offset(0, 0, 0, 1), Ok(8192));
    assert_eq!(l.offset(0, 0, 1, 0), Err(Error::Invalid));

    // Slice 1 round-trips and slice 0 stays untouched.
    let row: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let mut tiled = vec![0u8; t.size as usize];
    t.tile(&mut tiled, 1, &row, 256, 256).unwrap();
    assert!(tiled[..8192].iter().all(|&b| b == 0));
    assert_eq!(tiled.iter().filter(|&&b| b != 0).count(), 255);
    let mut back = vec![0u8; 256];
    t.detile(&tiled, 1, &mut back, 256, 256).unwrap();
    assert_eq!(back, row);
}

/// In 64 KiB blocks 2-byte elements make a 256×128 block, whose tail
/// takes mips up to 128×128. The `Z` mode enters the tail as if the
/// elements were 4 bytes (addrlib's dsMipmapHtileFix), so a 16-bit depth
/// mip must be 64×128 or smaller: a 256×256 D16 chain keeps mip 1 out of
/// the tail in `Z_X` but not in `S_X`.
#[test]
fn sixteen_bit_depth_enters_the_tail_one_mip_later() {
    let standard = texture(Dimension::Tex2d, 256, 256, 1, 0, D16, Swizzle::SW_64KB_S_X);
    let depth = texture(Dimension::Tex2d, 256, 256, 1, 0, D16, Swizzle::SW_64KB_Z_X);
    let (s, z) = (&standard.planes[0].layout, &depth.planes[0].layout);
    assert_eq!(s.block, [256, 128, 1]);
    assert_eq!(z.block, [256, 128, 1]);
    assert_eq!((s.first_tail_mip, z.first_tail_mip), (1, 2));

    // S_X: tail block at 0, mip 0 (256×256, two blocks) at 64 KiB.
    assert_eq!(s.mips[0].offset, 65536);
    assert_eq!(s.size, 196608);
    // Z_X: tail at 0, mip 1 (128×128 padded to one 256×128 block) at
    // 64 KiB, mip 0 at 128 KiB.
    assert_eq!(
        (z.mips[1].pitch, z.mips[1].rows, z.mips[1].offset),
        (256, 128, 65536)
    );
    assert_eq!(z.mips[0].offset, 131072);
    assert_eq!(z.size, 262144);
    assert!(z.mips[1].tail.is_none() && z.mips[2].tail.is_some());

    // Both XOR modes are sized but not addressed.
    assert!(!s.addressable() && !z.addressable());
    assert_eq!(z.offset(0, 0, 0, 0), Err(Error::Unsupported));
}

/// Whole-texture copies check the buffer against the texture's size (all
/// planes) before choosing a plane, and refuse a subresource past the end.
#[test]
fn texture_copies_check_the_whole_buffer_and_the_index() {
    let t = texture(Dimension::Tex2d, 4, 4, 2, 1, R8G8B8A8, Swizzle::LINEAR);
    assert_eq!(t.size, 2048);
    let mut tiled = vec![0u8; t.size as usize];
    let mut linear = vec![0u8; 64];
    assert_eq!(
        t.detile(&tiled[..2047], 0, &mut linear, 16, 64),
        Err(Error::TooSmall)
    );
    assert_eq!(
        t.detile(&tiled, 2, &mut linear, 16, 64),
        Err(Error::Invalid)
    );
    assert_eq!(t.tile(&mut tiled, 2, &linear, 16, 64), Err(Error::Invalid));
    // A row pitch narrower than a row is refused even with room to spare.
    assert_eq!(
        t.detile(&tiled, 0, &mut vec![0u8; 1024], 15, 64),
        Err(Error::TooSmall)
    );
    assert!(t.detile(&tiled, 1, &mut linear, 16, 64).is_ok());
}
