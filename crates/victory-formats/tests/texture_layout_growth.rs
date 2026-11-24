//! How layouts change as a description grows: wider, taller, deeper, more
//! array slices, more mips or wider elements never make a texture smaller,
//! and the parts a smaller description shares with a larger one keep their
//! sizes. Synthetic descriptions only: nothing here needs data from outside
//! the repository.

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

/// Every 2D mode GFX10 has, with and without XOR (sizes do not need
/// addressing).
const MODES_2D: [Swizzle; 14] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
    Swizzle::SW_64KB_S_T,
    Swizzle::SW_64KB_D_T,
    Swizzle::SW_4KB_S_X,
    Swizzle::SW_4KB_D_X,
    Swizzle::SW_64KB_Z_X,
    Swizzle::SW_64KB_S_X,
    Swizzle::SW_64KB_R_X,
];

/// The modes a 3D texture may use.
const MODES_3D: [Swizzle; 9] = [
    Swizzle::LINEAR,
    Swizzle::SW_4KB_S,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_S_T,
    Swizzle::SW_4KB_S_X,
    Swizzle::SW_64KB_Z_X,
    Swizzle::SW_64KB_S_X,
    Swizzle::SW_64KB_D_X,
    Swizzle::SW_64KB_R_X,
];

const ELEMENT_BYTES: [u32; 5] = [1, 2, 4, 8, 16];

fn layout(
    dimension: Dimension,
    (width, height, depth): (u32, u32, u32),
    mips: u32,
    element_bytes: u32,
    swizzle: Swizzle,
) -> Layout {
    Layout::new(&Desc {
        dimension,
        width,
        height,
        depth,
        mips,
        element_bytes,
        block: 1,
        samples: 1,
        swizzle,
    })
    .unwrap()
}

/// Mips in the full chain of the largest side.
fn full_chain(largest: u32) -> u32 {
    32 - largest.leading_zeros()
}

/// Growing the width (or the height) by one texel at a time, across the
/// point where mip 0 stops fitting the tail and every block boundary up to
/// 300 texels, never makes the texture smaller.
#[test]
fn sizes_never_shrink_as_a_side_grows() {
    for swizzle in MODES_2D {
        for bytes in ELEMENT_BYTES {
            for other in [1, 3, 17, 100] {
                for mips in [1, 3] {
                    let (mut by_width, mut by_height) = (0, 0);
                    for side in 1..=300 {
                        let mips = mips.min(full_chain(side.max(other)));
                        let wide = layout(Dimension::Tex2d, (side, other, 1), mips, bytes, swizzle);
                        let tall = layout(Dimension::Tex2d, (other, side, 1), mips, bytes, swizzle);
                        assert!(
                            wide.size >= by_width && tall.size >= by_height,
                            "{swizzle:?} {bytes} bytes, {side} by {other}, {mips} mips"
                        );
                        (by_width, by_height) = (wide.size, tall.size);
                    }
                }
            }
        }
    }
}

/// A volume deepened one slice at a time, across every thick slab
/// boundary, never shrinks either.
#[test]
fn volume_sizes_never_shrink_as_depth_grows() {
    for swizzle in MODES_3D {
        for bytes in ELEMENT_BYTES {
            let mut previous = 0;
            for depth in 1..=70 {
                let l = layout(Dimension::Tex3d, (20, 9, depth), 1, bytes, swizzle);
                assert!(
                    l.size >= previous,
                    "{swizzle:?} {bytes} bytes, depth {depth}"
                );
                previous = l.size;
            }
        }
    }
}

/// Array slices are whole copies of one slice: the slice's size and every
/// mip's placement do not depend on how many there are.
#[test]
fn array_slices_add_whole_slices_without_changing_one() {
    for swizzle in MODES_2D {
        for bytes in ELEMENT_BYTES {
            let one = layout(Dimension::Tex2d, (37, 21, 1), 4, bytes, swizzle);
            for array in [2, 3, 7, 64] {
                let many = layout(Dimension::Tex2d, (37, 21, array), 4, bytes, swizzle);
                assert_eq!(many.slices, array, "{swizzle:?} {bytes}");
                assert_eq!(many.slice_size, one.slice_size, "{swizzle:?} {bytes}");
                assert_eq!(
                    many.size,
                    one.size * u64::from(array),
                    "{swizzle:?} {bytes}"
                );
                assert_eq!(many.mips, one.mips, "{swizzle:?} {bytes}");
                assert_eq!(many.alignment, one.alignment, "{swizzle:?} {bytes}");
            }
        }
    }
}

/// Each extra mip adds a level without making the texture smaller, and the
/// levels a shorter chain has keep their element sizes in a longer one
/// (their placement may move: mips outside the tail sit smallest first).
#[test]
fn longer_chains_keep_the_sizes_of_their_first_mips() {
    for swizzle in MODES_2D {
        for bytes in ELEMENT_BYTES {
            for (w, h) in [(1, 200), (5, 200), (129, 200), (300, 7), (256, 256)] {
                let longest = layout(
                    Dimension::Tex2d,
                    (w, h, 1),
                    full_chain(w.max(h)),
                    bytes,
                    swizzle,
                );
                let mut previous = 0;
                for mips in 1..=longest.mips.len() as u32 {
                    let l = layout(Dimension::Tex2d, (w, h, 1), mips, bytes, swizzle);
                    let at = format!("{swizzle:?} {bytes} bytes, {w}x{h}, {mips} mips");
                    assert!(l.size >= previous, "{at}");
                    assert_eq!(l.mips.len() as u32, mips, "{at}");
                    for (short, long) in l.mips.iter().zip(&longest.mips) {
                        assert_eq!(
                            (short.width, short.height, short.depth),
                            (long.width, long.height, long.depth),
                            "{at}"
                        );
                    }
                    previous = l.size;
                }
            }
        }
    }
}

/// Mip `m + 1` of a texture twice as wide and tall holds as many elements
/// as mip `m` of the original: texels halve rounding down, so doubling the
/// sides cancels exactly one halving.
#[test]
fn doubling_the_sides_shifts_the_chain_by_one_mip() {
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_D] {
        for (w, h) in [(1, 1), (3, 5), (17, 2), (100, 63), (4096, 1)] {
            let mips = full_chain(w.max(h));
            let small = layout(Dimension::Tex2d, (w, h, 1), mips, 4, swizzle);
            let large = layout(Dimension::Tex2d, (2 * w, 2 * h, 1), mips + 1, 4, swizzle);
            for (m, mip) in small.mips.iter().enumerate() {
                let doubled = &large.mips[m + 1];
                assert_eq!(
                    (doubled.width, doubled.height),
                    (mip.width, mip.height),
                    "{swizzle:?} {w}x{h} mip {m}"
                );
            }
            assert!(large.size >= small.size, "{swizzle:?} {w}x{h}");
        }
    }
}

/// Wider elements never make a texture smaller, and a mode's alignment is
/// its block size whatever the element size.
#[test]
fn wider_elements_never_shrink_a_texture() {
    for swizzle in MODES_2D {
        for (w, h, mips) in [(1, 1, 1), (33, 9, 3), (200, 150, 8), (1000, 3, 10)] {
            let mut previous = 0;
            for bytes in ELEMENT_BYTES {
                let l = layout(Dimension::Tex2d, (w, h, 1), mips, bytes, swizzle);
                assert!(l.size >= previous, "{swizzle:?} {w}x{h} {bytes} bytes");
                if swizzle != Swizzle::LINEAR {
                    assert_eq!(l.alignment, 1 << swizzle.block_log2().unwrap());
                }
                previous = l.size;
            }
        }
    }
}
