//! Element addresses split into independent parts, through the public API:
//! in every addressable mode the column adds the same bytes on every row,
//! the row (and the depth inside a thick block) adds the same bytes in
//! every column, array slices and thick slabs step by whole slices, whole
//! blocks repeat one pattern, and that pattern fills every element slot of
//! its block exactly once. The tiled copies rely on the first property: they find
//! the contiguous runs of one row and repeat them on every row. Every
//! texture is synthetic and every expected value comes from
//! `Layout::offset` or the layout's own fields, so nothing here needs data
//! from outside the repository.

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

const THIN: [Swizzle; 7] = [
    Swizzle::LINEAR,
    Swizzle::SW_256B_S,
    Swizzle::SW_256B_D,
    Swizzle::SW_4KB_S,
    Swizzle::SW_4KB_D,
    Swizzle::SW_64KB_S,
    Swizzle::SW_64KB_D,
];

/// The 3D modes with an addressable layout: linear, and the 4 KiB and
/// 64 KiB `S` blocks, which are thick (3D `D` modes are `_X` only).
const VOLUME: [Swizzle; 3] = [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S];

/// Element bytes and texels per element side: every uncompressed size and
/// both block-compressed sizes.
const ELEMENTS: [(u32, u32); 7] = [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (8, 4), (16, 4)];

fn desc(
    dimension: Dimension,
    (w, h, d): (u32, u32, u32),
    (bytes, block): (u32, u32),
    swizzle: Swizzle,
) -> Desc {
    let largest = w
        .max(h)
        .max(if dimension == Dimension::Tex3d { d } else { 1 });
    Desc {
        dimension,
        width: w,
        height: h,
        depth: d,
        mips: 32 - largest.leading_zeros(),
        element_bytes: bytes,
        block,
        samples: 1,
        swizzle,
    }
}

fn layout(d: &Desc) -> Layout {
    let l = Layout::new(d).unwrap_or_else(|e| panic!("{d:?}: {e}"));
    assert!(l.addressable(), "{d:?}");
    l
}

/// Depth slices (3D) or array slices `offset` accepts for mip `mip`.
fn depth(l: &Layout, mip: u32) -> u32 {
    if l.desc.dimension == Dimension::Tex3d {
        l.mips[mip as usize].depth
    } else {
        l.slices
    }
}

/// For every mip, depth slice and element: offset(x, y, z) is the origin
/// plus what x alone adds plus what (y, z) alone adds.
fn assert_separable(l: &Layout) {
    for mip in 0..l.mips.len() as u32 {
        let m = l.mips[mip as usize];
        let origin = l.offset(mip, 0, 0, 0).unwrap();
        let columns: Vec<u64> = (0..m.width)
            .map(|x| l.offset(mip, x, 0, 0).unwrap() - origin)
            .collect();
        for z in 0..depth(l, mip) {
            for y in 0..m.height {
                let row = l.offset(mip, 0, y, z).unwrap() - origin;
                for (x, column) in columns.iter().enumerate() {
                    assert_eq!(
                        l.offset(mip, x as u32, y, z).unwrap(),
                        origin + row + column,
                        "{:?} mip {mip} ({x}, {y}, {z})",
                        l.desc
                    );
                }
            }
        }
    }
}

#[test]
fn columns_add_the_same_bytes_on_every_row_of_thin_textures() {
    let shapes = [
        (1, 1, 1),
        (5, 3, 2),
        (16, 16, 1),
        (19, 7, 2),
        (40, 1, 1),
        (1, 33, 1),
        (70, 18, 1),
    ];
    for swizzle in THIN {
        for element in ELEMENTS {
            for shape in shapes {
                assert_separable(&layout(&desc(Dimension::Tex2d, shape, element, swizzle)));
            }
        }
    }
}

#[test]
fn columns_add_the_same_bytes_on_every_row_and_slice_of_volumes() {
    let shapes = [(1, 1, 1), (4, 4, 4), (9, 5, 3), (16, 2, 17), (2, 16, 9)];
    for swizzle in VOLUME {
        for element in ELEMENTS {
            for shape in shapes {
                assert_separable(&layout(&desc(Dimension::Tex3d, shape, element, swizzle)));
            }
        }
    }
}

#[test]
fn one_dimensional_textures_separate_like_one_row() {
    for swizzle in [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_4KB_S,
        Swizzle::SW_64KB_S,
    ] {
        for element in ELEMENTS.iter().filter(|e| e.1 == 1) {
            for (w, arrays) in [(1, 1), (7, 3), (64, 1), (300, 2)] {
                assert_separable(&layout(&desc(
                    Dimension::Tex1d,
                    (w, 1, arrays),
                    *element,
                    swizzle,
                )));
            }
        }
    }
}

/// Every element of slice `s` sits exactly `s` slices past the same element
/// of slice 0, so array slices never interleave.
#[test]
fn array_slices_step_by_whole_slices() {
    for swizzle in THIN {
        for element in ELEMENTS {
            let l = layout(&desc(Dimension::Tex2d, (13, 6, 4), element, swizzle));
            assert_eq!(l.slices, 4);
            for mip in 0..l.mips.len() as u32 {
                let m = l.mips[mip as usize];
                for s in 0..l.slices {
                    for (x, y) in [
                        (0, 0),
                        (m.width - 1, 0),
                        (0, m.height - 1),
                        (m.width - 1, m.height - 1),
                    ] {
                        assert_eq!(
                            l.offset(mip, x, y, s).unwrap(),
                            l.offset(mip, x, y, 0).unwrap() + u64::from(s) * l.slice_size,
                            "{swizzle:?} {element:?} mip {mip} ({x}, {y}) slice {s}"
                        );
                    }
                }
            }
        }
    }
}

/// A thick block holds `block[2]` depth slices; slice `z` lands in slab
/// `z / block[2]` at the same place slice `z % block[2]` takes in slab 0.
#[test]
fn thick_slabs_step_by_whole_slices() {
    for swizzle in [Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
        for element in ELEMENTS {
            let l = layout(&desc(Dimension::Tex3d, (6, 5, 40), element, swizzle));
            assert!(l.thick, "{swizzle:?} {element:?}");
            let slab = l.block[2];
            assert!(slab > 1 && 40 > slab, "{swizzle:?} {element:?} slab {slab}");
            let m = l.mips[0];
            for z in 0..m.depth {
                for (x, y) in [(0, 0), (m.width - 1, m.height - 1)] {
                    assert_eq!(
                        l.offset(0, x, y, z).unwrap(),
                        l.offset(0, x, y, z % slab).unwrap() + u64::from(z / slab) * l.slice_size,
                        "{swizzle:?} {element:?} ({x}, {y}, {z})"
                    );
                }
            }
        }
    }
}

/// Outside the tail, the column part is periodic in whole blocks: element
/// `x + block[0]` sits one block after element `x`, and row `y + block[1]`
/// sits one row of blocks after row `y`.
#[test]
fn whole_blocks_repeat_their_pattern() {
    for swizzle in THIN.into_iter().skip(1) {
        for element in ELEMENTS {
            let l = layout(&desc(Dimension::Tex2d, (2048, 2048, 1), element, swizzle));
            let [bw, bh, _] = l.block;
            let m = l.mips[0];
            assert!(m.tail.is_none(), "{swizzle:?} {element:?}");
            assert!(
                m.width >= 2 * bw && m.height >= 2 * bh,
                "{swizzle:?} {element:?}"
            );
            let blocks_per_row = u64::from(m.pitch / bw);
            for (x, y) in [(0, 0), (1, 0), (0, 1), (bw - 1, bh - 1), (bw / 2, bh / 3)] {
                let at = l.offset(0, x, y, 0).unwrap();
                assert_eq!(
                    l.offset(0, x + bw, y, 0).unwrap(),
                    at + l.alignment,
                    "{swizzle:?} {element:?} ({x}, {y})"
                );
                assert_eq!(
                    l.offset(0, x, y + bh, 0).unwrap(),
                    at + blocks_per_row * l.alignment,
                    "{swizzle:?} {element:?} ({x}, {y})"
                );
            }
        }
    }
}

/// Inside one block of a thin tiled mode, every element lands in its own
/// element-aligned slot of that block: the pattern is a permutation of the
/// block's slots.
#[test]
fn a_block_pattern_is_a_permutation_of_its_slots() {
    for swizzle in THIN.into_iter().skip(1) {
        for element in ELEMENTS {
            let l = layout(&desc(Dimension::Tex2d, (2048, 2048, 1), element, swizzle));
            let [bw, bh, _] = l.block;
            let bytes = u64::from(element.0);
            let origin = l.offset(0, 0, 0, 0).unwrap();
            let mut seen = vec![false; (l.alignment / bytes) as usize];
            for y in 0..bh {
                for x in 0..bw {
                    let inside = l.offset(0, x, y, 0).unwrap() - origin;
                    assert!(
                        inside < l.alignment && inside.is_multiple_of(bytes),
                        "{swizzle:?} {element:?} ({x}, {y})"
                    );
                    let slot = &mut seen[(inside / bytes) as usize];
                    assert!(
                        !*slot,
                        "{swizzle:?} {element:?} ({x}, {y}) shares slot {inside}"
                    );
                    *slot = true;
                }
            }
            assert!(seen.iter().all(|s| *s), "{swizzle:?} {element:?}");
        }
    }
}
