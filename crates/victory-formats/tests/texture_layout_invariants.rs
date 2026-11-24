//! Cross-field invariants of `Layout`, checked over a seeded sweep of
//! synthetic descriptions (every known swizzle mode, every element size,
//! block compression, MSAA, arrays, 1D, 2D and 3D, full and partial mip
//! chains): the block holds exactly one block's bytes, the slice and the
//! texture are whole blocks, every mip fits its padding and its slice, the
//! mips outside the tail sit smallest first after the tail block, and the
//! tail holds exactly the mips from `first_tail_mip` on. Nothing here needs
//! data from outside the repository.

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

const MODES: [Swizzle; 16] = [
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
    Swizzle::SW_4KB_R_X,
    Swizzle::SW_64KB_Z_X,
    Swizzle::SW_64KB_S_X,
    Swizzle::SW_64KB_D_X,
    Swizzle::SW_64KB_R_X,
];

/// xorshift32: the same sweep on every run.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// Mostly small sizes, sometimes a power of two or one past it.
    fn extent(&mut self) -> u32 {
        match self.below(4) {
            0 => 1 << self.below(10),
            1 => (1 << self.below(10)) + 1,
            _ => 1 + self.below(70),
        }
    }
}

fn max_mips(d: &Desc) -> u32 {
    let side = d
        .width
        .max(d.height)
        .max(if d.dimension == Dimension::Tex3d {
            d.depth
        } else {
            1
        });
    32 - side.leading_zeros()
}

/// Valid descriptions only: `Layout::new` must accept every one.
fn descs() -> Vec<Desc> {
    let mut rng = Rng(0x3150_beef);
    let mut out = vec![];
    while out.len() < 4000 {
        let swizzle = MODES[rng.below(MODES.len() as u32) as usize];
        let dimension = match rng.below(3) {
            0 => Dimension::Tex1d,
            1 => Dimension::Tex2d,
            _ => Dimension::Tex3d,
        };
        if dimension == Dimension::Tex3d && !matches!(swizzle.0, 0 | 5 | 9 | 17 | 21 | 24..=27) {
            continue;
        }
        let block = if rng.below(4) == 0 { 4 } else { 1 };
        let element_bytes = if block == 4 {
            [8, 16][rng.below(2) as usize]
        } else {
            1 << rng.below(5)
        };
        let width = rng.extent();
        let height = if dimension == Dimension::Tex1d {
            1
        } else {
            rng.extent()
        };
        let depth = if dimension == Dimension::Tex3d {
            1 + rng.below(20)
        } else {
            1 + rng.below(4)
        };
        // MSAA only in tiled modes: linear layouts have no sample planes.
        let samples =
            if dimension == Dimension::Tex2d && swizzle != Swizzle::LINEAR && rng.below(5) == 0 {
                1 << rng.below(4)
            } else {
                1
            };
        let mut d = Desc {
            dimension,
            width,
            height,
            depth,
            mips: 1,
            element_bytes,
            block,
            samples,
            swizzle,
        };
        if samples == 1 {
            d.mips = 1 + rng.below(max_mips(&d));
        }
        out.push(d);
    }
    out
}

fn layouts() -> Vec<Layout> {
    descs()
        .iter()
        .map(|d| Layout::new(d).unwrap_or_else(|e| panic!("{d:?}: {e}")))
        .collect()
}

fn thick(d: &Desc) -> bool {
    d.dimension == Dimension::Tex3d && matches!(d.swizzle.0, 5 | 9 | 17 | 21 | 25 | 26)
}

#[test]
fn the_layout_keeps_its_description_and_one_mip_per_level() {
    for (d, l) in descs().iter().zip(layouts()) {
        assert_eq!(&l.desc, d);
        assert_eq!(l.mips.len(), d.mips as usize, "{d:?}");
        assert!(l.first_tail_mip <= d.mips, "{d:?}");
        assert_eq!(l.thick, thick(d), "{d:?}");
    }
}

#[test]
fn a_block_holds_exactly_one_blocks_bytes() {
    for l in layouts() {
        let d = l.desc;
        let log2 = d.swizzle.block_log2().unwrap();
        if log2 == 0 {
            assert_eq!(l.block, [256 / d.element_bytes, 1, 1], "{d:?}");
            assert_eq!(l.alignment, 256, "{d:?}");
            continue;
        }
        assert_eq!(l.alignment, 1 << log2, "{d:?}");
        let [x, y, z] = l.block;
        assert!(
            x.is_power_of_two() && y.is_power_of_two() && z.is_power_of_two(),
            "{d:?}"
        );
        let bytes = u64::from(x * y * z) * u64::from(d.element_bytes * d.samples);
        assert_eq!(bytes, l.alignment, "{d:?}");
        assert_eq!(z == 1, !l.thick, "{d:?}");
    }
}

#[test]
fn slices_and_the_texture_are_whole_blocks() {
    for l in layouts() {
        let d = l.desc;
        assert_eq!(l.size, l.slice_size * u64::from(l.slices), "{d:?}");
        assert!(l.slice_size > 0, "{d:?}");
        assert_eq!(l.slice_size % l.alignment, 0, "{d:?}");
        // Thick slabs cover the depth; everything else has one slice per
        // depth slice or array layer.
        if l.thick {
            assert_eq!(l.slices, d.depth.div_ceil(l.block[2]), "{d:?}");
        } else {
            assert_eq!(l.slices, d.depth, "{d:?}");
        }
    }
}

#[test]
fn every_mip_fits_its_padding_and_its_slice() {
    for l in layouts() {
        let d = l.desc;
        let slab = if l.thick { u64::from(l.block[2]) } else { 1 };
        for (i, m) in l.mips.iter().enumerate() {
            let what = format!("{d:?} mip {i}");
            assert!(m.width >= 1 && m.height >= 1 && m.depth >= 1, "{what}");
            assert!(m.pitch >= m.width, "{what}");
            assert!(m.rows >= m.height, "{what}");
            let expected_w = (d.width >> i).max(1).div_ceil(d.block);
            assert_eq!(m.width, expected_w, "{what}");
            if m.tail.is_none() {
                let bytes = u64::from(m.pitch) * u64::from(m.rows) * u64::from(d.element_bytes);
                let bytes = bytes * u64::from(d.samples) * slab;
                assert!(m.offset + bytes <= l.slice_size, "{what}");
            }
        }
    }
}

#[test]
fn the_tail_holds_exactly_the_mips_from_first_tail_mip_on() {
    for l in layouts() {
        let d = l.desc;
        for (i, m) in l.mips.iter().enumerate() {
            let in_tail = i as u32 >= l.first_tail_mip;
            assert_eq!(m.tail.is_some(), in_tail, "{d:?} mip {i}");
            if let Some(t) = m.tail {
                assert_eq!(m.offset, 0, "{d:?} mip {i}");
                assert!(u64::from(t.offset) < l.alignment, "{d:?} mip {i}");
            }
        }
        // Linear layouts, single mips and 256-byte blocks have no tail.
        let log2 = d.swizzle.block_log2().unwrap();
        if log2 <= 8 || d.mips == 1 {
            assert_eq!(l.first_tail_mip, d.mips, "{d:?}");
        }
    }
}

#[test]
fn mips_outside_the_tail_sit_smallest_first_after_the_tail_block() {
    for l in layouts() {
        let d = l.desc;
        let placed = &l.mips[..l.first_tail_mip as usize];
        for pair in placed.windows(2) {
            assert!(pair[0].offset > pair[1].offset, "{d:?}");
        }
        let Some(smallest) = placed.last() else {
            continue;
        };
        let expected = if l.first_tail_mip < d.mips && d.swizzle.block_log2() != Some(0) {
            l.alignment
        } else {
            0
        };
        assert_eq!(smallest.offset, expected, "{d:?}");
        // Tiled mips start on a block.
        if d.swizzle != Swizzle::LINEAR {
            for m in placed {
                assert_eq!(m.offset % l.alignment, 0, "{d:?}");
            }
        }
    }
}

#[test]
fn modes_gfx10_lacks_are_refused_for_any_description() {
    let d = descs()[0];
    for n in 0..64 {
        let swizzle = Swizzle(n);
        if MODES.contains(&swizzle) {
            continue;
        }
        assert!(swizzle.block_log2().is_none(), "{n}");
        assert!(Layout::new(&Desc { swizzle, ..d }).is_err(), "{n}");
    }
}
