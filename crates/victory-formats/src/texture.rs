//! Texture layouts of the console GPU (RDNA2, GFX10.3): a texture's size and
//! alignment, where each mip and array slice starts, the address of each
//! element, and copies between tiled memory and linear rows.
//!
//! Units: an element is a texel, or a 4x4 block of a block-compressed
//! format. Mips of one array slice (or one depth slab of a thick 3D texture)
//! are stored together, smallest first: the mip tail (the small mips sharing
//! one block) at offset 0, mip 0 last.
//!
//! The algorithm follows AMD's address library (addrlib), as shipped in Mesa
//! under `src/amd/addrlib`: block dimensions, mip placement and mip tails
//! from `gfx10addrlib.cpp` and `addrlib2.cpp`
//! (`ComputeSurfaceInfoMacroTiled`, `ComputeThinBlockDimension`,
//! `GetMipTailDim`, `GetMaxNumMipsInTail`, `HwlComputeSurfaceInfoLinear`,
//! `ComputeSurfaceAddrFromCoordMacroTiled`), and the 256-byte micro tile
//! patterns in `MICRO` from `gfx10SwizzlePattern.h` (the RB+ tables, as on
//! RDNA2). addrlib carries this notice:
//!
//! Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//! SPDX-License-Identifier: MIT
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including
//! without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit
//! persons to whom the Software is furnished to do so, subject to the
//! following conditions: the above copyright notice and this permission
//! notice shall be included in all copies or substantial portions of the
//! Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//! EXPRESS OR IMPLIED.

/// A swizzle mode, by its hardware number (addrlib's `ADDR_SW_*` order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub u32);

impl Swizzle {
    pub const LINEAR: Swizzle = Swizzle(0);
    pub const SW_256B_S: Swizzle = Swizzle(1);
    pub const SW_256B_D: Swizzle = Swizzle(2);
    pub const SW_4KB_S: Swizzle = Swizzle(5);
    pub const SW_4KB_D: Swizzle = Swizzle(6);
    pub const SW_64KB_S: Swizzle = Swizzle(9);
    pub const SW_64KB_D: Swizzle = Swizzle(10);
    pub const SW_64KB_S_T: Swizzle = Swizzle(17);
    pub const SW_64KB_D_T: Swizzle = Swizzle(18);
    pub const SW_4KB_S_X: Swizzle = Swizzle(21);
    pub const SW_4KB_D_X: Swizzle = Swizzle(22);
    pub const SW_4KB_R_X: Swizzle = Swizzle(23);
    pub const SW_64KB_Z_X: Swizzle = Swizzle(24);
    pub const SW_64KB_S_X: Swizzle = Swizzle(25);
    pub const SW_64KB_D_X: Swizzle = Swizzle(26);
    pub const SW_64KB_R_X: Swizzle = Swizzle(27);

    /// log2 of the block size in bytes (0 for linear); `None` for modes
    /// GFX10 does not have.
    pub fn block_log2(self) -> Option<u32> {
        Some(match self.0 {
            0 => 0,
            1 | 2 => 8,
            5 | 6 | 21..=23 => 12,
            9 | 10 | 17 | 18 | 24..=27 => 16,
            _ => return None,
        })
    }

    fn standard(self) -> bool {
        matches!(self.0, 1 | 5 | 9 | 17 | 21 | 25)
    }

    fn display(self) -> bool {
        matches!(self.0, 2 | 6 | 10 | 18 | 22 | 26)
    }

    fn z_order(self) -> bool {
        self.0 == 24
    }

    /// Pipe and bank XOR modes (`_T`, `_X`): their addresses depend on the
    /// GPU's pipe configuration.
    pub fn xor(self) -> bool {
        self.0 >= 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Tex1d,
    Tex2d,
    Tex3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The description makes no texture (zero size, bad element size, a
    /// mode this dimension cannot use, too many mips).
    Invalid,
    /// A valid layout whose addressing is not implemented here.
    Unsupported,
    /// A buffer is smaller than the layout or the rows need.
    TooSmall,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::Invalid => "texture: the description makes no texture",
            Error::Unsupported => "texture: layout addressing not implemented",
            Error::TooSmall => "texture: buffer smaller than the copy needs",
        })
    }
}

impl std::error::Error for Error {}

/// One plane of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Desc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth: u32,
    pub mips: u32,
    /// 1, 2, 4, 8 or 16.
    pub element_bytes: u32,
    /// Texels per element side: 1, or 4 for block compression.
    pub block: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

/// Elements along one side of mip `mip`, as data and copies see it: texels
/// halve rounding down (to at least 1), then round up to whole elements.
fn dim(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// Elements along one side of mip `mip` for placement (padding, tail,
/// offsets): addrlib halves mip 0's elements rounding up, which can be one
/// more than [`dim`] for odd sizes.
fn padded_dim(texels: u32, mip: u32, block: u32) -> u32 {
    texels.max(1).div_ceil(block).div_ceil(1 << mip)
}

/// Where one mip lives inside a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mip {
    /// Size in elements (depth 1 unless the texture is 3D).
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Padded size in elements: whole blocks, or the tail's share of one.
    pub pitch: u32,
    pub rows: u32,
    pub depth_pitch: u32,
    /// Byte offset of its first block in the slice (0 in the tail).
    pub offset: u64,
    /// In the mip tail: byte offset inside the tail block and the element
    /// position the mip starts at.
    pub tail: Option<Tail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tail {
    pub offset: u32,
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub desc: Desc,
    /// Block dimensions in elements (linear: the pitch alignment, 1, 1).
    pub block: [u32; 3],
    /// 3D blocks span `block[2]` depth slices (GFX10 3D `S` and `D` modes).
    pub thick: bool,
    /// Array slices, depth slabs of `block[2]` slices (thick), or depth
    /// slices (thin and linear 3D).
    pub slices: u32,
    /// Bytes of one slice: its whole mip chain.
    pub slice_size: u64,
    pub size: u64,
    pub alignment: u64,
    /// The first mip in the tail; `mips.len()` without one.
    pub first_tail_mip: u32,
    pub mips: Vec<Mip>,
    /// Byte offsets inside a block of each element coordinate bit, per
    /// axis; empty for linear and unsupported modes.
    pattern: [Vec<u32>; 3],
}

/// Micro tile (256 bytes) patterns by log2 of the element size: for each
/// address bit, the element coordinate bit it takes (`x0`, `y1`, `z0`...),
/// `--` for the bits inside an element. Thin `S`, thin `D`, thick `S`
/// (also the `D` 3D modes). From addrlib's `GFX10_SW_256_S_RBPLUS`,
/// `GFX10_SW_256_D_RBPLUS` and `GFX10_SW_4K_S3_RBPLUS` patterns; the larger
/// blocks add bits Y, X (thin) or Y, X, Z (thick) in turn.
const MICRO: [[&[u8; 16]; 5]; 3] = [
    [
        b"x0x1x2x3y0y1y2y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1y2x2",
        b"------x0y0y1x1x2",
        b"--------y0y1x0x1",
    ],
    [
        b"x0x1x2y1y0y2x3y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1x2y2",
        b"------x0y0x1x2y1",
        b"--------x0y0x1y1",
    ],
    [
        b"x0x1z0y0z1y1x2z2",
        b"--x0z0y0z1y1x1z2",
        b"----x0y0z0y1x1z1",
        b"------x0z0y0x1z1",
        b"--------z0y0x0z1",
    ],
];

/// For each axis, the address bit each coordinate bit lands on.
fn pattern(kind: usize, element_log2: u32, block_log2: u32) -> [Vec<u32>; 3] {
    let mut axes: [Vec<u32>; 3] = Default::default();
    let micro = MICRO[kind][element_log2 as usize];
    for bit in 0..8u32 {
        let (axis, index) = (micro[2 * bit as usize], micro[2 * bit as usize + 1]);
        let axis = match axis {
            b'x' => 0,
            b'y' => 1,
            b'z' => 2,
            _ => continue,
        };
        let index = usize::from(index - b'0');
        let to = &mut axes[axis];
        if to.len() <= index {
            to.resize(index + 1, 0);
        }
        to[index] = bit;
    }
    let order: &[usize] = if kind == 2 { &[1, 0, 2] } else { &[1, 0] };
    for (bit, axis) in (8..block_log2).zip(order.iter().cycle()) {
        axes[*axis].push(bit);
    }
    axes
}

/// Byte offsets inside a block for coordinates 0..2^bits of one axis.
fn spread(bits: &[u32]) -> Vec<u32> {
    (0..1u32 << bits.len())
        .map(|c| {
            bits.iter()
                .enumerate()
                .map(|(i, &to)| ((c >> i) & 1) << to)
                .sum()
        })
        .collect()
}

const MAX_MIPS: u32 = 16;

/// The largest width, height and depth (or array size) accepted, in texels.
/// GFX10's image descriptors hold at most 16384 texels per side, so no
/// texture the GPU can use is refused. The bound also keeps every size
/// computed below within `u64` (and its `u32` steps), which unchecked
/// values from a title would otherwise overflow.
const MAX_EXTENT: u32 = 1 << 14;

/// 256-byte micro tile dimensions by log2 element size, 2D and 3D.
const BLOCK_256_2D: [[u32; 2]; 5] = [[16, 16], [16, 8], [8, 8], [8, 4], [4, 4]];
const BLOCK_256_3D: [[u32; 3]; 5] = [[8, 4, 8], [4, 4, 8], [4, 4, 4], [4, 2, 4], [2, 2, 4]];
const BLOCK_1K_3D: [[u32; 3]; 5] = [[16, 8, 8], [8, 8, 8], [8, 8, 4], [8, 4, 4], [4, 4, 4]];

impl Layout {
    pub fn new(desc: &Desc) -> Result<Layout, Error> {
        let d = *desc;
        let block_log2 = d.swizzle.block_log2().ok_or(Error::Invalid)?;
        let valid = (1..=MAX_EXTENT).contains(&d.width)
            && (1..=MAX_EXTENT).contains(&d.height)
            && (1..=MAX_EXTENT).contains(&d.depth)
            && (1..=MAX_MIPS).contains(&d.mips)
            && d.element_bytes.is_power_of_two()
            && d.element_bytes <= 16
            && matches!(d.block, 1 | 4)
            && d.samples.is_power_of_two()
            && d.samples <= 8
            && (d.samples == 1 || (d.dimension == Dimension::Tex2d && d.mips == 1))
            && (d.dimension != Dimension::Tex1d || d.height == 1)
            && (d.dimension != Dimension::Tex3d
                || matches!(d.swizzle.0, 0 | 5 | 9 | 17 | 21 | 24..=27));
        if !valid {
            return Err(Error::Invalid);
        }
        if block_log2 == 0 {
            Ok(Layout::linear(d))
        } else {
            Ok(Layout::tiled(d, block_log2))
        }
    }

    /// `HwlComputeSurfaceInfoLinear`: rows aligned to 256 bytes, mips packed
    /// smallest first, a 3D texture's depth slices each hold the chain.
    fn linear(d: Desc) -> Layout {
        let bpe = u64::from(d.element_bytes);
        let align = 256 / d.element_bytes;
        let three_d = d.dimension == Dimension::Tex3d;
        let mut mips = vec![];
        let mut offset = 0;
        for i in (0..d.mips).rev() {
            let h = padded_dim(d.height, i, d.block);
            let pitch = padded_dim(d.width, i, d.block).next_multiple_of(align);
            mips.push(Mip {
                width: dim(d.width, i, d.block),
                height: dim(d.height, i, d.block),
                depth: if three_d { dim(d.depth, i, 1) } else { 1 },
                pitch,
                rows: h,
                depth_pitch: if three_d { d.depth } else { 1 },
                offset,
                tail: None,
            });
            offset += u64::from(pitch) * u64::from(h) * bpe;
        }
        mips.reverse();
        Layout {
            desc: d,
            block: [align, 1, 1],
            thick: false,
            slices: d.depth,
            slice_size: offset,
            size: offset * u64::from(d.depth),
            alignment: 256,
            first_tail_mip: d.mips,
            mips,
            pattern: Default::default(),
        }
    }

    fn tiled(d: Desc, block_log2: u32) -> Layout {
        let element_log2 = d.element_bytes.trailing_zeros();
        let bpe = u64::from(d.element_bytes);
        let block_bytes = 1u64 << block_log2;
        let thick = d.dimension == Dimension::Tex3d && (d.swizzle.standard() || d.swizzle.display());
        let block = if thick {
            // ComputeThickBlockDimension
            let amp = (block_log2 - 10) / 3;
            let rest = (block_log2 - 10) % 3;
            let b = BLOCK_1K_3D[element_log2 as usize];
            [b[0] << amp, b[1] << (amp + rest / 2), b[2] << (amp + u32::from(rest != 0))]
        } else {
            // ComputeThinBlockDimension
            let samples_log2 = d.samples.trailing_zeros();
            let elements_log2 = block_log2 - element_log2 - samples_log2;
            let wide = samples_log2 & 1 == 0 || block_log2 & 1 != 0;
            let w = (elements_log2 + u32::from(wide)) / 2;
            [1 << w, 1 << (elements_log2 - w), 1]
        };
        let slices = d.depth.next_multiple_of(block[2]);
        let three_d = d.dimension == Dimension::Tex3d;
        // Mip depths placed by a thick layout (thin ones store depth as
        // slices).
        let mip_depth = |i: u32| if thick { padded_dim(d.depth, i, 1) } else { 1 };
        let depth_of = |i: u32| if three_d { dim(d.depth, i, 1) } else { 1 };
        let (width, height) = (|i| dim(d.width, i, d.block), |i| dim(d.height, i, d.block));
        let padded = |i| (padded_dim(d.width, i, d.block), padded_dim(d.height, i, d.block));
        let mut mips = Vec::with_capacity(d.mips as usize);
        let mut first_tail_mip = d.mips;
        let mut slice_size = 0u64;
        if d.mips == 1 {
            mips.push(Mip {
                width: width(0),
                height: height(0),
                depth: depth_of(0),
                pitch: width(0).next_multiple_of(block[0]),
                rows: height(0).next_multiple_of(block[1]),
                depth_pitch: if three_d { slices } else { 1 },
                offset: 0,
                tail: None,
            });
            slice_size = u64::from(mips[0].pitch)
                * u64::from(mips[0].rows)
                * bpe
                * u64::from(d.samples);
        } else {
            // GetMipTailDim: the largest mip the tail takes.
            let mut tail_max = block;
            if thick {
                match block_log2 % 3 {
                    0 => tail_max[1] >>= 1,
                    1 => tail_max[0] >>= 1,
                    _ => tail_max[2] >>= 1,
                }
            } else {
                tail_max[0] >>= 1;
            }
            // GetMaxNumMipsInTail; 256-byte blocks have no tail
            // (`ComputeSurfaceInfoMicroTiled`).
            let effective = if thick {
                block_log2 - (block_log2 - 8) / 3
            } else {
                block_log2
            };
            let max_in_tail = match effective {
                8 => 0,
                9..=11 => 1 + (1 << (effective - 9)),
                _ => effective - 4,
            };
            // dsMipmapHtileFix: 8- and 16-bit depth enter the tail as 32-bit.
            let mut fixed = tail_max;
            if d.swizzle.z_order() && element_log2 <= 1 {
                let b = BLOCK_256_2D[element_log2 as usize];
                fixed[0] /= b[0] / BLOCK_256_2D[2][0];
                fixed[1] /= b[1] / BLOCK_256_2D[2][1];
            }
            let mut sizes = vec![];
            for i in 0..d.mips {
                let (w, h) = padded(i);
                if w <= fixed[0] && h <= fixed[1] && d.mips - i <= max_in_tail {
                    first_tail_mip = i;
                    slice_size += block_bytes / u64::from(block[2]);
                    break;
                }
                let pitch = w.next_multiple_of(block[0]);
                let rows = h.next_multiple_of(block[1]);
                let depth = mip_depth(i).next_multiple_of(block[2]);
                let one = u64::from(pitch) * u64::from(rows) * bpe;
                slice_size += one;
                sizes.push(one * u64::from(block[2]));
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if three_d { depth } else { 1 },
                    offset: 0,
                    tail: None,
                });
            }
            // Standard mips from the smallest up, after the tail block.
            let mut offset = if first_tail_mip < d.mips {
                block_bytes
            } else {
                0
            };
            for i in (0..first_tail_mip as usize).rev() {
                mips[i].offset = offset;
                offset += sizes[i];
            }
            let micro = if thick {
                let b = BLOCK_256_3D[element_log2 as usize];
                [b[0], b[1]]
            } else {
                BLOCK_256_2D[element_log2 as usize]
            };
            let (mut pitch, mut rows) = (tail_max[0], tail_max[1]);
            for i in first_tail_mip..d.mips {
                let m = max_in_tail - 1 - (i - first_tail_mip);
                let offset = if m > 6 { 16 << m } else { m << 8 };
                // Address bits 8.. of the offset, odd ones x, even ones y.
                let bits = |from: u32| (0..6).map(|b| ((offset >> (from + 2 * b)) & 1) << b).sum();
                let (mut x, mut y): (u32, u32) = (bits(9), bits(8));
                if block_log2 & 1 != 0 {
                    std::mem::swap(&mut x, &mut y);
                    if element_log2 & 1 != 0 {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                }
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if thick {
                        mip_depth(i).next_multiple_of(BLOCK_256_3D[element_log2 as usize][2])
                    } else {
                        1
                    },
                    offset: 0,
                    tail: Some(Tail {
                        offset,
                        x: x * micro[0],
                        y: y * micro[1],
                    }),
                });
                pitch = (pitch >> 1).max(micro[0]);
                rows = (rows >> 1).max(micro[1]);
            }
        }
        // Thick layouts store each depth slab's chain together.
        let slab = if thick { u64::from(block[2]) } else { 1 };
        let slice_size = slice_size * slab;
        let slices = slices / block[2];
        let addressable = !d.swizzle.xor() && d.samples == 1;
        let pattern = if addressable {
            let kind = if thick {
                2
            } else {
                usize::from(d.swizzle.display())
            };
            pattern(kind, element_log2, block_log2).map(|bits| spread(&bits))
        } else {
            Default::default()
        };
        Layout {
            desc: d,
            block,
            thick,
            slices,
            slice_size,
            size: slice_size * u64::from(slices),
            alignment: block_bytes,
            first_tail_mip,
            mips,
            pattern,
        }
    }

    /// Whether [`Layout::offset`] and the copies work for this layout:
    /// linear and the modes without pipe or bank XOR, single-sampled.
    ///
    /// shortcut: no addressing for the XOR modes (`_X`, `_T`) and MSAA;
    /// they need the GPU's pipe and bank configuration (Series X and S
    /// differ). Titles use them for render targets and depth, which the
    /// host renders; add them when a title copies into or out of one.
    pub fn addressable(&self) -> bool {
        self.desc.swizzle == Swizzle::LINEAR || !self.pattern[0].is_empty()
    }

    /// The slice and the depth inside it for `z` (an array slice or a 3D
    /// depth slice).
    fn split(&self, z: u32) -> (u32, u32) {
        if self.thick {
            (z / self.block[2], z % self.block[2])
        } else {
            (z, 0)
        }
    }

    /// Byte offset of element (`x`, `y`) of mip `mip` in array slice or
    /// depth slice `z`. Coordinates must lie inside the mip.
    pub fn offset(&self, mip: u32, x: u32, y: u32, z: u32) -> Result<u64, Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = if self.desc.dimension == Dimension::Tex3d {
            m.depth
        } else {
            self.slices
        };
        let inside = x < m.width && y < m.height && z < depth;
        if !inside || !self.addressable() {
            return Err(if inside {
                Error::Unsupported
            } else {
                Error::Invalid
            });
        }
        let (slice, z) = self.split(z);
        let base = u64::from(slice) * self.slice_size + m.offset;
        if self.desc.swizzle == Swizzle::LINEAR {
            let row = u64::from(y) * u64::from(m.pitch) + u64::from(x);
            return Ok(base + row * u64::from(self.desc.element_bytes));
        }
        Ok(base + self.row(m, y, z) + self.column(m, x))
    }

    /// The part of an address that depends on the row (and depth in a
    /// thick block).
    fn row(&self, m: &Mip, y: u32, z: u32) -> u64 {
        let [bw, bh, _] = self.block;
        let log2 = self.alignment.trailing_zeros();
        // In the tail the pattern of the shifted position gives the mip's
        // offset inside the tail block.
        let y = y + m.tail.map_or(0, |t| t.y);
        let blocks = u64::from(y / bh) * u64::from(m.pitch / bw);
        (blocks << log2) + u64::from(self.pattern[1][(y % bh) as usize] | self.pattern[2][z as usize])
    }

    /// The part of an address that depends on the column.
    fn column(&self, m: &Mip, x: u32) -> u64 {
        let bw = self.block[0];
        let x = x + m.tail.map_or(0, |t| t.x);
        (u64::from(x / bw) << self.alignment.trailing_zeros())
            + u64::from(self.pattern[0][(x % bw) as usize])
    }

    /// Depth slices a subresource of mip `mip` covers: the mip's depth for
    /// 3D textures, else 1 (one array slice).
    pub fn subresource_depth(&self, mip: u32) -> u32 {
        self.mips.get(mip as usize).map_or(0, |m| m.depth)
    }

    /// The tightly packed linear rows of one subresource of mip `mip`: the
    /// smallest row pitch and buffer [`Layout::detile`] and [`Layout::tile`]
    /// accept for it, with depth slices one after the other. `None` past
    /// the chain.
    pub fn footprint(&self, mip: u32) -> Option<Footprint> {
        let m = self.mips.get(mip as usize)?;
        let row_pitch = m.width as usize * self.desc.element_bytes as usize;
        let depth_pitch = row_pitch * m.height as usize;
        Some(Footprint {
            row_pitch,
            depth_pitch,
            size: depth_pitch * self.subresource_depth(mip) as usize,
        })
    }

    /// Copies one subresource out of tiled memory (the whole texture,
    /// [`Layout::size`] bytes) into linear rows: mip `mip` of array slice
    /// `slice` (3D: every depth slice of the mip, `slice` 0). Rows are
    /// `row_pitch` bytes apart, depth slices `depth_pitch`.
    pub fn detile(
        &self,
        tiled: &[u8],
        mip: u32,
        slice: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        self.walk(tiled.len(), mip, slice, linear.len(), row_pitch, depth_pitch, |t, l, len| {
            linear[l..][..len].copy_from_slice(&tiled[t..][..len]);
        })
    }

    /// The reverse of [`Layout::detile`]: linear rows into tiled memory.
    pub fn tile(
        &self,
        tiled: &mut [u8],
        mip: u32,
        slice: u32,
        linear: &[u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        self.walk(tiled.len(), mip, slice, linear.len(), row_pitch, depth_pitch, |t, l, len| {
            tiled[t..][..len].copy_from_slice(&linear[l..][..len]);
        })
    }

    /// Visits contiguous runs as (tiled byte offset, linear byte offset, length).
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        tiled_len: usize,
        mip: u32,
        slice: u32,
        linear_len: usize,
        row_pitch: usize,
        depth_pitch: usize,
        mut copy: impl FnMut(usize, usize, usize),
    ) -> Result<(), Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = m.depth;
        let first = if self.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err(Error::Invalid);
            }
            0
        } else {
            slice
        };
        if first >= self.slices {
            return Err(Error::Invalid);
        }
        if !self.addressable() {
            return Err(Error::Unsupported);
        }
        let n = self.desc.element_bytes as usize;
        let bytes = m.width as usize * n;
        let need = (depth as usize - 1)
            .checked_mul(depth_pitch)
            .zip((m.height as usize - 1).checked_mul(row_pitch))
            .and_then(|(depth_offset, row_offset)| depth_offset.checked_add(row_offset))
            .and_then(|offset| offset.checked_add(bytes))
            .ok_or(Error::TooSmall)?;
        if tiled_len < self.size as usize || linear_len < need || row_pitch < bytes {
            return Err(Error::TooSmall);
        }
        let linear = self.desc.swizzle == Swizzle::LINEAR;
        // The row and depth contribute one common base to every column, so
        // the same contiguous runs repeat on every row, including mip tails.
        let mut runs = Vec::new();
        if !linear {
            let mut first_linear = 0;
            let mut first_tiled = self.column(m, 0) as usize;
            for x in 1..m.width {
                let offset = self.column(m, x) as usize;
                let at = x as usize * n;
                if offset != first_tiled + (at - first_linear) {
                    runs.push((first_tiled, first_linear, at - first_linear));
                    first_tiled = offset;
                    first_linear = at;
                }
            }
            runs.push((first_tiled, first_linear, bytes - first_linear));
        }
        for z in 0..depth {
            let (s, inner) = self.split(first + z);
            let base = u64::from(s) * self.slice_size + m.offset;
            for y in 0..m.height {
                let linear_row = z as usize * depth_pitch + y as usize * row_pitch;
                if linear {
                    let tiled_row = base + u64::from(y) * u64::from(m.pitch) * n as u64;
                    copy(tiled_row as usize, linear_row, bytes);
                } else {
                    let tiled_row = (base + self.row(m, y, inner)) as usize;
                    for &(tiled, linear, len) in &runs {
                        copy(tiled_row + tiled, linear_row + linear, len);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Linear rows of one subresource, packed without padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    /// Bytes from one row of elements to the next.
    pub row_pitch: usize,
    /// Bytes from one depth slice to the next.
    pub depth_pitch: usize,
    /// Bytes of the whole subresource.
    pub size: usize,
}

/// What a DXGI format stores per element: bytes, texels per element side
/// (4 for block compression), and whether its stencil is a second plane
/// (the GPU keeps depth and stencil apart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub bytes: u32,
    pub block: u32,
    pub stencil: bool,
}

/// By DXGI format number, with the console's own numbers: 116/117 (10-bit
/// float color), 118-120 (16-bit depth with 8-bit stencil), 189
/// (R10G10B10_SNORM_A2_UNORM) and 190 (R4G4_UNORM). `None` for 96-bit,
/// 1-bit, packed 4:2:2 and video formats.
///
/// shortcut: those formats have no layout here; add them when a title uses
/// one.
pub fn format(dxgi: u32) -> Option<Format> {
    let (bytes, block, stencil) = match dxgi {
        1..=4 => (16, 1, false),
        9..=18 => (8, 1, false),
        19..=22 | 44..=47 => (4, 1, true),
        23..=43 | 67 | 87..=93 | 116 | 117 | 189 => (4, 1, false),
        48..=59 | 85 | 86 | 115 => (2, 1, false),
        118..=120 => (2, 1, true),
        60..=65 | 190 => (1, 1, false),
        70..=72 | 79..=81 => (8, 4, false),
        73..=78 | 82..=84 | 94..=99 => (16, 4, false),
        _ => return None,
    };
    Some(Format {
        bytes,
        block,
        stencil,
    })
}

/// A texture as the console allocates it: a D3D12-style description with
/// a swizzle mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth_or_array: u32,
    /// 0: the full chain.
    pub mips: u32,
    /// DXGI format number.
    pub format: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// From the start of the texture.
    pub offset: u64,
    pub stencil: bool,
    pub layout: Layout,
}

/// Every data plane of a texture (depth, then stencil), placed one after
/// the other.
///
/// shortcut: no compression metadata planes (HTILE, CMASK, FMASK, DCC);
/// the console adds them to render targets and depth buffers unless the
/// description denies them, so those are smaller here than on the console.
/// The host GPU renders them, so nothing reads that memory; add the planes
/// if a title sizes a pool from console sizes and runs out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    pub planes: Vec<Plane>,
    pub size: u64,
    pub alignment: u64,
}

impl Texture {
    pub fn new(desc: &TextureDesc) -> Result<Texture, Error> {
        let f = format(desc.format).ok_or(Error::Invalid)?;
        let depth3d = if desc.dimension == Dimension::Tex3d {
            desc.depth_or_array
        } else {
            1
        };
        let mips = match desc.mips {
            0 => 32 - desc.width.max(desc.height).max(depth3d).max(1).leading_zeros(),
            n => n,
        };
        let plane = |bytes| Desc {
            dimension: desc.dimension,
            width: desc.width,
            height: desc.height,
            depth: desc.depth_or_array,
            mips,
            element_bytes: bytes,
            block: f.block,
            samples: desc.samples.max(1),
            swizzle: desc.swizzle,
        };
        let mut planes = vec![Plane {
            offset: 0,
            stencil: false,
            layout: Layout::new(&plane(f.bytes))?,
        }];
        if f.stencil {
            let layout = Layout::new(&plane(1))?;
            let offset = planes[0].layout.size.next_multiple_of(layout.alignment);
            planes.push(Plane {
                offset,
                stencil: true,
                layout,
            });
        }
        let last = &planes[planes.len() - 1];
        let size = last.offset + last.layout.size;
        let alignment = planes.iter().map(|p| p.layout.alignment).max().unwrap_or(1);
        Ok(Texture {
            planes,
            size,
            alignment,
        })
    }

    /// Mips in each plane's chain (every plane has the same count).
    pub fn mips(&self) -> u32 {
        self.planes[0].layout.mips.len() as u32
    }

    /// Array slices of each plane: 1 for a 3D texture, whose depth slices
    /// belong to each mip's single subresource.
    pub fn array_size(&self) -> u32 {
        let d = &self.planes[0].layout.desc;
        if d.dimension == Dimension::Tex3d {
            1
        } else {
            d.depth
        }
    }

    /// Subresources across every plane: mips × array slices × planes.
    pub fn subresource_count(&self) -> u32 {
        self.mips() * self.array_size() * self.planes.len() as u32
    }

    /// The plane, mip and array slice of a D3D12 subresource index
    /// (`mip + slice * mips + plane * mips * array_size`). `None` past
    /// the last subresource.
    pub fn subresource(&self, index: u32) -> Option<Subresource> {
        if index >= self.subresource_count() {
            return None;
        }
        let mips = self.mips();
        let per_plane = mips * self.array_size();
        Some(Subresource {
            plane: index / per_plane,
            mip: index % mips,
            slice: index % per_plane / mips,
        })
    }

    /// The tightly packed linear rows of subresource `index` (see
    /// [`Layout::footprint`]).
    pub fn footprint(&self, index: u32) -> Option<Footprint> {
        let s = self.subresource(index)?;
        self.planes[s.plane as usize].layout.footprint(s.mip)
    }

    /// Copies subresource `index` out of the whole texture's memory
    /// ([`Texture::size`] bytes) into linear rows, as [`Layout::detile`]
    /// does for its plane.
    pub fn detile(
        &self,
        tiled: &[u8],
        index: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        let (p, s) = self.plane_of(tiled.len(), index)?;
        let at = p.offset as usize;
        p.layout.detile(
            &tiled[at..at + p.layout.size as usize],
            s.mip,
            s.slice,
            linear,
            row_pitch,
            depth_pitch,
        )
    }

    /// The reverse of [`Texture::detile`]: linear rows into the texture.
    pub fn tile(
        &self,
        tiled: &mut [u8],
        index: u32,
        linear: &[u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        let (p, s) = self.plane_of(tiled.len(), index)?;
        let at = p.offset as usize;
        p.layout.tile(
            &mut tiled[at..at + p.layout.size as usize],
            s.mip,
            s.slice,
            linear,
            row_pitch,
            depth_pitch,
        )
    }

    fn plane_of(&self, tiled_len: usize, index: u32) -> Result<(&Plane, Subresource), Error> {
        let s = self.subresource(index).ok_or(Error::Invalid)?;
        if (tiled_len as u64) < self.size {
            return Err(Error::TooSmall);
        }
        Ok((&self.planes[s.plane as usize], s))
    }
}

/// Where a D3D12 subresource index lands in a [`Texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Subresource {
    /// 0 data or depth, 1 stencil.
    pub plane: u32,
    pub mip: u32,
    /// Array slice (0 for a 3D texture).
    pub slice: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: u32, h: u32, mips: u32, bpe: u32, swizzle: Swizzle) -> Desc {
        Desc {
            dimension: Dimension::Tex2d,
            width: w,
            height: h,
            depth: 1,
            mips,
            element_bytes: bpe,
            block: 1,
            samples: 1,
            swizzle,
        }
    }

    /// Like the XMA and BCPack errors, texture errors name the format and
    /// can be boxed as `std::error::Error`.
    #[test]
    fn errors_name_the_format() {
        for e in [Error::Invalid, Error::Unsupported, Error::TooSmall] {
            let text = e.to_string();
            assert!(text.starts_with("texture: "), "{text}");
            let boxed: Box<dyn std::error::Error> = Box::new(e);
            assert_eq!(boxed.to_string(), text);
        }
        assert_ne!(Error::Invalid.to_string(), Error::TooSmall.to_string());
    }

    /// Sizes past the hardware's 16384 texels per side are refused instead
    /// of overflowing the size arithmetic (found by random-input testing).
    #[test]
    fn extents_past_the_hardware_limit_are_invalid() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_64KB_D_X] {
            let largest = Layout::new(&desc(MAX_EXTENT, MAX_EXTENT, 15, 16, swizzle)).unwrap();
            assert!(largest.size >= u64::from(MAX_EXTENT) * u64::from(MAX_EXTENT) * 16);
            for (w, h) in [
                (MAX_EXTENT + 1, 1),
                (1, MAX_EXTENT + 1),
                (u32::MAX, u32::MAX),
            ] {
                let result = Layout::new(&desc(w, h, 1, 16, swizzle));
                assert_eq!(result, Err(Error::Invalid), "{w}x{h}");
            }
            let mut deep = desc(1, 1, 1, 16, swizzle);
            deep.depth = u32::MAX;
            assert_eq!(Layout::new(&deep), Err(Error::Invalid));
        }
    }

    #[test]
    fn blocks_hold_their_size_in_elements() {
        for log2 in [8u32, 12, 16] {
            let swizzle = match log2 {
                8 => Swizzle::SW_256B_S,
                12 => Swizzle::SW_4KB_S,
                _ => Swizzle::SW_64KB_D,
            };
            for e in 0..5 {
                let l = Layout::new(&desc(1, 1, 1, 1 << e, swizzle)).unwrap();
                let [w, h, d] = l.block;
                assert_eq!(u64::from(w * h * d) << e, 1 << log2, "{log2} {e}");
                // The pattern covers exactly the block's coordinates.
                assert_eq!(l.pattern[0].len() as u32, w);
                assert_eq!(l.pattern[1].len() as u32, h);
            }
        }
        // 64 KiB, 16-byte elements: 64x64 (a 256x256 BC7 texture).
        let l = Layout::new(&desc(64, 64, 1, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.block, [64, 64, 1]);
        assert_eq!((l.size, l.alignment), (65536, 65536));
    }

    /// Every element of every mip and slice lands on its own address, inside
    /// the texture, and tiling then detiling gives the rows back.
    #[test]
    fn addresses_are_a_bijection_and_copies_round_trip() {
        let cases = [
            desc(64, 64, 7, 16, Swizzle::SW_64KB_D),
            desc(100, 60, 7, 4, Swizzle::SW_64KB_D),
            desc(250, 175, 8, 8, Swizzle::SW_4KB_S),
            desc(33, 17, 6, 1, Swizzle::SW_4KB_D),
            desc(40, 9, 3, 2, Swizzle::SW_64KB_S),
            desc(30, 20, 5, 4, Swizzle::LINEAR),
            desc(83, 31, 6, 1, Swizzle::SW_256B_S),
            Desc {
                block: 4,
                ..desc(109, 71, 6, 16, Swizzle::SW_256B_D)
            },
            Desc {
                depth: 3,
                ..desc(70, 70, 4, 16, Swizzle::SW_4KB_S)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 16,
                ..desc(64, 32, 7, 4, Swizzle::SW_4KB_S)
            },
        ];
        for d in cases {
            let l = Layout::new(&d).unwrap();
            let mut seen = vec![false; l.size as usize];
            let mut tiled = vec![0u8; l.size as usize];
            let n = d.element_bytes as usize;
            let slices = if d.dimension == Dimension::Tex3d { 1 } else { d.depth };
            for slice in 0..slices {
                for mip in 0..d.mips {
                    let m = l.mips[mip as usize];
                    let active = m.width as usize * n;
                    let row = active + 13;
                    let plane = row * m.height as usize + 11;
                    let linear: Vec<u8> = (0..plane * m.depth as usize)
                        .map(|i| (i * 7 + mip as usize * 13 + slice as usize) as u8)
                        .collect();
                    l.tile(&mut tiled, mip, slice, &linear, row, plane).unwrap();
                    let mut back = vec![0xcc; linear.len()];
                    l.detile(&tiled, mip, slice, &mut back, row, plane).unwrap();
                    for z in 0..m.depth {
                        let base = z as usize * plane;
                        for y in 0..m.height {
                            let start = base + y as usize * row;
                            assert_eq!(back[start..start + active], linear[start..start + active]);
                            assert!(back[start + active..start + row].iter().all(|&b| b == 0xcc));
                            for x in 0..m.width {
                                let at = l.offset(mip, x, y, slice + z).unwrap() as usize;
                                let source = start + x as usize * n;
                                assert_eq!(tiled[at..at + n], linear[source..source + n]);
                                assert!(at + n <= seen.len(), "{d:?} mip {mip} outside");
                                assert!(at.is_multiple_of(n));
                                assert!(!seen[at], "{d:?} mip {mip} ({x},{y},{z}) twice");
                                seen[at] = true;
                            }
                        }
                        assert!(back[base + plane - 11..base + plane].iter().all(|&b| b == 0xcc));
                    }
                }
            }
        }
    }

    /// Hand-checked: in a 64 KiB `D` block of 4-byte elements, x bits go to
    /// address bits 2, 3, 6, 9, 11, 13, 15 and y bits to 4, 5, 7, 8, 10, 12,
    /// 14; a 4 KiB `S` block of 16-byte elements is 16x16 with y first.
    #[test]
    fn hand_checked_addresses() {
        let l = Layout::new(&desc(128, 128, 1, 4, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 2));
        assert_eq!(l.offset(0, 4, 0, 0), Ok(1 << 6));
        assert_eq!(l.offset(0, 0, 4, 0), Ok(1 << 7));
        assert_eq!(l.offset(0, 0, 8, 0), Ok(1 << 8));
        assert_eq!(l.offset(0, 64, 64, 0), Ok((1 << 15) | (1 << 14)));
        let l = Layout::new(&desc(32, 32, 1, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.block, [16, 16, 1]);
        assert_eq!(l.offset(0, 0, 1, 0), Ok(1 << 4));
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 6));
        // Blocks row by row: (16, 0) is the second block, (0, 16) the third.
        assert_eq!(l.offset(0, 16, 0, 0), Ok(4096));
        assert_eq!(l.offset(0, 0, 16, 0), Ok(8192));
    }

    /// A 256x256 BC7 chain in 64 KiB `D`: one tail block holding mips 1-8,
    /// then mip 0; mip 1 sits at x = 32 elements, 32 KiB into the tail.
    #[test]
    fn mip_tail_comes_first() {
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.first_tail_mip, 1);
        assert_eq!(l.size, 2 * 65536);
        assert_eq!(l.mips[0].offset, 65536);
        assert_eq!(
            l.mips[1].tail,
            Some(Tail {
                offset: 32768,
                x: 32,
                y: 0
            })
        );
        assert_eq!(l.offset(1, 0, 0, 0), Ok(32768));
        // 4 KiB blocks hold fewer tail mips: 16x16 BC7 mips stay outside.
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.first_tail_mip, 3);
        assert_eq!(l.mips[2].offset, 4096);
        assert_eq!(l.mips[1].offset, 8192);
        assert_eq!(l.mips[0].offset, 8192 + 16384);
    }

    #[test]
    fn depth_stencil_is_two_planes() {
        let t = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 1920,
            height: 1080,
            depth_or_array: 1,
            mips: 1,
            format: 20,
            samples: 1,
            swizzle: Swizzle::SW_64KB_Z_X,
        })
        .unwrap();
        assert_eq!(t.planes.len(), 2);
        // 4-byte depth in 128x128 blocks, 1-byte stencil in 256x256 blocks.
        assert_eq!(t.planes[0].layout.size, 1920 * 1152 * 4);
        assert_eq!(t.planes[1].offset, t.planes[0].layout.size);
        assert_eq!(t.planes[1].layout.size, 2048 * 1280);
        assert!(!t.planes[0].layout.addressable());
    }

    /// The console's own format numbers have element sizes, so their
    /// textures get a console layout like any other.
    #[test]
    fn console_formats_have_layouts() {
        let element = |bytes, stencil| {
            Some(Format {
                bytes,
                block: 1,
                stencil,
            })
        };
        for (dxgi, want) in [
            // R10G10B10_7E3_A2_FLOAT, R10G10B10_6E4_A2_FLOAT.
            (116, element(4, false)),
            (117, element(4, false)),
            // D16_UNORM_S8_UINT and its typeless views: 16-bit depth, then
            // a stencil plane.
            (118, element(2, true)),
            (119, element(2, true)),
            (120, element(2, true)),
            // R10G10B10_SNORM_A2_UNORM, R4G4_UNORM.
            (189, element(4, false)),
            (190, element(1, false)),
        ] {
            assert_eq!(format(dxgi), want, "format {dxgi}");
            let t = Texture::new(&TextureDesc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 256,
                depth_or_array: 1,
                mips: 0,
                format: dxgi,
                samples: 1,
                swizzle: Swizzle::SW_64KB_R_X,
            })
            .unwrap_or_else(|e| panic!("format {dxgi}: {e:?}"));
            assert_eq!(t.planes[0].layout.desc.element_bytes, want.unwrap().bytes);
        }
        // Video and 1-bit formats still have none.
        for dxgi in [66, 103, 130] {
            assert_eq!(format(dxgi), None, "format {dxgi}");
        }
    }

    /// The standard DXGI numbers: element bytes by bit width, 4x4 blocks
    /// for BC1-BC7 (8 bytes for BC1/BC4, 16 for the rest), a stencil plane
    /// for the depth-stencil families, and nothing for 96-bit, 1-bit,
    /// packed 4:2:2, video or unassigned numbers. A texture of a format
    /// without a layout is refused rather than sized wrong.
    #[test]
    fn dxgi_formats_have_element_sizes() {
        let f = |bytes, block, stencil| {
            Some(Format {
                bytes,
                block,
                stencil,
            })
        };
        for (dxgi, want) in [
            // R32G32B32A32_FLOAT, R16G16B16A16_FLOAT, R32G32_FLOAT.
            (2, f(16, 1, false)),
            (10, f(8, 1, false)),
            (16, f(8, 1, false)),
            // D32_FLOAT_S8X24_UINT: 32-bit depth, then a stencil plane.
            (20, f(4, 1, true)),
            // R10G10B10A2_UNORM, R8G8B8A8_UNORM, D32_FLOAT, R32_FLOAT.
            (24, f(4, 1, false)),
            (28, f(4, 1, false)),
            (40, f(4, 1, false)),
            (41, f(4, 1, false)),
            // D24_UNORM_S8_UINT: depth and stencil kept apart.
            (45, f(4, 1, true)),
            // R8G8_UNORM, D16_UNORM, R16_FLOAT, R8_UNORM, A8_UNORM.
            (49, f(2, 1, false)),
            (55, f(2, 1, false)),
            (54, f(2, 1, false)),
            (61, f(1, 1, false)),
            (65, f(1, 1, false)),
            // R9G9B9E5_SHAREDEXP, B8G8R8A8_UNORM, B5G6R5, B4G4R4A4.
            (67, f(4, 1, false)),
            (87, f(4, 1, false)),
            (85, f(2, 1, false)),
            (115, f(2, 1, false)),
            // BC1, BC2, BC3, BC4, BC5, BC6H, BC7.
            (71, f(8, 4, false)),
            (74, f(16, 4, false)),
            (77, f(16, 4, false)),
            (80, f(8, 4, false)),
            (83, f(16, 4, false)),
            (95, f(16, 4, false)),
            (98, f(16, 4, false)),
            // UNKNOWN, R32G32B32 (96-bit), R1_UNORM, R8G8_B8G8 and
            // G8R8_G8B8 (4:2:2), NV12 and the other video formats, and
            // numbers past the console's own.
            (0, None),
            (6, None),
            (66, None),
            (68, None),
            (69, None),
            (103, None),
            (114, None),
            (121, None),
            (191, None),
            (u32::MAX, None),
        ] {
            assert_eq!(format(dxgi), want, "format {dxgi}");
            let t = Texture::new(&TextureDesc {
                dimension: Dimension::Tex2d,
                width: 64,
                height: 64,
                depth_or_array: 1,
                mips: 1,
                format: dxgi,
                samples: 1,
                swizzle: Swizzle::LINEAR,
            });
            match want {
                None => assert_eq!(t, Err(Error::Invalid), "format {dxgi}"),
                Some(w) => {
                    let t = t.unwrap_or_else(|e| panic!("format {dxgi}: {e:?}"));
                    assert_eq!(t.planes.len(), if w.stencil { 2 } else { 1 });
                    let d = &t.planes[0].layout.desc;
                    assert_eq!((d.element_bytes, d.block), (w.bytes, w.block));
                    // A stencil plane is one byte per texel, never blocked.
                    if let Some(s) = t.planes.get(1) {
                        assert!(s.stencil);
                        assert_eq!((s.layout.desc.element_bytes, s.layout.desc.block), (1, 1));
                        assert!(s.offset >= t.planes[0].layout.size);
                    }
                    assert!(t.size >= u64::from(64 / w.block).pow(2) * u64::from(w.bytes));
                }
            }
        }
    }

    /// `mips: 0` asks for the full chain, down to one texel on the largest
    /// side: the array size does not count, a 3D texture's depth does, and
    /// block-compressed formats count texels, not 4x4 blocks. A depth format
    /// gives its stencil plane the same chain.
    #[test]
    fn a_zero_mip_count_is_the_full_chain() {
        use Dimension::*;
        // R8G8B8A8_UNORM, BC1_UNORM, D32_FLOAT_S8X24_UINT.
        let (rgba8, bc1, d32s8) = (28, 71, 20);
        for (dimension, width, height, depth_or_array, dxgi, swizzle, want) in [
            (Tex2d, 1, 1, 1, rgba8, Swizzle::SW_64KB_S, 1),
            (Tex2d, 256, 256, 1, rgba8, Swizzle::SW_64KB_S, 9),
            (Tex2d, 257, 3, 1, rgba8, Swizzle::SW_4KB_S, 9),
            (Tex2d, 3, 1000, 1, rgba8, Swizzle::LINEAR, 10),
            (Tex2d, 4, 4, 1024, rgba8, Swizzle::SW_4KB_S, 3),
            (Tex1d, 4096, 1, 64, rgba8, Swizzle::LINEAR, 13),
            (Tex3d, 64, 64, 128, rgba8, Swizzle::SW_64KB_S, 8),
            (Tex3d, 8, 8, 1, rgba8, Swizzle::SW_4KB_S, 4),
            (Tex2d, 16, 16, 1, bc1, Swizzle::SW_64KB_S, 5),
            (Tex2d, 1920, 1080, 6, bc1, Swizzle::SW_64KB_S, 11),
            (Tex2d, MAX_EXTENT, 1, 1, rgba8, Swizzle::SW_64KB_S, 15),
            (Tex2d, 512, 512, 1, d32s8, Swizzle::SW_64KB_Z_X, 10),
        ] {
            let t = Texture::new(&TextureDesc {
                dimension,
                width,
                height,
                depth_or_array,
                mips: 0,
                format: dxgi,
                samples: 1,
                swizzle,
            })
            .unwrap_or_else(|e| panic!("{width}x{height}x{depth_or_array}: {e:?}"));
            for plane in &t.planes {
                let mips = &plane.layout.mips;
                let what = format!("{width}x{height}x{depth_or_array} format {dxgi}");
                assert_eq!(mips.len(), want, "{what}");
                // The last mip is one element (one texel, or the block that
                // holds it) on every side.
                let last = mips[want - 1];
                assert_eq!((last.width, last.height), (1, 1), "{what}");
                if dimension == Tex3d {
                    assert_eq!(last.depth, 1, "{what}");
                }
            }
            let stencil = format(dxgi).unwrap().stencil;
            assert_eq!(t.planes.len(), 1 + usize::from(stencil));
        }
    }

    #[test]
    fn texture_copies_reject_overflowing_pitches() {
        let layout = Layout::new(&desc(4, 2, 1, 4, Swizzle::LINEAR)).unwrap();
        let mut tiled = vec![0; layout.size as usize];
        let mut linear = [0; 32];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(
            layout.tile(&mut tiled, 0, 0, &linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
    }

    #[test]
    fn volume_copy_rejects_overflowing_depth_pitch() {
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 3,
            ..desc(4, 2, 1, 4, Swizzle::LINEAR)
        })
        .unwrap();
        let tiled = vec![0; layout.size as usize];
        let mut linear = [0; 96];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, 16, usize::MAX),
            Err(Error::TooSmall)
        );
    }

    #[test]
    fn bad_descriptions_and_buffers_are_refused() {
        assert_eq!(
            Layout::new(&desc(0, 4, 1, 4, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 12, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 4, Swizzle(3))),
            Err(Error::Invalid)
        );
        let l = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_S)).unwrap();
        let mut out = vec![0; 64 * 64 * 4];
        assert_eq!(
            l.detile(&[0; 100], 0, 0, &mut out, 256, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(l.offset(0, 64, 0, 0), Err(Error::Invalid));
    }

    /// The swizzle numbers GFX10 has: linear, 256 B `S`/`D`, 4 KiB and
    /// 64 KiB `S`/`D`, their `_T`/`_X` XOR forms and the `Z_X`/`R_X` modes.
    /// Each tiled mode has one kind, and XOR modes are exactly 16 and up.
    #[test]
    fn swizzle_numbers_classify() {
        let known = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27];
        for n in 0..64 {
            let s = Swizzle(n);
            assert_eq!(s.block_log2().is_some(), known.contains(&n), "{n}");
            let Some(log2) = s.block_log2() else {
                assert_eq!(
                    Layout::new(&desc(4, 4, 1, 4, s)),
                    Err(Error::Invalid),
                    "{n}"
                );
                continue;
            };
            let rotated = matches!(n, 23 | 27);
            let kinds = [s.standard(), s.display(), s.z_order(), rotated];
            let expected = usize::from(n != 0);
            assert_eq!(kinds.iter().filter(|&&k| k).count(), expected, "{n}");
            assert_eq!(s.xor(), n >= 16, "{n}");
            if n != 0 {
                // A 1x1 texture still takes one whole block.
                let l = Layout::new(&desc(1, 1, 1, 4, s)).unwrap();
                assert_eq!((l.size, l.alignment), (1 << log2, 1 << log2), "{n}");
            }
        }
    }

    /// XOR and multisampled layouts have sizes and mips, but their addresses
    /// depend on the GPU's pipes: in-range lookups and copies are
    /// `Unsupported`, out-of-range ones still `Invalid`, and the sizes match
    /// the non-XOR mode with the same block.
    #[test]
    fn xor_and_msaa_layouts_size_but_do_not_address() {
        let pairs = [
            (Swizzle::SW_64KB_S_T, Swizzle::SW_64KB_S),
            (Swizzle::SW_64KB_D_T, Swizzle::SW_64KB_D),
            (Swizzle::SW_4KB_S_X, Swizzle::SW_4KB_S),
            (Swizzle::SW_4KB_D_X, Swizzle::SW_4KB_D),
            (Swizzle::SW_64KB_S_X, Swizzle::SW_64KB_S),
            (Swizzle::SW_64KB_D_X, Swizzle::SW_64KB_D),
        ];
        for (xor, plain) in pairs {
            let d = desc(300, 200, 9, 4, xor);
            let l = Layout::new(&d).unwrap();
            let p = Layout::new(&Desc {
                swizzle: plain,
                ..d
            })
            .unwrap();
            assert!(!l.addressable() && p.addressable(), "{xor:?}");
            assert_eq!((l.size, l.alignment), (p.size, p.alignment), "{xor:?}");
            assert_eq!(l.mips, p.mips, "{xor:?}");
            assert_eq!(l.offset(0, 0, 0, 0), Err(Error::Unsupported));
            assert_eq!(l.offset(0, 300, 0, 0), Err(Error::Invalid));
            assert_eq!(l.offset(9, 0, 0, 0), Err(Error::Invalid));
            let tiled = vec![0u8; l.size as usize];
            let mut out = vec![0u8; 300 * 200 * 4];
            assert_eq!(
                l.detile(&tiled, 0, 0, &mut out, 1200, 0),
                Err(Error::Unsupported)
            );
            assert_eq!(
                l.detile(&tiled, 9, 0, &mut out, 1200, 0),
                Err(Error::Invalid)
            );
        }
        for swizzle in [
            Swizzle::SW_64KB_Z_X,
            Swizzle::SW_64KB_R_X,
            Swizzle::SW_4KB_R_X,
        ] {
            let l = Layout::new(&desc(64, 64, 1, 4, swizzle)).unwrap();
            assert_eq!(l.offset(0, 1, 1, 0), Err(Error::Unsupported), "{swizzle:?}");
        }
        let msaa = Layout::new(&Desc {
            samples: 4,
            ..desc(64, 64, 1, 4, Swizzle::SW_64KB_D)
        })
        .unwrap();
        assert!(!msaa.addressable());
        assert!(msaa.size >= 64 * 64 * 4 * 4);
        assert_eq!(msaa.offset(0, 0, 0, 0), Err(Error::Unsupported));
        // Multisampled textures have one mip and are 2D only.
        for bad in [
            Desc {
                samples: 4,
                ..desc(64, 64, 2, 4, Swizzle::SW_64KB_D)
            },
            Desc {
                samples: 3,
                ..desc(64, 64, 1, 4, Swizzle::SW_64KB_D)
            },
            Desc {
                samples: 16,
                ..desc(64, 64, 1, 4, Swizzle::SW_64KB_D)
            },
        ] {
            assert_eq!(Layout::new(&bad), Err(Error::Invalid), "{bad:?}");
        }
        // Past the chain a subresource covers nothing.
        assert_eq!(msaa.subresource_depth(0), 1);
        assert_eq!(msaa.subresource_depth(1), 0);
    }

    /// Random descriptions, coordinates and copy pitches give layouts or
    /// errors, never a panic or an overflow; every layout and copy stays
    /// inside the buffers it is given.
    #[test]
    fn noise_never_panics() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let dimensions = [Dimension::Tex1d, Dimension::Tex2d, Dimension::Tex3d];
        for _ in 0..3000 {
            // Mostly small sizes so copies are cheap, sometimes anything.
            let mut extent = |small: u64| match next() % 8 {
                0 => next() as u32,
                1 => MAX_EXTENT + (next() % 3) as u32 - 1,
                _ => (next() % small) as u32,
            };
            let (width, height, depth) = (extent(300), extent(300), extent(9));
            let dimension = dimensions[(next() % 3) as usize];
            // Each field is usually one the hardware takes, sometimes not.
            let mut pick = |good: &[u32], any: u64| {
                let r = next();
                if r % 8 == 0 {
                    (r >> 8) as u32 % any as u32
                } else {
                    good[(r >> 8) as usize % good.len()]
                }
            };
            let modes = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27];
            let d = Desc {
                dimension,
                width,
                height,
                depth,
                mips: pick(&[1, 1, 2, 3, 5, 9], 18),
                element_bytes: pick(&[1, 2, 4, 8, 16], 33),
                block: pick(&[1, 1, 4], 6),
                samples: pick(&[1, 1, 1, 2, 4, 8], 17),
                swizzle: Swizzle(pick(&modes, 34)),
            };
            let Ok(l) = Layout::new(&d) else { continue };
            assert!(l.alignment.is_power_of_two(), "{d:?}");
            if l.size > 1 << 22 {
                continue;
            }
            let mut tiled = vec![0u8; l.size as usize];
            for _ in 0..8 {
                let mip = (next() % (u64::from(d.mips) + 2)) as u32;
                // Mostly inside the mip, sometimes just past it.
                let mut coordinate = |n: u32| {
                    (next() % u64::from((n >> mip).div_ceil(d.block.max(1)).clamp(1, 512) + 1))
                        as u32
                };
                let (x, y, z) = (
                    coordinate(d.width),
                    coordinate(d.height),
                    coordinate(d.depth << mip),
                );
                if let Ok(offset) = l.offset(mip, x, y, z) {
                    assert!(
                        offset + u64::from(d.element_bytes) <= l.size,
                        "{d:?} {mip} {x} {y} {z}"
                    );
                }
                let slice = (next() % 10) as u32;
                let row_pitch = match next() % 4 {
                    0 => next() as usize,
                    _ => (next() % 5000) as usize,
                };
                let depth_pitch = match next() % 4 {
                    0 => next() as usize,
                    _ => (next() % 200_000) as usize,
                };
                let mut linear = vec![0u8; (next() % 300_000) as usize];
                let _ = l.detile(&tiled, mip, slice, &mut linear, row_pitch, depth_pitch);
                let _ = l.tile(&mut tiled, mip, slice, &linear, row_pitch, depth_pitch);
            }
        }
        for _ in 0..3000 {
            let desc = TextureDesc {
                dimension: dimensions[(next() % 3) as usize],
                width: (next() % 5000) as u32,
                height: (next() % 5000) as u32,
                depth_or_array: (next() % 70) as u32,
                mips: (next() % 18) as u32,
                format: (next() % 200) as u32,
                samples: [0, 1, 1, 1, 4, 3][(next() % 6) as usize],
                swizzle: Swizzle([0, 5, 9, 22, 24, 26, 27, 30][(next() % 8) as usize]),
            };
            if let Ok(t) = Texture::new(&desc) {
                let last = t.planes.last().unwrap();
                assert_eq!(t.size, last.offset + last.layout.size);
                assert!(t.planes.iter().all(|p| p.offset % p.layout.alignment == 0));
            }
        }
    }

    /// For random addressable layouts (every addressable mode, dimension,
    /// element size, block size, mip count and array size), tiling one
    /// subresource writes exactly that subresource's elements, each where
    /// [`Layout::offset`] says, and nothing else; detiling reads exactly
    /// those bytes back and leaves the linear padding alone. Mips stay
    /// padded to whole elements, and mips outside the tail sit block
    /// aligned, apart from each other, inside their slice.
    #[test]
    fn random_copies_touch_exactly_their_subresource() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let modes = [0, 1, 2, 5, 6, 9, 10];
        let mut checked = 0;
        while checked < 300 {
            let dimension =
                [Dimension::Tex1d, Dimension::Tex2d, Dimension::Tex3d][(next() % 3) as usize];
            let width = 1 + (next() % 150) as u32;
            let height = if dimension == Dimension::Tex1d {
                1
            } else {
                1 + (next() % 150) as u32
            };
            let depth = 1 + (next() % 12) as u32;
            let mips = 1 + (next() % 9) as u32;
            let d = Desc {
                dimension,
                width,
                height,
                depth,
                mips,
                element_bytes: 1 << (next() % 5),
                block: [1, 1, 4][(next() % 3) as usize],
                samples: 1,
                swizzle: Swizzle(modes[(next() % modes.len() as u64) as usize]),
            };
            let Ok(l) = Layout::new(&d) else { continue };
            if l.size > 1 << 21 {
                continue;
            }
            checked += 1;
            assert!(l.addressable(), "{d:?}");
            let three_d = d.dimension == Dimension::Tex3d;
            let mut placed: Vec<(u64, u64)> = vec![];
            for (i, m) in l.mips.iter().enumerate() {
                assert!(m.pitch >= m.width && m.rows >= m.height, "{d:?} mip {i}");
                assert_eq!(
                    m.tail.is_some(),
                    i as u32 >= l.first_tail_mip,
                    "{d:?} mip {i}"
                );
                if m.tail.is_none() && d.swizzle != Swizzle::LINEAR {
                    assert_eq!(m.offset % l.alignment, 0, "{d:?} mip {i}");
                }
                if m.tail.is_none() {
                    let bytes = u64::from(m.pitch)
                        * u64::from(m.rows)
                        * u64::from(if l.thick { l.block[2] } else { 1 })
                        * u64::from(d.element_bytes);
                    placed.push((m.offset, bytes));
                }
            }
            if d.swizzle != Swizzle::LINEAR {
                placed.sort_unstable();
                for w in placed.windows(2) {
                    assert!(w[0].0 + w[0].1 <= w[1].0, "{d:?} mips overlap: {placed:?}");
                }
                if let Some(&(offset, bytes)) = placed.last() {
                    assert!(offset + bytes <= l.slice_size, "{d:?} past its slice");
                }
            }
            let n = d.element_bytes as usize;
            let mip = (next() % u64::from(d.mips)) as u32;
            let m = l.mips[mip as usize];
            let slice = if three_d {
                0
            } else {
                (next() % u64::from(d.depth)) as u32
            };
            let row = m.width as usize * n + (next() % 9) as usize;
            let plane = row * m.height as usize + (next() % 9) as usize;
            let linear: Vec<u8> = (0..plane * m.depth as usize)
                .map(|_| (next() % 255) as u8 + 1)
                .collect();
            let mut tiled = vec![0u8; l.size as usize];
            l.tile(&mut tiled, mip, slice, &linear, row, plane).unwrap();
            let mut expect = vec![0u8; tiled.len()];
            for z in 0..m.depth {
                for y in 0..m.height {
                    for x in 0..m.width {
                        let at = l.offset(mip, x, y, slice + z).unwrap() as usize;
                        let from = z as usize * plane + y as usize * row + x as usize * n;
                        expect[at..at + n].copy_from_slice(&linear[from..from + n]);
                    }
                }
            }
            assert!(
                tiled == expect,
                "{d:?} mip {mip} slice {slice}: tile wrote elsewhere"
            );
            let mut back = vec![0u8; linear.len()];
            l.detile(&tiled, mip, slice, &mut back, row, plane).unwrap();
            for (i, (&b, &a)) in back.iter().zip(&linear).enumerate() {
                let (z, rest) = (i / plane, i % plane);
                let inside = rest / row < m.height as usize && rest % row < m.width as usize * n;
                assert_eq!(
                    b,
                    if inside { a } else { 0 },
                    "{d:?} mip {mip} byte {i} (slice {z})"
                );
            }
        }
    }

    /// Each dimension takes only the modes it can use: 1D needs a height of
    /// 1, 3D takes linear, the `S` modes and the 64 KiB `_X` modes (not 256 B
    /// or plain `D`), and only 2D can be multisampled. 3D `S` and `D` modes
    /// are thick: one block spans several depth slices, and the depth rounds
    /// up to whole blocks. 3D `Z_X`/`R_X` are thin: one slice per block.
    #[test]
    fn dimensions_take_only_their_modes() {
        let known = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27];
        let three_d = [0, 5, 9, 17, 21, 24, 25, 26, 27];
        let with = |dimension, w, h, depth, samples, n| Desc {
            dimension,
            depth,
            samples,
            ..desc(w, h, 1, 4, Swizzle(n))
        };
        for n in known {
            let s = Swizzle(n);
            let log2 = s.block_log2().unwrap();
            assert!(
                Layout::new(&with(Dimension::Tex1d, 64, 1, 3, 1, n)).is_ok(),
                "{n}"
            );
            for bad in [
                with(Dimension::Tex1d, 64, 2, 1, 1, n),
                with(Dimension::Tex1d, 64, 1, 1, 2, n),
                with(Dimension::Tex3d, 8, 8, 8, 2, n),
            ] {
                assert_eq!(Layout::new(&bad), Err(Error::Invalid), "{bad:?}");
            }
            assert!(
                Layout::new(&with(Dimension::Tex2d, 8, 8, 1, 2, n)).is_ok(),
                "{n}"
            );

            let d = with(Dimension::Tex3d, 20, 12, 9, 1, n);
            let Ok(l) = Layout::new(&d) else {
                assert!(!three_d.contains(&n), "{n} refused for 3D");
                continue;
            };
            assert!(three_d.contains(&n), "{n} accepted for 3D");
            assert_eq!(l.thick, s.standard() || s.display(), "{n}");
            let [bw, bh, bd] = l.block;
            if n == 0 {
                // Linear 3D: every depth slice holds the chain.
                assert_eq!((l.block, l.slices), ([64, 1, 1], 9));
            } else if l.thick {
                assert!(bd > 1, "{n}");
                assert_eq!(u64::from(bw * bh * bd * 4), 1u64 << log2, "{n}");
                assert_eq!(l.slices, 9u32.div_ceil(bd), "{n}");
                assert_eq!(l.mips[0].depth_pitch, 9u32.next_multiple_of(bd), "{n}");
            } else {
                assert_eq!(bd, 1, "{n}");
                assert_eq!(u64::from(bw * bh * 4), 1u64 << log2, "{n}");
                assert_eq!(l.slices, 9, "{n}");
            }
            assert_eq!(l.size, l.slice_size * u64::from(l.slices), "{n}");
            assert_eq!(l.mips[0].depth, 9, "{n}");
        }
    }

    /// Linear layouts (`HwlComputeSurfaceInfoLinear`): each row pads to 256
    /// bytes, mips pack smallest first with no gaps, every array or depth
    /// slice holds the whole chain, and an element's address is its row and
    /// column inside that. Hand-checked sizes for odd extents, BC blocks and
    /// 3D.
    #[test]
    fn linear_layouts_pack_rows_and_mips() {
        // (desc, per mip (pitch, rows, offset, width, height, depth), slice size)
        let cases = [
            (
                Desc {
                    depth: 3,
                    ..desc(30, 20, 3, 4, Swizzle::LINEAR)
                },
                vec![
                    (64, 20, 3840, 30, 20, 1),
                    (64, 10, 1280, 15, 10, 1),
                    (64, 5, 0, 7, 5, 1),
                ],
                8960,
            ),
            // Odd extents: placement rounds up, data rounds down.
            (
                desc(7, 5, 2, 16, Swizzle::LINEAR),
                vec![(16, 5, 768, 7, 5, 1), (16, 3, 0, 3, 2, 1)],
                2048,
            ),
            // BC1-sized 4x4 blocks: 64 texels are 16 elements.
            (
                Desc {
                    block: 4,
                    ..desc(64, 64, 2, 8, Swizzle::LINEAR)
                },
                vec![(32, 16, 2048, 16, 16, 1), (32, 8, 0, 8, 8, 1)],
                6144,
            ),
            (
                Desc {
                    dimension: Dimension::Tex3d,
                    depth: 4,
                    ..desc(8, 4, 3, 4, Swizzle::LINEAR)
                },
                vec![
                    (64, 4, 768, 8, 4, 4),
                    (64, 2, 256, 4, 2, 2),
                    (64, 1, 0, 2, 1, 1),
                ],
                1792,
            ),
        ];
        for (d, mips, slice_size) in cases {
            let l = Layout::new(&d).unwrap();
            let got: Vec<_> = l
                .mips
                .iter()
                .map(|m| (m.pitch, m.rows, m.offset, m.width, m.height, m.depth))
                .collect();
            assert_eq!(got, mips, "{d:?}");
            assert!(l.mips.iter().all(|m| m.tail.is_none()), "{d:?}");
            assert_eq!(l.slice_size, slice_size, "{d:?}");
            assert_eq!(
                (l.slices, l.size, l.alignment),
                (d.depth, slice_size * u64::from(d.depth), 256)
            );
            assert_eq!(l.first_tail_mip, d.mips, "{d:?}");
            assert!(l.addressable());
            let bpe = u64::from(d.element_bytes);
            for (i, m) in l.mips.iter().enumerate() {
                assert_eq!(u64::from(m.pitch) * bpe % 256, 0, "{d:?} mip {i}");
                let slices = if d.dimension == Dimension::Tex3d {
                    m.depth
                } else {
                    d.depth
                };
                for z in 0..slices {
                    for (x, y) in [
                        (0, 0),
                        (m.width - 1, 0),
                        (0, m.height - 1),
                        (m.width - 1, m.height - 1),
                    ] {
                        let row = u64::from(y) * u64::from(m.pitch) + u64::from(x);
                        let expected = u64::from(z) * slice_size + m.offset + row * bpe;
                        assert_eq!(l.offset(i as u32, x, y, z), Ok(expected), "{d:?} mip {i}");
                    }
                }
            }
        }
        // One 3D element by hand: depth slice 1 of mip 1, element (1, 1).
        let l = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 4,
            ..desc(8, 4, 3, 4, Swizzle::LINEAR)
        })
        .unwrap();
        assert_eq!(l.offset(1, 1, 1, 1), Ok(1792 + 256 + 65 * 4));
        assert_eq!(l.offset(1, 0, 0, 2), Err(Error::Invalid));
    }

    /// For random valid descriptions in every mode (addressable or not),
    /// the layout describes itself consistently: it keeps the description,
    /// holds one entry per mip with the halved sizes copies use, places the
    /// slices back to back in whole alignment units, keeps every mip padded
    /// and every mip outside the tail aligned, and marks exactly the mips
    /// from `first_tail_mip` on as tail mips.
    #[test]
    fn layouts_describe_themselves_consistently() {
        let mut seed = 0x853c_49e6_748f_ea9bu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let modes = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 23, 24, 25, 26, 27];
        let mut checked = 0;
        while checked < 4000 {
            let dimension =
                [Dimension::Tex1d, Dimension::Tex2d, Dimension::Tex3d][(next() % 3) as usize];
            let mut extent = |n: u64| match next() % 16 {
                0 => MAX_EXTENT - (next() % 3) as u32,
                _ => 1 + (next() % n) as u32,
            };
            let width = extent(2000);
            let height = if dimension == Dimension::Tex1d {
                1
            } else {
                extent(2000)
            };
            let depth = extent(40);
            let mips = 1 + (next() % u64::from(MAX_MIPS)) as u32;
            let samples = if dimension == Dimension::Tex2d && mips == 1 {
                [1, 1, 2, 4, 8][(next() % 5) as usize]
            } else {
                1
            };
            let d = Desc {
                dimension,
                width,
                height,
                depth,
                mips,
                element_bytes: 1 << (next() % 5),
                block: [1, 4][(next() % 2) as usize],
                samples,
                swizzle: Swizzle(modes[(next() % modes.len() as u64) as usize]),
            };
            let Ok(l) = Layout::new(&d) else {
                // Only 3D textures in thin-only modes are refused here.
                assert_eq!(dimension, Dimension::Tex3d, "{d:?}");
                continue;
            };
            checked += 1;
            assert_eq!(Layout::new(&d).as_ref(), Ok(&l), "{d:?}");
            assert_eq!(l.desc, d);
            assert!(l.alignment.is_power_of_two(), "{d:?}");
            assert_eq!(l.size, l.slice_size * u64::from(l.slices), "{d:?}");
            assert_eq!(l.slice_size % l.alignment, 0, "{d:?}");
            assert_eq!(l.mips.len(), d.mips as usize, "{d:?}");
            assert!(l.first_tail_mip <= d.mips, "{d:?}");
            if l.thick {
                assert_eq!(l.slices, d.depth.div_ceil(l.block[2]), "{d:?}");
            } else {
                assert_eq!(l.slices, d.depth, "{d:?}");
            }
            let three_d = dimension == Dimension::Tex3d;
            for (i, m) in l.mips.iter().enumerate() {
                let i = i as u32;
                let at = format!("{d:?} mip {i}");
                assert_eq!(m.width, dim(width, i, d.block), "{at}");
                assert_eq!(m.height, dim(height, i, d.block), "{at}");
                assert_eq!(m.depth, if three_d { dim(depth, i, 1) } else { 1 }, "{at}");
                assert_eq!(l.subresource_depth(i), m.depth, "{at}");
                assert!(m.pitch >= m.width && m.rows >= m.height, "{at}");
                assert_eq!(m.tail.is_some(), i >= l.first_tail_mip, "{at}");
                match m.tail {
                    Some(t) => assert!(u64::from(t.offset) < l.alignment, "{at}"),
                    None => {
                        assert_eq!(m.offset % l.alignment, 0, "{at}");
                        assert!(m.offset < l.slice_size, "{at}");
                        assert_eq!(m.pitch % l.block[0], 0, "{at}");
                    }
                }
                if i > 0 {
                    let p = &l.mips[i as usize - 1];
                    assert!(m.width <= p.width && m.height <= p.height, "{at}");
                    assert!(m.depth <= p.depth, "{at}");
                }
            }
            assert_eq!(l.subresource_depth(d.mips), 0, "{d:?}");
        }
    }

    /// The micro tile tables and the block bits added after them place
    /// every address bit above the element bits exactly once, and give the
    /// 256-byte, thin and thick block dimensions addrlib computes
    /// (`ComputeThinBlockDimension`, `ComputeThickBlockDimension`).
    #[test]
    fn patterns_cover_their_blocks() {
        for kind in 0..3 {
            let thick = kind == 2;
            for e in 0..5u32 {
                // The micro tile alone is 256 bytes.
                let micro = pattern(kind, e, 8);
                let side = |axis: usize| 1u32 << micro[axis].len();
                if thick {
                    let b = BLOCK_256_3D[e as usize];
                    assert_eq!([side(0), side(1), side(2)], b, "kind {kind} e {e}");
                } else {
                    assert_eq!(
                        [side(0), side(1)],
                        BLOCK_256_2D[e as usize],
                        "kind {kind} e {e}"
                    );
                    assert!(micro[2].is_empty(), "kind {kind} e {e}");
                }
                let blocks: &[u32] = if thick { &[12, 16] } else { &[8, 12, 16] };
                for &b in blocks {
                    let axes = pattern(kind, e, b);
                    let at = format!("kind {kind} e {e} block {b}");
                    // Bits 0..e address bytes inside an element; the rest
                    // are coordinate bits, each taken once.
                    let mut seen: Vec<u32> = (0..e).collect();
                    seen.extend(axes.iter().flatten());
                    seen.sort_unstable();
                    assert_eq!(seen, (0..b).collect::<Vec<_>>(), "{at}");
                    // Coordinate bits rise within an axis, so coordinates
                    // keep their order inside a block's rows.
                    for axis in &axes {
                        assert!(axis.windows(2).all(|w| w[0] < w[1]) || kind == 1, "{at}");
                    }
                    let dims = axes.each_ref().map(|a| 1u32 << a.len());
                    let expected = if thick {
                        let amp = (b - 10) / 3;
                        let rest = (b - 10) % 3;
                        let k = BLOCK_1K_3D[e as usize];
                        [
                            k[0] << amp,
                            k[1] << (amp + rest / 2),
                            k[2] << (amp + u32::from(rest != 0)),
                        ]
                    } else {
                        let elements = b - e;
                        let w = elements.div_ceil(2);
                        [1 << w, 1 << (elements - w), 1]
                    };
                    assert_eq!(dims, expected, "{at}");
                    assert_eq!(
                        (u64::from(dims[0]) * u64::from(dims[1]) * u64::from(dims[2])) << e,
                        1u64 << b,
                        "{at}"
                    );
                    // Spreading each axis and adding the element byte
                    // reaches every byte of the block once.
                    let [sx, sy, sz] = axes.each_ref().map(|a| spread(a));
                    let mut hit = vec![false; 1 << b];
                    for &x in &sx {
                        for &y in &sy {
                            for &z in &sz {
                                for byte in 0..1u32 << e {
                                    let o = (x | y | z | byte) as usize;
                                    assert_eq!(x + y + z + byte, o as u32, "{at}");
                                    assert!(!std::mem::replace(&mut hit[o], true), "{at}");
                                }
                            }
                        }
                    }
                    assert!(hit.iter().all(|&h| h), "{at}");
                }
            }
        }
    }

    /// `spread` maps coordinate `c` to the sum of its set bits' targets.
    #[test]
    fn spread_places_coordinate_bits() {
        assert_eq!(spread(&[]), vec![0]);
        assert_eq!(spread(&[3]), vec![0, 8]);
        assert_eq!(spread(&[4, 0]), vec![0, 16, 1, 17]);
        assert_eq!(spread(&[1, 5, 2]), vec![0, 2, 32, 34, 4, 6, 36, 38]);
    }

    /// Mip sizes: `dim` halves texels (to at least 1) and then rounds up to
    /// whole elements; `padded_dim` rounds to elements first and halves
    /// those rounding up, so it is never smaller and at most one larger.
    #[test]
    fn mip_dimensions_round_as_addrlib() {
        assert_eq!(dim(5, 1, 1), 2);
        assert_eq!(padded_dim(5, 1, 1), 3);
        assert_eq!(dim(5, 3, 1), 1);
        assert_eq!(padded_dim(5, 3, 1), 1);
        assert_eq!(dim(17, 0, 4), 5);
        assert_eq!(dim(17, 1, 4), 2);
        assert_eq!(padded_dim(17, 1, 4), 3);
        assert_eq!(dim(1, 15, 4), 1);
        assert_eq!(padded_dim(0, 0, 1), 1);
        assert_eq!(dim(0, 0, 1), 1);
        for texels in 1..=300u32 {
            for block in [1, 4] {
                assert_eq!(dim(texels, 0, block), padded_dim(texels, 0, block));
                for mip in 0..MAX_MIPS {
                    let (d, p) = (dim(texels, mip, block), padded_dim(texels, mip, block));
                    assert!(d >= 1 && (d..=d + 1).contains(&p), "{texels} {mip} {block}");
                    if texels.is_power_of_two() && block == 1 {
                        assert_eq!(d, p, "{texels} {mip}");
                    }
                }
            }
        }
    }

    /// Every mip of 2D arrays and 3D textures, in each addressable mode:
    /// the last element along each edge has an address whose element fits
    /// inside the layout, the first one past an edge (and any mip past the
    /// chain) is refused, and copies refuse slices the texture lacks.
    #[test]
    fn offsets_stop_at_the_edges() {
        let modes = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut checked = 0;
        for swizzle in modes {
            for dimension in [Dimension::Tex2d, Dimension::Tex3d] {
                for (w, h, depth, bpe, mips) in
                    [(37, 19, 5, 4, 6), (300, 7, 3, 8, 9), (1, 1, 2, 16, 1)]
                {
                    let d = Desc {
                        dimension,
                        depth,
                        ..desc(w, h, mips, bpe, swizzle)
                    };
                    let Ok(l) = Layout::new(&d) else {
                        // Thin-only modes have no 3D layout.
                        assert_eq!(dimension, Dimension::Tex3d, "{d:?}");
                        continue;
                    };
                    assert!(l.addressable(), "{d:?}");
                    checked += 1;
                    let three_d = dimension == Dimension::Tex3d;
                    for (i, m) in l.mips.iter().enumerate() {
                        let i = i as u32;
                        let at = format!("{d:?} mip {i}");
                        let zs = if three_d { m.depth } else { depth };
                        let (x, y, z) = (m.width - 1, m.height - 1, zs - 1);
                        for (cx, cy, cz) in [(0, 0, 0), (x, 0, 0), (0, y, 0), (0, 0, z), (x, y, z)]
                        {
                            let offset = l.offset(i, cx, cy, cz).unwrap();
                            assert!(offset + u64::from(bpe) <= l.size, "{at} ({cx},{cy},{cz})");
                        }
                        assert_eq!(l.offset(i, x + 1, 0, 0), Err(Error::Invalid), "{at}");
                        assert_eq!(l.offset(i, 0, y + 1, 0), Err(Error::Invalid), "{at}");
                        assert_eq!(l.offset(i, 0, 0, z + 1), Err(Error::Invalid), "{at}");
                        assert_eq!(
                            l.offset(i, u32::MAX, u32::MAX, u32::MAX),
                            Err(Error::Invalid),
                            "{at}"
                        );
                    }
                    let mips = l.mips.len() as u32;
                    assert_eq!(l.offset(mips, 0, 0, 0), Err(Error::Invalid), "{d:?}");
                    let mut tiled = vec![0; l.size as usize];
                    let row = w as usize * bpe as usize;
                    let mut linear = vec![0; row * h as usize * depth as usize];
                    let depth_pitch = row * h as usize;
                    let past = if three_d { 1 } else { depth };
                    assert_eq!(
                        l.detile(&tiled, 0, past, &mut linear, row, depth_pitch),
                        Err(Error::Invalid),
                        "{d:?}"
                    );
                    assert_eq!(
                        l.tile(&mut tiled, mips, 0, &linear, row, depth_pitch),
                        Err(Error::Invalid),
                        "{d:?}"
                    );
                    assert_eq!(
                        l.detile(&tiled, 0, 0, &mut linear, row, depth_pitch),
                        Ok(()),
                        "{d:?}"
                    );
                }
            }
        }
        assert!(checked >= 30, "{checked}");
    }
    /// Every mip of 2D arrays and 3D textures, linear and tiled: copies
    /// into exactly a mip's footprint succeed and round trip, while one
    /// byte less of buffer or of row pitch is refused.
    #[test]
    fn footprints_are_the_smallest_copies() {
        let mut checked = 0;
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_D] {
            for dimension in [Dimension::Tex2d, Dimension::Tex3d] {
                for (w, h, depth, bpe, mips) in [(37, 19, 3, 4, 6), (9, 70, 2, 16, 7)] {
                    let d = Desc {
                        dimension,
                        depth,
                        ..desc(w, h, mips, bpe, swizzle)
                    };
                    let Ok(l) = Layout::new(&d) else {
                        assert_eq!(dimension, Dimension::Tex3d, "{d:?}");
                        continue;
                    };
                    let three_d = dimension == Dimension::Tex3d;
                    let slices = if three_d { 1 } else { depth };
                    let mut tiled = vec![0u8; l.size as usize];
                    for mip in 0..mips {
                        let f = l.footprint(mip).unwrap();
                        let m = l.mips[mip as usize];
                        let at = format!("{d:?} mip {mip}");
                        assert_eq!(f.row_pitch, (m.width * bpe) as usize, "{at}");
                        assert_eq!(f.depth_pitch, f.row_pitch * m.height as usize, "{at}");
                        assert_eq!(
                            f.size,
                            f.depth_pitch * l.subresource_depth(mip) as usize,
                            "{at}"
                        );
                        for slice in 0..slices {
                            let fill: Vec<u8> = (0..f.size)
                                .map(|i| (i as u32 * 7 + mip * 13 + slice * 29) as u8)
                                .collect();
                            l.tile(&mut tiled, mip, slice, &fill, f.row_pitch, f.depth_pitch)
                                .unwrap();
                            let mut back = vec![0; f.size];
                            l.detile(&tiled, mip, slice, &mut back, f.row_pitch, f.depth_pitch)
                                .unwrap();
                            assert_eq!(back, fill, "{at} slice {slice}");
                            assert_eq!(
                                l.detile(
                                    &tiled,
                                    mip,
                                    slice,
                                    &mut back[1..],
                                    f.row_pitch,
                                    f.depth_pitch
                                ),
                                Err(Error::TooSmall),
                                "{at}"
                            );
                            assert_eq!(
                                l.detile(
                                    &tiled,
                                    mip,
                                    slice,
                                    &mut back,
                                    f.row_pitch - 1,
                                    f.depth_pitch
                                ),
                                Err(Error::TooSmall),
                                "{at}"
                            );
                        }
                        checked += 1;
                    }
                    assert_eq!(l.footprint(mips), None, "{d:?}");
                }
            }
        }
        assert!(checked >= 40, "{checked}");
    }

    /// D3D12 subresource indices run mip-fastest, then array slice, then
    /// plane, and copies by index land in their own plane: filling every
    /// subresource of a depth-stencil array through `Texture::tile` reads
    /// each back unchanged, and the two planes never overlap.
    #[test]
    fn subresource_indices_cover_every_plane() {
        for (dxgi, dimension, array, swizzle) in [
            (20, Dimension::Tex2d, 3, Swizzle::SW_4KB_S),
            (45, Dimension::Tex2d, 2, Swizzle::SW_64KB_D),
            (118, Dimension::Tex2d, 4, Swizzle::LINEAR),
            (28, Dimension::Tex2d, 5, Swizzle::SW_64KB_S),
            (71, Dimension::Tex2d, 2, Swizzle::SW_4KB_S),
            (10, Dimension::Tex3d, 6, Swizzle::SW_4KB_S),
            (28, Dimension::Tex1d, 3, Swizzle::LINEAR),
        ] {
            let t = Texture::new(&TextureDesc {
                dimension,
                width: 37,
                height: if dimension == Dimension::Tex1d { 1 } else { 23 },
                depth_or_array: array,
                mips: 0,
                format: dxgi,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let at = format!("format {dxgi} {dimension:?} {swizzle:?}");
            let planes = t.planes.len() as u32;
            assert_eq!(
                planes,
                if format(dxgi).unwrap().stencil { 2 } else { 1 },
                "{at}"
            );
            let mips = t.mips();
            // 37 texels wide: the full chain is 37, 18, 9, 4, 2, 1.
            assert_eq!(mips, 6, "{at}");
            let slices = if dimension == Dimension::Tex3d {
                1
            } else {
                array
            };
            assert_eq!(t.array_size(), slices, "{at}");
            assert_eq!(t.subresource_count(), mips * slices * planes, "{at}");
            assert_eq!(t.subresource(t.subresource_count()), None, "{at}");
            assert_eq!(t.footprint(t.subresource_count()), None, "{at}");

            let mut tiled = vec![0u8; t.size as usize];
            let mut fills = Vec::new();
            let mut index = 0;
            for plane in 0..planes {
                for slice in 0..slices {
                    for mip in 0..mips {
                        let s = t.subresource(index).unwrap();
                        assert_eq!(s, Subresource { plane, mip, slice }, "{at} {index}");
                        let f = t.footprint(index).unwrap();
                        let l = &t.planes[plane as usize].layout;
                        assert_eq!(Some(f), l.footprint(mip), "{at} {index}");
                        let fill: Vec<u8> = (0..f.size)
                            .map(|i| (i as u32 * 31 + index * 97 + 1) as u8)
                            .collect();
                        t.tile(&mut tiled, index, &fill, f.row_pitch, f.depth_pitch)
                            .unwrap();
                        fills.push(fill);
                        index += 1;
                    }
                }
            }
            for (index, fill) in fills.iter().enumerate() {
                let index = index as u32;
                let f = t.footprint(index).unwrap();
                let mut back = vec![0; f.size];
                t.detile(&tiled, index, &mut back, f.row_pitch, f.depth_pitch)
                    .unwrap();
                assert_eq!(&back, fill, "{at} subresource {index}");
            }
            // The stencil plane's bytes sit past the depth plane's.
            if planes == 2 {
                let depth_end = t.planes[0].layout.size as usize;
                assert!(t.planes[1].offset as usize >= depth_end, "{at}");
                let mut copy = tiled.clone();
                copy[t.planes[1].offset as usize..].fill(0);
                for index in 0..mips * slices {
                    let f = t.footprint(index).unwrap();
                    let mut back = vec![0; f.size];
                    t.detile(&copy, index, &mut back, f.row_pitch, f.depth_pitch)
                        .unwrap();
                    assert_eq!(back, fills[index as usize], "{at} depth {index}");
                }
            }
            // Past the end, a short texture buffer, and a short linear one.
            let f = t.footprint(0).unwrap();
            let mut back = vec![0; f.size];
            let n = t.subresource_count();
            assert_eq!(
                t.detile(&tiled, n, &mut back, f.row_pitch, f.depth_pitch),
                Err(Error::Invalid),
                "{at}"
            );
            assert_eq!(
                t.detile(&tiled[1..], 0, &mut back, f.row_pitch, f.depth_pitch),
                Err(Error::TooSmall),
                "{at}"
            );
            assert_eq!(
                t.tile(&mut tiled, 0, &back[1..], f.row_pitch, f.depth_pitch),
                Err(Error::TooSmall),
                "{at}"
            );
        }
    }
}
