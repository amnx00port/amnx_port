//! [`texture::format`] against the public `DXGI_FORMAT` enumeration, entry
//! by entry: every named format number from 0 to 132 is listed here by
//! name with the size of one element (a texel, or a 4x4 block) and whether
//! its stencil lives in a separate plane, or as having no layout. The
//! table is written from the format names alone (their channel widths and
//! the BCn block sizes), not from the source under test, so a slip in the
//! source's number ranges shows up as a named format. Nothing here reads
//! data from outside the repository.

use victory_formats::texture::{self, Format};

/// What a named format should store: element bytes, texels per element
/// side, and a separate stencil plane.
#[derive(Clone, Copy)]
enum Want {
    Plain(u32),
    Block(u32),
    /// The depth plane's bytes; the stencil plane is one byte.
    DepthStencil(u32),
    /// 96-bit, 1-bit, packed 4:2:2, video and palettized formats.
    NoLayout,
}

use Want::{Block, DepthStencil, NoLayout as No, Plain};

/// `DXGI_FORMAT` 0..=132 in enumeration order. Gaps in the enumeration
/// (133..) are not listed; numbers 116-120 and 189/190 are the console's
/// own and checked separately.
const DXGI: [(u32, &str, Want); 116] = [
    (0, "UNKNOWN", No),
    (1, "R32G32B32A32_TYPELESS", Plain(16)),
    (2, "R32G32B32A32_FLOAT", Plain(16)),
    (3, "R32G32B32A32_UINT", Plain(16)),
    (4, "R32G32B32A32_SINT", Plain(16)),
    (5, "R32G32B32_TYPELESS", No),
    (6, "R32G32B32_FLOAT", No),
    (7, "R32G32B32_UINT", No),
    (8, "R32G32B32_SINT", No),
    (9, "R16G16B16A16_TYPELESS", Plain(8)),
    (10, "R16G16B16A16_FLOAT", Plain(8)),
    (11, "R16G16B16A16_UNORM", Plain(8)),
    (12, "R16G16B16A16_UINT", Plain(8)),
    (13, "R16G16B16A16_SNORM", Plain(8)),
    (14, "R16G16B16A16_SINT", Plain(8)),
    (15, "R32G32_TYPELESS", Plain(8)),
    (16, "R32G32_FLOAT", Plain(8)),
    (17, "R32G32_UINT", Plain(8)),
    (18, "R32G32_SINT", Plain(8)),
    (19, "R32G8X24_TYPELESS", DepthStencil(4)),
    (20, "D32_FLOAT_S8X24_UINT", DepthStencil(4)),
    (21, "R32_FLOAT_X8X24_TYPELESS", DepthStencil(4)),
    (22, "X32_TYPELESS_G8X24_UINT", DepthStencil(4)),
    (23, "R10G10B10A2_TYPELESS", Plain(4)),
    (24, "R10G10B10A2_UNORM", Plain(4)),
    (25, "R10G10B10A2_UINT", Plain(4)),
    (26, "R11G11B10_FLOAT", Plain(4)),
    (27, "R8G8B8A8_TYPELESS", Plain(4)),
    (28, "R8G8B8A8_UNORM", Plain(4)),
    (29, "R8G8B8A8_UNORM_SRGB", Plain(4)),
    (30, "R8G8B8A8_UINT", Plain(4)),
    (31, "R8G8B8A8_SNORM", Plain(4)),
    (32, "R8G8B8A8_SINT", Plain(4)),
    (33, "R16G16_TYPELESS", Plain(4)),
    (34, "R16G16_FLOAT", Plain(4)),
    (35, "R16G16_UNORM", Plain(4)),
    (36, "R16G16_UINT", Plain(4)),
    (37, "R16G16_SNORM", Plain(4)),
    (38, "R16G16_SINT", Plain(4)),
    (39, "R32_TYPELESS", Plain(4)),
    (40, "D32_FLOAT", Plain(4)),
    (41, "R32_FLOAT", Plain(4)),
    (42, "R32_UINT", Plain(4)),
    (43, "R32_SINT", Plain(4)),
    (44, "R24G8_TYPELESS", DepthStencil(4)),
    (45, "D24_UNORM_S8_UINT", DepthStencil(4)),
    (46, "R24_UNORM_X8_TYPELESS", DepthStencil(4)),
    (47, "X24_TYPELESS_G8_UINT", DepthStencil(4)),
    (48, "R8G8_TYPELESS", Plain(2)),
    (49, "R8G8_UNORM", Plain(2)),
    (50, "R8G8_UINT", Plain(2)),
    (51, "R8G8_SNORM", Plain(2)),
    (52, "R8G8_SINT", Plain(2)),
    (53, "R16_TYPELESS", Plain(2)),
    (54, "R16_FLOAT", Plain(2)),
    (55, "D16_UNORM", Plain(2)),
    (56, "R16_UNORM", Plain(2)),
    (57, "R16_UINT", Plain(2)),
    (58, "R16_SNORM", Plain(2)),
    (59, "R16_SINT", Plain(2)),
    (60, "R8_TYPELESS", Plain(1)),
    (61, "R8_UNORM", Plain(1)),
    (62, "R8_UINT", Plain(1)),
    (63, "R8_SNORM", Plain(1)),
    (64, "R8_SINT", Plain(1)),
    (65, "A8_UNORM", Plain(1)),
    (66, "R1_UNORM", No),
    (67, "R9G9B9E5_SHAREDEXP", Plain(4)),
    (68, "R8G8_B8G8_UNORM", No),
    (69, "G8R8_G8B8_UNORM", No),
    (70, "BC1_TYPELESS", Block(8)),
    (71, "BC1_UNORM", Block(8)),
    (72, "BC1_UNORM_SRGB", Block(8)),
    (73, "BC2_TYPELESS", Block(16)),
    (74, "BC2_UNORM", Block(16)),
    (75, "BC2_UNORM_SRGB", Block(16)),
    (76, "BC3_TYPELESS", Block(16)),
    (77, "BC3_UNORM", Block(16)),
    (78, "BC3_UNORM_SRGB", Block(16)),
    (79, "BC4_TYPELESS", Block(8)),
    (80, "BC4_UNORM", Block(8)),
    (81, "BC4_SNORM", Block(8)),
    (82, "BC5_TYPELESS", Block(16)),
    (83, "BC5_UNORM", Block(16)),
    (84, "BC5_SNORM", Block(16)),
    (85, "B5G6R5_UNORM", Plain(2)),
    (86, "B5G5R5A1_UNORM", Plain(2)),
    (87, "B8G8R8A8_UNORM", Plain(4)),
    (88, "B8G8R8X8_UNORM", Plain(4)),
    (89, "R10G10B10_XR_BIAS_A2_UNORM", Plain(4)),
    (90, "B8G8R8A8_TYPELESS", Plain(4)),
    (91, "B8G8R8A8_UNORM_SRGB", Plain(4)),
    (92, "B8G8R8X8_TYPELESS", Plain(4)),
    (93, "B8G8R8X8_UNORM_SRGB", Plain(4)),
    (94, "BC6H_TYPELESS", Block(16)),
    (95, "BC6H_UF16", Block(16)),
    (96, "BC6H_SF16", Block(16)),
    (97, "BC7_TYPELESS", Block(16)),
    (98, "BC7_UNORM", Block(16)),
    (99, "BC7_UNORM_SRGB", Block(16)),
    (100, "AYUV", No),
    (101, "Y410", No),
    (102, "Y416", No),
    (103, "NV12", No),
    (104, "P010", No),
    (105, "P016", No),
    (106, "420_OPAQUE", No),
    (107, "YUY2", No),
    (108, "Y210", No),
    (109, "Y216", No),
    (110, "NV11", No),
    (111, "AI44", No),
    (112, "IA44", No),
    (113, "P8", No),
    (114, "A8P8", No),
    (115, "B4G4R4A4_UNORM", Plain(2)),
];

/// The tail of the enumeration past the console's numbers.
const DXGI_LATE: [(u32, &str, Want); 3] = [(130, "P208", No), (131, "V208", No), (132, "V408", No)];

fn expected(want: Want) -> Option<Format> {
    Some(match want {
        Plain(bytes) => Format {
            bytes,
            block: 1,
            stencil: false,
        },
        Block(bytes) => Format {
            bytes,
            block: 4,
            stencil: false,
        },
        DepthStencil(bytes) => Format {
            bytes,
            block: 1,
            stencil: true,
        },
        No => return None,
    })
}

/// The table is the enumeration in order, with no number skipped.
#[test]
fn the_table_lists_every_number_in_order() {
    for (i, &(n, name, _)) in DXGI.iter().enumerate() {
        assert_eq!(n, i as u32, "{name}");
    }
    for (i, &(n, name, _)) in DXGI_LATE.iter().enumerate() {
        assert_eq!(n, 130 + i as u32, "{name}");
    }
}

/// Every named format gets the element size its channels or block call
/// for, and only the depth-stencil families get a stencil plane.
#[test]
fn named_formats_match_their_channels() {
    for &(n, name, want) in DXGI.iter().chain(&DXGI_LATE) {
        assert_eq!(texture::format(n), expected(want), "{n} {name}");
    }
}

/// Typed and typeless members of one family store the same element.
#[test]
fn families_share_one_element() {
    let families: [&[u32]; 13] = [
        &[1, 2, 3, 4],
        &[9, 10, 11, 12, 13, 14],
        &[15, 16, 17, 18],
        &[19, 20, 21, 22],
        &[23, 24, 25],
        &[27, 28, 29, 30, 31, 32],
        &[33, 34, 35, 36, 37, 38],
        &[39, 40, 41, 42, 43],
        &[44, 45, 46, 47],
        &[48, 49, 50, 51, 52],
        &[53, 54, 55, 56, 57, 58, 59],
        &[60, 61, 62, 63, 64],
        &[87, 90, 91],
    ];
    for family in families {
        let first = texture::format(family[0]);
        assert!(first.is_some(), "{family:?}");
        for &n in family {
            assert_eq!(texture::format(n), first, "{n} in {family:?}");
        }
    }
}

/// The console's own numbers: 10-bit float color (116/117) and
/// R10G10B10_SNORM_A2_UNORM (189) are 32-bit, 16-bit depth with 8-bit
/// stencil (118-120) splits into a 2-byte depth plane and a stencil
/// plane, and R4G4_UNORM (190) is one byte.
#[test]
fn console_numbers_match_their_channels() {
    for (n, want) in [
        (116, Plain(4)),
        (117, Plain(4)),
        (118, DepthStencil(2)),
        (119, DepthStencil(2)),
        (120, DepthStencil(2)),
        (189, Plain(4)),
        (190, Plain(1)),
    ] {
        assert_eq!(texture::format(n), expected(want), "{n}");
    }
}

/// Numbers between the enumeration's end and the console's, and past the
/// console's, have no layout.
#[test]
fn unnamed_numbers_have_no_layout() {
    for n in (121..130).chain(133..189).chain(191..=1024) {
        assert_eq!(texture::format(n), None, "{n}");
    }
    assert_eq!(texture::format(u32::MAX), None);
}
