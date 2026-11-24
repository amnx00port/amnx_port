//! Native GFX10 programs that the console compiler stores in a `DXIL` part.
//!
//! An `XDAT` index after the program header lists blobs stored around the
//! bitcode; one or more of them hold native programs, zlib-packed (a vertex
//! shader can carry two: compiled for use with and without a geometry
//! shader). Layout as observed in shipped titles (index offsets are relative
//! to the part data):
//!
//! ```text
//! +0x18 "XDAT"  u32 entry count, u32 (0x24 seen), then per entry:
//!               u32 kind, u32 offset, u32 size
//! kind 0x0d     "XBCH" u32 version (3), u64 packed size, u64 unpacked size,
//!               zlib stream
//! unpacked      "FSSM" u32 version (0x101), u64 payload size, payload
//! payload +0x00 u64 graphics slot count (6)
//!         +0x08 u64 record offsets: LS HS ES GS VS PS (0x08..0x30),
//!               mesh 0x48, compute 0x60; unused ones are 0xDEADBEEF_DEADBEEF
//! record  +0x00 u64 code offset, +0x08 u64 code size (payload-relative),
//!         +0x14 u32 flags: bit 5 wave32 (else wave64)
//!         +0x28 u64 root layout table, +0x30 u64 used-resource table
//!               (payload-relative), +0x38 u32 and +0x3c u32 their counts
//!         +0x50 u32 bits 0-9 VGPRs used, bits 20-27 SGPRs used
//! layout  20 B: u32 root type, u32 table offset (dwords), u32 user SGPRs,
//!               u32 base register, u8 first user-data dword, u8 root slot,
//!               u16 0
//! used    20 B: u8 resource type, u8 root type, u16 0, u32 table offset
//!               (dwords), u32 lower bound, u32 stride, u8 descriptor
//!               dwords, u8 first user-data dword, u8 root slot, u8 0
//! ```
//!
//! Stage data follows the record; offsets from its end (E = record + 0x54):
//!
//! ```text
//! PS  E+0x24  32 inputs of 8 B: u8 usage, u8 signature row, u8 channel
//!             mask, u8 attribute (v_interp `attr`), u8 default, u8 second
//!             row (of a packed pair), u8 parameter slot, u8 flags (bit 0
//!             flat, 1 linear, 3 fp16, 4 fp16 packed); E+0x124 u32 count
//!     E+0x134 u32 SPI_PS_INPUT_ADDR, E+0x138 u32 DB_SHADER_CONTROL
//!     E+0x148 u32 export patch size (dwords), E+0x14c 8 patches, one per
//!             MRT: u32 code offset (dwords, -1 none), then the export code
//!             for each color format in 10 slots of 10 dwords (ZERO, 32_R,
//!             32_GR, 32_AR, FP16, UNORM16, SNORM16, UINT16, SINT16, 32_ABGR)
//! VS  E+0x24  32 inputs of 5 B: u8 usage, u8 row, u8 channel mask, u8 VGPR
//!             of channel 0 (channel k in VGPR + k), u8 highest channel;
//!             E+0xc4 u32 count
//!     E+0xc8  32 outputs of 6 B: u8 usage, u8 row, u8 parameter (`exp
//!             param<n>`), u8 row again, u8 slot, u8 flags (bit 0 fp16,
//!             bit 1 two fp16 values per dword); E+0x188 u32 count
//! CS  E+0x24  u32 COMPUTE_PGM_RSRC2, E+0x28 u32 x3 thread-group size
//! MS  E+0xc8  outputs as VS, E+0x188 count; the param byte's bit 7 marks a
//!             per-primitive output (`PSG1` row); the 3 bytes more are the
//!             second row of a packed pair, a slot, flags (bit 0 fp16, bit 1
//!             two fp16 elements packed)
//!     E+0x18c u32 SPI_VS_OUT_CONFIG, SPI_SHADER_POS_FORMAT,
//!             PA_CL_VS_OUT_CNTL, VGT_STRMOUT_CONFIG, SPI_SHADER_PGM_RSRC2_GS,
//!             VGT_GS_OUT_PRIM_TYPE, VGT_GS_INSTANCE_CNT, SPI_SHADER_IDX_FORMAT
//!     E+0x448 u16 thread-group size in lanes, u16 maximum vertices,
//!             u16 maximum primitives, u8 user SGPRs, u8 output topology
//!             (2 triangles); code follows at E+0x45c
//! record +0x4c u32 LDS bytes (`lds_byte_size`; mesh programs read it here)
//! ```
//!
//! Rows are registers of the container's signature parts (`ISG1`, `OSG1`).
//! Merged stages keep their stage data in another layout (not read).
//!
//! Other kinds seen: 0x07 program header + bitcode (or a stub when stripped),
//! 0x0a `xbop` (compiler options), 0x0e a table not decoded yet, 0x10 and
//! 0x12 16-byte blobs (0x10 is the shader hash).
//! Unknown versions are rejected so a format change shows up as an error.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{u32_at, u64_at};

const XDAT_OFFSET: usize = 0x18;
const MAX_XDAT_ENTRIES: usize = 64;
const KIND_NATIVE: u32 = 0x0d;
const XBCH_VERSION: u32 = 3;
const XBCH_HEADER: usize = 0x18;
const FSSM_VERSION: u32 = 0x101;
const FSSM_HEADER: usize = 0x10;
const GRAPHICS_SLOT_COUNT: u64 = 6;
/// Where each stage's record offset sits in the FSSM payload.
const SLOTS: [(HwStage, usize); 8] = [
    (HwStage::Ls, 0x08),
    (HwStage::Hs, 0x10),
    (HwStage::Es, 0x18),
    (HwStage::Gs, 0x20),
    (HwStage::Vs, 0x28),
    (HwStage::Ps, 0x30),
    (HwStage::Ms, 0x48),
    (HwStage::Cs, 0x60),
];
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const RECORD_SIZE: u64 = 0x54;
const RECORD_FLAGS: usize = 0x14;
const FLAG_WAVE32: u32 = 1 << 5;
const RECORD_REGS: usize = 0x50;
const RECORD_LAYOUT: usize = 0x28;
const RECORD_USED: usize = 0x30;
const RECORD_COUNTS: usize = 0x38;
const ENTRY_SIZE: u64 = 20;
/// `s0`-`s105`: the SGPRs a program can address (106 on is VCC).
const SGPRS: u32 = 106;
const MAX_SEMANTICS: u32 = 32;
const PS_INPUTS: usize = 0x24;
const PS_INPUT_COUNT: usize = 0x124;
const PS_INPUT_ADDR: usize = 0x134;
const PS_DB_SHADER_CONTROL: usize = 0x138;
const PS_PATCH_SIZE: usize = 0x148;
const PS_PATCHES: usize = 0x14c;
const PATCH_SLOT_DWORDS: usize = 10;
/// The variant that exports every channel as a raw 32-bit value.
const PATCH_FULL: usize = 9;
const PATCH_BYTES: usize = 4 + 10 * PATCH_SLOT_DWORDS * 4;
const MRTS: usize = 8;
const VS_INPUTS: usize = 0x24;
const VS_INPUT_COUNT: usize = 0xc4;
const VS_OUTPUTS: usize = 0xc8;
const VS_OUTPUT_COUNT: usize = 0x188;
const CS_RSRC2: usize = 0x24;
const CS_THREADS: usize = 0x28;
/// D3D12's `numthreads` limits: x and y up to 1024, z up to 64, and at
/// most 1024 threads in the group.
const MAX_GROUP: [u32; 3] = [1024, 1024, 64];
const MAX_GROUP_THREADS: u64 = 1024;
const RECORD_LDS: usize = 0x4c;
const MS_CLIP_CNTL: usize = 0x194;
const MS_INFO: usize = 0x448;
/// The LDS a GFX10 thread group can address.
const MAX_LDS_BYTES: u32 = 64 << 10;
/// D3D12's mesh limits: at most 128 threads in the group, and at most 256
/// vertices and 256 primitives out.
const MAX_MESH_THREADS: u32 = 128;
const MAX_MESH_OUTPUTS: u32 = 256;
/// The furthest dword a table read may name: D3D12's tier 2 limit of a
/// million descriptors in a shader-visible heap, of 8 dwords each.
const MAX_TABLE_DWORDS: u32 = 1_000_000 * 8;
/// Unpacked blobs are a few KiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: u64 = 16 << 20;

/// Hardware shader stage a native program runs as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HwStage {
    Ls,
    Hs,
    Es,
    Gs,
    Vs,
    Ps,
    /// Mesh shader.
    Ms,
    Cs,
}

impl HwStage {
    /// The SGPR holding user-data dword `index`: merged stages (and mesh
    /// shaders, which run as one) have 8 system SGPRs first.
    pub fn user_sgpr(self, index: u8) -> u8 {
        match self {
            HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs | HwStage::Ms => index + 8,
            HwStage::Vs | HwStage::Ps | HwStage::Cs => index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProgram {
    pub stage: HwStage,
    /// Runs as 64 lanes per wave (else 32); sizes lane masks.
    pub wave64: bool,
    /// VGPRs the program uses (v0 up to this), the bound of indexed access.
    pub vgprs: u16,
    /// Where the root signature's parameters arrive (user SGPRs).
    pub layout: Vec<RootParam>,
    /// The descriptors the program reads, by root parameter.
    pub used: Vec<UsedResource>,
    pub io: StageIo,
    /// GFX10 machine code, as stored.
    pub code: Vec<u8>,
}

impl NativeProgram {
    /// The code with each render target's export patched for its full
    /// 32-bit format, so exports carry the shader's values unconverted (as
    /// the console does at pipeline creation for that format).
    pub fn patched_code(&self) -> Vec<u8> {
        let mut code = self.code.clone();
        if let StageIo::Ps { patches, .. } = &self.io {
            for p in patches {
                let at = p.offset as usize * 4;
                for (i, word) in p.code.iter().enumerate() {
                    code[at + 4 * i..at + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        code
    }
}

/// What a semantic is, as the stage data names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Usage {
    Generic,
    ClipDistance,
    Other(u8),
}

impl Usage {
    fn from_raw(raw: u8) -> Self {
        match raw {
            6 => Usage::Generic,
            7 => Usage::ClipDistance,
            n => Usage::Other(n),
        }
    }
}

/// A pixel shader input: what `v_interp` of attribute `attr` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsInput {
    pub usage: Usage,
    /// Register in the input signature.
    pub row: u8,
    pub mask: u8,
    pub attr: u8,
    pub flat: bool,
    pub fp16: bool,
    /// Two 16-bit values per dword ([`VsOutput::pack2`]); `row2` is the
    /// second element's register.
    pub pack2: bool,
    pub row2: u8,
}

/// A vertex attribute the fetch shader loads: channel k into VGPR `vgpr + k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsInput {
    pub usage: Usage,
    pub row: u8,
    pub mask: u8,
    pub vgpr: u8,
}

/// What `exp param<param>` of a vertex shader writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsOutput {
    pub usage: Usage,
    /// Register in the output signature.
    pub row: u8,
    pub param: u8,
    /// 16-bit values, one per dword (low half) unless `pack2`: then two per
    /// dword, channel c of this row in dword c / 2 (odd channels in the
    /// high half), then a second 16-bit row from the next whole dword (the
    /// next 16-bit row of the signature no entry names).
    pub fp16: bool,
    pub pack2: bool,
}

/// The export code of one render target for its full 32-bit format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPatch {
    pub mrt: u8,
    /// Where it goes in the code, in dwords.
    pub offset: u32,
    pub code: Vec<u32>,
}

/// Inputs and outputs, for the stages whose stage data is read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum StageIo {
    #[default]
    None,
    Vs {
        inputs: Vec<VsInput>,
        outputs: Vec<VsOutput>,
    },
    Ps {
        inputs: Vec<PsInput>,
        /// `SPI_PS_INPUT_ADDR`: which system values arrive in VGPRs.
        input_addr: u32,
        /// `DB_SHADER_CONTROL`: depth export, kill.
        db_shader_control: u32,
        patches: Vec<ExportPatch>,
    },
    Cs(ComputeIo),
    Ms(MeshIo),
}

/// A mesh program's outputs and limits. It runs as one hardware NGG stage:
/// lane n of the group (`threads` lanes) exports vertex n (`exp pos`,
/// `exp param`) and primitive n (`exp prim`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshIo {
    /// `exp param<n>` targets; `param` bit 7 set: a per-primitive output,
    /// `row` then a row of the primitive signature (`PSG1`).
    pub outputs: Vec<VsOutput>,
    /// `PA_CL_VS_OUT_CNTL`: which clip and cull distances `pos1+` carry.
    pub clip_cntl: u32,
    /// Thread-group size in lanes (the HLSL group rounded up to waves).
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
    pub lds_bytes: u32,
}

/// A compute program's dispatch settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeIo {
    /// `COMPUTE_PGM_RSRC2`: user SGPRs, which system SGPRs and VGPRs
    /// follow them, LDS size.
    pub rsrc2: u32,
    /// Thread-group size (`numthreads`).
    pub threads: [u32; 3],
}

impl ComputeIo {
    /// User SGPRs before the system ones.
    pub fn user_sgprs(&self) -> u8 {
        (self.rsrc2 >> 1 & 0x1F) as u8
    }

    /// The SGPR of each enabled thread-group id component, in order after
    /// the user SGPRs, then the thread-group info SGPR if enabled.
    pub fn system_sgprs(&self) -> ([Option<u8>; 3], Option<u8>) {
        let mut next = self.user_sgprs();
        let mut take = |bit: u32| {
            (self.rsrc2 >> bit & 1 != 0).then(|| {
                next += 1;
                next - 1
            })
        };
        let ids = [take(7), take(8), take(9)];
        (ids, take(10))
    }

    /// SGPRs the hardware initializes: the user SGPRs, the enabled system
    /// ones, then the scratch wave offset when scratch is enabled (bit 0).
    /// The ones above hold whatever the previous wave left.
    pub fn initialized_sgprs(&self) -> u8 {
        let (ids, info) = self.system_sgprs();
        let end = ids.into_iter().chain([info]).flatten().map(|s| s + 1).max();
        end.unwrap_or(0).max(self.user_sgprs()) + (self.rsrc2 & 1) as u8
    }

    /// Thread id components in v0, v1, v2 (1 to 3). `TIDIG_COMP_CNT` is a
    /// 2-bit field, but a group id has only three components: 3 loads x, y
    /// and z like 2 does.
    pub fn thread_id_vgprs(&self) -> u8 {
        (self.rsrc2 >> 11 & 3).min(2) as u8 + 1
    }

    /// LDS bytes per thread group (512-byte granules).
    pub fn lds_bytes(&self) -> u32 {
        (self.rsrc2 >> 15 & 0x1FF) * 512
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootType {
    Constants,
    /// A 64-bit buffer address.
    Descriptor,
    /// The 32-bit heap address of a descriptor table range.
    Table,
    /// A sampler the root signature fixes.
    ImmediateSampler,
    /// The 32-bit address of the CBV/SRV/UAV heap (SM 6.6 dynamic resources).
    ResourceHeap,
    /// The 32-bit address of the sampler heap.
    SamplerHeap,
    Other(u32),
}

impl RootType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            3 => RootType::Table,
            4 => RootType::ImmediateSampler,
            5 => RootType::ResourceHeap,
            6 => RootType::SamplerHeap,
            n => RootType::Other(n),
        }
    }
}

/// One root parameter (or table range) and the user SGPRs holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParam {
    pub root: RootType,
    /// Of the range within its table.
    pub table_offset_dwords: u32,
    /// User SGPRs from `user_data` on.
    pub sgprs: u32,
    /// Shader register of the first descriptor (`b`/`t`/`u`/`s` number).
    pub base_register: u32,
    /// First user-data dword ([`HwStage::user_sgpr`]).
    pub user_data: u8,
    /// Index of the parameter in the root signature.
    pub slot: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Srv,
    Sampler,
    Cbv,
    Uav,
    Other(u8),
}

/// A descriptor range the program reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedResource {
    pub kind: ResourceType,
    pub root: RootType,
    pub table_offset_dwords: u32,
    /// Shader register of the first descriptor read.
    pub lower_bound: u32,
    pub stride: u32,
    /// Size of the descriptor (4 or 8).
    pub dwords: u8,
    pub user_data: u8,
    pub slot: u8,
}

/// The native programs in a `DXIL` part's data; empty for the PC layout.
pub fn native_programs(part: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    if part.get(XDAT_OFFSET..XDAT_OFFSET + 4) != Some(b"XDAT") {
        return Ok(Vec::new());
    }
    let count = u32_at(part, XDAT_OFFSET + 4)? as usize;
    if count > MAX_XDAT_ENTRIES {
        return Err(Malformed("implausible XDAT entry count"));
    }
    let mut out = Vec::new();
    for i in 0..count {
        let at = XDAT_OFFSET + 12 + 12 * i;
        if u32_at(part, at)? != KIND_NATIVE {
            continue;
        }
        let offset = u32_at(part, at + 4)?.into();
        let size = u32_at(part, at + 8)?.into();
        let blob = bytes(part, offset, size).ok_or(Malformed("XDAT entry out of bounds"))?;
        out.extend(programs(&unpack(blob)?)?);
    }
    Ok(out)
}

fn unpack(blob: &[u8]) -> Result<Vec<u8>, ShaderError> {
    if blob.get(0..4) != Some(b"XBCH") {
        return Err(Malformed("native blob lacks XBCH magic"));
    }
    if u32_at(blob, 4)? != XBCH_VERSION {
        return Err(Malformed("unknown XBCH version"));
    }
    let packed = u64_at(blob, 8)?;
    let unpacked = u64_at(blob, 16)?;
    if unpacked > MAX_UNPACKED {
        return Err(Malformed("implausible XBCH unpacked size"));
    }
    let stream =
        bytes(blob, XBCH_HEADER as u64, packed).ok_or(Malformed("XBCH stream out of bounds"))?;
    let out = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(stream, unpacked as usize)
        .map_err(|_| Malformed("XBCH stream does not inflate to its stated size"))?;
    if out.len() as u64 != unpacked {
        return Err(Malformed("XBCH stream does not inflate to its stated size"));
    }
    Ok(out)
}

fn programs(fssm: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    if fssm.get(0..4) != Some(b"FSSM") {
        return Err(Malformed("native blob lacks FSSM magic"));
    }
    if u32_at(fssm, 4)? != FSSM_VERSION {
        return Err(Malformed("unknown FSSM version"));
    }
    let payload = &fssm[FSSM_HEADER.min(fssm.len())..];
    if u64_at(fssm, 8)? != payload.len() as u64 {
        return Err(Malformed("FSSM payload size disagrees"));
    }
    if u64_at(payload, 0)? != GRAPHICS_SLOT_COUNT {
        return Err(Malformed("unexpected FSSM slot count"));
    }
    let mut out = Vec::new();
    for (stage, at) in SLOTS {
        let start = u64_at(payload, at)?;
        if start == ABSENT {
            continue;
        }
        let record =
            bytes(payload, start, RECORD_SIZE).ok_or(Malformed("FSSM record out of bounds"))?;
        let code = bytes(payload, u64_at(record, 0)?, u64_at(record, 8)?)
            .ok_or(Malformed("native code out of bounds"))?;
        let vgprs = (u32_at(record, RECORD_REGS)? & 0x3FF) as u16;
        if vgprs > 256 {
            return Err(Malformed("implausible VGPR count"));
        }
        let table = |at: usize, count_at: usize| -> Result<&[u8], ShaderError> {
            let len = ENTRY_SIZE * u64::from(u32_at(record, count_at)?);
            if len == 0 {
                return Ok(&[]);
            }
            bytes(payload, u64_at(record, at)?, len)
                .ok_or(Malformed("FSSM root table out of bounds"))
        };
        let layout = table(RECORD_LAYOUT, RECORD_COUNTS)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| RootParam {
                root: RootType::from_raw(le32(e, 0)),
                table_offset_dwords: le32(e, 4),
                sgprs: le32(e, 8),
                base_register: le32(e, 12),
                user_data: e[16],
                slot: e[17],
            })
            .collect::<Vec<RootParam>>();
        // The kinds above are the ones shipped layouts use, and each part of
        // the translation decides what a parameter is by its kind: resource
        // tracking takes an address only from a table, heap or descriptor,
        // lowering reads only constants and tables as user data, the host
        // fills user data only for tables and constants, and the shader
        // cache stores a kind as a code (`6 + n` for an unknown one, which
        // overflows near `u32::MAX`). A kind none of them knows is a format
        // change to name here, not one to guess at, like a read of a kind
        // not known below.
        if layout.iter().any(|p| matches!(p.root, RootType::Other(_))) {
            return Err(Malformed("root parameter of unknown kind"));
        }
        // A parameter's SGPRs must exist: `user_sgpr` and the high half of a
        // root descriptor are computed from these bytes ([`sgpr_span`]). A
        // fixed sampler taking none may name the end of them, but no
        // further: resource tracking still adds the stage's system SGPRs to
        // its byte, which would overflow from dword 248 of a merged stage.
        if layout.iter().any(|p| {
            let (first, span) = sgpr_span(stage, p);
            u64::from(first) + u64::from(span) > u64::from(SGPRS)
        }) {
            return Err(Malformed("root parameter outside the SGPRs"));
        }
        // The SGPRs a parameter takes are filled from the stage's user-data
        // registers, of which GFX10 has 32 per graphics stage and 16 for
        // compute (`COMPUTE_USER_DATA_0..15`). Lowering declares the stage's
        // user data up to the last dword read and the host makes it root
        // constants, so a parameter at, say, user-data dword 90 would ask
        // for a root signature past D3D12's 64 dwords, which the device
        // refuses to create.
        if !params_in_user_data(stage, &layout) {
            return Err(Malformed("root parameter past the user data"));
        }
        // A table or heap parameter is one 32-bit address. Resource tracking
        // maps each SGPR it states to the parameter (`resource::param_of`),
        // and the host fills every dword of a table with that address
        // (`draw::user_value`; lowering reads a heap's as 0), so an SGPR past
        // the first would silently read the address too.
        if layout.iter().any(|p| {
            matches!(
                p.root,
                RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
            ) && p.sgprs > 1
        }) {
            return Err(Malformed("root address wider than one SGPR"));
        }
        // Resource tracking maps an SGPR to the first parameter covering it
        // (`resource::param_of`), so a second parameter over the same SGPR
        // would have its reads attributed to the first. The ranges of one
        // table share its SGPR (and its address), so only different slots
        // are refused.
        if !params_apart(stage, &layout) {
            return Err(Malformed("root parameters share an SGPR"));
        }
        let used = table(RECORD_USED, RECORD_COUNTS + 4)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| UsedResource {
                kind: match e[0] {
                    0 => ResourceType::Srv,
                    1 => ResourceType::Sampler,
                    2 => ResourceType::Cbv,
                    3 => ResourceType::Uav,
                    n => ResourceType::Other(n),
                },
                root: RootType::from_raw(e[1].into()),
                table_offset_dwords: le32(e, 4),
                lower_bound: le32(e, 8),
                stride: le32(e, 12),
                dwords: e[16],
                user_data: e[17],
                slot: e[18],
            })
            .collect::<Vec<UsedResource>>();
        // Lowering picks a read's access from its kind (UAV if `Uav`, a
        // constant buffer if `Cbv`, else a read-only view), while the host
        // makes only an SRV or UAV view ready for the shader (`draw::readable`:
        // texture upload, resource state). A read of a kind neither knows
        // would be lowered as a view the host never prepares, reading stale
        // data with no error, and counts as a view, not a sampler, for the
        // one-heap check below.
        if used
            .iter()
            .any(|u| matches!(u.kind, ResourceType::Other(_)))
        {
            return Err(Malformed("used descriptor of unknown kind"));
        }
        // A read is likewise found and bound by its root kind: lowering
        // classifies only table and root-descriptor reads (`used_kind`), the
        // host binds only table reads (`draw::tables`), and the checks below
        // pair a read with its parameter by kind. A read of a kind the
        // layout could not hold (refused above) names no parameter any of
        // them could find, so it is a format change to name here too.
        if used.iter().any(|u| matches!(u.root, RootType::Other(_))) {
            return Err(Malformed("used descriptor of unknown root kind"));
        }
        // The host binds a table read by its root slot (`draw::tables`),
        // while lowering finds what a read is by the table address's SGPR
        // (`used_kind`). A read naming no table parameter with both would be
        // classified from one table and its descriptors bound from another,
        // or from none. A root descriptor's read is likewise found by its
        // SGPR (`handle`, which picks CBV or SRV from it) and bound by the
        // parameter's slot, so it must name that parameter with both too.
        let named = |u: &UsedResource| {
            layout
                .iter()
                .any(|p| p.root == u.root && p.slot == u.slot && p.user_data == u.user_data)
        };
        if used.iter().any(|u| u.root == RootType::Table && !named(u)) {
            return Err(Malformed("used descriptor names no root table"));
        }
        if used
            .iter()
            .any(|u| u.root == RootType::Descriptor && !named(u))
        {
            return Err(Malformed("used descriptor names no root descriptor"));
        }
        // The host makes a table holding a constant buffer afresh with a slot
        // for every descriptor up to the furthest read (`draw::tables`: its
        // offset / 8 + 1), reading each from title memory to key it before
        // asking the heap for that many slots. An offset past any heap would
        // read gigabytes past the table (536 million slots near `u32::MAX`)
        // before the allocation fails.
        if used
            .iter()
            .any(|u| u.root == RootType::Table && u.table_offset_dwords >= MAX_TABLE_DWORDS)
        {
            return Err(Malformed("used descriptor past any heap"));
        }
        if !used_one_way(&used) {
            return Err(Malformed("used descriptors read one descriptor two ways"));
        }
        if !tables_one_heap(&used) {
            return Err(Malformed("root table mixes samplers and views"));
        }
        // The record is in bounds, so its end is too.
        let stage_data = &payload[(start + RECORD_SIZE) as usize..];
        let io = match stage {
            HwStage::Ps => ps_io(stage_data, code.len())?,
            HwStage::Vs => vs_outputs_per_vertex(vs_inputs_fit(vs_io(stage_data)?, vgprs)?)?,
            HwStage::Cs => StageIo::Cs(compute_io(stage_data, &layout, vgprs)?),
            HwStage::Ms => ms_io(stage_data, u32_at(record, RECORD_LDS)?)?,
            _ => StageIo::None,
        };
        out.push(NativeProgram {
            stage,
            wave64: u32_at(record, RECORD_FLAGS)? & FLAG_WAVE32 == 0,
            vgprs,
            layout,
            used,
            io,
            code: code.to_vec(),
        });
    }
    Ok(out)
}

/// A compute program's dispatch settings. `COMPUTE_PGM_RSRC2.LDS_SIZE` is a
/// 9-bit count of 512-byte granules, so it can state up to 255.5 KiB, which
/// lowering would declare as the group's shared memory; like a mesh group's,
/// a compute group's LDS can't exceed 64 KiB.
///
/// The group ids and group info arrive in the SGPRs after the user SGPRs
/// (`ComputeIo::system_sgprs`), and raising labels them there. A root
/// parameter covering one of those SGPRs would make the same register both a
/// root parameter and a group id, so such a layout is refused. The thread
/// ids arrive in v0 on, which must be among the program's VGPRs.
fn compute_io(data: &[u8], layout: &[RootParam], vgprs: u16) -> Result<ComputeIo, ShaderError> {
    let cs = ComputeIo {
        rsrc2: u32_at(data, CS_RSRC2)?,
        threads: group_size(data)?,
    };
    if cs.lds_bytes() > MAX_LDS_BYTES {
        return Err(Malformed("implausible LDS size"));
    }
    // Raising labels v0 up to the thread-id count as the thread id in the
    // group (`io::raise`), so each of those must be one of the program's
    // VGPRs, as a vertex input must.
    if u16::from(cs.thread_id_vgprs()) > vgprs {
        return Err(Malformed("thread ids outside the VGPRs"));
    }
    let (ids, info) = cs.system_sgprs();
    // The layout was checked first, so `first + span` stays below `SGPRS`.
    let mut system = ids.into_iter().chain([info]).flatten().map(u32::from);
    if system.any(|s| {
        layout.iter().any(|p| {
            let (first, span) = sgpr_span(HwStage::Cs, p);
            (first..first + span).contains(&s)
        })
    }) {
        return Err(Malformed("root parameter over a system SGPR"));
    }
    Ok(cs)
}

/// The first SGPR of a root parameter and how many it takes. Resource
/// tracking reads a root descriptor's high half from the SGPR after its
/// first whatever the stated count, so a descriptor spans at least two. A
/// fixed sampler is part of the root signature, not of the user data: one
/// stating no SGPRs takes none (`resource::param_of` never maps an SGPR to
/// it), so its user-data byte only has to keep `user_sgpr` in range.
fn sgpr_span(stage: HwStage, p: &RootParam) -> (u32, u32) {
    let first = u32::from(p.user_data) + u32::from(stage.user_sgpr(0));
    let span = match p.root {
        RootType::Descriptor => p.sgprs.max(2),
        RootType::ImmediateSampler => p.sgprs,
        _ => p.sgprs.max(1),
    };
    (first, span)
}

/// User-data registers a stage has: `SPI_SHADER_USER_DATA_*_0..31` for the
/// graphics stages, `COMPUTE_USER_DATA_0..15` for compute.
fn user_data_dwords(stage: HwStage) -> u32 {
    match stage {
        HwStage::Cs => 16,
        _ => 32,
    }
}

/// Whether every SGPR a root parameter takes ([`sgpr_span`]: a root
/// descriptor's high half included) is one of the stage's user-data
/// registers. Only a fixed sampler stating no SGPRs takes none: a table,
/// heap or constants entry stating none still names its first SGPR, which
/// lowering reads as user data (a table's address) like any other.
fn params_in_user_data(stage: HwStage, layout: &[RootParam]) -> bool {
    layout.iter().all(|p| {
        let (_, span) = sgpr_span(stage, p);
        span == 0 || u64::from(p.user_data) + u64::from(span) <= u64::from(user_data_dwords(stage))
    })
}

/// Whether no SGPR holds two root parameters: each covers the SGPRs it
/// takes ([`sgpr_span`]: none only for a fixed sampler stating none, at
/// least its first for any other kind, and a root descriptor also the high
/// half resource tracking reads after its first). Only the ranges of one
/// table (one slot, one table address SGPR) may share; a slot number alone
/// does not make two entries one parameter, as a table range and constants
/// stating the same slot would still be read as whichever comes first.
fn params_apart(stage: HwStage, layout: &[RootParam]) -> bool {
    // The layout was checked first, so every covered SGPR is below `SGPRS`.
    let mut owner = [None; SGPRS as usize];
    layout.iter().all(|p| {
        let (first, span) = sgpr_span(stage, p);
        let key = (p.slot, p.root, first);
        (first..first + span).all(|s| match owner[s as usize].replace(key) {
            None => true,
            Some(held) => held == key && p.root == RootType::Table,
        })
    })
}

/// The descriptor a table or root-descriptor read names: its parameter and,
/// in a table, its offset. A root descriptor is one view, so all its reads
/// name the same one.
fn read_of(u: &UsedResource) -> Option<(RootType, u8, u8, u32)> {
    match u.root {
        RootType::Table => Some((u.root, u.slot, u.user_data, u.table_offset_dwords)),
        RootType::Descriptor => Some((u.root, u.slot, u.user_data, 0)),
        _ => None,
    }
}

/// Whether no descriptor is read as two kinds. Lowering classifies a read
/// by one entry (`used_kind`: a root descriptor's first, a table's last at
/// or below the offset) and picks CBV, SRV or UAV access from its kind,
/// while the host binds each entry by its own kind (`draw::readable`): a
/// second entry of another kind would bind the view one way and read it
/// another.
fn used_one_way(used: &[UsedResource]) -> bool {
    used.iter().enumerate().all(|(k, a)| {
        used[..k]
            .iter()
            .all(|b| read_of(a).is_none() || read_of(b) != read_of(a) || b.kind == a.kind)
    })
}

/// Whether each root table holds samplers only or views only. D3D12 puts
/// samplers in their own heap, so one table can't hold both, and the host
/// makes a table with a constant buffer afresh in its view heap
/// (`draw::tables`), where a sampler has no place.
fn tables_one_heap(used: &[UsedResource]) -> bool {
    let tables = || used.iter().filter(|u| u.root == RootType::Table);
    let sampler = |u: &UsedResource| u.kind == ResourceType::Sampler;
    tables().all(|a| {
        tables().all(|b| (a.slot, a.user_data) != (b.slot, b.user_data) || sampler(a) == sampler(b))
    })
}

/// A compute program's thread-group size. Lowering multiplies the
/// components and writes them as the DXIL `numthreads`, so each must be one
/// that HLSL allows: a zero or oversized group is refused here rather than
/// overflowing that product (a debug panic) or emitting an invalid shader.
fn group_size(data: &[u8]) -> Result<[u32; 3], ShaderError> {
    let mut threads = [0; 3];
    for (k, n) in threads.iter_mut().enumerate() {
        *n = u32_at(data, CS_THREADS + 4 * k)?;
    }
    // Each component is checked first, so the product fits a u64.
    if threads
        .iter()
        .zip(MAX_GROUP)
        .any(|(&n, max)| n == 0 || n > max)
        || threads.iter().map(|&n| u64::from(n)).product::<u64>() > MAX_GROUP_THREADS
    {
        return Err(Malformed("implausible thread-group size"));
    }
    Ok(threads)
}

/// The entries of `size` bytes from `at`, with their count at `count_at`.
fn semantics(data: &[u8], at: usize, size: usize, count_at: usize) -> Result<&[u8], ShaderError> {
    let count = u32_at(data, count_at)?;
    if count > MAX_SEMANTICS {
        return Err(Malformed("implausible semantic count"));
    }
    data.get(at..at + size * count as usize)
        .ok_or(Malformed("stage data out of bounds"))
}

fn ps_io(data: &[u8], code_len: usize) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, PS_INPUTS, 8, PS_INPUT_COUNT)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|e| PsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            attr: e[3],
            flat: e[7] & 1 != 0,
            fp16: e[7] & 8 != 0,
            pack2: e[7] & 0x10 != 0,
            row2: e[5],
        })
        .collect::<Vec<_>>();
    // Raising resolves `v_interp` of attribute n to the first input naming
    // n, so a second input with the same attribute would be dropped.
    if !distinct(inputs.iter().map(|i| i.attr)) {
        return Err(Malformed("pixel inputs share an attribute"));
    }
    // Inputs packed into one signature row (disjoint channels) are fine, but
    // lowering finds a row's 16-bit layout as the first fp16 input naming
    // it: a second input on that row read as 32-bit, or packed differently,
    // would be decoded with the first's layout.
    let packing = |i: &PsInput| i.fp16.then_some((i.pack2, i.row2));
    if inputs.iter().enumerate().any(|(k, a)| {
        inputs[..k]
            .iter()
            .any(|b| b.row == a.row && packing(b) != packing(a))
    }) {
        return Err(Malformed("pixel inputs read a row two ways"));
    }
    // A packed input's halves past its own row's come from its second row
    // (`row2`), which lowering reads as that row's signature element
    // (`packed_half`). Another input naming that row, or a packed input of
    // another row claiming it too, would read one element as two inputs.
    let second = |i: &PsInput| (i.fp16 && i.pack2 && i.row2 != i.row).then_some(i.row2);
    if inputs.iter().any(|a| {
        second(a).is_some_and(|r2| {
            inputs
                .iter()
                .any(|b| b.row == r2 || (b.row != a.row && second(b) == Some(r2)))
        })
    }) {
        return Err(Malformed("pixel inputs claim a packed row"));
    }
    let size = u32_at(data, PS_PATCH_SIZE)? as usize;
    if size > PATCH_SLOT_DWORDS {
        return Err(Malformed("implausible export patch size"));
    }
    let mut patches = Vec::new();
    for mrt in 0..MRTS {
        let at = PS_PATCHES + mrt * PATCH_BYTES;
        let offset = u32_at(data, at)?;
        if offset == u32::MAX {
            continue;
        }
        let start = u64::from(offset);
        let end = start + size as u64;
        if end > (code_len / 4) as u64 {
            return Err(Malformed("export patch outside the code"));
        }
        // `patched_code` writes the patches in MRT order: a later one over an
        // earlier one's dwords would clobber that export with no error.
        if patches.iter().any(|p: &ExportPatch| {
            start < u64::from(p.offset) + size as u64 && u64::from(p.offset) < end
        }) {
            return Err(Malformed("export patches overlap"));
        }
        let full = at + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        let code = (0..size)
            .map(|i| u32_at(data, full + 4 * i))
            .collect::<Result<_, _>>()?;
        patches.push(ExportPatch {
            mrt: mrt as u8,
            offset,
            code,
        });
    }
    Ok(StageIo::Ps {
        inputs,
        input_addr: u32_at(data, PS_INPUT_ADDR)?,
        db_shader_control: u32_at(data, PS_DB_SHADER_CONTROL)?,
        patches,
    })
}

fn vs_io(data: &[u8]) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, VS_INPUTS, 5, VS_INPUT_COUNT)?
        .as_chunks::<5>()
        .0
        .iter()
        .map(|e| VsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            vgpr: e[3],
        })
        .collect();
    let outputs = semantics(data, VS_OUTPUTS, 6, VS_OUTPUT_COUNT)?
        .as_chunks::<6>()
        .0
        .iter()
        .map(|e| VsOutput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            param: e[2],
            fp16: e[5] & 1 != 0,
            pack2: e[5] & 2 != 0,
        })
        .collect::<Vec<_>>();
    // Raising resolves `exp param<n>` to the first output whose parameter
    // (bit 7, a mesh program's per-primitive flag, aside) is n, so a second
    // output with the same parameter would be dropped.
    if !distinct(outputs.iter().map(|o| o.param & 0x7F)) {
        return Err(Malformed("vertex outputs share a parameter"));
    }
    // Raising turns each `exp param<n>` into a write of its output's
    // signature row, and lowering keys 16-bit outputs by that row
    // (`half_outputs`): two outputs naming one row would write the same
    // element, the later export silently replacing the earlier. Per-vertex
    // rows (`OSG1`) and per-primitive rows (`PSG1`) are separate signatures.
    let rows = |primitive: bool| {
        outputs
            .iter()
            .filter(move |o| (o.param & 0x80 != 0) == primitive)
            .map(|o| o.row)
    };
    if !distinct(rows(false)) || !distinct(rows(true)) {
        return Err(Malformed("vertex outputs share a row"));
    }
    Ok(StageIo::Vs { inputs, outputs })
}

/// Whether no byte appears twice.
fn distinct(bytes: impl IntoIterator<Item = u8>) -> bool {
    let mut seen = [false; 256];
    bytes
        .into_iter()
        .all(|b| !std::mem::replace(&mut seen[usize::from(b)], true))
}

/// The VGPRs a vertex program receives its vertex, primitive and instance
/// id in.
const VS_SYSTEM_VGPRS: [u32; 3] = [0, 2, 3];

/// Raising loads a vertex input's channel `c` from `v[vgpr + c]`: every
/// enabled channel must land on one of the program's `vgprs` registers (at
/// most 256, so within the 8-bit register number too). The fetch fills only
/// the VGPRs the wave is given, and relative VGPR reads (`ir::build`) treat
/// those past `vgprs` as absent, so an input past them would be labeled in a
/// register the program never has.
fn vs_inputs_fit(io: StageIo, vgprs: u16) -> Result<StageIo, ShaderError> {
    if let StageIo::Vs { inputs, .. } = &io
        && inputs.iter().any(|i| {
            i.mask
                .checked_ilog2()
                .is_some_and(|last| u32::from(i.vgpr) + last >= u32::from(vgprs))
        })
    {
        return Err(Malformed("vertex input outside the VGPRs"));
    }
    // Raising also labels v0, v2 and v3 as the vertex, primitive and
    // instance id (`io::raise`) and takes the first label a register has, so
    // a channel loaded into one of them would silently read as that id.
    if let StageIo::Vs { inputs, .. } = &io
        && inputs.iter().any(|i| {
            (0..4)
                .filter(|c| i.mask >> c & 1 != 0)
                .any(|c| VS_SYSTEM_VGPRS.contains(&(u32::from(i.vgpr) + c)))
        })
    {
        return Err(Malformed("vertex input over a system VGPR"));
    }
    // Two inputs loading one VGPR would leave it labeled as the first; the
    // fetch would have overwritten it with the second, which raising drops.
    if let StageIo::Vs { inputs, .. } = &io {
        let mut taken = [false; 256];
        for i in inputs {
            for c in (0..4).filter(|c| i.mask >> c & 1 != 0) {
                // The first check keeps `vgpr + c` within v255.
                let r = usize::from(i.vgpr) + c;
                if std::mem::replace(&mut taken[r], true) {
                    return Err(Malformed("vertex inputs overlap"));
                }
            }
        }
    }
    Ok(io)
}

/// Bit 7 of an output's parameter marks a mesh program's per-primitive
/// output (a `PSG1` row). A vertex program has no primitive signature: such
/// an output's exports would be written to an element that doesn't exist
/// and silently dropped, and its row would escape the shared-row check.
fn vs_outputs_per_vertex(io: StageIo) -> Result<StageIo, ShaderError> {
    if let StageIo::Vs { outputs, .. } = &io
        && outputs.iter().any(|o| o.param & 0x80 != 0)
    {
        return Err(Malformed("vertex output marked per-primitive"));
    }
    Ok(io)
}

fn ms_io(data: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let StageIo::Vs { outputs, .. } = vs_io(data)? else {
        unreachable!()
    };
    let info = data
        .get(MS_INFO..MS_INFO + 8)
        .ok_or(Malformed("stage data out of bounds"))?;
    let half = |at: usize| u32::from(u16::from_le_bytes([info[at], info[at + 1]]));
    // The group's lanes become the DXIL `numthreads` x and, for containers
    // without the HLSL count, the stride of the group id: an empty group
    // would emit `numthreads(0, 1, 1)` and give every group id 0.
    // A group past 128 lanes is one no mesh shader can declare (the lanes
    // are the HLSL group rounded up to waves, and 128 is a whole number of
    // either), so it would emit a `numthreads` the validator rejects.
    let threads = half(0);
    if threads == 0 || threads > MAX_MESH_THREADS {
        return Err(Malformed("implausible thread-group size"));
    }
    // Lowering writes the vertex and primitive limits and the topology as
    // the DXIL mesh state and the PSV mesh info as they are (the limits as
    // 16-bit values): past 256 outputs, or a topology neither lines (1) nor
    // triangles (2), the emitted shader is one D3D12 refuses to create.
    let (max_vertices, max_primitives) = (half(2), half(4));
    if max_vertices > MAX_MESH_OUTPUTS || max_primitives > MAX_MESH_OUTPUTS {
        return Err(Malformed("implausible mesh output count"));
    }
    let topology = info[7];
    if !matches!(topology, 1 | 2) {
        return Err(Malformed("unknown mesh topology"));
    }
    // Lowering declares the LDS as `lds_bytes.div_ceil(4)` dwords and states
    // it again in bytes (`lds_dwords * 4`), a u32 product that overflows for
    // sizes near `u32::MAX`; a group's LDS can't exceed 64 KiB anyway.
    if lds_bytes > MAX_LDS_BYTES {
        return Err(Malformed("implausible LDS size"));
    }
    Ok(StageIo::Ms(MeshIo {
        outputs,
        clip_cntl: u32_at(data, MS_CLIP_CNTL)?,
        threads,
        max_vertices,
        max_primitives,
        topology,
        lds_bytes,
    }))
}

/// For entries already sized by `as_chunks`.
fn le32(entry: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(entry[at..at + 4].try_into().unwrap())
}

/// `data[offset..offset + len]` with untrusted 64-bit bounds.
fn bytes(data: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    data.get(start..start.checked_add(usize::try_from(len).ok()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS_CODE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const PS_CODE: [u8; 4] = [9, 10, 11, 12];
    const VGPRS: u32 = 28;

    fn slot(stage: HwStage) -> usize {
        SLOTS.iter().find(|s| s.0 == stage).unwrap().1
    }

    /// Stage data: no semantics, and every export patch absent.
    const STAGE_DATA: usize = PS_PATCHES + MRTS * PATCH_BYTES;

    fn empty_stage_data() -> Vec<u8> {
        let mut data = vec![0; STAGE_DATA];
        for mrt in 0..MRTS {
            let at = PS_PATCHES + mrt * PATCH_BYTES;
            data[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        data
    }

    /// Where the selected stage's data starts in the blob.
    fn stage_data_at(blob: &[u8], stage: HwStage) -> usize {
        FSSM_HEADER
            + u64_at(blob, FSSM_HEADER + slot(stage)).unwrap() as usize
            + RECORD_SIZE as usize
    }

    /// FSSM blob with `programs` as (stage, code) pairs.
    fn fssm(programs: &[(HwStage, &[u8])]) -> Vec<u8> {
        let mut payload = GRAPHICS_SLOT_COUNT.to_le_bytes().to_vec();
        payload.resize(slot(HwStage::Cs) + 8, 0);
        for (_, at) in SLOTS {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        for &(stage, code) in programs {
            let slot = slot(stage);
            let record = payload.len() as u64;
            payload[slot..slot + 8].copy_from_slice(&record.to_le_bytes());
            let code_at = record + RECORD_SIZE + STAGE_DATA as u64;
            payload.extend_from_slice(&code_at.to_le_bytes());
            payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
            let flags = if stage == HwStage::Ps { FLAG_WAVE32 } else { 0 };
            payload.extend_from_slice(&[0; 4]);
            payload.extend_from_slice(&flags.to_le_bytes());
            payload.resize(record as usize + RECORD_REGS, 0);
            payload.extend_from_slice(&(VGPRS | 31 << 20).to_le_bytes());
            let mut data = empty_stage_data();
            // numthreads(1, 1, 1): a compute or mesh group is never empty.
            if stage == HwStage::Cs {
                for k in 0..3 {
                    data[CS_THREADS + 4 * k..][..4].copy_from_slice(&1u32.to_le_bytes());
                }
            }
            // One lane drawing triangles: a mesh group is never empty and
            // always names a topology.
            if stage == HwStage::Ms {
                data[MS_INFO..][..2].copy_from_slice(&1u16.to_le_bytes());
                data[MS_INFO + 7] = 2;
            }
            payload.extend_from_slice(&data);
            payload.extend_from_slice(code);
        }
        let mut out = b"FSSM".to_vec();
        out.extend_from_slice(&FSSM_VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn xbch(unpacked: &[u8]) -> Vec<u8> {
        let stream = miniz_oxide::deflate::compress_to_vec_zlib(unpacked, 6);
        let mut out = b"XBCH".to_vec();
        out.extend_from_slice(&XBCH_VERSION.to_le_bytes());
        out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        out.extend_from_slice(&(unpacked.len() as u64).to_le_bytes());
        out.extend_from_slice(&stream);
        out
    }

    /// `DXIL` part data: program header, XDAT index, then the blobs.
    fn part(blobs: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = vec![0u8; XDAT_OFFSET];
        out.extend_from_slice(b"XDAT");
        out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x24u32.to_le_bytes());
        let mut offset = out.len() + 12 * blobs.len();
        for (kind, blob) in blobs {
            for v in [*kind, offset as u32, blob.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            offset += blob.len();
        }
        for (_, blob) in blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    #[test]
    fn finds_graphics_programs() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE), (HwStage::Ps, &PS_CODE)]));
        let data = part(&[(0x0a, b"xbop"), (KIND_NATIVE, &native)]);
        let found = native_programs(&data).unwrap();
        assert_eq!(
            found,
            [
                NativeProgram {
                    stage: HwStage::Vs,
                    wave64: true,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Vs {
                        inputs: vec![],
                        outputs: vec![],
                    },
                    code: VS_CODE.to_vec()
                },
                NativeProgram {
                    stage: HwStage::Ps,
                    wave64: false,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Ps {
                        inputs: vec![],
                        input_addr: 0,
                        db_shader_control: 0,
                        patches: vec![],
                    },
                    code: PS_CODE.to_vec()
                },
            ]
        );
    }

    #[test]
    fn reads_root_tables() {
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Ps)).unwrap() as usize;
        let tables = (blob.len() - FSSM_HEADER) as u64;
        // A table range in s9 (root slot 1), and an SRV read from it.
        for v in [3u32, 2, 1, 5] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[9, 1, 0, 0, 0, 3, 0, 0]);
        for v in [2u32, 5, 4] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[8, 9, 1, 0]);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&tables.to_le_bytes());
        blob[record + RECORD_USED..][..8].copy_from_slice(&(tables + ENTRY_SIZE).to_le_bytes());
        blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        assert_eq!(
            p.layout,
            [RootParam {
                root: RootType::Table,
                table_offset_dwords: 2,
                sgprs: 1,
                base_register: 5,
                user_data: 9,
                slot: 1
            }]
        );
        assert_eq!(
            p.used,
            [UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 2,
                lower_bound: 5,
                stride: 4,
                dwords: 8,
                user_data: 9,
                slot: 1
            }]
        );

        // Parameters must sit in SGPRs the program can address: s9 plus 98
        // dwords is past s105, and so is a user-data byte past the SGPR
        // file. They must also sit in the stage's 32 user-data dwords: from
        // dword 9, 23 end at dword 31 and 24 are past it. (Constants: a
        // table takes one SGPR, and the read is then not a table's.)
        let entry = FSSM_HEADER + tables as usize;
        let layout = |blob: &[u8]| native_programs(&part(&[(KIND_NATIVE, &xbch(blob))]));
        let mut wide = blob.clone();
        wide[entry..entry + 4].copy_from_slice(&0u32.to_le_bytes());
        wide[entry + ENTRY_SIZE as usize + 1] = 0;
        wide[entry + 8..entry + 12].copy_from_slice(&23u32.to_le_bytes());
        assert_eq!(layout(&wide).unwrap()[0].layout[0].sgprs, 23);
        for sgprs in [24u32, 97] {
            wide[entry + 8..entry + 12].copy_from_slice(&sgprs.to_le_bytes());
            assert_eq!(
                refused_blob(&xbch(&wide)),
                "root parameter past the user data",
                "{sgprs}"
            );
        }
        wide[entry + 8..entry + 12].copy_from_slice(&98u32.to_le_bytes());
        assert_eq!(
            refused_blob(&xbch(&wide)),
            "root parameter outside the SGPRs"
        );
        let mut far = blob.clone();
        far[entry + 16] = 0xFF;
        assert_eq!(
            refused_blob(&xbch(&far)),
            "root parameter outside the SGPRs"
        );

        // A root descriptor's high half is the SGPR after its first, even
        // when the record states fewer than two: one at user-data dword 30
        // fits (30, 31), one at 31 would take dword 32 as its high half,
        // and one in s105 would take VCC_LO.
        for sgprs in [0u32, 1, 2] {
            let mut desc = blob.clone();
            desc[entry..entry + 4].copy_from_slice(&1u32.to_le_bytes());
            // The SRV read through it is then a root descriptor's.
            desc[entry + ENTRY_SIZE as usize + 1] = 1;
            desc[entry + 8..entry + 12].copy_from_slice(&sgprs.to_le_bytes());
            desc[entry + 16] = 30;
            desc[entry + ENTRY_SIZE as usize + 17] = 30;
            assert_eq!(
                layout(&desc).unwrap()[0].layout[0].root,
                RootType::Descriptor
            );
            desc[entry + 16] = 31;
            desc[entry + ENTRY_SIZE as usize + 17] = 31;
            assert_eq!(
                refused_blob(&xbch(&desc)),
                "root parameter past the user data"
            );
            desc[entry + 16] = 105;
            desc[entry + ENTRY_SIZE as usize + 17] = 105;
            assert_eq!(
                refused_blob(&xbch(&desc)),
                "root parameter outside the SGPRs"
            );
            // Other kinds of parameter still take one SGPR at dword 31.
            desc[entry + 16] = 31;
            desc[entry + ENTRY_SIZE as usize + 17] = 31;
            desc[entry..entry + 4].copy_from_slice(&3u32.to_le_bytes());
            desc[entry + ENTRY_SIZE as usize + 1] = 3;
            if sgprs <= 1 {
                assert!(layout(&desc).is_ok());
            }
        }

        // The SRV is read through the table in s9, root slot 1: a read
        // naming another slot or SGPR names no table the layout has.
        let used = entry + ENTRY_SIZE as usize;
        for (at, byte) in [(17, 8u8), (17, 10), (18, 0), (18, 2)] {
            let mut stray = blob.clone();
            stray[used + at] = byte;
            assert_eq!(
                refused_blob(&xbch(&stray)),
                "used descriptor names no root table",
                "byte {at} = {byte}"
            );
        }
        // Nor may it name a parameter of another kind there.
        let mut constants = blob.clone();
        constants[entry..entry + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            refused_blob(&xbch(&constants)),
            "used descriptor names no root table"
        );
        // A read of another kind of root is not a table read.
        let mut desc = constants.clone();
        desc[used + 1] = 0;
        assert!(layout(&desc).is_ok());

        // A root descriptor read must name the root descriptor in its SGPR
        // and slot: not another SGPR or slot, nor a table or constants there.
        let mut root = blob.clone();
        root[entry..entry + 4].copy_from_slice(&1u32.to_le_bytes());
        root[entry + 8..entry + 12].copy_from_slice(&2u32.to_le_bytes());
        root[used + 1] = 1;
        assert!(layout(&root).is_ok());
        for (at, byte) in [(17, 8u8), (17, 10), (18, 0), (18, 2)] {
            let mut stray = root.clone();
            stray[used + at] = byte;
            assert_eq!(
                refused_blob(&xbch(&stray)),
                "used descriptor names no root descriptor",
                "byte {at} = {byte}"
            );
        }
        for kind in [0u32, 3] {
            let mut other = root.clone();
            other[entry..entry + 4].copy_from_slice(&kind.to_le_bytes());
            // A table takes one SGPR.
            other[entry + 8..entry + 12].copy_from_slice(&1u32.to_le_bytes());
            assert_eq!(
                refused_blob(&xbch(&other)),
                "used descriptor names no root descriptor",
                "parameter kind {kind}"
            );
        }

        blob[record + RECORD_COUNTS] = 9;
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn reads_pixel_io_and_applies_the_full_export_patch() {
        let code = [0xBF80_0000u32; 4]; // s_nop
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut blob = fssm(&[(HwStage::Ps, &bytes)]);
        let e = stage_data_at(&blob, HwStage::Ps);
        // TEXCOORD in row 1 (xy), read as attr0, flat.
        blob[e + PS_INPUTS..][..8].copy_from_slice(&[6, 1, 3, 0, 0, 0, 1, 1]);
        blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[e + PS_INPUT_ADDR..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&2u32.to_le_bytes());
        // MRT1 at dword 1: the 32_ABGR variant is `exp mrt1 v0-v3 done`.
        let patch = e + PS_PATCHES + PATCH_BYTES;
        blob[patch..][..4].copy_from_slice(&1u32.to_le_bytes());
        let full = patch + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        blob[full..][..8].copy_from_slice(&[0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let StageIo::Ps {
            inputs,
            input_addr,
            patches,
            ..
        } = &p.io
        else {
            panic!("{:?}", p.io)
        };
        let input = PsInput {
            usage: Usage::Generic,
            row: 1,
            mask: 3,
            attr: 0,
            flat: true,
            fp16: false,
            pack2: false,
            row2: 0,
        };
        assert_eq!((&inputs[..], *input_addr), (&[input][..], 2));
        assert_eq!(patches.len(), 1);
        assert_eq!((patches[0].mrt, patches[0].offset), (1, 1));
        let patched = p.patched_code();
        assert_eq!(patched[..4], p.code[..4]);
        assert_eq!(patched[4..12], [0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);
        assert_eq!(patched[12..], p.code[12..]);

        blob[patch..][..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn export_patches_use_complete_dwords_at_their_stated_offsets() {
        let size = PATCH_SLOT_DWORDS;
        let code = vec![0x55; 2 * size * 4 + 3];
        let mut blob = fssm(&[(HwStage::Ps, &code)]);
        let data = stage_data_at(&blob, HwStage::Ps);
        blob[data + PS_PATCH_SIZE..][..4].copy_from_slice(&(size as u32).to_le_bytes());
        // The last MRT patches the first half, and MRT0 patches the second.
        let last = data + PS_PATCHES + (MRTS - 1) * PATCH_BYTES;
        for (at, offset, fill) in [(data + PS_PATCHES, size as u32, 0xaa), (last, 0, 0xbb)] {
            blob[at..][..4].copy_from_slice(&offset.to_le_bytes());
            let full = at + 4 + PATCH_FULL * size * 4;
            blob[full..][..size * 4].fill(fill);
        }
        let programs = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap();
        let mut expected = code.clone();
        expected[..size * 4].fill(0xbb);
        expected[size * 4..2 * size * 4].fill(0xaa);
        assert_eq!(programs[0].patched_code(), expected);
        assert_eq!(programs[0].code, code);

        // Moving MRT0 one dword would leave only three bytes for its last word.
        // Large offsets must also be refused on hosts with 32-bit byte indices.
        for offset in [size as u32 + 1, 1 << 30, u32::MAX - 1] {
            blob[data + PS_PATCHES..][..4].copy_from_slice(&offset.to_le_bytes());
            assert_eq!(refused_blob(&xbch(&blob)), "export patch outside the code");
        }
    }

    #[test]
    fn finds_mesh_and_compute_programs() {
        for stage in [HwStage::Ms, HwStage::Cs] {
            let data = part(&[(KIND_NATIVE, &xbch(&fssm(&[(stage, &VS_CODE)])))]);
            let found = native_programs(&data).unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].stage, stage);
        }
    }

    #[test]
    fn reads_mesh_outputs_and_limits() {
        let mut blob = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&blob, HwStage::Ms);
        // param0 is row 1 per vertex, param2 row 0 of the primitive signature.
        blob[e + VS_OUTPUTS..][..12].copy_from_slice(&[6, 1, 0, 0, 0, 0, 6, 0, 0x82, 0, 0, 0]);
        blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + MS_CLIP_CNTL..][..4].copy_from_slice(&0x0040_000Fu32.to_le_bytes());
        // 128 lanes, 128 vertices, 64 primitives, 13 user SGPRs, triangles.
        blob[e + MS_INFO..][..8].copy_from_slice(&[0x80, 0, 0x80, 0, 0x40, 0, 13, 2]);
        let record = e - RECORD_SIZE as usize;
        blob[record + RECORD_LDS..][..4].copy_from_slice(&16u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let out = |row, param| VsOutput {
            usage: Usage::Generic,
            row,
            param,
            fp16: false,
            pack2: false,
        };
        assert_eq!(
            p.io,
            StageIo::Ms(MeshIo {
                outputs: vec![out(1, 0), out(0, 0x82)],
                clip_cntl: 0x0040_000F,
                threads: 128,
                max_vertices: 128,
                max_primitives: 64,
                topology: 2,
                lds_bytes: 16,
            })
        );
    }

    #[test]
    fn compute_dispatch_settings() {
        // 5 user SGPRs, group id x and y, thread ids in v0-v1, 1 KiB LDS.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.user_sgprs(), 5);
        assert_eq!(cs.system_sgprs(), ([Some(5), Some(6), None], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (2, 1024));
        // 6 user SGPRs, all three group ids, v0-v2, no LDS.
        let cs = ComputeIo {
            rsrc2: 0x138C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.system_sgprs(), ([Some(6), Some(7), Some(8)], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (3, 0));
    }

    /// A compute program with this `COMPUTE_PGM_RSRC2` and one (root type,
    /// SGPRs, user data) parameter.
    fn compute_program(
        rsrc2: u32,
        root: u32,
        sgprs: u32,
        user_data: u8,
    ) -> Result<Vec<NativeProgram>, ShaderError> {
        let mut blob = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Cs)).unwrap() as usize;
        let e = record + RECORD_SIZE as usize;
        blob[e + CS_RSRC2..][..4].copy_from_slice(&rsrc2.to_le_bytes());
        let table = (blob.len() - FSSM_HEADER) as u64;
        for v in [root, 0, sgprs, 0] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[user_data, 0, 0, 0]);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
        blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
    }

    #[test]
    fn compute_root_params_stay_off_system_sgprs() {
        // 4 user SGPRs, then group id x in s4, y in s5 and the group info
        // in s6.
        let rsrc2: u32 = 4 << 1 | 1 << 7 | 1 << 8 | 1 << 10;
        let program = compute_program;
        let (constants, descriptor, table) = (0, 1, 3);
        // s0-s3 are the user SGPRs; a table in s3 takes one of them.
        let p = program(rsrc2, constants, 4, 0).unwrap();
        assert_eq!(p[0].layout[0].sgprs, 4);
        assert!(program(rsrc2, table, 1, 3).is_ok());
        // Past the system SGPRs nothing collides.
        assert!(program(rsrc2, constants, 1, 7).is_ok());
        // With no group ids or info enabled, s4 is free.
        assert!(program(4 << 1, constants, 1, 4).is_ok());
        // Compute has 16 user-data dwords: constants may end at dword 15,
        // and a descriptor's high half may be it, but neither may pass it.
        assert!(program(0, constants, 16, 0).is_ok());
        assert!(program(0, descriptor, 2, 14).is_ok());
        for (root, sgprs, user_data, what) in [
            (constants, 17, 0, "seventeen dwords"),
            (constants, 1, 16, "constants at dword 16"),
            (descriptor, 1, 15, "a descriptor's high half at dword 16"),
            (table, 1, 40, "a table at dword 40"),
        ] {
            assert!(
                matches!(
                    program(0, root, sgprs, user_data),
                    Err(Malformed("root parameter past the user data"))
                ),
                "{what}"
            );
        }
        for (root, sgprs, user_data, what) in [
            (constants, 5, 0, "five dwords from s0 reach group id x"),
            (descriptor, 1, 3, "a descriptor's high half in s4"),
            (constants, 1, 5, "group id y"),
            (table, 1, 6, "the group info"),
        ] {
            assert!(
                matches!(
                    program(rsrc2, root, sgprs, user_data),
                    Err(Malformed("root parameter over a system SGPR"))
                ),
                "{what}"
            );
        }
    }

    #[test]
    fn compute_system_sgprs_skip_disabled_components() {
        // Only group id z: it takes the first SGPR after the user data, not
        // the third.
        let cs = |rsrc2: u32| ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        };
        assert_eq!(
            cs(3 << 1 | 1 << 9).system_sgprs(),
            ([None, None, Some(3)], None)
        );
        // Group id x and z with the info: y's SGPR is not left free.
        assert_eq!(
            cs(2 << 1 | 1 << 7 | 1 << 9 | 1 << 10).system_sgprs(),
            ([Some(2), None, Some(3)], Some(4))
        );
        // The info alone, after no user SGPRs.
        assert_eq!(cs(1 << 10).system_sgprs(), ([None; 3], Some(0)));
        // The 5-bit user SGPR field at its most: the ids follow s30.
        let full = cs(0x1F << 1 | 0xF << 7);
        assert_eq!(full.user_sgprs(), 31);
        assert_eq!(
            full.system_sgprs(),
            ([Some(31), Some(32), Some(33)], Some(34))
        );
        // Bit 6 (trap present) and the LDS field sit between and after the
        // fields read here and move none of them.
        assert_eq!(
            cs(1 << 1 | 1 << 6 | 1 << 8 | 0x1FF << 15).system_sgprs(),
            ([None, Some(1), None], None)
        );
    }

    #[test]
    fn compute_root_params_over_a_disabled_component_are_free() {
        // Group id z alone lands in s2, right after two user SGPRs; s3 and
        // up are free though x and y would have taken them.
        let rsrc2: u32 = 2 << 1 | 1 << 9;
        let constants = 0;
        assert_eq!(
            why_refused(compute_program(rsrc2, constants, 1, 2)),
            "root parameter over a system SGPR"
        );
        for user_data in [0, 1, 3, 4] {
            assert!(
                compute_program(rsrc2, constants, 1, user_data).is_ok(),
                "dword {user_data}"
            );
        }
    }

    #[test]
    fn a_fixed_sampler_takes_no_sgpr() {
        let (constants, sampler) = (0, 4);
        // No user SGPRs and group id x in s0: a fixed sampler whose byte
        // names dword 0 holds no SGPR, so it can't collide with the id.
        let rsrc2: u32 = 1 << 7 | 1 << 10;
        assert!(compute_program(rsrc2, sampler, 0, 0).is_ok());
        assert!(compute_program(rsrc2, sampler, 0, 1).is_ok());
        // Constants there still do, and so would a sampler stating an SGPR.
        for root in [constants, sampler] {
            assert_eq!(
                why_refused(compute_program(rsrc2, root, 1, 0)),
                "root parameter over a system SGPR",
                "{root}"
            );
        }
        // Past the user data, up to the end of the SGPRs (s106, just after
        // s105), it is still fine, since it takes none; further on its byte
        // could overflow `user_sgpr` in a merged stage, so that is refused.
        assert!(compute_program(0, sampler, 0, 106).is_ok());
        assert_eq!(
            why_refused(compute_program(0, sampler, 0, 107)),
            "root parameter outside the SGPRs"
        );
        assert_eq!(
            why_refused(compute_program(0, sampler, 0, 0xFF)),
            "root parameter outside the SGPRs"
        );
        // In a pixel program, a fixed sampler beside a root descriptor's
        // high half, and at the user data's end, is fine too.
        assert!(program(&[(1, 2, 0, 0), (sampler, 0, 1, 1)]).is_ok());
        assert!(program(&[(sampler, 0, 106, 1)]).is_ok());
    }

    #[test]
    fn thread_id_count_three_is_still_three_components() {
        // TIDIG_COMP_CNT (bits 11-12) of 3: still x, y and z in v0-v2,
        // never a fourth component in v3.
        let cs = ComputeIo {
            rsrc2: 3 << 11,
            threads: [4, 4, 4],
        };
        assert_eq!(cs.thread_id_vgprs(), 3);
    }

    #[test]
    fn thread_ids_stay_inside_the_vgprs() {
        let cs = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let record = FSSM_HEADER + u64_at(&cs, FSSM_HEADER + slot(HwStage::Cs)).unwrap() as usize;
        let e = stage_data_at(&cs, HwStage::Cs);
        // `count` thread id components (field 0-3) in a program of `vgprs`.
        let program = |vgprs: u32, count: u32| {
            let mut blob = cs.clone();
            blob[record + RECORD_REGS..][..4].copy_from_slice(&vgprs.to_le_bytes());
            blob[e + CS_RSRC2..][..4].copy_from_slice(&(count << 11).to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        for (vgprs, count) in [(1, 0), (2, 1), (3, 2), (3, 3), (VGPRS, 2)] {
            assert!(
                program(vgprs, count).is_ok(),
                "{vgprs} VGPRs, field {count}"
            );
        }
        for (vgprs, count) in [(0, 0), (1, 1), (2, 2), (2, 3)] {
            assert!(
                matches!(
                    program(vgprs, count),
                    Err(Malformed("thread ids outside the VGPRs"))
                ),
                "{vgprs} VGPRs, field {count}"
            );
        }
    }

    #[test]
    fn the_wave_size_is_flag_bit_five_in_every_stage() {
        // `flags` in a lone `stage` program's record.
        let program = |stage: HwStage, flags: u32| {
            let mut blob = fssm(&[(stage, &VS_CODE)]);
            let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
            blob[record + RECORD_FLAGS..][..4].copy_from_slice(&flags.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Only bit 5 says wave32, in a merged, mesh or compute program as
        // in a pixel one; every other bit leaves it wave64.
        for (stage, _) in SLOTS {
            for (flags, wave64) in [
                (0, true),
                (FLAG_WAVE32, false),
                (!FLAG_WAVE32, true),
                (u32::MAX, false),
            ] {
                let p = program(stage, flags)
                    .unwrap_or_else(|e| panic!("{stage:?}: {flags:#x}: {e:?}"));
                assert_eq!((p[0].stage, p[0].wave64), (stage, wave64), "{flags:#x}");
            }
        }
    }

    #[test]
    fn the_vgpr_count_is_the_low_ten_bits() {
        let cs = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let record = FSSM_HEADER + u64_at(&cs, FSSM_HEADER + slot(HwStage::Cs)).unwrap() as usize;
        let program = |regs: u32| {
            let mut blob = cs.clone();
            blob[record + RECORD_REGS..][..4].copy_from_slice(&regs.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Bits above the 10-bit count belong to other fields and change
        // nothing: the program still has the low bits' VGPRs.
        for (regs, vgprs) in [
            (0xFFFF_FC00 | VGPRS, VGPRS as u16),
            (1 << 10 | 1, 1),
            (0xFFFF_FC00 | 256, 256),
        ] {
            assert_eq!(program(regs).unwrap()[0].vgprs, vgprs, "{regs:#x}");
        }
        // So a count of 0x400 is no VGPRs at all, and the thread id in v0
        // has nowhere to arrive, while 0x3FF is past any program.
        assert_eq!(
            why_refused(program(1 << 10)),
            "thread ids outside the VGPRs"
        );
        assert_eq!(why_refused(program(0x3FF)), "implausible VGPR count");
        assert_eq!(why_refused(program(0x7FF)), "implausible VGPR count");
    }

    #[test]
    fn a_root_descriptor_stating_no_sgprs_still_takes_two() {
        let (constants, descriptor) = (0, 1);
        // Its high half is read from the SGPR after its first anyway, so
        // in compute's 16 dwords it may start at dword 14, not 15.
        assert!(compute_program(0, descriptor, 0, 14).is_ok());
        assert_eq!(
            why_refused(compute_program(0, descriptor, 0, 15)),
            "root parameter past the user data"
        );
        // The same high half collides with constants after it, and with a
        // group id right after the user SGPRs.
        assert_eq!(
            why_refused(program(&[(descriptor, 0, 0, 0), (constants, 1, 1, 1)])),
            "root parameters share an SGPR"
        );
        assert!(program(&[(descriptor, 0, 0, 0), (constants, 1, 2, 1)]).is_ok());
        let id_in_s1: u32 = 1 << 1 | 1 << 7;
        assert_eq!(
            why_refused(compute_program(id_in_s1, descriptor, 0, 0)),
            "root parameter over a system SGPR"
        );
        // In a pixel program, s104-s105 exist but are past the user data;
        // from dword 105 its high half would be s106, which no program has.
        assert_eq!(
            why_refused(program(&[(descriptor, 0, 104, 0)])),
            "root parameter past the user data"
        );
        assert_eq!(
            why_refused(program(&[(descriptor, 0, 105, 0)])),
            "root parameter outside the SGPRs"
        );
    }

    /// A compute program with this `COMPUTE_PGM_RSRC2` and these (root
    /// type, SGPRs, user data, slot) parameters.
    fn compute_layout(
        rsrc2: u32,
        params: &[(u32, u32, u8, u8)],
    ) -> Result<Vec<NativeProgram>, ShaderError> {
        let mut blob = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Cs)).unwrap() as usize;
        let e = record + RECORD_SIZE as usize;
        blob[e + CS_RSRC2..][..4].copy_from_slice(&rsrc2.to_le_bytes());
        let table = (blob.len() - FSSM_HEADER) as u64;
        for &(root, sgprs, user_data, slot) in params {
            for v in [root, 0, sgprs, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[user_data, slot, 0, 0]);
        }
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
        let count = params.len() as u64;
        blob[record + RECORD_COUNTS..][..8].copy_from_slice(&count.to_le_bytes());
        native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
    }

    #[test]
    fn every_compute_parameter_is_checked_against_the_system_sgprs() {
        let (constants, table, sampler) = (0, 3, 4);
        // Three user SGPRs, then group id x in s3.
        let rsrc2: u32 = 3 << 1 | 1 << 7;
        assert!(compute_layout(rsrc2, &[(constants, 2, 0, 0), (table, 1, 2, 1)]).is_ok());
        // A fixed sampler beside them names s3 but takes no SGPR.
        assert!(compute_layout(rsrc2, &[(constants, 3, 0, 0), (sampler, 0, 3, 1)]).is_ok());
        // A later parameter over the id is found whatever comes first, and
        // wherever it sits in the list.
        for params in [
            &[(constants, 2, 0, 0), (table, 1, 3, 1)][..],
            &[(table, 1, 3, 1), (constants, 2, 0, 0)],
            &[
                (constants, 1, 0, 0),
                (sampler, 0, 1, 1),
                (constants, 1, 3, 2),
            ],
        ] {
            assert_eq!(
                why_refused(compute_layout(rsrc2, params)),
                "root parameter over a system SGPR",
                "{params:?}"
            );
        }
        // A table range stating no SGPRs still names its table's address
        // SGPR, so over the id it is refused too.
        assert_eq!(
            why_refused(compute_layout(rsrc2, &[(table, 0, 3, 1)])),
            "root parameter over a system SGPR"
        );
        assert!(compute_layout(rsrc2, &[(table, 0, 2, 1)]).is_ok());
    }

    #[test]
    fn a_root_descriptor_stating_more_than_two_sgprs_takes_them_all() {
        let (constants, descriptor) = (0, 1);
        // In a pixel program's 32 dwords, four SGPRs may start at dword 28
        // but not 29.
        assert!(program(&[(descriptor, 4, 28, 0)]).is_ok());
        assert_eq!(
            why_refused(program(&[(descriptor, 4, 29, 0)])),
            "root parameter past the user data"
        );
        // Its fourth SGPR is still its own: constants there collide, and
        // right after it they don't.
        assert_eq!(
            why_refused(program(&[(descriptor, 4, 0, 0), (constants, 1, 3, 1)])),
            "root parameters share an SGPR"
        );
        assert!(program(&[(descriptor, 4, 0, 0), (constants, 1, 4, 1)]).is_ok());
        // In compute's 16 dwords it may start at 12, not 13, and with four
        // user SGPRs and group id x in s4 its last SGPR may be s3, not s4.
        assert!(compute_program(0, descriptor, 4, 12).is_ok());
        assert_eq!(
            why_refused(compute_program(0, descriptor, 4, 13)),
            "root parameter past the user data"
        );
        let id_in_s4: u32 = 4 << 1 | 1 << 7;
        assert!(compute_program(id_in_s4, descriptor, 4, 0).is_ok());
        assert_eq!(
            why_refused(compute_program(id_in_s4, descriptor, 4, 1)),
            "root parameter over a system SGPR"
        );
    }

    #[test]
    fn a_fixed_sampler_stating_sgprs_takes_them_all() {
        let (constants, sampler) = (0, 4);
        // A fixed sampler stating no SGPRs takes none, but one stating some
        // covers each like constants: beside the constants it is fine, in
        // their last dword it collides.
        assert!(program(&[(constants, 2, 0, 0), (sampler, 1, 2, 1)]).is_ok());
        assert!(program(&[(constants, 2, 0, 0), (sampler, 0, 1, 1)]).is_ok());
        assert_eq!(
            why_refused(program(&[(constants, 2, 0, 0), (sampler, 1, 1, 1)])),
            "root parameters share an SGPR"
        );
        // Its SGPRs must fit the user data: two may start at dword 30 of a
        // pixel program's 32, not 31; stating none, dword 31 is fine.
        assert!(program(&[(sampler, 2, 30, 0)]).is_ok());
        assert!(program(&[(sampler, 0, 31, 0)]).is_ok());
        assert_eq!(
            why_refused(program(&[(sampler, 2, 31, 0)])),
            "root parameter past the user data"
        );
        // In compute, with group id x in s3 after three user SGPRs, a
        // sampler stating one SGPR there covers the id; stating none it
        // doesn't.
        let id_in_s3: u32 = 3 << 1 | 1 << 7;
        assert!(compute_program(id_in_s3, sampler, 0, 3).is_ok());
        assert!(compute_program(id_in_s3, sampler, 1, 2).is_ok());
        assert_eq!(
            why_refused(compute_program(id_in_s3, sampler, 1, 3)),
            "root parameter over a system SGPR"
        );
        assert_eq!(
            why_refused(compute_program(id_in_s3, sampler, 2, 2)),
            "root parameter over a system SGPR"
        );
    }

    #[test]
    fn fixed_samplers_share_an_sgpr_only_by_taking_none() {
        let (constants, descriptor, table, sampler, resource_heap) = (0, 1, 3, 4, 5);
        // Stating none, any number of fixed samplers may name one dword,
        // whatever their slots, and sit in a table's SGPR (of its own slot
        // too), a heap's, or a root descriptor's high half, in either order.
        for params in [
            &[(sampler, 0, 4, 1), (sampler, 0, 4, 1)][..],
            &[(sampler, 0, 4, 1), (sampler, 0, 4, 2), (sampler, 0, 4, 3)],
            &[(table, 1, 4, 2), (sampler, 0, 4, 2)],
            &[(sampler, 0, 4, 2), (table, 1, 4, 2)],
            &[(resource_heap, 1, 4, 0), (sampler, 0, 4, 1)],
            &[(descriptor, 2, 4, 0), (sampler, 0, 5, 1)],
        ] {
            assert!(program(params).is_ok(), "{params:?}");
        }
        // Stating one, a fixed sampler holds its SGPR like constants: only a
        // table's ranges share one, so a second sampler there (even the same
        // entry listed twice), a table range of the same slot, a heap, or a
        // root descriptor's high half each collide, in either order.
        for params in [
            &[(sampler, 1, 4, 1), (sampler, 1, 4, 1)][..],
            &[(sampler, 1, 4, 1), (sampler, 1, 4, 2)],
            &[(sampler, 0, 4, 1), (sampler, 1, 4, 1), (sampler, 1, 4, 1)],
            &[(table, 1, 4, 2), (sampler, 1, 4, 2)],
            &[(sampler, 1, 4, 2), (table, 0, 4, 2)],
            &[(resource_heap, 1, 4, 0), (sampler, 1, 4, 1)],
            &[(descriptor, 2, 4, 0), (sampler, 1, 5, 1)],
            &[(sampler, 2, 3, 1), (constants, 1, 4, 0)],
        ] {
            assert_eq!(
                why_refused(program(params)),
                "root parameters share an SGPR",
                "{params:?}"
            );
        }
        // Right after a sampler's last stated SGPR, the next is free.
        assert!(program(&[(sampler, 2, 3, 1), (sampler, 1, 5, 2), (table, 1, 6, 3)]).is_ok());
    }

    #[test]
    fn compute_user_data_is_refused_before_system_sgprs() {
        // Sixteen user SGPRs (all of compute's user data) put group id x in
        // s16. Constants at dword 16 would cover it too, but are first a
        // parameter past the user data: the layout is checked before the
        // stage data, so it is named for that.
        let rsrc2: u32 = 16 << 1 | 1 << 7;
        let constants = 0;
        assert!(compute_program(rsrc2, constants, 16, 0).is_ok());
        assert_eq!(
            why_refused(compute_program(rsrc2, constants, 1, 16)),
            "root parameter past the user data"
        );
        // With the 5-bit field at 31, the ids sit past the user data
        // altogether, so no parameter that fits it can reach them.
        let full: u32 = 0x1F << 1 | 0xF << 7;
        for user_data in [0, 8, 15] {
            assert!(
                compute_program(full, constants, 1, user_data).is_ok(),
                "dword {user_data}"
            );
        }
    }

    #[test]
    fn initialized_sgprs_covers_user_system_and_scratch() {
        // 5 user SGPRs, group ids x and y (s5, s6), no scratch: s0..=s6.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 7);
        // All three group ids after 6 user SGPRs, plus the info SGPR: s0..=s9.
        let cs = ComputeIo {
            rsrc2: 0x78C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 10);
        // The same with scratch enabled adds the wave offset: s0..=s10.
        let cs = ComputeIo {
            rsrc2: 0x78D,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 11);
    }

    #[test]
    fn initialized_sgprs_count_only_enabled_registers() {
        let cs = |rsrc2: u32| ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        };
        // Nothing enabled initializes nothing; scratch alone puts its wave
        // offset in s0.
        assert_eq!(cs(0).initialized_sgprs(), 0);
        assert_eq!(cs(1).initialized_sgprs(), 1);
        // User SGPRs alone end where they do.
        assert_eq!(cs(16 << 1).initialized_sgprs(), 16);
        // Group id z alone takes the SGPR right after the user ones: a
        // disabled x and y leave no gap.
        assert_eq!(cs(3 << 1 | 1 << 9).initialized_sgprs(), 4);
        // The info alone, then with scratch after it.
        assert_eq!(cs(2 << 1 | 1 << 10).initialized_sgprs(), 3);
        assert_eq!(cs(2 << 1 | 1 << 10 | 1).initialized_sgprs(), 4);
        // The thread-id count and LDS size are VGPR and memory settings:
        // they leave the SGPR count alone.
        let base = 5 << 1 | 1 << 7;
        assert_eq!(cs(base).initialized_sgprs(), 6);
        assert_eq!(cs(base | 3 << 11 | 0x80 << 15).initialized_sgprs(), 6);
    }

    #[test]
    fn the_scratch_offset_follows_every_system_sgpr_and_moves_none() {
        let cs = |rsrc2: u32| ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        };
        // Bit 0 adds the wave offset after whatever else is initialized: the
        // user and system SGPRs stay where they were, and the count grows
        // by exactly one, whichever ids and info are enabled.
        for rest in [
            0,
            3 << 1,
            1 << 9,
            2 << 1 | 1 << 8 | 1 << 10,
            0x1F << 1 | 0xF << 7,
        ] {
            let (off, on) = (cs(rest), cs(rest | 1));
            assert_eq!(on.user_sgprs(), off.user_sgprs(), "{rest:#x}");
            assert_eq!(on.system_sgprs(), off.system_sgprs(), "{rest:#x}");
            assert_eq!(
                on.initialized_sgprs(),
                off.initialized_sgprs() + 1,
                "{rest:#x}"
            );
        }
        // Every field at its most: s0..=s30 user, s31..=s34 system, the
        // wave offset in s35.
        assert_eq!(cs(0x1F << 1 | 0xF << 7 | 1).initialized_sgprs(), 36);
        // A parsed program keeps the bit: all sixteen user-data dwords as
        // constants, below the ids, with scratch on.
        let rsrc2: u32 = 16 << 1 | 0xF << 7 | 1;
        let p = compute_program(rsrc2, 0, 16, 0).unwrap();
        assert!(
            matches!(p[0].io, StageIo::Cs(c) if c.rsrc2 == rsrc2 && c.initialized_sgprs() == 21)
        );
    }

    #[test]
    fn a_parsed_compute_program_initializes_its_root_parameters() {
        // 4 user SGPRs, group id x and the info, scratch: s0..=s6.
        let rsrc2: u32 = 4 << 1 | 1 << 7 | 1 << 10 | 1;
        let p = compute_program(rsrc2, 0, 4, 0).unwrap();
        let StageIo::Cs(cs) = p[0].io else {
            panic!("{:?}", p[0].io)
        };
        assert_eq!(cs.rsrc2, rsrc2);
        assert_eq!(cs.initialized_sgprs(), 7);
        // Every SGPR the root parameter covers is below the initialized
        // count, so none of them reads as zero.
        let (first, span) = sgpr_span(HwStage::Cs, &p[0].layout[0]);
        assert!(first + span <= u32::from(cs.initialized_sgprs()));
    }

    #[test]
    fn reads_every_native_blob() {
        let es = xbch(&fssm(&[(HwStage::Es, &VS_CODE)]));
        let vs = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let data = part(&[(0x12, &[0; 16]), (KIND_NATIVE, &es), (KIND_NATIVE, &vs)]);
        let stages: Vec<_> = native_programs(&data)
            .unwrap()
            .iter()
            .map(|p| p.stage)
            .collect();
        assert_eq!(stages, [HwStage::Es, HwStage::Vs]);
    }

    #[test]
    fn pc_layout_has_none() {
        assert!(native_programs(&[0u8; 64]).unwrap().is_empty());
        assert!(
            native_programs(&part(&[(0x0a, b"xbop")]))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_corrupt_blobs() {
        let good = xbch(&fssm(&[(HwStage::Ls, &VS_CODE)]));
        let corrupt = |at: usize, byte: u8| {
            let mut blob = good.clone();
            blob[at] = byte;
            native_programs(&part(&[(KIND_NATIVE, &blob)]))
        };
        assert!(corrupt(4, 9).is_err(), "unknown version");
        assert!(corrupt(16, 0xFF).is_err(), "stated size");
        assert!(corrupt(XBCH_HEADER + 2, 0xFF).is_err(), "zlib stream");
        assert!(native_programs(&part(&[(KIND_NATIVE, &good[..20])])).is_err());

        let mut bad_record = fssm(&[(HwStage::Ls, &VS_CODE)]);
        let at = FSSM_HEADER + slot(HwStage::Ls);
        bad_record[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&bad_record))])).is_err());
    }

    /// Why `data` (a `DXIL` part) is refused.
    fn refused(data: &[u8]) -> &'static str {
        match native_programs(data) {
            Err(Malformed(why)) => why,
            other => panic!("expected a malformed blob, got {other:?}"),
        }
    }

    /// Why a part holding `blob` as its only native entry is refused.
    fn refused_blob(blob: &[u8]) -> &'static str {
        refused(&part(&[(KIND_NATIVE, blob)]))
    }

    #[test]
    fn names_each_xdat_refusal() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let mut data = part(&[(KIND_NATIVE, &native)]);
        data[XDAT_OFFSET + 4..][..4].copy_from_slice(&(MAX_XDAT_ENTRIES as u32 + 1).to_le_bytes());
        assert_eq!(refused(&data), "implausible XDAT entry count");

        // A native entry whose size runs past the part.
        let mut data = part(&[(KIND_NATIVE, &native)]);
        let size = XDAT_OFFSET + 12 + 8;
        data[size..][..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(refused(&data), "XDAT entry out of bounds");

        // Entries of other kinds are never read, so their bounds do not matter.
        let mut data = part(&[(0x0e, b"tbl."), (KIND_NATIVE, &native)]);
        data[size..][..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(native_programs(&data).unwrap().len(), 1);
    }

    #[test]
    fn several_native_entries_are_read_in_index_order() {
        let vs = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let ps_cs = xbch(&fssm(&[(HwStage::Ps, &PS_CODE), (HwStage::Cs, &VS_CODE)]));
        let stages = |data: &[u8]| -> Vec<HwStage> {
            native_programs(data)
                .unwrap()
                .iter()
                .map(|p| p.stage)
                .collect()
        };
        // Each entry's programs follow the last entry's, other kinds between
        // them adding none.
        let data = part(&[(KIND_NATIVE, &ps_cs), (0x0a, b"xbop"), (KIND_NATIVE, &vs)]);
        assert_eq!(stages(&data), [HwStage::Ps, HwStage::Cs, HwStage::Vs]);
        let data = part(&[(KIND_NATIVE, &vs), (KIND_NATIVE, &ps_cs)]);
        assert_eq!(stages(&data), [HwStage::Vs, HwStage::Ps, HwStage::Cs]);
        // The same blob named twice is read twice.
        let data = part(&[(KIND_NATIVE, &vs), (KIND_NATIVE, &vs)]);
        assert_eq!(stages(&data), [HwStage::Vs, HwStage::Vs]);

        // A part with no entries, or none native, has no programs.
        assert!(native_programs(&part(&[])).unwrap().is_empty());
        assert!(stages(&part(&[(0x0a, b"xbop"), (0x0e, b"tbl.")])).is_empty());
    }

    #[test]
    fn one_bad_native_entry_refuses_the_whole_part() {
        let good = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let mut bad = good.clone();
        bad[0..4].copy_from_slice(b"XBCX");
        // Wherever the bad entry sits, nothing from the good ones is kept.
        for data in [
            part(&[(KIND_NATIVE, &good), (KIND_NATIVE, &bad)]),
            part(&[(KIND_NATIVE, &bad), (KIND_NATIVE, &good)]),
            part(&[(KIND_NATIVE, &good), (0x0a, b"xbop"), (KIND_NATIVE, &bad)]),
        ] {
            assert_eq!(refused(&data), "native blob lacks XBCH magic");
        }
    }

    #[test]
    fn xdat_entry_counts_up_to_the_limit_are_read() {
        // The most entries a part may state, all of another kind: read, and
        // none of them native.
        let others: Vec<(u32, &[u8])> = vec![(0x0a, b"xbop"); MAX_XDAT_ENTRIES];
        assert!(native_programs(&part(&others)).unwrap().is_empty());
        // The last of them native is still found.
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let mut entries = others.clone();
        entries[MAX_XDAT_ENTRIES - 1] = (KIND_NATIVE, &native);
        assert_eq!(native_programs(&part(&entries)).unwrap().len(), 1);

        // An index stating more entries than the part holds is cut short,
        // even when every entry it does hold is of another kind.
        let mut data = part(&[(0x0a, b"xbop")]);
        data[XDAT_OFFSET + 4..][..4].copy_from_slice(&(MAX_XDAT_ENTRIES as u32).to_le_bytes());
        assert_eq!(refused(&data), "truncated");
        // An index cut inside its count.
        assert_eq!(refused(&data[..XDAT_OFFSET + 6]), "truncated");
    }

    #[test]
    fn an_index_without_its_exact_magic_is_the_pc_layout() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let good = part(&[(KIND_NATIVE, &native)]);
        assert_eq!(native_programs(&good).unwrap().len(), 1);
        // Any other four bytes where the magic belongs, even the same letters
        // in another case, mean the part holds no native programs.
        for magic in [b"xdat", b"XDAX", b"TADX"] {
            let mut data = good.clone();
            data[XDAT_OFFSET..][..4].copy_from_slice(magic);
            assert!(native_programs(&data).unwrap().is_empty(), "{magic:?}");
        }
        // A part ending before or inside the magic has none either.
        for len in [0, XDAT_OFFSET, XDAT_OFFSET + 3] {
            assert!(native_programs(&good[..len]).unwrap().is_empty(), "{len}");
        }
        // The magic alone, with no count after it, is cut short.
        assert_eq!(refused(&good[..XDAT_OFFSET + 4]), "truncated");
    }

    #[test]
    fn a_native_entry_is_read_exactly_within_its_bounds() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        // The entry's offset and size fields.
        let offset_at = XDAT_OFFSET + 12 + 4;
        let size_at = offset_at + 4;
        let entry = |data: &mut Vec<u8>, offset: u32, size: u32| {
            data[offset_at..][..4].copy_from_slice(&offset.to_le_bytes());
            data[size_at..][..4].copy_from_slice(&size.to_le_bytes());
        };

        // A blob ending the part exactly is read; bytes after it are ignored.
        let data = part(&[(KIND_NATIVE, &native)]);
        assert_eq!(native_programs(&data).unwrap().len(), 1);
        let mut padded = data.clone();
        padded.extend_from_slice(&[0xAB; 7]);
        assert_eq!(native_programs(&padded).unwrap().len(), 1);

        // One byte past the part, by size or by offset, is out of bounds.
        let start = (data.len() - native.len()) as u32;
        let mut over = data.clone();
        entry(&mut over, start, native.len() as u32 + 1);
        assert_eq!(refused(&over), "XDAT entry out of bounds");
        let mut over = data.clone();
        entry(&mut over, start + 1, native.len() as u32);
        assert_eq!(refused(&over), "XDAT entry out of bounds");
        // Both fields at their largest do not wrap round into the part.
        let mut over = data.clone();
        entry(&mut over, u32::MAX, u32::MAX);
        assert_eq!(refused(&over), "XDAT entry out of bounds");

        // A blob one byte short is read as far as it goes, and refused there.
        let mut short = data.clone();
        entry(&mut short, start, native.len() as u32 - 1);
        assert_eq!(refused(&short), "XBCH stream out of bounds");
        // An empty native entry, wherever it points, has no XBCH magic.
        for offset in [0, start, data.len() as u32] {
            let mut empty = data.clone();
            entry(&mut empty, offset, 0);
            assert_eq!(refused(&empty), "native blob lacks XBCH magic", "{offset}");
        }
        // One pointing at the index itself is read as a blob like any other.
        let mut index = data.clone();
        entry(&mut index, XDAT_OFFSET as u32, 16);
        assert_eq!(refused(&index), "native blob lacks XBCH magic");
    }

    #[test]
    fn a_native_entry_cut_inside_the_index_is_truncated() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let data = part(&[(KIND_NATIVE, &native)]);
        // Cut inside the entry's kind, offset or size field.
        for cut in [2, 6, 10] {
            assert_eq!(
                refused(&data[..XDAT_OFFSET + 12 + cut]),
                "truncated",
                "{cut}"
            );
        }
        // The third index word is not read: any value leaves the entries as
        // they were.
        for word in [0, 0x0c, u32::MAX] {
            let mut data = data.clone();
            data[XDAT_OFFSET + 8..][..4].copy_from_slice(&word.to_le_bytes());
            assert_eq!(native_programs(&data).unwrap().len(), 1, "{word:#x}");
        }
    }

    #[test]
    fn names_each_xbch_refusal() {
        let good = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let with = |at: usize, value: &[u8]| {
            let mut blob = good.clone();
            blob[at..at + value.len()].copy_from_slice(value);
            blob
        };
        assert_eq!(
            refused_blob(&with(0, b"XBCX")),
            "native blob lacks XBCH magic"
        );
        assert_eq!(
            refused_blob(&with(4, &(XBCH_VERSION + 1).to_le_bytes())),
            "unknown XBCH version"
        );
        assert_eq!(
            refused_blob(&with(16, &(MAX_UNPACKED + 1).to_le_bytes())),
            "implausible XBCH unpacked size"
        );
        let packed = u64_at(&good, 8).unwrap();
        assert_eq!(
            refused_blob(&with(8, &(packed + 1).to_le_bytes())),
            "XBCH stream out of bounds"
        );
        // A stated size either side of what the stream holds.
        let unpacked = u64_at(&good, 16).unwrap();
        for stated in [unpacked - 1, unpacked + 1] {
            assert_eq!(
                refused_blob(&with(16, &stated.to_le_bytes())),
                "XBCH stream does not inflate to its stated size"
            );
        }
        // A header with no stream after it.
        assert_eq!(refused_blob(&good[..20]), "truncated");
    }

    #[test]
    fn an_xbch_stream_is_read_exactly_as_stated() {
        let fssm_blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let good = xbch(&fssm_blob);
        let packed = u64_at(&good, 8).unwrap();
        let with = |at: usize, value: u64| {
            let mut blob = good.clone();
            blob[at..at + 8].copy_from_slice(&value.to_le_bytes());
            blob
        };

        // Bytes after the stated stream are never read.
        let mut trailing = good.clone();
        trailing.extend_from_slice(&[0xFF; 9]);
        assert_eq!(
            native_programs(&part(&[(KIND_NATIVE, &trailing)]))
                .unwrap()
                .len(),
            1
        );
        // A stated stream one byte short of the real one does not inflate.
        assert_eq!(
            refused_blob(&with(8, packed - 1)),
            "XBCH stream does not inflate to its stated size"
        );
        // Nor does an empty one.
        assert_eq!(
            refused_blob(&with(8, 0)),
            "XBCH stream does not inflate to its stated size"
        );
        // The largest packed size does not wrap round into the blob.
        assert_eq!(
            refused_blob(&with(8, u64::MAX)),
            "XBCH stream out of bounds"
        );

        // The largest plausible unpacked size passes the limit, then fails to
        // match what the stream holds; the largest of all is implausible.
        assert_eq!(
            refused_blob(&with(16, MAX_UNPACKED)),
            "XBCH stream does not inflate to its stated size"
        );
        assert_eq!(
            refused_blob(&with(16, u64::MAX)),
            "implausible XBCH unpacked size"
        );
        // The unpacked limit is checked before the stream's bounds.
        let mut both = with(16, MAX_UNPACKED + 1);
        both[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(refused_blob(&both), "implausible XBCH unpacked size");

        // A stream inflating to bytes that are not FSSM is refused there.
        assert_eq!(
            refused_blob(&xbch(b"not fssm")),
            "native blob lacks FSSM magic"
        );
        // As is one inflating to nothing at all.
        assert_eq!(refused_blob(&xbch(&[])), "native blob lacks FSSM magic");
    }

    #[test]
    fn only_the_exact_native_kind_is_read() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        // The native blob under a neighbouring or wider kind is skipped.
        for kind in [
            KIND_NATIVE - 1,
            KIND_NATIVE + 1,
            KIND_NATIVE | 1 << 8,
            0,
            u32::MAX,
        ] {
            assert!(
                native_programs(&part(&[(kind, &native)]))
                    .unwrap()
                    .is_empty(),
                "{kind:#x}"
            );
        }
        // Two native entries naming the same bytes read them twice.
        let mut data = part(&[(KIND_NATIVE, &native), (KIND_NATIVE, b"")]);
        let first = XDAT_OFFSET + 12;
        let (offset, size) = (
            u32_at(&data, first + 4).unwrap(),
            u32_at(&data, first + 8).unwrap(),
        );
        data[first + 12 + 4..][..4].copy_from_slice(&offset.to_le_bytes());
        data[first + 12 + 8..][..4].copy_from_slice(&size.to_le_bytes());
        assert_eq!(native_programs(&data).unwrap().len(), 2);
    }

    #[test]
    fn an_xbch_header_cut_anywhere_is_truncated() {
        let good = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        // Cut inside the version, the packed size or the unpacked size: each
        // field is read whole or not at all.
        for len in [4, 6, 8, 12, 16, 23] {
            assert_eq!(refused_blob(&good[..len]), "truncated", "{len}");
        }
        // Cut inside the magic, there is no magic to find.
        for len in [1, 3] {
            assert_eq!(
                refused_blob(&good[..len]),
                "native blob lacks XBCH magic",
                "{len}"
            );
        }
        // The header alone, stating an empty stream, reads no stream at all.
        let mut header = good[..XBCH_HEADER].to_vec();
        header[8..16].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            refused_blob(&header),
            "XBCH stream does not inflate to its stated size"
        );
    }

    #[test]
    fn an_xbch_stream_must_be_sound_zlib() {
        let fssm_blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let expect = native_programs(&part(&[(KIND_NATIVE, &xbch(&fssm_blob))])).unwrap();
        // Any compression level, stored blocks included, reads the same.
        for level in [0, 1, 9, 10] {
            let stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm_blob, level);
            let mut blob = xbch(&fssm_blob);
            blob.truncate(XBCH_HEADER);
            blob[8..16].copy_from_slice(&(stream.len() as u64).to_le_bytes());
            blob.extend_from_slice(&stream);
            assert_eq!(
                native_programs(&part(&[(KIND_NATIVE, &blob)])).unwrap(),
                expect,
                "{level}"
            );
        }

        let with_stream = |stream: &[u8]| {
            let mut blob = xbch(&fssm_blob);
            blob.truncate(XBCH_HEADER);
            blob[8..16].copy_from_slice(&(stream.len() as u64).to_le_bytes());
            blob.extend_from_slice(stream);
            refused_blob(&blob)
        };
        let why = "XBCH stream does not inflate to its stated size";
        // A raw deflate stream, with no zlib header, is refused.
        assert_eq!(
            with_stream(&miniz_oxide::deflate::compress_to_vec(&fssm_blob, 6)),
            why
        );
        // As is a zlib stream whose checksum no longer matches its bytes.
        let mut stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm_blob, 0);
        let last = stream.len() - 1;
        stream[last] ^= 1;
        assert_eq!(with_stream(&stream), why);
        // Or whose header names another method.
        let mut stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm_blob, 6);
        stream[0] = 0x79;
        assert_eq!(with_stream(&stream), why);
        // Or whose final block is cut off.
        let stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm_blob, 6);
        assert_eq!(with_stream(&stream[..stream.len() - 5]), why);
        // Bytes that are not a stream at all.
        assert_eq!(with_stream(&[0xFF; 32]), why);
    }

    #[test]
    fn the_unpacked_limit_holds_for_real_streams() {
        // A stream really inflating to the largest plausible size passes the
        // limit and is read as far as its FSSM magic.
        let zeros = vec![0u8; MAX_UNPACKED as usize];
        assert_eq!(refused_blob(&xbch(&zeros)), "native blob lacks FSSM magic");
        // One byte more is implausible before a byte of it is inflated.
        let mut over = zeros.clone();
        over.push(0);
        assert_eq!(refused_blob(&xbch(&over)), "implausible XBCH unpacked size");
        // Stated as the limit, it does not inflate to its stated size: the
        // stream is never cut to fit the limit and read as far as it goes.
        let mut blob = xbch(&over);
        blob[16..24].copy_from_slice(&MAX_UNPACKED.to_le_bytes());
        assert_eq!(
            refused_blob(&blob),
            "XBCH stream does not inflate to its stated size"
        );
    }

    #[test]
    fn a_stated_stream_holding_more_than_one_zlib_stream_reads_the_first() {
        let fssm_blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let expect = native_programs(&part(&[(KIND_NATIVE, &xbch(&fssm_blob))])).unwrap();
        let with_stream = |stream: &[u8]| {
            let mut blob = xbch(&fssm_blob);
            blob.truncate(XBCH_HEADER);
            blob[8..16].copy_from_slice(&(stream.len() as u64).to_le_bytes());
            blob.extend_from_slice(stream);
            native_programs(&part(&[(KIND_NATIVE, &blob)]))
        };
        let stream = miniz_oxide::deflate::compress_to_vec_zlib(&fssm_blob, 6);
        // A second whole stream after the first, inside the stated size,
        // adds nothing to what is inflated.
        let mut twice = stream.clone();
        twice.extend_from_slice(&stream);
        assert_eq!(with_stream(&twice).unwrap(), expect);
    }

    #[test]
    fn only_native_entries_need_their_bounds_in_the_index() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        // The last entry of another kind cut after its kind: its offset and
        // size are never read, so the part still yields the native programs.
        let data = part(&[(KIND_NATIVE, &native), (0x0a, b"xbop")]);
        let last_kind = XDAT_OFFSET + 12 + 12;
        let mut index_only = data[..last_kind + 4].to_vec();
        // Keep the native entry's blob reachable after the cut index.
        let blob_at = index_only.len() as u32;
        index_only[XDAT_OFFSET + 12 + 4..][..4].copy_from_slice(&blob_at.to_le_bytes());
        index_only.extend_from_slice(&native);
        assert_eq!(native_programs(&index_only).unwrap().len(), 1);
        // The same cut after a native kind is truncated.
        let data = part(&[(0x0a, b"xbop"), (KIND_NATIVE, &native)]);
        assert_eq!(refused(&data[..last_kind + 4]), "truncated");

        // An index stating no entries reads none, whatever follows it.
        let mut empty = part(&[(KIND_NATIVE, &native)]);
        empty[XDAT_OFFSET + 4..][..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(native_programs(&empty).unwrap().is_empty());
        // Nor does it need anything after its count.
        assert!(
            native_programs(&empty[..XDAT_OFFSET + 8])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn names_each_fssm_header_refusal() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let with = |at: usize, value: &[u8]| {
            let mut blob = good.clone();
            blob[at..at + value.len()].copy_from_slice(value);
            xbch(&blob)
        };
        assert_eq!(
            refused_blob(&with(0, b"FSSX")),
            "native blob lacks FSSM magic"
        );
        assert_eq!(
            refused_blob(&with(4, &(FSSM_VERSION + 1).to_le_bytes())),
            "unknown FSSM version"
        );
        let payload = (good.len() - FSSM_HEADER) as u64;
        assert_eq!(
            refused_blob(&with(8, &(payload - 1).to_le_bytes())),
            "FSSM payload size disagrees"
        );
        assert_eq!(
            refused_blob(&with(FSSM_HEADER, &(GRAPHICS_SLOT_COUNT + 1).to_le_bytes())),
            "unexpected FSSM slot count"
        );
        assert_eq!(
            refused_blob(&with(
                FSSM_HEADER + slot(HwStage::Vs),
                &payload.to_le_bytes()
            )),
            "FSSM record out of bounds"
        );
    }

    #[test]
    fn only_the_exact_xbch_and_fssm_versions_are_read() {
        let fssm_blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let good = xbch(&fssm_blob);
        // Each version is its whole dword: a neighbour, a stray high byte, the
        // other header's version or every bit set is refused.
        for version in [
            0,
            XBCH_VERSION - 1,
            XBCH_VERSION + 1,
            XBCH_VERSION | 1 << 8,
            XBCH_VERSION | 1 << 31,
            FSSM_VERSION,
            u32::MAX,
        ] {
            let mut blob = good.clone();
            blob[4..8].copy_from_slice(&version.to_le_bytes());
            assert_eq!(refused_blob(&blob), "unknown XBCH version", "{version:#x}");
        }
        for version in [
            0,
            FSSM_VERSION - 1,
            FSSM_VERSION + 1,
            FSSM_VERSION & 0xFF,
            FSSM_VERSION | 1 << 16,
            XBCH_VERSION,
            u32::MAX,
        ] {
            let mut blob = fssm_blob.clone();
            blob[4..8].copy_from_slice(&version.to_le_bytes());
            assert_eq!(
                refused_blob(&xbch(&blob)),
                "unknown FSSM version",
                "{version:#x}"
            );
        }
        assert_eq!(
            native_programs(&part(&[(KIND_NATIVE, &good)]))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn xbch_and_fssm_headers_breaking_several_rules_are_named_for_the_first_checked() {
        let fssm_blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let good = xbch(&fssm_blob);
        let xbch_with = |fields: &[(usize, &[u8])]| {
            let mut blob = good.clone();
            for &(at, value) in fields {
                blob[at..at + value.len()].copy_from_slice(value);
            }
            refused_blob(&blob)
        };
        let bad_version = (XBCH_VERSION + 1).to_le_bytes();
        let implausible = (MAX_UNPACKED + 1).to_le_bytes();
        let past_end = u64::MAX.to_le_bytes();
        // Magic, then version, then the unpacked limit, then the stream's
        // bounds: each is named while every later rule is broken too.
        assert_eq!(
            xbch_with(&[(0, b"XBCX"), (4, &bad_version), (16, &implausible)]),
            "native blob lacks XBCH magic"
        );
        assert_eq!(
            xbch_with(&[(4, &bad_version), (8, &past_end), (16, &implausible)]),
            "unknown XBCH version"
        );
        // A bad version is named even when the header is cut after it.
        let mut cut = good[..8].to_vec();
        cut[4..8].copy_from_slice(&bad_version);
        assert_eq!(refused_blob(&cut), "unknown XBCH version");

        let fssm_with = |fields: &[(usize, &[u8])]| {
            let mut blob = fssm_blob.clone();
            for &(at, value) in fields {
                blob[at..at + value.len()].copy_from_slice(value);
            }
            refused_blob(&xbch(&blob))
        };
        let bad_version = (FSSM_VERSION + 1).to_le_bytes();
        let wrong_payload = 0u64.to_le_bytes();
        let wrong_slots = (GRAPHICS_SLOT_COUNT + 1).to_le_bytes();
        // Magic, then version, then the payload size, then the slot count.
        assert_eq!(
            fssm_with(&[
                (0, b"FSSX"),
                (4, &bad_version),
                (8, &wrong_payload),
                (FSSM_HEADER, &wrong_slots)
            ]),
            "native blob lacks FSSM magic"
        );
        assert_eq!(
            fssm_with(&[
                (4, &bad_version),
                (8, &wrong_payload),
                (FSSM_HEADER, &wrong_slots)
            ]),
            "unknown FSSM version"
        );
        assert_eq!(
            fssm_with(&[(8, &wrong_payload), (FSSM_HEADER, &wrong_slots)]),
            "FSSM payload size disagrees"
        );
        // A bad FSSM version is named even with nothing after it.
        let mut cut = fssm_blob[..8].to_vec();
        cut[4..8].copy_from_slice(&bad_version);
        assert_eq!(refused_blob(&xbch(&cut)), "unknown FSSM version");
    }

    #[test]
    fn an_fssm_header_or_slot_table_cut_anywhere_is_truncated() {
        let good = fssm(&[]);
        // `good[..len]` with its stated payload size made to agree, so only
        // the cut itself can be refused.
        let cut = |len: usize| {
            let mut blob = good[..len].to_vec();
            if len >= FSSM_HEADER {
                let payload = (len - FSSM_HEADER) as u64;
                blob[8..16].copy_from_slice(&payload.to_le_bytes());
            }
            refused_blob(&xbch(&blob))
        };
        // Cut inside the magic, there is no magic to find.
        for len in [0, 1, 3] {
            assert_eq!(cut(len), "native blob lacks FSSM magic", "{len}");
        }
        // Cut inside the version or the payload size: each is read whole or
        // not at all.
        for len in [4, 7, 8, 12, 15] {
            assert_eq!(cut(len), "truncated", "{len}");
        }
        // An empty payload has no slot count, and a slot table cut inside
        // any slot is read no further than the cut.
        for len in [0, 4, 7, 8, 12, 0x2f, 0x30, 0x47, 0x5f, 0x67] {
            assert_eq!(cut(FSSM_HEADER + len), "truncated", "payload {len:#x}");
        }
        // The whole table, every slot absent, holds no programs.
        let blob = xbch(&good);
        assert!(
            native_programs(&part(&[(KIND_NATIVE, &blob)]))
                .unwrap()
                .is_empty()
        );
        assert_eq!(good.len(), FSSM_HEADER + slot(HwStage::Cs) + 8);
    }

    #[test]
    fn a_record_must_fit_whole_in_the_payload() {
        // A local-shader record (no stage data) of empty code ending exactly
        // at the payload end is read; one byte less is refused as a record,
        // not as a field read past the end.
        let mut blob = fssm(&[(HwStage::Ls, &[])]);
        let record = stage_data_at(&blob, HwStage::Ls) - RECORD_SIZE as usize;
        blob[record..][..8].copy_from_slice(&0u64.to_le_bytes());
        let with_len = |len: usize| {
            let mut blob = blob[..len].to_vec();
            let payload = (len - FSSM_HEADER) as u64;
            blob[8..16].copy_from_slice(&payload.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let end = record + RECORD_SIZE as usize;
        let programs = with_len(end).unwrap();
        assert_eq!(programs.len(), 1);
        assert_eq!(programs[0].stage, HwStage::Ls);
        assert!(programs[0].code.is_empty());
        for len in [end - 1, end - 4, record + 1, record] {
            assert!(
                matches!(with_len(len), Err(Malformed("FSSM record out of bounds"))),
                "{len}"
            );
        }
    }

    /// A one-program blob whose stage data is cut to `len` bytes: the code
    /// is moved to the payload start (empty) so the stage data ends the blob.
    fn cut_stage_data(stage: HwStage, len: usize) -> Result<Vec<NativeProgram>, ShaderError> {
        let mut blob = fssm(&[(stage, &[])]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
        blob[record..][..8].copy_from_slice(&0u64.to_le_bytes());
        let e = stage_data_at(&blob, stage);
        blob.truncate(e + len);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
    }

    #[test]
    fn refuses_stage_data_cut_short() {
        // The fewest bytes each stage reads: a vertex program up to its
        // output count, a mesh program up to its group info (after its
        // outputs and clip control), compute up to the group size, a pixel
        // program up to the last MRT's patch offset (every patch absent).
        let needs = [
            (HwStage::Vs, VS_OUTPUT_COUNT + 4),
            (HwStage::Ms, MS_INFO + 8),
            (HwStage::Cs, CS_THREADS + 12),
            (HwStage::Ps, PS_PATCHES + (MRTS - 1) * PATCH_BYTES + 4),
            (HwStage::Ls, 0),
            (HwStage::Gs, 0),
        ];
        for (stage, need) in needs {
            assert!(
                cut_stage_data(stage, need).is_ok(),
                "{stage:?} with {need} bytes"
            );
            for len in 0..need {
                // Only a mesh program's group info is read as one span, named
                // as such; every other field is a lone read past the end.
                let why = if stage == HwStage::Ms && len >= VS_OUTPUT_COUNT + 4 {
                    "stage data out of bounds"
                } else {
                    "truncated"
                };
                assert!(
                    matches!(cut_stage_data(stage, len), Err(Malformed(w)) if w == why),
                    "{stage:?} with {len} of {need} bytes"
                );
            }
        }
    }

    #[test]
    fn names_each_record_refusal() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let record = stage_data_at(&good, HwStage::Vs) - RECORD_SIZE as usize;
        let with = |at: usize, value: &[u8]| {
            let mut blob = good.clone();
            blob[at..at + value.len()].copy_from_slice(value);
            xbch(&blob)
        };
        // One byte more code than the payload holds.
        assert_eq!(
            refused_blob(&with(record + 8, &(VS_CODE.len() as u64 + 1).to_le_bytes())),
            "native code out of bounds"
        );
        assert_eq!(
            refused_blob(&with(record + RECORD_REGS, &257u32.to_le_bytes())),
            "implausible VGPR count"
        );
        // 256 is the most a program can have.
        let blob = with(record + RECORD_REGS, &256u32.to_le_bytes());
        assert_eq!(
            native_programs(&part(&[(KIND_NATIVE, &blob)])).unwrap()[0].vgprs,
            256
        );
        // One root parameter, at an offset past the payload.
        let mut blob = good.clone();
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&u64::MAX.to_le_bytes());
        blob[record + RECORD_COUNTS..][..4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "FSSM root table out of bounds");
    }

    #[test]
    fn names_each_stage_data_refusal() {
        let vs = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let e = stage_data_at(&vs, HwStage::Vs);
        for count_at in [VS_INPUT_COUNT, VS_OUTPUT_COUNT] {
            let mut blob = vs.clone();
            blob[e + count_at..][..4].copy_from_slice(&(MAX_SEMANTICS + 1).to_le_bytes());
            assert_eq!(refused_blob(&xbch(&blob)), "implausible semantic count");
        }
        // One input in a program of `vgprs` VGPRs. With all 256, x-w of
        // v252 end at v255 and fit; from v253, w would be v256.
        let record = FSSM_HEADER + u64_at(&vs, FSSM_HEADER + slot(HwStage::Vs)).unwrap() as usize;
        let input_of = |vgprs: u32, vgpr: u8, mask: u8| {
            let mut blob = vs.clone();
            blob[record + RECORD_REGS..][..4].copy_from_slice(&vgprs.to_le_bytes());
            blob[e + VS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
            blob[e + VS_INPUTS..][..5].copy_from_slice(&[0, 0, mask, vgpr, 0]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let input = |vgpr: u8, mask: u8| input_of(256, vgpr, mask);
        assert!(input(252, 0xf).is_ok());
        assert!(input(255, 0x1).is_ok());
        assert!(input(255, 0).is_ok(), "no channel is loaded");
        // With 28, x-w of v24 end at v27 and fit; w of v25 would be v28.
        assert!(input_of(VGPRS, 24, 0xf).is_ok());
        assert!(input_of(VGPRS, 200, 0).is_ok(), "no channel is loaded");
        for (vgprs, vgpr, mask) in [
            (256, 253, 0xf),
            (256, 255, 0x2),
            (VGPRS, 25, 0xf),
            (VGPRS, 28, 0x1),
            (VGPRS, 252, 0xf),
            (0, 4, 0x1),
        ] {
            assert!(
                matches!(
                    input_of(vgprs, vgpr, mask),
                    Err(Malformed("vertex input outside the VGPRs"))
                ),
                "{vgprs} VGPRs: v{vgpr} mask {mask:#x}"
            );
        }
        // v1 and v4 up hold no id: x in v1 and x-w from v4 load. Any channel
        // on v0, v2 or v3 is refused, including y and z of an input in v1.
        assert!(input(1, 0x1).is_ok());
        assert!(input(4, 0xf).is_ok());
        assert!(input(0, 0).is_ok(), "no channel is loaded");
        for (vgpr, mask) in [(0, 0x1), (1, 0x2), (2, 0x1), (3, 0x1), (1, 0x4)] {
            assert!(
                matches!(
                    input(vgpr, mask),
                    Err(Malformed("vertex input over a system VGPR"))
                ),
                "v{vgpr} mask {mask:#x}"
            );
        }
        // Two inputs: x-w in v4 and the second at `vgpr` with `mask`.
        let inputs = |vgpr: u8, mask: u8| {
            let mut blob = vs.clone();
            blob[e + VS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            blob[e + VS_INPUTS..][..10].copy_from_slice(&[0, 0, 0xf, 4, 0, 0, 1, mask, vgpr, 0]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        assert!(inputs(8, 0xf).is_ok());
        assert!(inputs(5, 0).is_ok(), "no channel is loaded");
        // x in v1 sits below the first input; any channel on v4-v7 overlaps
        // it, including w of an input in v1.
        assert!(inputs(1, 0x1).is_ok());
        for (vgpr, mask) in [(4, 0xf), (7, 0x1), (6, 0x2), (1, 0x8), (5, 0x4)] {
            assert!(
                matches!(inputs(vgpr, mask), Err(Malformed("vertex inputs overlap"))),
                "v{vgpr} mask {mask:#x}"
            );
        }

        let ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let mut blob = ps.clone();
        blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&(MAX_SEMANTICS + 1).to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "implausible semantic count");
        let mut blob = ps.clone();
        let size = PATCH_SLOT_DWORDS as u32 + 1;
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&size.to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "implausible export patch size");
        // A one-dword patch at the code's last dword fits; one past it does not.
        let mut blob = ps.clone();
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&1u32.to_le_bytes());
        let last = (PS_CODE.len() / 4) as u32;
        blob[e + PS_PATCHES..][..4].copy_from_slice(&(last - 1).to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_ok());
        blob[e + PS_PATCHES..][..4].copy_from_slice(&last.to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "export patch outside the code");
        // Two-dword patches for MRT0 and MRT1: adjacent fits, sharing a
        // dword does not; with no dwords to patch, nothing can overlap.
        let code: Vec<u8> = [0xBF80_0000u32; 4] // s_nop
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let ps4 = fssm(&[(HwStage::Ps, &code)]);
        let patches = |size: u32, offsets: [u32; 2]| {
            let mut blob = ps4.clone();
            blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&size.to_le_bytes());
            for (mrt, offset) in offsets.into_iter().enumerate() {
                let at = e + PS_PATCHES + mrt * PATCH_BYTES;
                blob[at..][..4].copy_from_slice(&offset.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        assert!(patches(2, [0, 2]).is_ok());
        assert!(patches(2, [2, 0]).is_ok());
        assert!(patches(0, [1, 1]).is_ok());
        for offsets in [[0, 1], [1, 0], [1, 1]] {
            assert!(
                matches!(
                    patches(2, offsets),
                    Err(Malformed("export patches overlap"))
                ),
                "{offsets:?}"
            );
        }

        // numthreads at each limit fits; past one, or a zero component, does
        // not. 65536 x 65536 would overflow lowering's u32 thread count.
        let cs = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let e = stage_data_at(&cs, HwStage::Cs);
        let group = |threads: [u32; 3]| {
            let mut blob = cs.clone();
            for (k, n) in threads.into_iter().enumerate() {
                blob[e + CS_THREADS + 4 * k..][..4].copy_from_slice(&n.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        for fits in [[1024, 1, 1], [1, 1024, 1], [1, 16, 64], [32, 32, 1]] {
            let p = group(fits).unwrap();
            assert!(
                matches!(p[0].io, StageIo::Cs(ComputeIo { threads, .. }) if threads == fits),
                "{fits:?}"
            );
        }
        for bad in [
            [0, 1, 1],
            [1, 1, 0],
            [1025, 1, 1],
            [1, 1, 65],
            [64, 32, 1],
            [65536, 65536, 1],
            [u32::MAX, u32::MAX, u32::MAX],
        ] {
            assert!(
                matches!(group(bad), Err(Malformed("implausible thread-group size"))),
                "{bad:?}"
            );
        }

        // A compute group's LDS (512-byte granules in RSRC2 bits 15-23):
        // 128 granules is 64 KiB and fits; one more, or the field's maximum
        // of 511, does not.
        let cs_lds = |granules: u32| {
            let mut blob = cs.clone();
            blob[e + CS_RSRC2..][..4].copy_from_slice(&(granules << 15).to_le_bytes());
            xbch(&blob)
        };
        let p = native_programs(&part(&[(KIND_NATIVE, &cs_lds(128))])).unwrap();
        assert!(matches!(p[0].io, StageIo::Cs(c) if c.lds_bytes() == MAX_LDS_BYTES));
        for bad in [129, 0x1FF] {
            assert_eq!(refused_blob(&cs_lds(bad)), "implausible LDS size", "{bad}");
        }

        // A mesh group of one lane fits; an empty one does not.
        let ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&ms))])).unwrap();
        assert!(matches!(&p[0].io, StageIo::Ms(m) if m.threads == 1));
        let mut blob = ms.clone();
        blob[e + MS_INFO..][..2].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "implausible thread-group size");

        // Mesh limits: 128 lanes and 256 vertices and primitives fit, with
        // lines or triangles; a lane, an output past them, or another
        // topology does not.
        let info = |bytes: [u8; 8]| {
            let mut blob = ms.clone();
            blob[e + MS_INFO..][..8].copy_from_slice(&bytes);
            xbch(&blob)
        };
        for topology in [1, 2] {
            let p = native_programs(&part(&[(
                KIND_NATIVE,
                &info([128, 0, 0, 1, 0, 1, 0, topology]),
            )]))
            .unwrap();
            assert!(matches!(
                &p[0].io,
                StageIo::Ms(m) if (m.threads, m.max_vertices, m.max_primitives, m.topology)
                    == (128, 256, 256, topology)
            ));
        }
        assert_eq!(
            refused_blob(&info([129, 0, 1, 0, 1, 0, 0, 2])),
            "implausible thread-group size"
        );
        assert_eq!(
            refused_blob(&info([0xff, 0xff, 1, 0, 1, 0, 0, 2])),
            "implausible thread-group size"
        );
        for bad in [
            [1, 0, 1, 1, 1, 0, 0, 2],
            [1, 0, 1, 0, 1, 1, 0, 2],
            [1, 0, 0xff, 0xff, 1, 0, 0, 2],
        ] {
            assert_eq!(
                refused_blob(&info(bad)),
                "implausible mesh output count",
                "{bad:?}"
            );
        }
        for topology in [0, 3, 0xff] {
            assert_eq!(
                refused_blob(&info([1, 0, 1, 0, 1, 0, 0, topology])),
                "unknown mesh topology",
                "{topology}"
            );
        }

        // A mesh group's LDS: 64 KiB fits; a byte more, or u32::MAX, does not.
        let lds = |bytes: u32| {
            let mut blob = ms.clone();
            let record = e - RECORD_SIZE as usize;
            blob[record + RECORD_LDS..][..4].copy_from_slice(&bytes.to_le_bytes());
            xbch(&blob)
        };
        let p = native_programs(&part(&[(KIND_NATIVE, &lds(MAX_LDS_BYTES))])).unwrap();
        assert!(matches!(&p[0].io, StageIo::Ms(m) if m.lds_bytes == 64 << 10));
        for bad in [MAX_LDS_BYTES + 1, u32::MAX] {
            assert_eq!(refused_blob(&lds(bad)), "implausible LDS size", "{bad}");
        }
    }

    #[test]
    fn a_compute_group_inside_each_component_limit_is_still_capped_by_its_product() {
        let cs = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let e = stage_data_at(&cs, HwStage::Cs);
        let group = |threads: [u32; 3]| {
            let mut blob = cs.clone();
            for (k, n) in threads.into_iter().enumerate() {
                blob[e + CS_THREADS + 4 * k..][..4].copy_from_slice(&n.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Exactly 1024 lanes over all three components fits, as does one
        // lane short of it.
        for fits in [[4, 4, 64], [8, 8, 16], [16, 1, 64], [31, 33, 1], [1, 1, 64]] {
            let p = group(fits).unwrap();
            assert!(
                matches!(p[0].io, StageIo::Cs(ComputeIo { threads, .. }) if threads == fits),
                "{fits:?}"
            );
        }
        // Every component below is within its own limit (1024, 1024, 64);
        // only the product, one lane or more past 1024, is refused.
        for bad in [
            [5, 5, 41],
            [3, 7, 49],
            [2, 513, 1],
            [1025 / 5 + 1, 5, 1],
            [1024, 1024, 64],
        ] {
            assert!(
                matches!(group(bad), Err(Malformed("implausible thread-group size"))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn compute_lds_is_read_from_rsrc2_bits_15_to_23_alone() {
        let cs = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let e = stage_data_at(&cs, HwStage::Cs);
        let lds_field = 0x1FFu32 << 15;
        let with_rsrc2 = |rsrc2: u32| {
            let mut blob = cs.clone();
            blob[e + CS_RSRC2..][..4].copy_from_slice(&rsrc2.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Every bit outside the field set (three thread ids fit the 28
        // VGPRs, and with no root parameters the system SGPRs clash with
        // nothing): the LDS is still the field's granules alone.
        for (granules, bytes) in [(0, 0), (1, 512), (128, MAX_LDS_BYTES)] {
            let rsrc2 = !lds_field | granules << 15;
            let p = with_rsrc2(rsrc2).unwrap();
            assert!(
                matches!(p[0].io, StageIo::Cs(c) if c.rsrc2 == rsrc2 && c.lds_bytes() == bytes),
                "{granules} granules"
            );
        }
        // A granule past 64 KiB is refused whatever the other bits hold.
        for others in [0, !lds_field] {
            assert!(
                matches!(
                    with_rsrc2(others | 129 << 15),
                    Err(Malformed("implausible LDS size"))
                ),
                "{others:#x}"
            );
        }
    }

    #[test]
    fn the_mesh_info_user_sgpr_byte_never_refuses_a_mesh() {
        let ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        // One lane, one vertex and primitive, lines; byte 6 is the user
        // SGPR count, which the limits and topology checks never read.
        for user_sgprs in [0u8, 13, 0x80, 0xff] {
            let mut blob = ms.clone();
            blob[e + MS_INFO..][..8].copy_from_slice(&[1, 0, 1, 0, 1, 0, user_sgprs, 1]);
            let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap();
            assert!(
                matches!(
                    &p[0].io,
                    StageIo::Ms(m) if (m.threads, m.max_vertices, m.max_primitives, m.topology)
                        == (1, 1, 1, 1)
                ),
                "{user_sgprs}"
            );
        }
    }

    #[test]
    fn mesh_info_breaking_several_rules_is_named_for_the_first_checked() {
        let ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        let mesh = |info: [u8; 8], lds: u32, shared_param: bool| {
            let mut blob = ms.clone();
            blob[e + MS_INFO..][..8].copy_from_slice(&info);
            let record = e - RECORD_SIZE as usize;
            blob[record + RECORD_LDS..][..4].copy_from_slice(&lds.to_le_bytes());
            if shared_param {
                // Two outputs on parameter 3, one per vertex and one per
                // primitive: bit 7 aside, they name the same parameter.
                blob[e + VS_OUTPUTS..][..12]
                    .copy_from_slice(&[6, 0, 3, 0, 0, 0, 6, 1, 0x83, 0, 0, 0]);
                blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            }
            xbch(&blob)
        };
        let (no_lanes, too_many_vertices, no_topology) = (0u8, 0xffu8, 0u8);
        let too_much_lds = MAX_LDS_BYTES + 1;
        // Each mesh below also breaks every rule named after it: the outputs
        // are checked first, then lanes, output counts, topology and LDS.
        for (info, lds, shared, why) in [
            (
                [no_lanes, 0, 1, too_many_vertices, 1, 0, 0, no_topology],
                too_much_lds,
                true,
                "vertex outputs share a parameter",
            ),
            (
                [no_lanes, 0, 1, too_many_vertices, 1, 0, 0, no_topology],
                too_much_lds,
                false,
                "implausible thread-group size",
            ),
            (
                [1, 0, 1, too_many_vertices, 1, 0, 0, no_topology],
                too_much_lds,
                false,
                "implausible mesh output count",
            ),
            (
                [1, 0, 1, 0, 1, 0, 0, no_topology],
                too_much_lds,
                false,
                "unknown mesh topology",
            ),
            (
                [1, 0, 1, 0, 1, 0, 0, 2],
                too_much_lds,
                false,
                "implausible LDS size",
            ),
        ] {
            assert_eq!(refused_blob(&mesh(info, lds, shared)), why, "{info:?}");
        }
        // With every rule kept, the same mesh is read.
        let p = native_programs(&part(&[(
            KIND_NATIVE,
            &mesh([1, 0, 1, 0, 1, 0, 0, 2], 0, false),
        )]))
        .unwrap();
        assert!(matches!(&p[0].io, StageIo::Ms(m) if (m.threads, m.topology) == (1, 2)));
    }

    #[test]
    fn sparse_vertex_inputs_can_interleave_at_the_last_vgpr() {
        let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let data = stage_data_at(&blob, HwStage::Vs);
        let regs = data - RECORD_SIZE as usize + RECORD_REGS;
        blob[regs..][..4].copy_from_slice(&256u32.to_le_bytes());
        blob[data + VS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        let inputs = &mut blob[data + VS_INPUTS..][..10];
        // Both rows start at v252. Row 0 loads x and z, row 1 loads y and w.
        inputs.copy_from_slice(&[0, 0, 0x5, 252, 0, 0, 1, 0xa, 252, 0]);
        let parse = |blob: &[u8]| native_programs(&part(&[(KIND_NATIVE, &xbch(blob))]));
        let programs = parse(&blob).unwrap();
        let StageIo::Vs { inputs, .. } = &programs[0].io else {
            panic!("expected vertex inputs");
        };
        assert_eq!((inputs[0].mask, inputs[1].mask), (0x5, 0xa));

        // Enabling z in the second row makes both rows load v254.
        blob[data + VS_INPUTS + 7] |= 0x4;
        assert_eq!(why_refused(parse(&blob)), "vertex inputs overlap");
    }

    #[test]
    fn a_vertex_input_uses_its_own_stages_vgpr_budget() {
        // Store the pixel record first, opposite the slot traversal order.
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE), (HwStage::Vs, &VS_CODE)]);
        let vs = stage_data_at(&blob, HwStage::Vs);
        let ps = stage_data_at(&blob, HwStage::Ps);
        blob[vs + VS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[vs + VS_INPUTS..][..5].copy_from_slice(&[0, 0, 0x1, 7, 0]);
        let mut program = |vs_count: u32, ps_count: u32| {
            for (data, count) in [(vs, vs_count), (ps, ps_count)] {
                let regs = data - RECORD_SIZE as usize + RECORD_REGS;
                blob[regs..][..4].copy_from_slice(&count.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let programs = program(8, 7).unwrap();
        assert_eq!((programs[0].stage, programs[0].vgprs), (HwStage::Vs, 8));
        assert_eq!((programs[1].stage, programs[1].vgprs), (HwStage::Ps, 7));
        assert_eq!(why_refused(program(7, 8)), "vertex input outside the VGPRs");
    }

    #[test]
    fn refuses_outputs_sharing_a_parameter_and_inputs_sharing_an_attribute() {
        // Two outputs: param0 and `param`. Bit 7 (per-primitive) does not
        // make a separate index: `exp param<n>` matches n either way.
        for stage in [HwStage::Vs, HwStage::Ms] {
            let base = fssm(&[(stage, &VS_CODE)]);
            let e = stage_data_at(&base, stage);
            let outputs = |param: u8| {
                let mut blob = base.clone();
                blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
                blob[e + VS_OUTPUTS..][..12]
                    .copy_from_slice(&[6, 0, 0, 0, 0, 0, 6, 1, param, 1, 0, 0]);
                native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
            };
            assert!(outputs(1).is_ok(), "{stage:?}");
            assert!(outputs(31).is_ok(), "{stage:?}");
            for param in [0, 0x80] {
                assert!(
                    matches!(
                        outputs(param),
                        Err(Malformed("vertex outputs share a parameter"))
                    ),
                    "{stage:?} param {param:#x}"
                );
            }
        }

        // Two inputs: attr0 and `attr`.
        let ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let inputs = |attr: u8| {
            let mut blob = ps.clone();
            blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            blob[e + PS_INPUTS..][..16]
                .copy_from_slice(&[6, 0, 0xf, 0, 0, 0, 0, 0, 6, 1, 0xf, attr, 0, 0, 1, 0]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        assert!(inputs(1).is_ok());
        assert!(matches!(
            inputs(0),
            Err(Malformed("pixel inputs share an attribute"))
        ));
    }

    #[test]
    fn refuses_pixel_inputs_reading_a_row_two_ways() {
        // Two inputs, attr0 and attr1, with these (row, row2, flags): flag
        // bit 3 is fp16, bit 4 two values per dword.
        let ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let inputs = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            let mut blob = ps.clone();
            blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            blob[e + PS_INPUTS..][..16].copy_from_slice(&[
                6, a.0, 0x3, 0, 0, a.1, 0, a.2, 6, b.0, 0xc, 1, 0, b.1, 1, b.2,
            ]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let (fp16, pack2) = (0x8, 0x18);
        // Different rows may differ; one row packed the same way (or 32-bit
        // in both, or a 32-bit input's unused row2 differing) is fine.
        for (a, b) in [
            ((0, 0, 0), (1, 1, fp16)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 5, 0)),
            ((2, 2, fp16), (2, 2, fp16)),
            ((2, 3, pack2), (2, 3, pack2)),
            ((0, 0, 0x1), (0, 0, 0x1)),
        ] {
            assert!(inputs(a, b).is_ok(), "{a:?} {b:?}");
        }
        for (a, b, what) in [
            ((0, 0, 0), (0, 0, fp16), "32-bit then fp16"),
            ((0, 0, fp16), (0, 0, 0), "fp16 then 32-bit"),
            ((2, 2, fp16), (2, 2, pack2), "one packed, one not"),
            ((2, 3, pack2), (2, 4, pack2), "different second rows"),
        ] {
            assert!(
                matches!(
                    inputs(a, b),
                    Err(Malformed("pixel inputs read a row two ways"))
                ),
                "{what}"
            );
        }
    }

    #[test]
    fn refuses_pixel_inputs_claiming_a_packed_row() {
        // Two inputs, attr0 and attr1, with these (row, row2, flags), as in
        // `refuses_pixel_inputs_reading_a_row_two_ways`.
        let ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let inputs = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            let mut blob = ps.clone();
            blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            blob[e + PS_INPUTS..][..16].copy_from_slice(&[
                6, a.0, 0x3, 0, 0, a.1, 0, a.2, 6, b.0, 0xc, 1, 0, b.1, 1, b.2,
            ]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let (fp16, pack2) = (0x8, 0x18);
        // A second row no input names; one row's inputs sharing theirs; a
        // packed input with no second row; an unpacked input's unused row2.
        for (a, b) in [
            ((2, 3, pack2), (4, 5, pack2)),
            ((2, 3, pack2), (2, 3, pack2)),
            ((2, 2, pack2), (3, 3, fp16)),
            ((2, 3, fp16), (3, 3, fp16)),
            ((2, 3, 0), (3, 3, 0)),
        ] {
            assert!(inputs(a, b).is_ok(), "{a:?} {b:?}");
        }
        for (a, b, what) in [
            ((2, 3, pack2), (3, 3, 0), "32-bit input on the second row"),
            ((2, 3, pack2), (3, 3, fp16), "fp16 input on the second row"),
            ((3, 3, fp16), (2, 3, pack2), "named before it is claimed"),
            ((2, 4, pack2), (3, 4, pack2), "two rows claiming one"),
            ((2, 3, pack2), (3, 2, pack2), "each claiming the other"),
        ] {
            assert!(
                matches!(
                    inputs(a, b),
                    Err(Malformed("pixel inputs claim a packed row"))
                ),
                "{what}"
            );
        }
    }

    #[test]
    fn a_pixel_input_packed_without_fp16_is_read_as_32_bit() {
        // Two inputs, attr0 and attr1, with these (row, row2, flags), as in
        // `refuses_pixel_inputs_reading_a_row_two_ways`. Flag bit 4 (two
        // values per dword) is kept as read, but only a 16-bit input packs:
        // without bit 3 an input is 32-bit, so its row2 claims no row.
        let ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let inputs = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            let mut blob = ps.clone();
            blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
            blob[e + PS_INPUTS..][..16].copy_from_slice(&[
                6, a.0, 0x3, 0, 0, a.1, 0, a.2, 6, b.0, 0xc, 1, 0, b.1, 1, b.2,
            ]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let (fp16, pack2, bare) = (0x8, 0x18, 0x10);
        let p = inputs((0, 0, bare), (1, 1, 0)).unwrap();
        let StageIo::Ps { inputs: read, .. } = &p[0].io else {
            panic!("{:?}", p[0].io)
        };
        assert!(read[0].pack2 && !read[0].fp16);
        // Sharing a row with a 32-bit input; a row2 another input names, or
        // that a second bare input names too.
        for (a, b) in [
            ((0, 0, bare), (0, 0, 0)),
            ((0, 0, 0), (0, 0, bare)),
            ((2, 3, bare), (3, 3, 0)),
            ((2, 3, bare), (3, 3, pack2)),
            ((2, 3, bare), (4, 3, bare)),
        ] {
            assert!(inputs(a, b).is_ok(), "{a:?} {b:?}");
        }
        // It still reads its row as 32-bit, so a 16-bit input there clashes.
        for (a, b) in [
            ((0, 0, bare), (0, 0, fp16)),
            ((2, 3, bare), (2, 3, pack2)),
            ((2, 3, pack2), (2, 3, bare)),
        ] {
            assert!(
                matches!(
                    inputs(a, b),
                    Err(Malformed("pixel inputs read a row two ways"))
                ),
                "{a:?} {b:?}"
            );
        }
    }

    #[test]
    fn refuses_outputs_sharing_a_row() {
        // Two outputs: param0 in row 0, param1 (or per-primitive param1) in
        // `row`.
        for stage in [HwStage::Vs, HwStage::Ms] {
            let base = fssm(&[(stage, &VS_CODE)]);
            let e = stage_data_at(&base, stage);
            let outputs = |param: u8, row: u8| {
                let mut blob = base.clone();
                blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
                blob[e + VS_OUTPUTS..][..12]
                    .copy_from_slice(&[6, 0, 0, 0, 0, 0, 6, row, param, row, 0, 0]);
                native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
            };
            assert!(outputs(1, 1).is_ok(), "{stage:?}");
            // A per-primitive output's row is in the primitive signature, so
            // row 0 there is not the vertex output's row 0. Only a mesh
            // program has that signature.
            if stage == HwStage::Ms {
                assert!(outputs(0x81, 0).is_ok());
            } else {
                for (param, row) in [(0x81, 0), (0x81, 1), (0xFF, 7)] {
                    assert!(
                        matches!(
                            outputs(param, row),
                            Err(Malformed("vertex output marked per-primitive"))
                        ),
                        "{param:#x} in row {row}"
                    );
                }
            }
            assert!(
                matches!(outputs(1, 0), Err(Malformed("vertex outputs share a row"))),
                "{stage:?}"
            );
        }
        // Two per-primitive outputs in one row clash too.
        let base = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&base, HwStage::Ms);
        let mut blob = base.clone();
        blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + VS_OUTPUTS..][..12].copy_from_slice(&[6, 2, 0x80, 2, 0, 0, 6, 2, 0x81, 2, 0, 0]);
        assert!(matches!(
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])),
            Err(Malformed("vertex outputs share a row"))
        ));
    }

    /// A pixel program with these (root type, SGPRs, user data, slot)
    /// parameters.
    fn program(params: &[(u32, u32, u8, u8)]) -> Result<Vec<NativeProgram>, ShaderError> {
        stage_layout(HwStage::Ps, params)
    }

    #[test]
    fn refuses_a_root_parameter_of_unknown_kind() {
        // Constants, descriptor, table, fixed sampler, resource heap and
        // sampler heap are the kinds there are; 2 is not one of them.
        for (root, sgprs) in [(0, 1), (1, 2), (3, 1), (4, 0), (5, 1), (6, 1)] {
            assert!(program(&[(root, sgprs, 4, 2)]).is_ok(), "kind {root}");
        }
        // Whether or not it states SGPRs, and next to a known parameter;
        // `u32::MAX` would overflow the shader cache's code for it.
        for params in [
            &[(2, 1, 4, 2)][..],
            &[(7, 0, 4, 2)],
            &[(u32::MAX, 1, 4, 2)],
            &[(0, 2, 0, 0), (2, 1, 4, 1)],
        ] {
            assert_eq!(
                why_refused(program(params)),
                "root parameter of unknown kind",
                "{params:?}"
            );
        }
    }

    #[test]
    fn refuses_root_parameters_sharing_an_sgpr() {
        let (constants, descriptor, table, sampler) = (0, 1, 3, 4);
        // Side by side, two ranges of one table, and a fixed sampler (no
        // SGPRs) at a taken user-data dword are all fine.
        assert!(program(&[(constants, 2, 0, 0), (descriptor, 2, 2, 1)]).is_ok());
        assert!(program(&[(table, 1, 4, 2), (table, 1, 4, 2)]).is_ok());
        assert!(program(&[(table, 1, 4, 2), (table, 1, 4, 2), (table, 1, 4, 2)]).is_ok());
        assert!(program(&[(constants, 2, 0, 0), (sampler, 0, 0, 3)]).is_ok());
        for (params, what) in [
            (
                [(constants, 2, 0, 0), (table, 1, 1, 1)],
                "a table in the constants' second dword",
            ),
            (
                [(table, 1, 3, 0), (table, 1, 3, 1)],
                "two tables in one SGPR",
            ),
            (
                [(descriptor, 1, 0, 0), (constants, 1, 1, 1)],
                "constants in a descriptor's high half",
            ),
            (
                [(table, 1, 3, 2), (constants, 1, 3, 2)],
                "constants in a table's SGPR stating its slot",
            ),
            (
                [(constants, 2, 0, 0), (constants, 2, 1, 0)],
                "one slot's constants overlapping themselves",
            ),
            (
                [(constants, 1, 5, 4), (constants, 1, 5, 4)],
                "constants listed twice",
            ),
            (
                [(descriptor, 2, 0, 1), (descriptor, 2, 0, 1)],
                "a root descriptor listed twice",
            ),
        ] {
            assert_eq!(
                match program(&params) {
                    Err(Malformed(why)) => why,
                    other => panic!("{what}: expected a refusal, got {other:?}"),
                },
                "root parameters share an SGPR",
                "{what}"
            );
        }

        // A table or heap is one 32-bit address: it may state one SGPR (or
        // none, as a further range of a table), never two.
        let (resource_heap, sampler_heap) = (5, 6);
        for root in [table, resource_heap, sampler_heap] {
            for sgprs in [0, 1] {
                assert!(program(&[(root, sgprs, 4, 2)]).is_ok(), "{root}/{sgprs}");
            }
            for sgprs in [2, 8] {
                assert_eq!(
                    match program(&[(root, sgprs, 4, 2)]) {
                        Err(Malformed(why)) => why,
                        other => panic!("{root}/{sgprs}: expected a refusal, got {other:?}"),
                    },
                    "root address wider than one SGPR",
                    "{root}/{sgprs}"
                );
            }
        }
        // Constants and a root descriptor still take what they state.
        assert!(program(&[(constants, 8, 0, 0), (descriptor, 2, 8, 1)]).is_ok());
    }

    #[test]
    fn heaps_hold_one_sgpr_of_their_own() {
        let (constants, table, resource_heap, sampler_heap) = (0, 3, 5, 6);
        // The two heaps side by side, beside constants, and at the last
        // user-data dword are fine.
        assert!(program(&[(resource_heap, 1, 4, 0), (sampler_heap, 1, 5, 1)]).is_ok());
        assert!(program(&[(constants, 4, 0, 0), (resource_heap, 1, 4, 1)]).is_ok());
        for root in [resource_heap, sampler_heap] {
            assert!(program(&[(root, 1, 31, 0)]).is_ok(), "{root} at dword 31");
            assert_eq!(
                why_refused(program(&[(root, 1, 32, 0)])),
                "root parameter past the user data",
                "{root} at dword 32"
            );
        }
        // Only a table's ranges may share an SGPR: two heaps, a heap in the
        // constants' last dword, or a heap and a table of the same slot
        // would each have one read as the other.
        for (params, what) in [
            (
                [(resource_heap, 1, 4, 0), (sampler_heap, 1, 4, 1)],
                "both heaps in s4",
            ),
            (
                [(resource_heap, 1, 4, 0), (resource_heap, 1, 4, 0)],
                "the resource heap listed twice",
            ),
            (
                [(constants, 4, 0, 0), (sampler_heap, 1, 3, 1)],
                "the sampler heap in the constants' last dword",
            ),
            (
                [(table, 1, 4, 2), (resource_heap, 1, 4, 2)],
                "a heap stating a table's slot and SGPR",
            ),
        ] {
            assert_eq!(
                why_refused(program(&params)),
                "root parameters share an SGPR",
                "{what}"
            );
        }
        // In compute, with only the group info enabled (s0), a heap there
        // is refused even stating no SGPRs: its address is read from the
        // SGPR its byte names. Right after it, s1 is free.
        let info_in_s0: u32 = 1 << 10;
        for root in [resource_heap, sampler_heap] {
            for sgprs in [0, 1] {
                assert_eq!(
                    why_refused(compute_program(info_in_s0, root, sgprs, 0)),
                    "root parameter over a system SGPR",
                    "{root}/{sgprs}"
                );
                assert!(
                    compute_program(info_in_s0, root, sgprs, 1).is_ok(),
                    "{root}/{sgprs} at s1"
                );
            }
            // Compute's user data ends at dword 15 for a heap too.
            assert!(compute_program(0, root, 1, 15).is_ok(), "{root}");
            assert_eq!(
                why_refused(compute_program(0, root, 1, 16)),
                "root parameter past the user data",
                "{root}"
            );
        }
    }

    #[test]
    fn user_data_follows_each_stage_system_sgprs() {
        // Merged stages and mesh shaders have 8 system SGPRs before their
        // user data; the legacy stages and compute start it at s0.
        for stage in [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Ms,
        ] {
            assert_eq!(
                (stage.user_sgpr(0), stage.user_sgpr(31)),
                (8, 39),
                "{stage:?}"
            );
        }
        for stage in [HwStage::Vs, HwStage::Ps, HwStage::Cs] {
            assert_eq!(
                (stage.user_sgpr(0), stage.user_sgpr(31)),
                (0, 31),
                "{stage:?}"
            );
        }
    }

    #[test]
    fn merged_stage_root_params_count_from_s8() {
        // One (root type, SGPRs, user data) parameter in a `stage` program.
        let program = |stage: HwStage, root: u32, sgprs: u32, user_data: u8| {
            let mut blob = fssm(&[(stage, &VS_CODE)]);
            let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
            let table = (blob.len() - FSSM_HEADER) as u64;
            for v in [root, 0, sgprs, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[user_data, 0, 0, 0]);
            let payload = (blob.len() - FSSM_HEADER) as u64;
            blob[8..16].copy_from_slice(&payload.to_le_bytes());
            blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
            blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let (constants, descriptor) = (0, 1);
        for stage in [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Ms,
        ] {
            // All 32 user-data dwords are s8-s39, and a descriptor's high
            // half may be the last of them.
            let p = program(stage, constants, 32, 0).unwrap();
            assert_eq!((p[0].stage, p[0].layout[0].sgprs), (stage, 32));
            assert!(program(stage, descriptor, 2, 30).is_ok(), "{stage:?}");
            // Dword 32 is past the user data while its SGPR (s40) exists;
            // dword 97 is s105, the last SGPR, and dword 98 would be s106,
            // which no program has.
            for (root, sgprs, user_data) in
                [(constants, 1, 32), (descriptor, 1, 31), (constants, 1, 97)]
            {
                assert_eq!(
                    why_refused(program(stage, root, sgprs, user_data)),
                    "root parameter past the user data",
                    "{stage:?}: {root}/{sgprs} at dword {user_data}"
                );
            }
            for (root, sgprs, user_data) in [
                (constants, 1, 98),
                (descriptor, 2, 97),
                (constants, 1, 0xFF),
            ] {
                assert_eq!(
                    why_refused(program(stage, root, sgprs, user_data)),
                    "root parameter outside the SGPRs",
                    "{stage:?}: {root}/{sgprs} at dword {user_data}"
                );
            }
        }
        // The same dword 98 is still an SGPR (s98) in a vertex program, so
        // it is refused only as past the user data there; s106 is the first
        // dword that program can't address.
        assert_eq!(
            why_refused(program(HwStage::Vs, constants, 1, 98)),
            "root parameter past the user data"
        );
        assert_eq!(
            why_refused(program(HwStage::Vs, constants, 1, 106)),
            "root parameter outside the SGPRs"
        );
    }

    /// A `stage` program with these (root type, SGPRs, user data, slot)
    /// parameters.
    fn stage_layout(
        stage: HwStage,
        params: &[(u32, u32, u8, u8)],
    ) -> Result<Vec<NativeProgram>, ShaderError> {
        let code: &[u8] = if stage == HwStage::Ps {
            &PS_CODE
        } else {
            &VS_CODE
        };
        let mut blob = fssm(&[(stage, code)]);
        set_stage_layout(&mut blob, stage, params);
        native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
    }

    fn set_stage_layout(blob: &mut Vec<u8>, stage: HwStage, params: &[(u32, u32, u8, u8)]) {
        let record = FSSM_HEADER + u64_at(blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
        let table = (blob.len() - FSSM_HEADER) as u64;
        for &(root, sgprs, user_data, slot) in params {
            for v in [root, 0, sgprs, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[user_data, slot, 0, 0]);
        }
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
        let count = params.len() as u32;
        blob[record + RECORD_COUNTS..][..4].copy_from_slice(&count.to_le_bytes());
    }

    #[test]
    fn an_entry_stating_no_sgprs_still_takes_its_first_unless_a_fixed_sampler() {
        let (constants, table, sampler, resource_heap, sampler_heap) = (0, 3, 4, 5, 6);
        // A table, heap or constants entry stating no SGPRs names its first
        // one all the same: lowering reads a table's address from it and
        // declares the user data up to it, so it must be user data and its
        // own. In a pixel program dword 31 is the last of the 32.
        for root in [constants, table, resource_heap, sampler_heap] {
            assert!(program(&[(root, 0, 31, 1)]).is_ok(), "kind {root}");
            for (params, why) in [
                (&[(root, 0, 32, 1)][..], "root parameter past the user data"),
                (
                    &[(constants, 1, 4, 0), (root, 0, 4, 1)],
                    "root parameters share an SGPR",
                ),
                (
                    &[(root, 0, 4, 1), (constants, 1, 4, 0)],
                    "root parameters share an SGPR",
                ),
            ] {
                assert_eq!(why_refused(program(params)), why, "kind {root}: {params:?}");
            }
            // Next to it, the following SGPR is free.
            assert!(program(&[(root, 0, 4, 1), (constants, 1, 5, 0)]).is_ok());
        }
        // A fixed sampler stating none takes none, so it may sit past the
        // user data (in the SGPRs) and share an SGPR with anything.
        assert!(program(&[(sampler, 0, 32, 1)]).is_ok());
        assert!(program(&[(constants, 1, 4, 0), (sampler, 0, 4, 1)]).is_ok());
        // Ranges of one table stating none share its address SGPR, as ranges
        // stating one do; another table there does not.
        assert!(program(&[(table, 0, 4, 1), (table, 1, 4, 1), (table, 0, 4, 1)]).is_ok());
        assert_eq!(
            why_refused(program(&[(table, 0, 4, 1), (table, 0, 4, 2)])),
            "root parameters share an SGPR"
        );
        // Compute has 16 user-data dwords and runs the same rule.
        let cs = |params: &[(u32, u32, u8, u8)]| stage_layout(HwStage::Cs, params);
        assert!(cs(&[(table, 0, 15, 1)]).is_ok());
        assert_eq!(
            why_refused(cs(&[(table, 0, 16, 1)])),
            "root parameter past the user data"
        );
    }

    #[test]
    fn root_layouts_are_local_to_each_stage_record() {
        let (cs, ps) = (HwStage::Cs, HwStage::Ps);
        let (descriptor, table) = (1, 3);
        let parse = |cs_start, ps_start| {
            let mut blob = fssm(&[(cs, &VS_CODE), (ps, &PS_CODE)]);
            set_stage_layout(&mut blob, cs, &[(descriptor, 1, cs_start, 0)]);
            set_stage_layout(
                &mut blob,
                ps,
                &[(table, 1, 15, 1), (descriptor, 1, ps_start, 0)],
            );
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };

        // The compute descriptor uses s14 and s15. Pixel can reuse s15 for
        // its table and put its own descriptor at the end of its 32 dwords.
        let programs = parse(14, 30).unwrap();
        let [pixel, compute] = programs.as_slice() else {
            panic!("expected pixel and compute records");
        };
        assert_eq!((pixel.stage, pixel.layout.len()), (ps, 2));
        assert_eq!((compute.stage, compute.layout.len()), (cs, 1));
        assert_eq!(pixel.layout[1].user_data, 30);
        assert_eq!(compute.layout[0].user_data, 14);

        assert_eq!(
            why_refused(parse(15, 30)),
            "root parameter past the user data"
        );
        assert_eq!(
            why_refused(parse(14, 31)),
            "root parameter past the user data"
        );
        assert_eq!(why_refused(parse(14, 14)), "root parameters share an SGPR");
    }

    #[test]
    fn root_sgpr_span_keeps_the_full_width_of_the_count() {
        // At dword 1 this count would wrap its exclusive end to zero in u32.
        // In a merged stage the eight system SGPRs also contribute to the end.
        for stage in [HwStage::Vs, HwStage::Gs] {
            assert_eq!(
                why_refused(stage_layout(stage, &[(0, u32::MAX, 1, 0)])),
                "root parameter outside the SGPRs",
                "{stage:?}"
            );
        }
    }

    #[test]
    fn merged_stage_root_params_share_sgprs_as_a_legacy_stage_does() {
        let (constants, descriptor, table, sampler) = (0, 1, 3, 4);
        // Sharing is judged in user-data dwords, which every stage moves by
        // the same system SGPRs: what a pixel program allows or refuses, a
        // merged stage or mesh shader (from s8) does too.
        for stage in [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Ms,
            HwStage::Vs,
            HwStage::Ps,
        ] {
            for params in [
                &[(constants, 2, 0, 0), (descriptor, 2, 2, 1)][..],
                &[(table, 1, 4, 2), (table, 1, 4, 2)],
                &[(constants, 2, 0, 0), (sampler, 0, 0, 3)],
                // Constants in the last two dwords, a descriptor before them.
                &[(descriptor, 2, 28, 0), (constants, 2, 30, 1)],
            ] {
                let p = stage_layout(stage, params)
                    .unwrap_or_else(|e| panic!("{stage:?}: {params:?}: {e:?}"));
                // The layout keeps the dword, not the SGPR it lands in.
                let dwords: Vec<u8> = p[0].layout.iter().map(|r| r.user_data).collect();
                let stated: Vec<u8> = params.iter().map(|&(_, _, d, _)| d).collect();
                assert_eq!(dwords, stated, "{stage:?}: {params:?}");
            }
            for params in [
                &[(constants, 2, 0, 0), (table, 1, 1, 1)][..],
                &[(table, 1, 3, 0), (table, 1, 3, 1)],
                &[(descriptor, 1, 0, 0), (constants, 1, 1, 1)],
                &[(descriptor, 2, 30, 0), (constants, 1, 31, 1)],
                &[(constants, 1, 31, 0), (constants, 1, 31, 0)],
            ] {
                assert_eq!(
                    why_refused(stage_layout(stage, params)),
                    "root parameters share an SGPR",
                    "{stage:?}: {params:?}"
                );
            }
        }
    }

    #[test]
    fn a_merged_stage_fixed_sampler_may_name_the_end_of_the_sgprs() {
        // One fixed sampler stating no SGPRs at user-data dword `user_data`
        // in a `stage` program.
        let program = |stage: HwStage, user_data: u8| {
            let mut blob = fssm(&[(stage, &VS_CODE)]);
            let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
            let table = (blob.len() - FSSM_HEADER) as u64;
            for v in [4u32, 0, 0, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[user_data, 0, 0, 0]);
            let payload = (blob.len() - FSSM_HEADER) as u64;
            blob[8..16].copy_from_slice(&payload.to_le_bytes());
            blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
            blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // From s8, dword 98 names s106, just past the last SGPR, which a
        // sampler taking none may; dword 99 (s107) and the byte's top
        // would only grow `user_sgpr` toward its overflow, so are refused.
        for stage in [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Ms,
        ] {
            let p = program(stage, 98).unwrap();
            assert_eq!(
                (p[0].layout[0].root, p[0].layout[0].user_data),
                (RootType::ImmediateSampler, 98),
                "{stage:?}"
            );
            for user_data in [99, 0xFF] {
                assert_eq!(
                    why_refused(program(stage, user_data)),
                    "root parameter outside the SGPRs",
                    "{stage:?}: dword {user_data}"
                );
            }
        }
        // A vertex program counts from s0, so its end is dword 106.
        assert!(program(HwStage::Vs, 106).is_ok());
        assert_eq!(
            why_refused(program(HwStage::Vs, 107)),
            "root parameter outside the SGPRs"
        );
    }

    #[test]
    fn a_layout_breaking_several_rules_is_named_for_the_first_checked() {
        let (constants, table, unknown) = (0, 3, 2);
        // Each layout below also breaks every rule named after it: the
        // checks run as kind, SGPRs, user data, address width, then sharing.
        for (params, why) in [
            (
                &[(unknown, 2, 0xFF, 0)][..],
                "root parameter of unknown kind",
            ),
            (&[(table, 2, 0xFF, 0)], "root parameter outside the SGPRs"),
            (&[(table, 2, 40, 0)], "root parameter past the user data"),
            (
                &[(constants, 2, 0, 0), (table, 2, 1, 1)],
                "root address wider than one SGPR",
            ),
            (
                &[(constants, 2, 0, 0), (table, 1, 1, 1)],
                "root parameters share an SGPR",
            ),
        ] {
            assert_eq!(why_refused(program(params)), why, "{params:?}");
        }
        // Wherever the breaking parameter sits in the list.
        assert_eq!(
            why_refused(program(&[(table, 2, 40, 1), (unknown, 1, 4, 2)])),
            "root parameter of unknown kind"
        );
        assert_eq!(
            why_refused(program(&[(table, 2, 4, 1), (constants, 1, 0xFF, 2)])),
            "root parameter outside the SGPRs"
        );
    }

    /// A pixel program with a table in s4 (slot 2), a root descriptor in
    /// s6-s7 (slot 3), and these (kind, root type, table offset) reads.
    fn reading(reads: &[(u8, u8, u32)]) -> Result<Vec<NativeProgram>, ShaderError> {
        let named = reads
            .iter()
            .map(|&(kind, root, offset)| {
                let (user_data, slot) = if root == 3 { (4, 2) } else { (6, 3) };
                (kind, root, offset, user_data, slot)
            })
            .collect::<Vec<_>>();
        reading_named(&named)
    }

    /// A read's (kind, root type, table offset, user data, slot).
    type Read = (u8, u8, u32, u8, u8);

    /// As [`reading`], with each read's (kind, root type, table offset,
    /// user data, slot) stated in full.
    fn reading_named(reads: &[Read]) -> Result<Vec<NativeProgram>, ShaderError> {
        reading_with(&[(3, 1, 4, 2), (1, 2, 6, 3)], reads)
    }

    /// As [`reading_named`], over these (root type, SGPRs, user data, slot)
    /// parameters instead of a table in s4 and a root descriptor in s6.
    fn reading_with(
        params: &[(u32, u32, u8, u8)],
        reads: &[Read],
    ) -> Result<Vec<NativeProgram>, ShaderError> {
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
        set_stage_reads(&mut blob, HwStage::Ps, params, reads);
        native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
    }

    /// Sets `stage`'s record in `blob` to these (root type, SGPRs, user
    /// data, slot) parameters and these reads, appended to the payload.
    fn set_stage_reads(
        blob: &mut Vec<u8>,
        stage: HwStage,
        params: &[(u32, u32, u8, u8)],
        reads: &[Read],
    ) {
        let record = FSSM_HEADER + u64_at(blob, FSSM_HEADER + slot(stage)).unwrap() as usize;
        let tables = (blob.len() - FSSM_HEADER) as u64;
        for &(root, sgprs, user_data, slot) in params {
            for v in [root, 0, sgprs, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[user_data, slot, 0, 0]);
        }
        for &(kind, root, offset, user_data, slot) in reads {
            blob.extend_from_slice(&[kind, root, 0, 0]);
            for v in [offset, 0, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[8, user_data, slot, 0]);
        }
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&tables.to_le_bytes());
        let used = tables + params.len() as u64 * ENTRY_SIZE;
        blob[record + RECORD_USED..][..8].copy_from_slice(&used.to_le_bytes());
        blob[record + RECORD_COUNTS..][..4].copy_from_slice(&(params.len() as u32).to_le_bytes());
        blob[record + RECORD_COUNTS + 4..][..4]
            .copy_from_slice(&(reads.len() as u32).to_le_bytes());
    }

    #[test]
    fn reads_name_only_their_own_records_parameters() {
        let (srv, sampler, uav) = (0u8, 1u8, 3u8);
        let (descriptor, table) = (1u8, 3u8);
        let (vs, ps) = (HwStage::Vs, HwStage::Ps);
        // A vertex and a pixel program, each with these parameters and reads.
        type Params<'a> = &'a [(u32, u32, u8, u8)];
        let parse = |vs_side: (Params, &[Read]), ps_side: (Params, &[Read])| {
            let mut blob = fssm(&[(vs, &VS_CODE), (ps, &PS_CODE)]);
            set_stage_reads(&mut blob, vs, vs_side.0, vs_side.1);
            set_stage_reads(&mut blob, ps, ps_side.0, ps_side.1);
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // A table in s4 (slot 2) and a root descriptor in s6-s7 (slot 3).
        let both = &[(3, 1, 4, 2), (1, 2, 6, 3)][..];
        let srv_in_table = &[(srv, table, 0, 4, 2)][..];
        let srv_in_descriptor = &[(srv, descriptor, 0, 6, 3)][..];

        // Each program reading its own parameters is fine, in either one.
        let p = parse((both, srv_in_table), (both, srv_in_descriptor)).unwrap();
        assert_eq!((p[0].stage, p[0].used.len()), (vs, 1));
        assert_eq!((p[1].stage, p[1].used.len()), (ps, 1));

        // A read naming a parameter only the other program has names none:
        // the host binds a stage's reads from that stage's root parameters.
        for (vs_side, ps_side, why) in [
            (
                (both, &[][..]),
                (&[][..], srv_in_table),
                "used descriptor names no root table",
            ),
            (
                (&[][..], srv_in_table),
                (both, &[][..]),
                "used descriptor names no root table",
            ),
            (
                (both, &[][..]),
                (&[][..], srv_in_descriptor),
                "used descriptor names no root descriptor",
            ),
            (
                (&[][..], srv_in_descriptor),
                (both, &[][..]),
                "used descriptor names no root descriptor",
            ),
        ] {
            assert_eq!(
                why_refused(parse(vs_side, ps_side)),
                why,
                "{vs_side:?} {ps_side:?}"
            );
        }

        // One table's descriptor read as an SRV by one program and a UAV by
        // the other, or its table holding samplers in one and views in the
        // other, breaks neither rule: each program binds its own tables.
        let as_uav = &[(uav, table, 0, 4, 2)][..];
        let as_sampler = &[(sampler, table, 0, 4, 2)][..];
        assert!(parse((both, srv_in_table), (both, as_uav)).is_ok());
        assert!(parse((both, as_sampler), (both, srv_in_table)).is_ok());
        // Within one program they are still refused.
        let mut two_ways = srv_in_table.to_vec();
        two_ways.extend_from_slice(as_uav);
        assert_eq!(
            why_refused(parse((both, &two_ways), (both, srv_in_table))),
            "used descriptors read one descriptor two ways"
        );
        let mut mixed = srv_in_table.to_vec();
        mixed.push((sampler, table, 8, 4, 2));
        assert_eq!(
            why_refused(parse((both, srv_in_table), (both, &mixed))),
            "root table mixes samplers and views"
        );
    }
    #[test]
    fn refuses_a_descriptor_read_two_ways() {
        let (srv, cbv, uav) = (0u8, 2u8, 3u8);
        let (descriptor, table) = (1u8, 3u8);
        // Distinct table entries, the same entry listed twice as one kind,
        // and a root descriptor read twice as one kind are fine.
        assert!(reading(&[(srv, table, 0), (uav, table, 8), (cbv, table, 16)]).is_ok());
        assert!(reading(&[(srv, table, 8), (srv, table, 8)]).is_ok());
        assert!(reading(&[(cbv, descriptor, 0), (cbv, descriptor, 0)]).is_ok());
        for (reads, what) in [
            (
                [(srv, table, 8), (uav, table, 8)],
                "an SRV and a UAV at one offset",
            ),
            (
                [(cbv, table, 0), (srv, table, 0)],
                "a CBV and an SRV at one offset",
            ),
            (
                [(cbv, descriptor, 0), (srv, descriptor, 0)],
                "a root descriptor as a CBV and an SRV",
            ),
            (
                [(srv, descriptor, 0), (uav, descriptor, 4)],
                "a root descriptor at two offsets",
            ),
        ] {
            assert_eq!(
                why_refused(reading(&reads)),
                "used descriptors read one descriptor two ways",
                "{what}"
            );
        }
        // A table and a root descriptor are different descriptors even at
        // the same stated offset.
        assert!(reading(&[(srv, table, 0), (cbv, descriptor, 0)]).is_ok());
    }

    #[test]
    fn reads_name_a_parameter_by_its_first_sgpr_and_root_descriptors_stay_apart() {
        let (srv, cbv, uav) = (0u8, 2u8, 3u8);
        let (descriptor, table) = (1u8, 3u8);
        // A table in s4 (slot 2), a root descriptor in s6-s7 (slot 3) and a
        // root descriptor stating four SGPRs in s10-s13 (slot 5).
        let params = [(3, 1, 4, 2), (1, 2, 6, 3), (1, 4, 10, 5)];
        assert!(reading_with(&params, &[(srv, descriptor, 0, 6, 3)]).is_ok());
        assert!(reading_with(&params, &[(srv, descriptor, 0, 10, 5)]).is_ok());
        // A read is matched to its parameter's first SGPR alone: the SGPRs
        // after it (a root descriptor's high half, or the rest of a wider
        // count) hold no parameter a read can name, nor does a parameter's
        // first SGPR stated with another's slot.
        for (read, refusal) in [
            (
                (srv, descriptor, 0, 7, 3),
                "used descriptor names no root descriptor",
            ),
            (
                (srv, descriptor, 0, 12, 5),
                "used descriptor names no root descriptor",
            ),
            (
                (srv, descriptor, 0, 13, 5),
                "used descriptor names no root descriptor",
            ),
            (
                (srv, descriptor, 0, 10, 3),
                "used descriptor names no root descriptor",
            ),
            (
                (srv, descriptor, 0, 6, 5),
                "used descriptor names no root descriptor",
            ),
            ((srv, table, 0, 7, 3), "used descriptor names no root table"),
            // A root descriptor's own SGPR and slot are no table either.
            ((srv, table, 0, 6, 3), "used descriptor names no root table"),
            // Nor is a table's a root descriptor.
            (
                (cbv, descriptor, 0, 4, 2),
                "used descriptor names no root descriptor",
            ),
        ] {
            assert_eq!(
                why_refused(reading_with(&params, &[read])),
                refusal,
                "{read:?}"
            );
        }
        // Each root descriptor is one view of its own: two may be read as
        // different kinds, and one read twice as one kind at any stated
        // offset is still that view.
        assert!(
            reading_with(
                &params,
                &[(cbv, descriptor, 0, 6, 3), (uav, descriptor, 0, 10, 5)]
            )
            .is_ok()
        );
        assert!(
            reading_with(
                &params,
                &[(srv, descriptor, 0, 6, 3), (srv, descriptor, 4, 6, 3)]
            )
            .is_ok()
        );
        // One of them read two ways is refused even with the other read
        // between, and with the second read at another stated offset.
        assert_eq!(
            why_refused(reading_with(
                &params,
                &[
                    (srv, descriptor, 0, 6, 3),
                    (uav, descriptor, 0, 10, 5),
                    (cbv, descriptor, 8, 6, 3),
                ]
            )),
            "used descriptors read one descriptor two ways"
        );
        // The parsed reads keep each root descriptor's SGPR and slot.
        let p = &reading_with(
            &params,
            &[(cbv, descriptor, 0, 6, 3), (uav, descriptor, 0, 10, 5)],
        )
        .unwrap()[0];
        assert_eq!(
            p.used
                .iter()
                .map(|u| (u.kind, u.root, u.user_data, u.slot))
                .collect::<Vec<_>>(),
            [
                (ResourceType::Cbv, RootType::Descriptor, 6, 3),
                (ResourceType::Uav, RootType::Descriptor, 10, 5),
            ]
        );
    }

    #[test]
    fn refuses_a_table_mixing_samplers_and_views() {
        let (srv, sampler, cbv) = (0u8, 1u8, 2u8);
        let table = 3u8;
        assert!(reading(&[(sampler, table, 0), (sampler, table, 4)]).is_ok());
        assert!(reading(&[(srv, table, 0), (cbv, table, 8)]).is_ok());
        for reads in [
            [(sampler, table, 0), (srv, table, 8)],
            [(cbv, table, 0), (sampler, table, 8)],
        ] {
            assert_eq!(
                why_refused(reading(&reads)),
                "root table mixes samplers and views",
                "{reads:?}"
            );
        }
    }

    #[test]
    fn separate_root_tables_are_separate_descriptors_and_heaps() {
        let (srv, sampler, uav) = (0u8, 1u8, 3u8);
        let table = 3u8;
        // Tables in s4 (slot 2) and s5 (slot 3): one may hold samplers and
        // the other views, and the same offset in each may be read as a
        // different kind, as they are different descriptors.
        let two = [(3, 1, 4, 2), (3, 1, 5, 3)];
        assert!(reading_with(&two, &[(sampler, table, 0, 4, 2), (srv, table, 0, 5, 3)]).is_ok());
        assert!(reading_with(&two, &[(srv, table, 8, 4, 2), (uav, table, 8, 5, 3)]).is_ok());
        // Two tables stating one slot from different SGPRs are still two:
        // each read names its table by both.
        let one_slot = [(3, 1, 4, 2), (3, 1, 5, 2)];
        assert!(
            reading_with(
                &one_slot,
                &[(sampler, table, 0, 4, 2), (srv, table, 0, 5, 2)]
            )
            .is_ok()
        );
        // Two ranges of one table are one table: they share its heap and
        // its descriptors.
        let ranges = [(3, 1, 4, 2), (3, 1, 4, 2)];
        assert_eq!(
            why_refused(reading_with(
                &ranges,
                &[(sampler, table, 0, 4, 2), (srv, table, 8, 4, 2)]
            )),
            "root table mixes samplers and views"
        );
        assert_eq!(
            why_refused(reading_with(
                &ranges,
                &[(srv, table, 8, 4, 2), (uav, table, 8, 4, 2)]
            )),
            "used descriptors read one descriptor two ways"
        );
        // A read naming one table's SGPR with the other's slot names neither.
        assert_eq!(
            why_refused(reading_with(&two, &[(srv, table, 0, 4, 3)])),
            "used descriptor names no root table"
        );
    }

    #[test]
    fn heap_and_fixed_sampler_reads_keep_none_of_the_table_rules() {
        let (srv, sampler, cbv, uav) = (0u8, 1u8, 2u8, 3u8);
        let (table, fixed, resource_heap, sampler_heap) = (3u8, 4u8, 5u8, 6u8);
        // A table in s4 (slot 2), a root descriptor in s6-s7 (slot 3), the
        // resource heap in s8 (slot 4) and the sampler heap in s9 (slot 5).
        let params = [(3, 1, 4, 2), (1, 2, 6, 3), (5, 1, 8, 4), (6, 1, 9, 5)];
        // A heap or fixed-sampler read is neither found by its SGPR nor bound
        // by its slot, and names no descriptor of a table: each read below
        // is kept as stated, even where the same read of the table is
        // refused for the reason beside it.
        for (kind, root, offset, user_data, slot) in [
            (srv, resource_heap, 0, 8, 4),
            (sampler, sampler_heap, 0, 9, 5),
            (sampler, fixed, 0, 30, 7),
        ] {
            let p = &reading_with(&params, &[(kind, root, offset, user_data, slot)]).unwrap()[0];
            assert_eq!(
                (p.used[0].kind, p.used[0].table_offset_dwords),
                (
                    [
                        ResourceType::Srv,
                        ResourceType::Sampler,
                        ResourceType::Cbv,
                        ResourceType::Uav
                    ][usize::from(kind)],
                    offset
                )
            );
            assert_eq!((p.used[0].user_data, p.used[0].slot), (user_data, slot));
        }
        for (reads, table_refusal) in [
            (vec![(srv, 20, 9)], "used descriptor names no root table"),
            (
                vec![(srv, 8, 4), (uav, 8, 4)],
                "used descriptors read one descriptor two ways",
            ),
            (
                vec![(sampler, 0, 4), (cbv, 8, 4)],
                "root table mixes samplers and views",
            ),
            (
                vec![(cbv, MAX_TABLE_DWORDS, 4)],
                "used descriptor past any heap",
            ),
            (vec![(cbv, u32::MAX, 4)], "used descriptor past any heap"),
        ] {
            // Through each heap (its own SGPR and slot, or another's).
            for root in [resource_heap, sampler_heap, fixed] {
                let heap_reads = reads
                    .iter()
                    .map(|&(kind, offset, user_data)| {
                        (kind, root, offset, user_data, user_data / 2)
                    })
                    .collect::<Vec<_>>();
                assert!(
                    reading_with(&params, &heap_reads).is_ok(),
                    "root {root}: {heap_reads:?}"
                );
            }
            // Through the table in s4 (slot 2) unless the read names another.
            let table_reads = reads
                .iter()
                .map(|&(kind, offset, user_data)| {
                    let slot = if user_data == 4 { 2 } else { user_data / 2 };
                    (kind, table, offset, user_data, slot)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                why_refused(reading_with(&params, &table_reads)),
                table_refusal,
                "{table_reads:?}"
            );
        }
    }

    #[test]
    fn constants_reads_keep_none_of_the_table_or_descriptor_rules() {
        let (srv, sampler, cbv, uav) = (0u8, 1u8, 2u8, 3u8);
        let (constants, descriptor, table) = (0u8, 1u8, 3u8);
        // A table in s4 (slot 2), a root descriptor in s6-s7 (slot 3) and
        // two constants in s8-s9 (slot 4).
        let params = [(3, 1, 4, 2), (1, 2, 6, 3), (0, 2, 8, 4)];
        // Lowering classifies only table and root-descriptor reads, and the
        // host binds only table reads, so a read through constants names no
        // parameter and no descriptor: it is kept as stated, at any offset
        // and whatever SGPR and slot it states, its own or none.
        for (kind, offset, user_data, slot) in [
            (cbv, 0, 8, 4),
            (srv, u32::MAX, 8, 4),
            (sampler, MAX_TABLE_DWORDS, 30, 7),
            (uav, 4, 4, 2),
        ] {
            let p =
                &reading_with(&params, &[(kind, constants, offset, user_data, slot)]).unwrap()[0];
            assert_eq!(p.used[0].root, RootType::Constants);
            assert_eq!(
                (
                    p.used[0].table_offset_dwords,
                    p.used[0].user_data,
                    p.used[0].slot
                ),
                (offset, user_data, slot)
            );
        }
        // Beside the table's and the root descriptor's own reads, naming
        // their SGPR and slot with another kind, it is neither a second way
        // of reading their descriptor nor a sampler in the view table.
        let beside = [
            (cbv, table, 4, 4, 2),
            (sampler, constants, 4, 4, 2),
            (srv, descriptor, 0, 6, 3),
            (uav, constants, 0, 6, 3),
        ];
        assert!(reading_with(&params, &beside).is_ok());
        // The same reads through the table or the root descriptor are
        // refused.
        for (reads, refusal) in [
            (
                [(cbv, table, 4, 4, 2), (sampler, table, 4, 4, 2)],
                "used descriptors read one descriptor two ways",
            ),
            (
                [(cbv, table, 4, 4, 2), (sampler, table, 8, 4, 2)],
                "root table mixes samplers and views",
            ),
            (
                [(srv, descriptor, 0, 6, 3), (uav, descriptor, 0, 6, 3)],
                "used descriptors read one descriptor two ways",
            ),
        ] {
            assert_eq!(
                why_refused(reading_with(&params, &reads)),
                refusal,
                "{reads:?}"
            );
        }
    }

    #[test]
    fn refuses_a_descriptor_read_of_unknown_kind() {
        let (descriptor, table) = (1u8, 3u8);
        // SRV, sampler, CBV and UAV are the kinds there are.
        for kind in 0..4u8 {
            assert!(reading(&[(kind, table, 0)]).is_ok(), "kind {kind}");
        }
        for (kind, root) in [(4u8, table), (0xFF, table), (4, descriptor)] {
            assert_eq!(
                why_refused(reading(&[(kind, root, 0)])),
                "used descriptor of unknown kind",
                "kind {kind}, root {root}"
            );
        }
    }

    #[test]
    fn refuses_a_descriptor_read_of_unknown_root_kind() {
        let srv = 0u8;
        // A table and a root descriptor are the kinds a read names here.
        assert!(reading(&[(srv, 3, 0)]).is_ok());
        assert!(reading(&[(srv, 1, 0)]).is_ok());
        // 2 is the gap in the kinds, 7 the first past the sampler heap, and
        // 0xFF the furthest a read's byte states; each is refused before
        // the parameter it would name is looked for.
        for root in [2u8, 7, 0xFF] {
            assert_eq!(
                why_refused(reading(&[(srv, root, 0)])),
                "used descriptor of unknown root kind",
                "root {root}"
            );
        }
        // An unknown root kind is refused even beside a valid read.
        assert_eq!(
            why_refused(reading(&[(srv, 3, 0), (srv, 2, 0)])),
            "used descriptor of unknown root kind"
        );
    }

    #[test]
    fn entry_padding_is_never_read_but_a_layout_kind_is_its_whole_dword() {
        // A table in s4 (slot 2) and an SRV read from it.
        let build = |pad: u8, layout_kind: u32| {
            let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
            set_stage_reads(&mut blob, HwStage::Ps, &[(3, 1, 4, 2)], &[(0, 3, 0, 4, 2)]);
            let used = blob.len() - ENTRY_SIZE as usize;
            let layout = used - ENTRY_SIZE as usize;
            blob[layout..][..4].copy_from_slice(&layout_kind.to_le_bytes());
            // A root entry's last two bytes, and a read's bytes 2-3 and 19.
            blob[layout + 18..][..2].fill(pad);
            blob[used + 2..][..2].fill(pad);
            blob[used + 19] = pad;
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let clean = build(0, 3).unwrap();
        let padded = build(0xFF, 3).unwrap();
        assert_eq!(clean[0].layout, padded[0].layout);
        assert_eq!(clean[0].used, padded[0].used);
        assert_eq!(padded[0].used[0].root, RootType::Table);
        // A read states its kinds in one byte each, but a root entry's kind
        // is all four: a table's byte with any higher bit set is no table.
        for kind in [0x103, 0x1_0003, 0x0100_0003, 0x8000_0003] {
            assert_eq!(
                why_refused(build(0, kind)),
                "root parameter of unknown kind",
                "{kind:#x}"
            );
        }
    }

    #[test]
    fn refuses_a_table_read_past_any_heap() {
        let (srv, cbv, sampler) = (0u8, 2u8, 1u8);
        let (descriptor, table) = (1u8, 3u8);
        // The last descriptor of a full heap is fine, as is a root
        // descriptor's read, which has no table to index.
        let last = MAX_TABLE_DWORDS - 8;
        assert!(reading(&[(cbv, table, last)]).is_ok());
        assert!(reading(&[(sampler, table, MAX_TABLE_DWORDS - 4)]).is_ok());
        assert!(reading(&[(srv, descriptor, u32::MAX)]).is_ok());
        for (reads, what) in [
            (vec![(cbv, table, MAX_TABLE_DWORDS)], "one descriptor past"),
            (vec![(srv, table, u32::MAX)], "the furthest offset"),
            (vec![(sampler, table, u32::MAX - 3)], "a sampler table"),
            (
                vec![(cbv, table, 0), (cbv, table, 1 << 30)],
                "past a read in bounds",
            ),
        ] {
            assert_eq!(
                why_refused(reading(&reads)),
                "used descriptor past any heap",
                "{what}"
            );
        }
    }

    #[test]
    fn reads_breaking_several_rules_are_named_for_the_first_checked() {
        let (srv, sampler, cbv, uav) = (0u8, 1u8, 2u8, 3u8);
        let (descriptor, table, unknown) = (1u8, 3u8, 2u8);
        // The reads breaking each rule, in the order the rules are checked.
        // Each row below holds the reads breaking its rule and every later
        // one, the first-checked rule's read listed last.
        let rules: [(&[Read], &str); 7] = [
            (&[(4, table, 0, 4, 2)], "used descriptor of unknown kind"),
            (
                &[(srv, unknown, 0, 4, 2)],
                "used descriptor of unknown root kind",
            ),
            (
                &[(srv, table, 0, 5, 2)],
                "used descriptor names no root table",
            ),
            (
                &[(cbv, descriptor, 0, 8, 3)],
                "used descriptor names no root descriptor",
            ),
            (
                &[(cbv, table, MAX_TABLE_DWORDS, 4, 2)],
                "used descriptor past any heap",
            ),
            (
                &[(srv, table, 16, 4, 2), (uav, table, 16, 4, 2)],
                "used descriptors read one descriptor two ways",
            ),
            (
                &[(sampler, table, 0, 4, 2), (srv, table, 8, 4, 2)],
                "root table mixes samplers and views",
            ),
        ];
        for (k, &(_, why)) in rules.iter().enumerate() {
            let reads = rules[k..]
                .iter()
                .rev()
                .flat_map(|(reads, _)| reads.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(why_refused(reading_named(&reads)), why, "{reads:?}");
            // Listed first, too.
            let first = rules[k..]
                .iter()
                .flat_map(|(reads, _)| reads.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(why_refused(reading_named(&first)), why, "{first:?}");
        }
        // Each rule's reads named in full, without the others, are well
        // formed once the breaking byte is put right.
        assert!(
            reading_named(&[
                (srv, table, 0, 4, 2),
                (cbv, descriptor, 0, 6, 3),
                (cbv, table, MAX_TABLE_DWORDS - 8, 4, 2),
                (uav, table, 16, 4, 2),
            ])
            .is_ok()
        );
    }

    fn why_refused(r: Result<Vec<NativeProgram>, ShaderError>) -> &'static str {
        match r {
            Err(Malformed(why)) => why,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn semantic_entries_decode_usage_flags_and_low_mask_bits() {
        // A pixel input: clip distance, the mask's high nibble set, flags
        // linear (bit 1, not read), fp16 and packed with its pair in row 4.
        let mut ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        ps[e + PS_INPUTS..][..8].copy_from_slice(&[7, 2, 0xF3, 5, 0, 4, 0, 0x1A]);
        ps[e + PS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&ps))])).unwrap()[0];
        let StageIo::Ps { inputs, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!(
            inputs[..],
            [PsInput {
                usage: Usage::ClipDistance,
                row: 2,
                mask: 3,
                attr: 5,
                flat: false,
                fp16: true,
                pack2: true,
                row2: 4,
            }]
        );

        // A vertex input of a usage not named (kept as its byte) with the
        // mask's high nibble set, and a clip-distance output of two packed
        // 16-bit values per dword.
        let mut vs = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let e = stage_data_at(&vs, HwStage::Vs);
        vs[e + VS_INPUTS..][..5].copy_from_slice(&[0x2A, 1, 0xF1, 4, 0]);
        vs[e + VS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        vs[e + VS_OUTPUTS..][..6].copy_from_slice(&[7, 3, 2, 3, 0, 3]);
        vs[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&vs))])).unwrap()[0];
        assert_eq!(
            p.io,
            StageIo::Vs {
                inputs: vec![VsInput {
                    usage: Usage::Other(0x2A),
                    row: 1,
                    mask: 1,
                    vgpr: 4,
                }],
                outputs: vec![VsOutput {
                    usage: Usage::ClipDistance,
                    row: 3,
                    param: 2,
                    fp16: true,
                    pack2: true,
                }],
            }
        );
        // Only the low nibble loads: the high bits of 0xF1 name no VGPR, so
        // an input at the last VGPR still fits.
        let regs = e - RECORD_SIZE as usize + RECORD_REGS;
        vs[regs..][..4].copy_from_slice(&5u32.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&vs))])).is_ok());
    }

    #[test]
    fn full_semantic_tables_are_read_to_their_last_entry() {
        let n = MAX_SEMANTICS as usize;
        // Each table at its limit ends exactly where its count is stored.
        assert_eq!(PS_INPUTS + 8 * n, PS_INPUT_COUNT);
        assert_eq!(VS_INPUTS + 5 * n, VS_INPUT_COUNT);
        assert_eq!(VS_OUTPUTS + 6 * n, VS_OUTPUT_COUNT);

        // 32 pixel inputs, input k in row k read as attribute 31 - k.
        let mut ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        for k in 0..n {
            let attr = (n - 1 - k) as u8;
            ps[e + PS_INPUTS + 8 * k..][..8].copy_from_slice(&[6, k as u8, 0xf, attr, 0, 0, 0, 0]);
        }
        ps[e + PS_INPUT_COUNT..][..4].copy_from_slice(&MAX_SEMANTICS.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&ps))])).unwrap()[0];
        let StageIo::Ps { inputs, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!(inputs.len(), n);
        assert_eq!((inputs[n - 1].row, inputs[n - 1].attr), (31, 0));

        // 32 vertex inputs, x-w of input k in v4 + 4k, and 32 outputs,
        // output k param 31 - k in row k.
        let mut vs = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let e = stage_data_at(&vs, HwStage::Vs);
        let regs = e - RECORD_SIZE as usize + RECORD_REGS;
        vs[regs..][..4].copy_from_slice(&256u32.to_le_bytes());
        for k in 0..n {
            let vgpr = 4 + 4 * k as u8;
            vs[e + VS_INPUTS + 5 * k..][..5].copy_from_slice(&[6, k as u8, 0xf, vgpr, 0]);
            let param = (n - 1 - k) as u8;
            vs[e + VS_OUTPUTS + 6 * k..][..6].copy_from_slice(&[6, k as u8, param, 0, 0, 0]);
        }
        for count_at in [VS_INPUT_COUNT, VS_OUTPUT_COUNT] {
            vs[e + count_at..][..4].copy_from_slice(&MAX_SEMANTICS.to_le_bytes());
        }
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&vs))])).unwrap()[0];
        let StageIo::Vs { inputs, outputs } = &p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!((inputs.len(), outputs.len()), (n, n));
        assert_eq!((inputs[n - 1].row, inputs[n - 1].vgpr), (31, 128));
        assert_eq!((outputs[n - 1].row, outputs[n - 1].param), (31, 0));

        // A mesh program reads the same 32 outputs, the odd ones
        // per-primitive: rows repeat across the two signatures only.
        let mut ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        for k in 0..n {
            let param = k as u8 | if k % 2 == 1 { 0x80 } else { 0 };
            let row = (k / 2) as u8;
            ms[e + VS_OUTPUTS + 6 * k..][..6].copy_from_slice(&[6, row, param, 0, 0, 0]);
        }
        ms[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&MAX_SEMANTICS.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&ms))])).unwrap()[0];
        let StageIo::Ms(m) = &p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!(m.outputs.len(), n);
        assert_eq!((m.outputs[n - 1].row, m.outputs[n - 1].param), (15, 0x9F));

        // The last entry is checked like the others: a 32nd output reusing
        // the first's parameter is refused.
        ms[e + VS_OUTPUTS + 6 * (n - 1) + 2] = 0x80;
        assert_eq!(refused_blob(&xbch(&ms)), "vertex outputs share a parameter");
    }

    #[test]
    fn merged_stages_read_no_stage_data() {
        // Merged stages keep their stage data in another layout: bytes that
        // would be refused as vertex or pixel data are not read at all.
        for stage in [HwStage::Ls, HwStage::Hs, HwStage::Es, HwStage::Gs] {
            let mut blob = fssm(&[(stage, &VS_CODE)]);
            let e = stage_data_at(&blob, stage);
            blob[e..e + STAGE_DATA].fill(0xFF);
            let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap();
            assert_eq!((p[0].stage, &p[0].io), (stage, &StageIo::None));
            assert_eq!(p[0].code, VS_CODE);
        }
        // The same bytes in a vertex program are read, and refused.
        let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let e = stage_data_at(&blob, HwStage::Vs);
        blob[e..e + STAGE_DATA].fill(0xFF);
        assert_eq!(refused_blob(&xbch(&blob)), "implausible semantic count");
    }

    #[test]
    fn a_mesh_program_keeps_no_vertex_inputs() {
        // A mesh program has no fetch shader: the input table it shares with
        // a vertex program's layout is bounded by its count, but no entry is
        // kept or checked against the VGPRs.
        let ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        let vs = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let ve = stage_data_at(&vs, HwStage::Vs);
        // Inputs a vertex program refuses: x-w over the vertex id in v0,
        // x-w of v4 twice, and w of v27 past the program's 28 VGPRs.
        for (entries, why) in [
            (vec![[0, 0, 0xf, 0, 0]], "vertex input over a system VGPR"),
            (
                vec![[0, 0, 0xf, 4, 0], [0, 1, 0xf, 4, 0]],
                "vertex inputs overlap",
            ),
            (vec![[0, 0, 0xf, 27, 0]], "vertex input outside the VGPRs"),
        ] {
            let count = (entries.len() as u32).to_le_bytes();
            let mut mesh = ms.clone();
            mesh[e + VS_INPUT_COUNT..][..4].copy_from_slice(&count);
            mesh[e + VS_INPUTS..][..5 * entries.len()].copy_from_slice(entries.as_flattened());
            let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&mesh))])).unwrap();
            assert!(
                matches!(&p[0].io, StageIo::Ms(m) if m.outputs.is_empty()),
                "{why}"
            );

            let mut vertex = vs.clone();
            vertex[ve + VS_INPUT_COUNT..][..4].copy_from_slice(&count);
            vertex[ve + VS_INPUTS..][..5 * entries.len()].copy_from_slice(entries.as_flattened());
            assert_eq!(refused_blob(&xbch(&vertex)), why);
        }
        // The count is still the semantic table's: 32 entries are read, and
        // one more is refused as it is for a vertex program.
        let mut mesh = ms.clone();
        mesh[e + VS_INPUT_COUNT..][..4].copy_from_slice(&MAX_SEMANTICS.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&mesh))])).is_ok());
        mesh[e + VS_INPUT_COUNT..][..4].copy_from_slice(&(MAX_SEMANTICS + 1).to_le_bytes());
        assert_eq!(refused_blob(&xbch(&mesh)), "implausible semantic count");
    }

    #[test]
    fn an_empty_export_patch_is_still_bounded_by_the_code() {
        // Seven bytes: one whole dword and three more, so the code ends at
        // dword 1 for a patch.
        let code = [0x5a; 7];
        let ps = fssm(&[(HwStage::Ps, &code)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let patch_at = |offset: u32| {
            let mut blob = ps.clone();
            blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&0u32.to_le_bytes());
            blob[e + PS_PATCHES..][..4].copy_from_slice(&offset.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // At the start or at the end of the whole dwords, it is kept and
        // patches nothing.
        for offset in [0, 1] {
            let p = &patch_at(offset).unwrap()[0];
            let StageIo::Ps { patches, .. } = &p.io else {
                panic!("{:?}", p.io)
            };
            assert_eq!(
                patches[..],
                [ExportPatch {
                    mrt: 0,
                    offset,
                    code: vec![],
                }]
            );
            assert_eq!(p.patched_code(), code);
        }
        // Past them it is refused like any patch, the last offset before
        // the absent marker included; the marker itself is no patch.
        for offset in [2, 1 << 30, u32::MAX - 1] {
            assert!(
                matches!(
                    patch_at(offset),
                    Err(Malformed("export patch outside the code"))
                ),
                "dword {offset}"
            );
        }
        let p = &patch_at(u32::MAX).unwrap()[0];
        assert!(matches!(&p.io, StageIo::Ps { patches, .. } if patches.is_empty()));
    }

    #[test]
    fn empty_export_patches_share_an_offset_but_sized_ones_do_not() {
        // Two dwords of code; every MRT patches dword 1.
        let code = [0x5a; 8];
        let ps = fssm(&[(HwStage::Ps, &code)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let every_mrt_at_one = |size: u32| {
            let mut blob = ps.clone();
            blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&size.to_le_bytes());
            for mrt in 0..MRTS {
                let at = e + PS_PATCHES + mrt * PATCH_BYTES;
                blob[at..][..4].copy_from_slice(&1u32.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Empty patches write no dword, so none clobbers another: all are
        // kept, in MRT order, and the code is left as stored.
        let p = &every_mrt_at_one(0).unwrap()[0];
        let StageIo::Ps { patches, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        let read: Vec<_> = patches
            .iter()
            .map(|p| (p.mrt, p.offset, p.code.len()))
            .collect();
        assert_eq!(read, (0..MRTS as u8).map(|m| (m, 1, 0)).collect::<Vec<_>>());
        assert_eq!(p.patched_code(), code);
        // One dword each, the second MRT already lands on the first's.
        assert!(matches!(
            every_mrt_at_one(1),
            Err(Malformed("export patches overlap"))
        ));
    }

    #[test]
    fn patched_code_writes_every_mrt_and_keeps_the_stored_code() {
        let code = [0xBF80_0000u32; 6]; // s_nop
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut blob = fssm(&[(HwStage::Ps, &bytes)]);
        let e = stage_data_at(&blob, HwStage::Ps);
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&2u32.to_le_bytes());
        // MRT0 at dword 4 (the last two), MRT7 at dword 0; each variant other
        // than the full one is filled with a marker that must not be used.
        for (mrt, offset, word) in [(0, 4u32, 0xAAAA_0000u32), (7, 0, 0x7777_0000)] {
            let at = e + PS_PATCHES + mrt * PATCH_BYTES;
            blob[at..][..4].copy_from_slice(&offset.to_le_bytes());
            blob[at + 4..at + PATCH_BYTES].fill(0xEE);
            let full = at + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
            for i in 0..2 {
                blob[full + 4 * i..][..4].copy_from_slice(&(word + i as u32).to_le_bytes());
            }
        }
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        assert_eq!(p.code, bytes, "the stored code is kept as read");
        let StageIo::Ps { patches, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!(
            patches[..],
            [
                ExportPatch {
                    mrt: 0,
                    offset: 4,
                    code: vec![0xAAAA_0000, 0xAAAA_0001],
                },
                ExportPatch {
                    mrt: 7,
                    offset: 0,
                    code: vec![0x7777_0000, 0x7777_0001],
                },
            ]
        );
        let words: Vec<u32> = p
            .patched_code()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        assert_eq!(
            words,
            [
                0x7777_0000,
                0x7777_0001,
                0xBF80_0000,
                0xBF80_0000,
                0xAAAA_0000,
                0xAAAA_0001,
            ]
        );

        // An empty patch is still a patch, and changes nothing.
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&0u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let StageIo::Ps { patches, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        assert!(patches.iter().all(|p| p.code.is_empty()));
        assert_eq!(patches.len(), 2);
        assert_eq!(p.patched_code(), bytes);

        // Other stages have no patches to apply.
        let vs = &native_programs(&part(&[(
            KIND_NATIVE,
            &xbch(&fssm(&[(HwStage::Vs, &VS_CODE)])),
        )]))
        .unwrap()[0];
        assert_eq!(vs.patched_code(), VS_CODE);
    }

    #[test]
    fn an_export_patch_is_kept_apart_from_every_earlier_mrt() {
        // One two-dword patch per MRT at these offsets (`None`: absent) in
        // twice as many dwords of code as there are MRTs.
        let code = vec![0u8; 8 * MRTS * 4];
        let ps = fssm(&[(HwStage::Ps, &code)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        let patches = |offsets: &[Option<u32>]| {
            let mut blob = ps.clone();
            blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&2u32.to_le_bytes());
            for (mrt, offset) in offsets.iter().enumerate() {
                let at = e + PS_PATCHES + mrt * PATCH_BYTES;
                let offset = offset.unwrap_or(u32::MAX);
                blob[at..][..4].copy_from_slice(&offset.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Every MRT side by side, last first, fits and is kept in MRT order.
        let all: Vec<_> = (0..MRTS as u32)
            .map(|m| Some(2 * (MRTS as u32 - 1 - m)))
            .collect();
        let p = &patches(&all).unwrap()[0];
        let StageIo::Ps { patches: read, .. } = &p.io else {
            panic!("{:?}", p.io)
        };
        let mrts: Vec<u8> = read.iter().map(|p| p.mrt).collect();
        assert_eq!(mrts, (0..MRTS as u8).collect::<Vec<_>>());
        // Absent MRTs between two patches neither part them nor hide an
        // overlap: the last MRT is checked against the first, not only
        // against the one before it.
        let mut gaps = vec![None; MRTS];
        gaps[0] = Some(4);
        gaps[3] = Some(0);
        for (last, ok) in [(2, true), (6, true), (3, false), (4, false), (5, false)] {
            gaps[MRTS - 1] = Some(last);
            let read = patches(&gaps);
            if ok {
                let p = &read.unwrap()[0];
                assert!(
                    matches!(&p.io, StageIo::Ps { patches, .. } if patches.len() == 3),
                    "dword {last}"
                );
            } else {
                assert!(
                    matches!(read, Err(Malformed("export patches overlap"))),
                    "dword {last}"
                );
            }
        }
    }

    /// The stages a blob's programs are read as, in order.
    fn stages_of(blob: &[u8]) -> Vec<HwStage> {
        native_programs(&part(&[(KIND_NATIVE, &xbch(blob))]))
            .unwrap()
            .iter()
            .map(|p| p.stage)
            .collect()
    }

    #[test]
    fn programs_come_in_slot_order_whatever_the_record_order() {
        // Records laid out pixel before vertex and compute before mesh are
        // still read in slot order: LS HS ES GS VS PS, then mesh, compute.
        let blob = fssm(&[
            (HwStage::Cs, &VS_CODE),
            (HwStage::Ps, &PS_CODE),
            (HwStage::Ms, &VS_CODE),
            (HwStage::Vs, &VS_CODE),
            (HwStage::Ls, &VS_CODE),
        ]);
        assert_eq!(
            stages_of(&blob),
            [
                HwStage::Ls,
                HwStage::Vs,
                HwStage::Ps,
                HwStage::Ms,
                HwStage::Cs
            ]
        );
        // Every slot filled, in reverse.
        let all: Vec<(HwStage, &[u8])> = SLOTS
            .iter()
            .rev()
            .map(|&(stage, _)| (stage, &VS_CODE[..]))
            .collect();
        let want: Vec<HwStage> = SLOTS.iter().map(|s| s.0).collect();
        assert_eq!(stages_of(&fssm(&all)), want);
        // A blob with every slot absent has no programs.
        assert!(stages_of(&fssm(&[])).is_empty());
    }

    #[test]
    fn payload_words_between_the_slots_are_never_read() {
        // 0x38, 0x40, 0x50 and 0x58 sit among the slots but name no stage:
        // whatever they hold, the programs read are the same.
        let good = fssm(&[(HwStage::Vs, &VS_CODE), (HwStage::Cs, &VS_CODE)]);
        let want = native_programs(&part(&[(KIND_NATIVE, &xbch(&good))])).unwrap();
        for fill in [0x00, 0xFF, 0xDE] {
            let mut blob = good.clone();
            for at in [0x38, 0x40, 0x50, 0x58] {
                blob[FSSM_HEADER + at..][..8].fill(fill);
            }
            assert_eq!(
                native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap(),
                want,
                "{fill:#x}"
            );
        }
    }

    #[test]
    fn only_the_exact_absent_marker_skips_a_slot() {
        // Half the marker, or the marker with one bit off, is an offset like
        // any other: far past the payload.
        for near in [
            0xDEAD_BEEFu64,
            0xDEAD_BEEF_0000_0000,
            ABSENT ^ 1,
            ABSENT ^ 1 << 63,
        ] {
            let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
            blob[FSSM_HEADER + slot(HwStage::Ps)..][..8].copy_from_slice(&near.to_le_bytes());
            assert_eq!(
                refused_blob(&xbch(&blob)),
                "FSSM record out of bounds",
                "{near:#x}"
            );
        }
        // A slot of 0 names the slot table itself as a record: its code
        // offset is the slot count and its code size the LS slot, absent and
        // so far past the payload.
        let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        blob[FSSM_HEADER + slot(HwStage::Ps)..][..8].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(refused_blob(&xbch(&blob)), "native code out of bounds");
    }

    #[test]
    fn two_slots_may_name_one_record() {
        // Nothing ties a record to one slot: a pixel slot naming the vertex
        // record reads it again, its stage data now as a pixel program's
        // (empty here: no inputs, every patch absent).
        let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let vs = u64_at(&blob, FSSM_HEADER + slot(HwStage::Vs)).unwrap();
        blob[FSSM_HEADER + slot(HwStage::Ps)..][..8].copy_from_slice(&vs.to_le_bytes());
        let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].stage, p[1].stage), (HwStage::Vs, HwStage::Ps));
        assert_eq!(p[0].code, p[1].code);
        // The wave size is the record's, not the stage's.
        assert_eq!((p[0].wave64, p[1].wave64), (true, true));
        assert_eq!(
            p[1].io,
            StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            }
        );
    }

    #[test]
    fn code_is_read_at_its_payload_relative_offset() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let record = stage_data_at(&good, HwStage::Vs) - RECORD_SIZE as usize;
        let code_at = u64_at(&good, record).unwrap();
        let payload = (good.len() - FSSM_HEADER) as u64;
        // The code ends the payload.
        assert_eq!(code_at + VS_CODE.len() as u64, payload);
        let read = |offset: u64, size: u64| {
            let mut blob = good.clone();
            blob[record..][..8].copy_from_slice(&offset.to_le_bytes());
            blob[record + 8..][..8].copy_from_slice(&size.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Offsets count from the payload, not the FSSM header: the slot
        // count is the first word read from offset 0.
        assert_eq!(
            read(0, 8).unwrap()[0].code,
            GRAPHICS_SLOT_COUNT.to_le_bytes()
        );
        // A window inside the code reads just those bytes.
        assert_eq!(read(code_at + 2, 3).unwrap()[0].code, VS_CODE[2..5]);
        // Empty code is read anywhere up to the payload's end, not past it.
        assert!(read(payload, 0).unwrap()[0].code.is_empty());
        assert_eq!(
            refused_blob(&{
                let mut blob = good.clone();
                blob[record..][..8].copy_from_slice(&(payload + 1).to_le_bytes());
                blob[record + 8..][..8].fill(0);
                xbch(&blob)
            }),
            "native code out of bounds"
        );
        // Offset and size at their largest do not wrap round into the payload.
        assert!(matches!(
            read(u64::MAX, 2),
            Err(Malformed("native code out of bounds"))
        ));
        assert!(matches!(
            read(2, u64::MAX),
            Err(Malformed("native code out of bounds"))
        ));
    }

    #[test]
    fn root_tables_are_read_exactly_within_the_payload() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let record = stage_data_at(&good, HwStage::Vs) - RECORD_SIZE as usize;
        // One constants parameter in user-data dword 0 ending the payload.
        let mut blob = good.clone();
        let table = (blob.len() - FSSM_HEADER) as u64;
        for v in [0u32, 0, 1, 0] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[0; 4]);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        assert_eq!(table + ENTRY_SIZE, payload);
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        let read = |layout: (u64, u32), used: (u64, u32)| {
            let mut blob = blob.clone();
            blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&layout.0.to_le_bytes());
            blob[record + RECORD_USED..][..8].copy_from_slice(&used.0.to_le_bytes());
            blob[record + RECORD_COUNTS..][..4].copy_from_slice(&layout.1.to_le_bytes());
            blob[record + RECORD_COUNTS + 4..][..4].copy_from_slice(&used.1.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // A table ending exactly at the payload's end is read whole.
        let p = read((table, 1), (0, 0)).unwrap();
        assert_eq!(
            p[0].layout,
            [RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0
            }]
        );
        assert!(p[0].used.is_empty());
        // An empty table's offset is never read, whatever it holds.
        for offset in [0, payload, payload + 1, u64::MAX] {
            let p = read((offset, 0), (offset, 0)).unwrap();
            assert!(
                p[0].layout.is_empty() && p[0].used.is_empty(),
                "{offset:#x}"
            );
        }
        // One entry more, or one byte later, is past the payload, for either
        // table.
        for bad in [(table, 2), (table + 1, 1), (payload, 1), (u64::MAX, 1)] {
            assert!(
                matches!(
                    read(bad, (0, 0)),
                    Err(Malformed("FSSM root table out of bounds"))
                ),
                "layout {bad:?}"
            );
            assert!(
                matches!(
                    read((0, 0), bad),
                    Err(Malformed("FSSM root table out of bounds"))
                ),
                "used {bad:?}"
            );
        }
        // The largest count does not wrap round into the payload.
        for offset in [0, table] {
            assert!(matches!(
                read((offset, u32::MAX), (0, 0)),
                Err(Malformed("FSSM root table out of bounds"))
            ));
            assert!(matches!(
                read((0, 0), (offset, u32::MAX)),
                Err(Malformed("FSSM root table out of bounds"))
            ));
        }
        // Offsets count from the payload, not the FSSM header: a table read
        // from offset 0 is the slot count (6, a sampler heap) over the
        // absent LS slot (0xDEADBEEF SGPRs), not the FSSM magic, which would
        // be a parameter of unknown kind.
        assert!(matches!(
            read((0, 1), (0, 0)),
            Err(Malformed("root parameter outside the SGPRs"))
        ));
    }

    #[test]
    fn one_entry_may_be_both_a_parameter_and_a_read_each_by_its_own_fields() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let record = stage_data_at(&good, HwStage::Vs) - RECORD_SIZE as usize;
        // Both tables at the same 20 bytes, one entry each.
        let shared = |entry: [u8; ENTRY_SIZE as usize]| {
            let mut blob = good.clone();
            let table = (blob.len() - FSSM_HEADER) as u64;
            blob.extend_from_slice(&entry);
            let payload = (blob.len() - FSSM_HEADER) as u64;
            blob[8..16].copy_from_slice(&payload.to_le_bytes());
            for at in [RECORD_LAYOUT, RECORD_USED] {
                blob[record + at..][..8].copy_from_slice(&table.to_le_bytes());
            }
            for at in [RECORD_COUNTS, RECORD_COUNTS + 4] {
                blob[record + at..][..4].copy_from_slice(&1u32.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        let mut entry = [0u8; ENTRY_SIZE as usize];
        entry[0] = 3;
        entry[4..8].copy_from_slice(&8u32.to_le_bytes());
        entry[8..12].copy_from_slice(&1u32.to_le_bytes());
        entry[12..16].copy_from_slice(&2u32.to_le_bytes());
        entry[16..20].copy_from_slice(&[4, 5, 7, 0xAA]);
        // As a parameter, byte 0 starts a whole-dword kind (3, a table) and
        // bytes 16 and 17 are its user data and slot; as a read, byte 0 is
        // its kind (3, a UAV), byte 1 its root kind (0, constants), and bytes
        // 16, 17 and 18 its size, user data and slot. A constants read keeps
        // none of the table rules, so the table it shares bytes with is no
        // parameter it must name.
        let p = &shared(entry).unwrap()[0];
        assert_eq!(
            p.layout,
            [RootParam {
                root: RootType::Table,
                table_offset_dwords: 8,
                sgprs: 1,
                base_register: 2,
                user_data: 4,
                slot: 5,
            }]
        );
        assert_eq!(
            p.used,
            [UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Constants,
                table_offset_dwords: 8,
                lower_bound: 1,
                stride: 2,
                dwords: 4,
                user_data: 5,
                slot: 7,
            }]
        );
        // A table read needs byte 1 to be 3, which makes the parameter's
        // kind 0x303: shared bytes never hold a table read, and the layout,
        // checked first, names why.
        entry[1] = 3;
        assert_eq!(why_refused(shared(entry)), "root parameter of unknown kind");
        // A parameter of kind 0 with byte 1 at 3 would still be 0x300.
        entry[0] = 0;
        assert_eq!(why_refused(shared(entry)), "root parameter of unknown kind");
    }

    #[test]
    fn records_and_code_need_no_alignment() {
        // Every record is found through its slot and every offset counts
        // from the payload, so a record moved 1..7 bytes later (its slot and
        // code offset following it) reads exactly as it did aligned.
        let good = fssm(&[(HwStage::Vs, &VS_CODE), (HwStage::Cs, &VS_CODE)]);
        let want = native_programs(&part(&[(KIND_NATIVE, &xbch(&good))])).unwrap();
        assert_eq!(want.len(), 2);
        for pad in 1..8usize {
            for stage in [HwStage::Vs, HwStage::Cs] {
                let slot_at = FSSM_HEADER + slot(stage);
                let record = u64_at(&good, slot_at).unwrap();
                let at = FSSM_HEADER + record as usize;
                let mut blob = good.clone();
                blob.splice(at..at, std::iter::repeat_n(0xA5, pad));
                // Every offset at or past the insertion moves with it.
                for s in [HwStage::Vs, HwStage::Cs] {
                    let s_at = FSSM_HEADER + slot(s);
                    let r = u64_at(&blob, s_at).unwrap();
                    let r = if r >= record { r + pad as u64 } else { r };
                    blob[s_at..][..8].copy_from_slice(&r.to_le_bytes());
                    let code_at = FSSM_HEADER + r as usize;
                    let code = u64_at(&blob, code_at).unwrap();
                    let code = if code >= record {
                        code + pad as u64
                    } else {
                        code
                    };
                    blob[code_at..][..8].copy_from_slice(&code.to_le_bytes());
                }
                let payload = (blob.len() - FSSM_HEADER) as u64;
                blob[8..16].copy_from_slice(&payload.to_le_bytes());
                assert_eq!(
                    native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap(),
                    want,
                    "{stage:?} moved {pad}"
                );
            }
        }
    }

    #[test]
    fn the_payload_size_and_slot_count_are_whole_64_bit_words() {
        let good = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let payload = (good.len() - FSSM_HEADER) as u64;
        let with = |at: usize, value: u64| {
            let mut blob = good.clone();
            blob[at..][..8].copy_from_slice(&value.to_le_bytes());
            refused_blob(&xbch(&blob))
        };
        // The stated size must be the payload's, neither more nor less, and
        // its high half counts.
        for size in [0, payload + 1, payload | 1 << 32, u64::MAX] {
            assert_eq!(with(8, size), "FSSM payload size disagrees", "{size:#x}");
        }
        // The slot count is six in all 64 bits: a high bit set, or a count of
        // the six graphics slots plus mesh and compute, is not it.
        for count in [
            GRAPHICS_SLOT_COUNT | 1 << 32,
            GRAPHICS_SLOT_COUNT | 1 << 63,
            8,
            0,
        ] {
            assert_eq!(
                with(FSSM_HEADER, count),
                "unexpected FSSM slot count",
                "{count:#x}"
            );
        }
        // Bytes after the code, counted in the stated size, are never read.
        let mut padded = good.clone();
        padded.extend_from_slice(&[0xCC; 13]);
        let size = (padded.len() - FSSM_HEADER) as u64;
        padded[8..16].copy_from_slice(&size.to_le_bytes());
        assert_eq!(
            native_programs(&part(&[(KIND_NATIVE, &xbch(&padded))])).unwrap(),
            native_programs(&part(&[(KIND_NATIVE, &xbch(&good))])).unwrap()
        );
    }

    #[test]
    fn compute_stage_data_breaking_several_rules_is_named_for_the_first_checked() {
        let base = fssm(&[(HwStage::Cs, &VS_CODE)]);
        let e = stage_data_at(&base, HwStage::Cs);
        let record = e - RECORD_SIZE as usize;
        // Root constants in s4, with these VGPRs, `COMPUTE_PGM_RSRC2` and
        // numthreads.
        let cs = |vgprs: u32, rsrc2: u32, threads: [u32; 3]| {
            let mut blob = base.clone();
            blob[record + RECORD_REGS..][..4].copy_from_slice(&vgprs.to_le_bytes());
            blob[e + CS_RSRC2..][..4].copy_from_slice(&rsrc2.to_le_bytes());
            for (k, n) in threads.into_iter().enumerate() {
                blob[e + CS_THREADS + 4 * k..][..4].copy_from_slice(&n.to_le_bytes());
            }
            let table = (blob.len() - FSSM_HEADER) as u64;
            for v in [0u32, 0, 1, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&[4, 0, 0, 0]);
            let payload = (blob.len() - FSSM_HEADER) as u64;
            blob[8..16].copy_from_slice(&payload.to_le_bytes());
            blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&table.to_le_bytes());
            blob[record + RECORD_COUNTS..][..4].copy_from_slice(&1u32.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // 4 user SGPRs, then group id x in s4, under the constants.
        let group_id_in_s4 = 4 << 1 | 1 << 7;
        let (three_ids, too_much_lds) = (2 << 11, 129 << 15);
        let (no_threads, one_thread) = ([0, 1, 1], [1, 1, 1]);
        let every = group_id_in_s4 | three_ids | too_much_lds;
        // Each program below also breaks every rule named after it: the
        // record's VGPRs are checked first, then the group size, LDS, the
        // thread ids' VGPRs and the system SGPRs.
        for (vgprs, rsrc2, threads, why) in [
            (257, every, no_threads, "implausible VGPR count"),
            (2, every, no_threads, "implausible thread-group size"),
            (2, every, one_thread, "implausible LDS size"),
            (
                2,
                group_id_in_s4 | three_ids,
                one_thread,
                "thread ids outside the VGPRs",
            ),
            (
                3,
                group_id_in_s4 | three_ids,
                one_thread,
                "root parameter over a system SGPR",
            ),
        ] {
            assert_eq!(
                why_refused(cs(vgprs, rsrc2, threads)),
                why,
                "{vgprs} VGPRs, RSRC2 {rsrc2:#x}, {threads:?}"
            );
        }
        // With group id x off, s4 is the constants' alone.
        let p = cs(3, 4 << 1 | three_ids, one_thread).unwrap();
        assert!(matches!(p[0].io, StageIo::Cs(c) if c.thread_id_vgprs() == 3));
    }

    #[test]
    fn vertex_stage_data_breaking_several_rules_is_named_for_the_first_checked() {
        let base = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let e = stage_data_at(&base, HwStage::Vs);
        let record = e - RECORD_SIZE as usize;
        // `inputs` entries with `input_count` stated, and `outputs`.
        let vs = |vgprs: u32, input_count: u32, inputs: &[[u8; 5]], outputs: &[[u8; 6]]| {
            let mut blob = base.clone();
            blob[record + RECORD_REGS..][..4].copy_from_slice(&vgprs.to_le_bytes());
            blob[e + VS_INPUT_COUNT..][..4].copy_from_slice(&input_count.to_le_bytes());
            blob[e + VS_INPUTS..][..5 * inputs.len()].copy_from_slice(inputs.as_flattened());
            let output_count = outputs.len() as u32;
            blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&output_count.to_le_bytes());
            blob[e + VS_OUTPUTS..][..6 * outputs.len()].copy_from_slice(outputs.as_flattened());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Inputs as (usage, row, mask, VGPR, -): x-w of v255 is past the
        // 28 VGPRs; x of v0 is the vertex id, twice; x of v4 twice overlaps.
        let past = [6, 2, 0xf, 255, 0];
        let (v0, v0_again) = ([6, 0, 0x1, 0, 0], [6, 1, 0x1, 0, 0]);
        let (v4, v4_again, v5) = ([6, 0, 0x1, 4, 0], [6, 1, 0x1, 4, 0], [6, 1, 0x1, 5, 0]);
        // Outputs as (usage, row, param, -, -, flags): per-primitive param0
        // twice in row 0, then param0 and param1 in row 0, then rows 0 and 1.
        let same_param = [[6, 0, 0x80, 0, 0, 0], [6, 0, 0x80, 0, 0, 0]];
        let same_row = [[6, 0, 0x80, 0, 0, 0], [6, 0, 0x81, 0, 0, 0]];
        let per_primitive = [[6, 0, 0x80, 0, 0, 0], [6, 1, 0x81, 0, 0, 0]];
        let too_many = MAX_SEMANTICS + 1;
        let all_bad = [past, v0, v0_again];
        // Each program below also breaks every rule named after it: the
        // record's VGPRs, then the semantic counts, the outputs' parameters
        // and rows, the inputs' VGPRs, and per-primitive outputs last.
        for (vgprs, count, inputs, outputs, why) in [
            (
                257,
                too_many,
                &all_bad[..],
                same_param,
                "implausible VGPR count",
            ),
            (
                VGPRS,
                too_many,
                &all_bad,
                same_param,
                "implausible semantic count",
            ),
            (
                VGPRS,
                3,
                &all_bad,
                same_param,
                "vertex outputs share a parameter",
            ),
            (VGPRS, 3, &all_bad, same_row, "vertex outputs share a row"),
            (
                VGPRS,
                3,
                &all_bad,
                per_primitive,
                "vertex input outside the VGPRs",
            ),
            (
                VGPRS,
                2,
                &all_bad[1..],
                per_primitive,
                "vertex input over a system VGPR",
            ),
            (
                VGPRS,
                2,
                &[v4, v4_again],
                per_primitive,
                "vertex inputs overlap",
            ),
            (
                VGPRS,
                2,
                &[v4, v5],
                per_primitive,
                "vertex output marked per-primitive",
            ),
        ] {
            assert_eq!(why_refused(vs(vgprs, count, inputs, &outputs)), why);
        }
        // With every rule kept, the same program is read.
        let per_vertex = [[6, 0, 0, 0, 0, 0], [6, 1, 1, 0, 0, 0]];
        let p = vs(VGPRS, 2, &[v4, v5], &per_vertex).unwrap();
        assert!(matches!(&p[0].io, StageIo::Vs { inputs, outputs }
            if inputs.len() == 2 && outputs.len() == 2));
    }

    #[test]
    fn pixel_stage_data_breaking_several_rules_is_named_for_the_first_checked() {
        let code: Vec<u8> = [0xBF80_0000u32; 4] // s_nop
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let base = fssm(&[(HwStage::Ps, &code)]);
        let e = stage_data_at(&base, HwStage::Ps);
        // `inputs` entries with `input_count` stated, and patches of `size`
        // dwords for MRT0 and MRT1 at these offsets.
        let ps = |input_count: u32, inputs: &[[u8; 8]], size: u32, offsets: [u32; 2]| {
            let mut blob = base.clone();
            blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&input_count.to_le_bytes());
            blob[e + PS_INPUTS..][..8 * inputs.len()].copy_from_slice(inputs.as_flattened());
            blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&size.to_le_bytes());
            for (mrt, offset) in offsets.into_iter().enumerate() {
                let at = e + PS_PATCHES + mrt * PATCH_BYTES;
                blob[at..][..4].copy_from_slice(&offset.to_le_bytes());
            }
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]))
        };
        // Inputs as (usage, row, mask, attr, -, row2, -, flags): attr0 packs
        // rows 2 and 3; a 32-bit attr0 (or attr1) names row 3, its second
        // row; a 32-bit attr2 reads row 2, which attr0 reads packed.
        let packed = [6, 2, 0x3, 0, 0, 3, 0, 0x18];
        let (on_its_row, on_its_row_attr1) =
            ([6, 3, 0xc, 0, 0, 3, 0, 0], [6, 3, 0xc, 1, 0, 3, 0, 0]);
        let row_two_ways = [6, 2, 0xc, 2, 0, 2, 0, 0];
        let too_many = MAX_SEMANTICS + 1;
        let too_big = PATCH_SLOT_DWORDS as u32 + 1;
        // Two-dword patches: MRT1 at dword 3 runs past the four-dword code
        // and over MRT0 at dword 2; at dword 1 it only overlaps MRT0.
        let (past_and_over, over) = ([2, 3], [2, 1]);
        let every = [packed, on_its_row, row_two_ways];
        // Each program below also breaks every rule named after it: the
        // input count, then attributes, rows read two ways, claimed packed
        // rows, the patch size, patches past the code and overlaps.
        for (count, inputs, size, offsets, why) in [
            (
                too_many,
                &every[..],
                too_big,
                past_and_over,
                "implausible semantic count",
            ),
            (
                3,
                &every,
                too_big,
                past_and_over,
                "pixel inputs share an attribute",
            ),
            (
                3,
                &[packed, on_its_row_attr1, row_two_ways],
                too_big,
                past_and_over,
                "pixel inputs read a row two ways",
            ),
            (
                2,
                &[packed, on_its_row_attr1],
                too_big,
                past_and_over,
                "pixel inputs claim a packed row",
            ),
            (
                0,
                &[],
                too_big,
                past_and_over,
                "implausible export patch size",
            ),
            (0, &[], 2, past_and_over, "export patch outside the code"),
            (0, &[], 2, over, "export patches overlap"),
        ] {
            assert_eq!(why_refused(ps(count, inputs, size, offsets)), why);
        }
        // With every rule kept, the same program is read.
        let p = ps(1, &[packed], 2, [2, 0]).unwrap();
        assert!(matches!(&p[0].io, StageIo::Ps { inputs, patches, .. }
            if inputs.len() == 1 && patches.len() == 2));
    }

    #[test]
    fn record_words_between_its_fields_are_never_read() {
        // 0x10, 0x18-0x27 and 0x40-0x4b of a record name nothing, and 0x4c
        // is a mesh program's LDS alone (compute states its own in RSRC2):
        // whatever they hold, every other program reads the same.
        for (stage, _) in SLOTS {
            let good = fssm(&[(stage, &VS_CODE)]);
            let want = native_programs(&part(&[(KIND_NATIVE, &xbch(&good))])).unwrap();
            let record = FSSM_HEADER + u64_at(&good, FSSM_HEADER + slot(stage)).unwrap() as usize;
            for fill in [0xFF, 0xDE] {
                let mut blob = good.clone();
                blob[record + 0x10..][..4].fill(fill);
                blob[record + 0x18..record + RECORD_LAYOUT].fill(fill);
                blob[record + 0x40..record + RECORD_LDS].fill(fill);
                let read = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]));
                assert_eq!(read.unwrap(), want, "{stage:?}: {fill:#x}");
                blob[record + RECORD_LDS..][..4].fill(fill);
                let read = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))]));
                if stage == HwStage::Ms {
                    assert_eq!(why_refused(read), "implausible LDS size", "{fill:#x}");
                } else {
                    assert_eq!(read.unwrap(), want, "{stage:?}: {fill:#x}");
                }
            }
        }
    }

    #[test]
    fn stage_registers_are_carried_whole_and_only_clip_cntl_of_a_mesh() {
        // A pixel program's input and depth-control words keep every bit.
        let mut ps = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let e = stage_data_at(&ps, HwStage::Ps);
        ps[e + PS_INPUT_ADDR..][..4].copy_from_slice(&u32::MAX.to_le_bytes());
        ps[e + PS_DB_SHADER_CONTROL..][..4].copy_from_slice(&0x8000_0001u32.to_le_bytes());
        let p = native_programs(&part(&[(KIND_NATIVE, &xbch(&ps))])).unwrap();
        assert!(matches!(
            &p[0].io,
            StageIo::Ps {
                input_addr: u32::MAX,
                db_shader_control: 0x8000_0001,
                ..
            }
        ));
        // Of a mesh program's eight registers only PA_CL_VS_OUT_CNTL is
        // read, whole; the other seven change nothing whatever they hold.
        let ms = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&ms, HwStage::Ms);
        let regs = VS_OUTPUT_COUNT + 4;
        let read = |fill: u8, clip: u32| {
            let mut blob = ms.clone();
            blob[e + regs..][..32].fill(fill);
            blob[e + MS_CLIP_CNTL..][..4].copy_from_slice(&clip.to_le_bytes());
            native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()
        };
        let want = read(0, u32::MAX);
        assert!(matches!(&want[0].io, StageIo::Ms(m) if m.clip_cntl == u32::MAX));
        for fill in [0xFF, 0xDE] {
            assert_eq!(read(fill, u32::MAX), want, "{fill:#x}");
        }
        assert!(matches!(&read(0xFF, 0)[0].io, StageIo::Ms(m) if m.clip_cntl == 0));
    }

    #[test]
    fn descriptor_bounds_strides_and_sizes_are_carried_whole_and_unchecked() {
        let (srv, cbv) = (0u8, 2u8);
        let (descriptor, table) = (1u8, 3u8);
        // A table in s4 (slot 2) and a root descriptor in s6 (slot 3), each
        // read once; then every field no check reads is set to a value no
        // shipped layout states, byte by byte distinct so a field read from
        // its neighbour's bytes, or narrowed, would show.
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
        set_stage_reads(
            &mut blob,
            HwStage::Ps,
            &[(3, 1, 4, 2), (1, 2, 6, 3)],
            &[(srv, table, 0x0012_3456, 4, 2), (cbv, descriptor, 0, 6, 3)],
        );
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Ps)).unwrap() as usize;
        let layout = FSSM_HEADER + u64_at(&blob, record + RECORD_LAYOUT).unwrap() as usize;
        let used = FSSM_HEADER + u64_at(&blob, record + RECORD_USED).unwrap() as usize;
        let set = |blob: &mut Vec<u8>, at: usize, v: u32| {
            blob[at..][..4].copy_from_slice(&v.to_le_bytes())
        };
        let entry = ENTRY_SIZE as usize;
        // The table's range offset and base register, and the descriptor's.
        set(&mut blob, layout + 4, u32::MAX);
        set(&mut blob, layout + 12, 0xA1B2_C3D4);
        set(&mut blob, layout + entry + 4, 0x0102_0304);
        set(&mut blob, layout + entry + 12, u32::MAX);
        // The table read's lower bound, stride and size; the root
        // descriptor's offset, which indexes no table, at its furthest.
        set(&mut blob, used + 8, u32::MAX);
        set(&mut blob, used + 12, 0x5566_7788);
        blob[used + 16] = 0xFF;
        set(&mut blob, used + entry + 4, u32::MAX);
        set(&mut blob, used + entry + 8, 0x1122_3344);
        set(&mut blob, used + entry + 12, 0);
        blob[used + entry + 16] = 0;
        let read = |blob: &[u8]| native_programs(&part(&[(KIND_NATIVE, &xbch(blob))]));
        let p = &read(&blob).unwrap()[0];
        assert_eq!(
            p.layout,
            [
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: u32::MAX,
                    sgprs: 1,
                    base_register: 0xA1B2_C3D4,
                    user_data: 4,
                    slot: 2
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0x0102_0304,
                    sgprs: 2,
                    base_register: u32::MAX,
                    user_data: 6,
                    slot: 3
                },
            ]
        );
        assert_eq!(
            p.used,
            [
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0x0012_3456,
                    lower_bound: u32::MAX,
                    stride: 0x5566_7788,
                    dwords: 0xFF,
                    user_data: 4,
                    slot: 2
                },
                UsedResource {
                    kind: ResourceType::Cbv,
                    root: RootType::Descriptor,
                    table_offset_dwords: u32::MAX,
                    lower_bound: 0x1122_3344,
                    stride: 0,
                    dwords: 0,
                    user_data: 6,
                    slot: 3
                },
            ]
        );
        // Of all these, only a table read's own offset is bounded: moved to
        // the heap's end it is refused, though its parameter's range offset
        // at `u32::MAX` was not.
        let mut past = blob.clone();
        set(&mut past, used + 4, MAX_TABLE_DWORDS);
        assert_eq!(why_refused(read(&past)), "used descriptor past any heap");
        // Two reads of one descriptor as one kind are one way, however
        // their bounds, strides and sizes differ.
        let mut twice = fssm(&[(HwStage::Ps, &PS_CODE)]);
        set_stage_reads(
            &mut twice,
            HwStage::Ps,
            &[(3, 1, 4, 2), (1, 2, 6, 3)],
            &[
                (srv, table, 8, 4, 2),
                (srv, table, 8, 4, 2),
                (cbv, descriptor, 0, 6, 3),
                (cbv, descriptor, 0, 6, 3),
            ],
        );
        let used = FSSM_HEADER + u64_at(&twice, record + RECORD_USED).unwrap() as usize;
        for (k, (bound, stride, dwords)) in [(0, 4, 4), (u32::MAX, 0, 8), (7, 16, 0), (0, 0, 0xFF)]
            .into_iter()
            .enumerate()
        {
            let at = used + k * entry;
            set(&mut twice, at + 8, bound);
            set(&mut twice, at + 12, stride);
            twice[at + 16] = dwords;
        }
        let p = &read(&twice).unwrap()[0];
        assert_eq!(
            p.used
                .iter()
                .map(|u| (u.lower_bound, u.stride, u.dwords))
                .collect::<Vec<_>>(),
            [(0, 4, 4), (u32::MAX, 0, 8), (7, 16, 0), (0, 0, 0xFF)]
        );
    }

    #[test]
    fn a_record_breaking_several_rules_is_named_for_the_first_checked() {
        // A record's checks, in the order they are made: its code, its VGPR
        // count, its layout table and the layout's rules, its read table and
        // the reads' rules, then its stage data. The read table is not even
        // found until the layout has passed. Record `from` breaks every rule
        // from that one on.
        const RULES: [&str; 7] = [
            "native code out of bounds",
            "implausible VGPR count",
            "FSSM root table out of bounds",
            "root parameter of unknown kind",
            "FSSM root table out of bounds",
            "used descriptor of unknown kind",
            "implausible semantic count",
        ];
        let (constants, unknown_root, srv, unknown_kind) = (0, 2, 0, 9);
        let blob = |from: usize| {
            let breaks = |k: usize| k >= from;
            let mut blob = fssm(&[(HwStage::Vs, &VS_CODE)]);
            let root = if breaks(3) { unknown_root } else { constants };
            let kind = if breaks(5) { unknown_kind } else { srv };
            // A constants read keeps none of the table or descriptor rules.
            set_stage_reads(
                &mut blob,
                HwStage::Vs,
                &[(root, 1, 0, 0)],
                &[(kind, constants as u8, 0, 0, 0)],
            );
            let e = stage_data_at(&blob, HwStage::Vs);
            let record = e - RECORD_SIZE as usize;
            if breaks(0) {
                blob[record + 8..][..8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            if breaks(1) {
                blob[record + RECORD_REGS..][..4].copy_from_slice(&257u32.to_le_bytes());
            }
            if breaks(2) {
                blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            if breaks(4) {
                blob[record + RECORD_USED..][..8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            if breaks(6) {
                blob[e + VS_INPUT_COUNT..][..4].copy_from_slice(&(MAX_SEMANTICS + 1).to_le_bytes());
            }
            xbch(&blob)
        };
        for (from, why) in RULES.into_iter().enumerate() {
            assert_eq!(refused_blob(&blob(from)), why, "breaking rules {from}..");
        }
        let p = native_programs(&part(&[(KIND_NATIVE, &blob(RULES.len()))])).unwrap();
        assert_eq!((p[0].layout.len(), p[0].used.len()), (1, 1));
    }

    #[test]
    fn of_several_bad_records_the_first_slot_is_named_whatever_the_record_order() {
        // Pixel laid out before vertex, then compute, each able to break a
        // rule of its own. Slots are read in order (VS, PS, then CS), so the
        // first bad slot names the refusal wherever its record sits, and a
        // bad later slot refuses the programs already read before it.
        let blob = |bad: &[HwStage]| {
            let mut blob = fssm(&[
                (HwStage::Ps, &PS_CODE),
                (HwStage::Vs, &VS_CODE),
                (HwStage::Cs, &VS_CODE),
            ]);
            for &stage in bad {
                let e = stage_data_at(&blob, stage);
                let record = e - RECORD_SIZE as usize;
                match stage {
                    HwStage::Vs => {
                        blob[record + RECORD_REGS..][..4].copy_from_slice(&257u32.to_le_bytes())
                    }
                    HwStage::Ps => blob[record + 8..][..8].copy_from_slice(&u64::MAX.to_le_bytes()),
                    _ => blob[e + CS_THREADS..][..4].copy_from_slice(&0u32.to_le_bytes()),
                }
            }
            xbch(&blob)
        };
        let (vs, ps, cs) = (HwStage::Vs, HwStage::Ps, HwStage::Cs);
        for (bad, why) in [
            (&[cs, ps, vs][..], "implausible VGPR count"),
            (&[cs, vs], "implausible VGPR count"),
            (&[cs, ps], "native code out of bounds"),
            (&[ps], "native code out of bounds"),
            (&[cs], "implausible thread-group size"),
        ] {
            assert_eq!(refused_blob(&blob(bad)), why, "{bad:?}");
        }
        let p = native_programs(&part(&[(KIND_NATIVE, &blob(&[]))])).unwrap();
        assert_eq!(p.iter().map(|p| p.stage).collect::<Vec<_>>(), [vs, ps, cs]);
    }
}
