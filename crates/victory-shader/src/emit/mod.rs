//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod sampler;
mod shader;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
    })?;
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
    })
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    /// End-to-end pixel shader `exp ... vm` (Valid Mask) and multi-MRT color
    /// export tests, validated with `dxil.dll` and rendered on hardware GPU
    /// and WARP into cleared 4x4 RGBA32F render targets.
    #[test]
    #[cfg(windows)]
    fn pixel_export_valid_mask_and_multi_mrt_match_rdna2() {
        use super::bitcode::{Attr, Term};
        use super::host::{Draw, Host};

        // Pass-through fullscreen vertex shader: POSITION -> SV_Position.
        let fullscreen_vs = || -> super::HostShader {
            let mut s = Shader::new(false);
            s.inputs.push(SigElement {
                name: "POSITION".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 15,
                half: false,
            });
            s.outputs.push(SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 15,
                half: false,
            });
            let b = s.entry;
            let undef = s.m.undef(s.t.i32);
            for col in 0..4u8 {
                let id = s.m.i32(0);
                let row = s.m.i32(0);
                let c = s.m.i8(col);
                let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &[id, row, c, undef]);
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &[id, row, c, v]);
            }
            s.m.terminate(b, Term::Ret);
            let mut dxil = s.finish();
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut dxil).expect("VS DXIL validation failed");
            }
            super::HostShader {
                stage: super::HwStage::Vs,
                dxil,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            }
        };

        let exp = |target: u8, en: u8, done: bool, vm: bool, vsrc: [u8; 4]| -> [u32; 2] {
            let w0 = 0xF800_0000
                | (u32::from(vm) << 12)
                | (u32::from(done) << 11)
                | (u32::from(target) << 4)
                | u32::from(en & 0xF);
            let w1 = u32::from(vsrc[0])
                | (u32::from(vsrc[1]) << 8)
                | (u32::from(vsrc[2]) << 16)
                | (u32::from(vsrc[3]) << 24);
            [w0, w1]
        };

        let build_ps = |words: &[u32], mrt_count: u32| -> super::HostShader {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![],
                    input_addr: 1 << 8, // v0 = SV_Position.x
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let pos = SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 1,
                half: false,
            };
            let targets: Vec<SigElement> = (0..mrt_count)
                .map(|n| SigElement {
                    name: "SV_Target".into(),
                    index: n,
                    kind: 16,
                    comp: 9,
                    interp: 0,
                    row: Some(n),
                    col: 0,
                    cols: 4,
                    usage: 15,
                    half: false,
                })
                .collect();
            let isg1 = super::container::signature(&[pos], true);
            let osg1 = super::container::signature(&targets, false);
            let bytes = crate::dxbc::build(&[
                (super::FourCC::ISG1, &isg1),
                (super::FourCC::OSG1, &osg1),
            ]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let mut out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                let ours = out.dxil[4..20].to_vec();
                v.validate(&mut out.dxil).expect("PS DXIL validation failed");
                assert_eq!(out.dxil[4..20], ours[..], "container hash mismatch");
            }
            out
        };

        const V_MOV_V1_1: u32 = 0x7E02_02F2; // v_mov_b32 v1, 1.0
        const V_MOV_V2_0: u32 = 0x7E04_0280; // v_mov_b32 v2, 0.0
        // v_cmp_lt_f32 vcc_lo, v0, 2.0 (VOP3 op 0x001, sdst = 0x6a, src0 = v0 (256), src1 = 2.0 (244))
        const CMP_X_LT_2: [u32; 2] = [0xD400_0000 | (0x001 << 16) | 0x6A, 256 | (244 << 9)];
        const SAVEEXEC: u32 = 0xBE80_3C6A; // s_and_saveexec_b32 s0, vcc_lo
        const RESTORE_EXEC: u32 = 0xBEFE_0300; // s_mov_b32 exec_lo, s0
        const S_ENDPGM: u32 = 0xBF81_0000;

        const RED_V: [u8; 4] = [1, 2, 2, 1];
        const GREEN_V: [u8; 4] = [2, 1, 2, 1];
        const BLUE_V: [u8; 4] = [2, 2, 1, 1];
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
        const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

        // Case 1: exp mrt0 (vm=1, done=0) under x<2, restore EXEC, exp mrt1 (vm=0, done=1) full EXEC.
        // Right half (x>=2) MUST be discarded even though done=1 ran under full EXEC.
        let mut w1 = vec![V_MOV_V1_1, V_MOV_V2_0, CMP_X_LT_2[0], CMP_X_LT_2[1], SAVEEXEC];
        w1.extend_from_slice(&exp(0, 0xF, false, true, RED_V));
        w1.push(RESTORE_EXEC);
        w1.extend_from_slice(&exp(1, 0xF, true, false, BLUE_V));
        w1.push(S_ENDPGM);
        let ps_vm_then_done = build_ps(&w1, 2);

        // Case 2: exp mrt0 (vm=1, done=0) full EXEC, narrow to x<2, exp mrt1 (vm=0, done=1).
        // Right half (x>=2) MUST NOT be discarded (MRT0 stays RED) because done=1 has vm=0.
        let mut w2 = vec![V_MOV_V1_1, V_MOV_V2_0];
        w2.extend_from_slice(&exp(0, 0xF, false, true, RED_V));
        w2.extend_from_slice(&[CMP_X_LT_2[0], CMP_X_LT_2[1], SAVEEXEC]);
        w2.extend_from_slice(&exp(1, 0xF, true, false, BLUE_V));
        w2.push(S_ENDPGM);
        let ps_vm_full_done_partial = build_ps(&w2, 2);

        // Case 3: exp mrt0 (vm=0, done=0) full EXEC, narrow to x<2, exp null (vm=1, done=1).
        // Left half survives with MRT0=RED; right half is discarded.
        let mut w3 = vec![V_MOV_V1_1, V_MOV_V2_0];
        w3.extend_from_slice(&exp(0, 0xF, false, false, RED_V));
        w3.extend_from_slice(&[CMP_X_LT_2[0], CMP_X_LT_2[1], SAVEEXEC]);
        w3.extend_from_slice(&exp(9, 0x0, true, true, [0, 0, 0, 0]));
        w3.push(S_ENDPGM);
        let ps_mrt0_then_null_vm = build_ps(&w3, 1);

        // Case 4: Two VM=1 exports — last VM=1 export governs survival.
        // exp mrt0 (vm=1) under x<2, restore EXEC, exp mrt1 (vm=1, done=1) full EXEC.
        // All 16 pixels survive (MRT1=BLUE everywhere; right half is not killed by the first vm).
        let mut w4 = vec![V_MOV_V1_1, V_MOV_V2_0, CMP_X_LT_2[0], CMP_X_LT_2[1], SAVEEXEC];
        w4.extend_from_slice(&exp(0, 0xF, false, true, RED_V));
        w4.push(RESTORE_EXEC);
        w4.extend_from_slice(&exp(1, 0xF, true, true, BLUE_V));
        w4.push(S_ENDPGM);
        let ps_last_vm_wins = build_ps(&w4, 2);

        // Case 5: Conditional per-MRT color overwrite (non-done MRT0 write under partial EXEC).
        // exp mrt0 RED (full), narrow to x<2, exp mrt0 GREEN, restore EXEC, exp mrt1 BLUE done vm.
        let mut w5 = vec![V_MOV_V1_1, V_MOV_V2_0];
        w5.extend_from_slice(&exp(0, 0xF, false, false, RED_V));
        w5.extend_from_slice(&[CMP_X_LT_2[0], CMP_X_LT_2[1], SAVEEXEC]);
        w5.extend_from_slice(&exp(0, 0xF, false, false, GREEN_V));
        w5.push(RESTORE_EXEC);
        w5.extend_from_slice(&exp(1, 0xF, true, true, BLUE_V));
        w5.push(S_ENDPGM);
        let ps_conditional_mrt_overwrite = build_ps(&w5, 2);

        // Case 6: Mixed-join CFG — left half exports mrt0 done vm, right half skips to shared s_endpgm.
        let mut w6 = vec![
            V_MOV_V1_1,
            V_MOV_V2_0,
            CMP_X_LT_2[0],
            CMP_X_LT_2[1],
            SAVEEXEC,
            0xBF88_0002, // s_cbranch_execz +2 -> S_ENDPGM
        ];
        w6.extend_from_slice(&exp(0, 0xF, true, true, RED_V));
        w6.push(S_ENDPGM);
        let ps_mixed_join = build_ps(&w6, 1);

        // Case 7: Early-kill fallthrough (`v_cmpx_lt_f32` + `s_cbranch_execnz`):
        // dead lanes fall through to `exp null done vm` with EXEC=0 (guard=Bool::FALSE),
        // alive lanes jump to `exp mrt0 RED done vm`.
        // v_cmpx_lt_f32 exec, v0, 2.0 (VOP3 op 0x011, sdst = 0x7e)
        let mut w7 = vec![
            V_MOV_V1_1,
            V_MOV_V2_0,
            0xD400_0000 | (0x011 << 16) | 0x7E,
            256 | (244 << 9),
            0xBF89_0003, // s_cbranch_execnz +3 -> alive block (after 2-dword exp null + s_endpgm)
        ];
        w7.extend_from_slice(&exp(9, 0x0, true, true, [0, 0, 0, 0]));
        w7.push(S_ENDPGM);
        w7.extend_from_slice(&exp(0, 0xF, true, true, RED_V));
        w7.push(S_ENDPGM);
        let ps_early_kill_null_vm = build_ps(&w7, 1);

        let vs = fullscreen_vs();
        let verts: [f32; 12] = [
            -1.0, -1.0, 0.0, 1.0,
            3.0, -1.0, 0.0, 1.0,
            -1.0, 3.0, 0.0, 1.0,
        ];
        let draw_desc = Draw {
            roots: vec![],
            textures: &[],
            samplers: &[],
            vertices: &verts,
            stride: 16,
            size: (4, 4),
            mesh_groups: 0,
        };

        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            ran += 1;

            let render = |ps: &super::HostShader, rtvs: usize| -> Vec<Vec<[f32; 4]>> {
                let (root, pso) = host
                    .pipeline(&vs, ps)
                    .unwrap_or_else(|e| panic!("{label}: pipeline failed: {e}"));
                host.draw_mrt(&root, &pso, &draw_desc, rtvs)
                    .unwrap_or_else(|e| panic!("{label}: draw_mrt failed: {e}"))
            };

            // Case 1: left survives (RED, BLUE), right discarded (CLEAR, CLEAR).
            let r1 = render(&ps_vm_then_done, 2);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    let (want0, want1) = if x < 2 { (RED, BLUE) } else { (CLEAR, CLEAR) };
                    assert_eq!(r1[0][i], want0, "{label}: case1 MRT0 ({x},{y})");
                    assert_eq!(r1[1][i], want1, "{label}: case1 MRT1 ({x},{y})");
                }
            }

            // Case 2: ALL pixels survive (MRT0 = RED everywhere); left MRT1 = BLUE.
            let r2 = render(&ps_vm_full_done_partial, 2);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    assert_eq!(r2[0][i], RED, "{label}: case2 MRT0 ({x},{y})");
                    if x < 2 {
                        assert_eq!(r2[1][i], BLUE, "{label}: case2 MRT1 ({x},{y})");
                    }
                }
            }

            // Case 3: exp mrt0 full -> exp null done vm partial.
            let r3 = render(&ps_mrt0_then_null_vm, 1);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    let want = if x < 2 { RED } else { CLEAR };
                    assert_eq!(r3[0][i], want, "{label}: case3 MRT0 ({x},{y})");
                }
            }

            // Case 4: two VM=1 exports; last VM=1 (full EXEC) keeps all pixels alive.
            let r4 = render(&ps_last_vm_wins, 2);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    assert_eq!(r4[1][i], BLUE, "{label}: case4 MRT1 ({x},{y})");
                    if x < 2 {
                        assert_eq!(r4[0][i], RED, "{label}: case4 MRT0 ({x},{y})");
                    }
                }
            }

            // Case 5: conditional per-MRT color overwrite.
            let r5 = render(&ps_conditional_mrt_overwrite, 2);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    let want0 = if x < 2 { GREEN } else { RED };
                    assert_eq!(r5[0][i], want0, "{label}: case5 MRT0 ({x},{y})");
                    assert_eq!(r5[1][i], BLUE, "{label}: case5 MRT1 ({x},{y})");
                }
            }

            // Case 6: mixed-join CFG — left survives (RED), right discarded (CLEAR).
            let r6 = render(&ps_mixed_join, 1);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    let want = if x < 2 { RED } else { CLEAR };
                    assert_eq!(r6[0][i], want, "{label}: case6 MRT0 ({x},{y})");
                }
            }

            // Case 7: early-kill fallthrough (`exp null done vm` under EXEC=0).
            let r7 = render(&ps_early_kill_null_vm, 1);
            for y in 0..4 {
                for x in 0..4 {
                    let i = y * 4 + x;
                    let want = if x < 2 { RED } else { CLEAR };
                    assert_eq!(r7[0][i], want, "{label}: case7 MRT0 ({x},{y})");
                }
            }
        }
        assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
    }
}
