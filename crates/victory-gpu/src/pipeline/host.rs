//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// `None` if the stream supplied an invalid input layout.
    input_layout: Option<InputLayout>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            input_layout: Some(InputLayout::default()),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`. When an input layout
    /// has a nonzero, in-range count and a non-null, aligned descriptor
    /// pointer, that pointer must reference an array of that many initialized
    /// elements. Each non-null semantic name read must point to a C string.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    self.input_layout = InputLayout::read(layout);
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs = self.input_layout.as_ref().map(InputLayout::describe);
        let inputs = inputs.as_deref().unwrap_or("<invalid>");
        format!(
            "{:?} {} {:?} {:?} {inputs} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }
}

#[derive(Default)]
struct InputLayout {
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
}

impl InputLayout {
    /// # Safety
    /// The input layout's pointers must meet the requirements of `GraphicsState::read`.
    unsafe fn read(layout: D3D12_INPUT_LAYOUT_DESC) -> Option<Self> {
        let count = layout.NumElements;
        if count == 0 {
            return Some(Self::default());
        }
        let ptr = layout.pInputElementDescs;
        if count > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT
            || ptr.is_null()
            || !ptr.is_aligned()
        {
            return None;
        }

        let count = count as usize;
        // SAFETY: the pointer passed the shape checks and the caller supplies
        // the complete array.
        let elements = unsafe { std::slice::from_raw_parts(ptr, count) };
        let mut owned = Self {
            inputs: Vec::with_capacity(count),
            names: Vec::with_capacity(count),
        };
        for e in elements {
            if e.InputSlot >= D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT || e.SemanticName.is_null()
            {
                return None;
            }
            // SAFETY: the caller supplies a C string for each non-null name.
            let name = unsafe { CStr::from_ptr(e.SemanticName.0.cast()) }.to_owned();
            owned.inputs.push(D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                Format: DXGI_FORMAT(crate::resource::pc_format(e.Format.0 as u32) as i32),
                ..*e
            });
            owned.names.push(name);
        }
        Some(owned)
    }

    fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate,
                )
            })
            .collect();
        format!("{inputs:?}")
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs.iter().fold(0, |m, e| m | 1 << e.InputSlot)
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Option<Vec<Input>> {
        let mut next = [0u32; D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT as usize];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot as usize;
                let format = e.Format.0 as u32;
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    // The input assembler caps natural element alignment at 4 bytes.
                    let alignment = bytes.min(4).next_power_of_two();
                    next[slot].checked_next_multiple_of(alignment)?
                } else {
                    e.AlignedByteOffset
                };
                next[slot] = offset.checked_add(bytes)?;
                Some(Input {
                    slot: e.InputSlot,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                })
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
            ] {
                for &slot in slots {
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso = match mesh_pipeline(&host.device, &root, code(0), code(1), g, &name) {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }

        let Some(layout) = &g.input_layout else {
            return refuse("invalid input layout", shaders);
        };
        let Some(inputs) = layout.elements() else {
            return refuse("input layout byte offset overflow", shaders);
        };
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: layout.inputs.as_ptr(),
                NumElements: layout.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1))],
            &state,
        );
        let made = super::library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: layout.slots(),
            inputs,
        })
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout or topology).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_subobject<T>(
        state: &mut GraphicsState,
        kind: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        data: T,
    ) {
        #[repr(C, align(8))]
        struct Subobject<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

        let subobject = Subobject(kind, data);
        // SAFETY: the test supplies the payload type for the given kind and
        // keeps any data it references alive for the call.
        unsafe { state.read(kind.0 as u32, (&raw const subobject).cast()) };
    }

    fn read_layout(state: &mut GraphicsState, elements: &[D3D12_INPUT_ELEMENT_DESC]) {
        read_subobject(
            state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
            D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: elements.as_ptr(),
                NumElements: elements.len() as u32,
            },
        );
    }

    fn input(slot: u32, format: DXGI_FORMAT, offset: u32) -> D3D12_INPUT_ELEMENT_DESC {
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(c"SYNTHETIC".as_ptr().cast()),
            Format: format,
            InputSlot: slot,
            AlignedByteOffset: offset,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        }
    }

    #[test]
    fn empty_input_layout_replaces_existing_elements() {
        let mut state = GraphicsState::default();
        let element = input(3, DXGI_FORMAT_R32_FLOAT, 0);
        for ptr in [std::ptr::null(), &raw const element] {
            read_layout(&mut state, &[element]);
            assert_eq!(state.input_layout.as_ref().unwrap().slots(), 1 << 3);

            read_subobject(
                &mut state,
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: ptr,
                    NumElements: 0,
                },
            );
            let layout = state.input_layout.as_ref().unwrap();
            assert_eq!(layout.slots(), 0);
            assert!(layout.elements().unwrap().is_empty());
            assert_eq!(state.describe(), GraphicsState::default().describe());
        }
    }

    #[test]
    fn oversized_input_layout_is_discarded() {
        let mut state = GraphicsState::default();
        let element = input(0, DXGI_FORMAT_R32_FLOAT, 0);
        let elements: [_; 33] = std::array::from_fn(|i| D3D12_INPUT_ELEMENT_DESC {
            SemanticIndex: i as u32,
            ..element
        });
        read_layout(&mut state, &elements[..32]);
        assert_eq!(state.input_layout.as_ref().unwrap().inputs.len(), 32);

        read_layout(&mut state, &elements);
        assert!(state.input_layout.is_none());
    }

    #[test]
    fn out_of_range_input_slot_is_discarded() {
        let mut state = GraphicsState::default();
        read_layout(&mut state, &[input(32, DXGI_FORMAT_R32_FLOAT, 0)]);

        assert!(state.input_layout.is_none());
    }

    #[test]
    fn nonempty_input_layout_requires_an_aligned_non_null_array() {
        let mut state = GraphicsState::default();
        let element = input(0, DXGI_FORMAT_R32_FLOAT, 0);
        let misaligned = (&raw const element).cast::<u8>().wrapping_add(1).cast();
        for ptr in [std::ptr::null(), misaligned] {
            read_layout(&mut state, &[element]);
            read_subobject(
                &mut state,
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: ptr,
                    NumElements: 1,
                },
            );
            assert!(state.input_layout.is_none());
        }
    }

    #[test]
    fn null_semantic_name_discards_the_layout() {
        let mut state = GraphicsState::default();
        let valid = input(0, DXGI_FORMAT_R32_FLOAT, 0);
        let invalid = D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR::null(),
            ..input(1, DXGI_FORMAT_R32_FLOAT, 0)
        };
        read_layout(&mut state, &[valid]);
        read_layout(&mut state, &[valid, invalid]);
        assert!(state.input_layout.is_none());
        assert_ne!(state.describe(), GraphicsState::default().describe());

        read_layout(&mut state, &[valid]);
        assert_eq!(state.input_layout.as_ref().unwrap().slots(), 1);
    }

    #[test]
    fn input_layout_owns_semantic_names_and_describes_their_values() {
        let mut state = GraphicsState::default();
        let element = D3D12_INPUT_ELEMENT_DESC {
            SemanticIndex: 7,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
            InstanceDataStepRate: 3,
            ..input(2, DXGI_FORMAT(119), 12)
        };
        {
            let name = CString::new("SYNTHETIC").unwrap();
            let source = D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                ..element
            };
            read_layout(&mut state, &[source]);
            assert_ne!(
                state.input_layout.as_ref().unwrap().inputs[0].SemanticName,
                source.SemanticName,
            );
        }
        let stored = state.input_layout.as_ref().unwrap().inputs[0];
        // SAFETY: the state still owns the copied string.
        let copied_name = unsafe { CStr::from_ptr(stored.SemanticName.0.cast()) };
        assert_eq!(copied_name, c"SYNTHETIC");
        assert_eq!(
            stored,
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: stored.SemanticName,
                Format: DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                ..element
            },
        );

        let mut other = GraphicsState::default();
        read_layout(&mut other, &[element]);
        assert_eq!(state.describe(), other.describe());
        read_layout(
            &mut other,
            &[D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"OTHER".as_ptr().cast()),
                ..element
            }],
        );
        assert_ne!(state.describe(), other.describe());
    }

    #[test]
    fn appended_inputs_respect_format_alignment_per_slot() {
        let mut state = GraphicsState::default();
        let append = D3D12_APPEND_ALIGNED_ELEMENT;
        let mut instance = input(31, DXGI_FORMAT_R16_UINT, append);
        instance.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
        instance.InstanceDataStepRate = 2;
        let mut instance_next = instance;
        instance_next.Format = DXGI_FORMAT_R32_FLOAT;
        read_layout(
            &mut state,
            &[
                input(0, DXGI_FORMAT_R8_UINT, 0),
                instance,
                input(0, DXGI_FORMAT_R16_UINT, append),
                instance_next,
                input(0, DXGI_FORMAT_R32G32B32_FLOAT, append),
                input(0, DXGI_FORMAT_R8_UINT, append),
                input(0, DXGI_FORMAT_R32_FLOAT, 24),
                input(0, DXGI_FORMAT_R32_FLOAT, append),
            ],
        );

        let layout = state.input_layout.as_ref().unwrap();
        assert_eq!(layout.slots(), 1 | (1 << 31));
        let elements = layout.elements().unwrap();
        let locations: Vec<_> = elements
            .iter()
            .map(|e| (e.slot, e.offset, e.format, e.per_instance))
            .collect();
        assert_eq!(
            locations,
            [
                (0, 0, 62, false),
                (31, 0, 57, true),
                (0, 2, 57, false),
                (31, 4, 41, true),
                (0, 4, 6, false),
                (0, 16, 62, false),
                (0, 24, 41, false),
                (0, 28, 41, false),
            ],
        );
    }

    #[test]
    fn input_offset_overflow_is_rejected() {
        let mut state = GraphicsState::default();
        read_layout(&mut state, &[input(0, DXGI_FORMAT_R32_FLOAT, u32::MAX - 3)]);
        assert!(state.input_layout.as_ref().unwrap().elements().is_none());

        read_layout(
            &mut state,
            &[
                input(0, DXGI_FORMAT_R8_UINT, u32::MAX - 1),
                input(0, DXGI_FORMAT_R32_FLOAT, D3D12_APPEND_ALIGNED_ELEMENT),
            ],
        );
        assert!(state.input_layout.as_ref().unwrap().elements().is_none());
    }

    #[test]
    fn omitted_subobjects_use_d3d12_defaults() {
        let state = GraphicsState::default();
        assert!(!state.blend.AlphaToCoverageEnable.as_bool());
        assert!(!state.blend.IndependentBlendEnable.as_bool());
        for target in state.blend.RenderTarget {
            assert!(!target.BlendEnable.as_bool());
            assert!(!target.LogicOpEnable.as_bool());
            assert_eq!(target.SrcBlend, D3D12_BLEND_ONE);
            assert_eq!(target.DestBlend, D3D12_BLEND_ZERO);
            assert_eq!(target.BlendOp, D3D12_BLEND_OP_ADD);
            assert_eq!(target.SrcBlendAlpha, D3D12_BLEND_ONE);
            assert_eq!(target.DestBlendAlpha, D3D12_BLEND_ZERO);
            assert_eq!(target.BlendOpAlpha, D3D12_BLEND_OP_ADD);
            assert_eq!(target.LogicOp, D3D12_LOGIC_OP_NOOP);
            assert_eq!(target.RenderTargetWriteMask, 0xf);
        }
        assert_eq!(state.sample_mask, u32::MAX);
        assert_eq!(state.rasterizer.FillMode, D3D12_FILL_MODE_SOLID);
        assert_eq!(state.rasterizer.CullMode, D3D12_CULL_MODE_BACK);
        assert!(!state.rasterizer.FrontCounterClockwise.as_bool());
        assert!(state.rasterizer.DepthClipEnable.as_bool());
        assert_eq!(state.rasterizer.DepthBias, 0);
        assert!(state.depth.DepthEnable.as_bool());
        assert_eq!(state.depth.DepthWriteMask, D3D12_DEPTH_WRITE_MASK_ALL);
        assert_eq!(state.depth.DepthFunc, D3D12_COMPARISON_FUNC_LESS);
        assert!(!state.depth.StencilEnable.as_bool());
        assert_eq!(state.depth.StencilReadMask, 0xff);
        assert_eq!(state.depth.StencilWriteMask, 0xff);
        for face in [state.depth.FrontFace, state.depth.BackFace] {
            assert_eq!(face.StencilFailOp, D3D12_STENCIL_OP_KEEP);
            assert_eq!(face.StencilDepthFailOp, D3D12_STENCIL_OP_KEEP);
            assert_eq!(face.StencilPassOp, D3D12_STENCIL_OP_KEEP);
            assert_eq!(face.StencilFunc, D3D12_COMPARISON_FUNC_ALWAYS);
        }
        assert_eq!(state.strip_cut, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED);
        assert_eq!(state.topology, D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED);
        assert_eq!(state.targets.NumRenderTargets, 0);
        assert_eq!(state.targets.RTFormats, [DXGI_FORMAT_UNKNOWN; 8]);
        assert_eq!(state.depth_format, DXGI_FORMAT_UNKNOWN);
        assert_eq!(
            state.samples,
            DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0
            }
        );
    }

    #[test]
    fn reads_blend_and_rasterizer_state() {
        let mut state = GraphicsState::default();
        let mut blend = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: true.into(),
            IndependentBlendEnable: true.into(),
            RenderTarget: state.blend.RenderTarget,
        };
        blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_SRC_ALPHA,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_REV_SUBTRACT,
            SrcBlendAlpha: D3D12_BLEND_DEST_ALPHA,
            DestBlendAlpha: D3D12_BLEND_INV_DEST_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_MAX,
            RenderTargetWriteMask: 0x5,
            ..blend.RenderTarget[0]
        };
        blend.RenderTarget[7].LogicOpEnable = true.into();
        blend.RenderTarget[7].LogicOp = D3D12_LOGIC_OP_XOR;
        let raster = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_WIREFRAME,
            CullMode: D3D12_CULL_MODE_FRONT,
            FrontCounterClockwise: true.into(),
            DepthBias: -7,
            DepthBiasClamp: 0.5,
            SlopeScaledDepthBias: -1.25,
            DepthClipEnable: false.into(),
            MultisampleEnable: true.into(),
            AntialiasedLineEnable: true.into(),
            ForcedSampleCount: 4,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON,
        };
        read_subobject(&mut state, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend);
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            raster,
        );
        assert_eq!(state.blend, blend);
        assert_eq!(state.rasterizer, raster);
    }

    #[test]
    fn reads_both_depth_stencil_subobjects() {
        let mut state = GraphicsState::default();
        let depth = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_GREATER,
            StencilEnable: true.into(),
            StencilReadMask: 0x31,
            StencilWriteMask: 0xe7,
            FrontFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_ZERO,
                StencilDepthFailOp: D3D12_STENCIL_OP_REPLACE,
                StencilPassOp: D3D12_STENCIL_OP_INCR_SAT,
                StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
            },
            BackFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_INVERT,
                StencilDepthFailOp: D3D12_STENCIL_OP_DECR,
                StencilPassOp: D3D12_STENCIL_OP_INCR,
                StencilFunc: D3D12_COMPARISON_FUNC_NOT_EQUAL,
            },
        };
        let depth1 = D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL,
            StencilEnable: false.into(),
            StencilReadMask: 0x52,
            StencilWriteMask: 0xa5,
            FrontFace: depth.BackFace,
            BackFace: depth.FrontFace,
            DepthBoundsTestEnable: true.into(),
        };
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            depth,
        );
        assert_eq!(state.depth, depth);
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
            depth1,
        );
        assert_eq!(
            state.depth,
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL,
                StencilEnable: false.into(),
                StencilReadMask: 0x52,
                StencilWriteMask: 0xa5,
                FrontFace: depth.BackFace,
                BackFace: depth.FrontFace,
            },
        );
    }

    #[test]
    fn translates_render_target_and_depth_formats() {
        let mut state = GraphicsState::default();
        let mut formats = [DXGI_FORMAT_R8G8B8A8_UNORM; 8];
        formats[6] = DXGI_FORMAT(119);
        formats[7] = DXGI_FORMAT(120);
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            D3D12_RT_FORMAT_ARRAY {
                RTFormats: formats,
                NumRenderTargets: 8,
            },
        );
        formats[6] = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        formats[7] = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
        assert_eq!(state.targets.NumRenderTargets, 8);
        assert_eq!(state.targets.RTFormats, formats);

        for (title, native) in [
            (118, DXGI_FORMAT_D24_UNORM_S8_UINT),
            (40, DXGI_FORMAT_D32_FLOAT),
            (0, DXGI_FORMAT_UNKNOWN),
        ] {
            read_subobject(
                &mut state,
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                DXGI_FORMAT(title),
            );
            assert_eq!(state.depth_format, native);
        }
    }

    #[test]
    fn reads_scalar_and_sample_state_and_ignores_unknown_subobjects() {
        let mut state = GraphicsState::default();
        let mut storage = [0u64; 2];
        let kind = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK.0 as u32;
        let at = storage.as_mut_ptr().cast::<u8>().wrapping_add(1);
        let mask = 0xfe01_1234u32;
        // SAFETY: both u32 writes are inside the storage at the stream's
        // type/data offsets. The read is of that whole subobject.
        unsafe {
            at.cast::<u32>().write_unaligned(kind);
            at.add(4).cast::<u32>().write_unaligned(mask);
            state.read(kind, at);
        }
        assert_eq!(state.sample_mask, mask);

        let strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
        let topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        let samples = DXGI_SAMPLE_DESC {
            Count: 4,
            Quality: 3,
        };
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
            strip_cut,
        );
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            topology,
        );
        read_subobject(
            &mut state,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
            samples,
        );
        assert_eq!(state.strip_cut, strip_cut);
        assert_eq!(state.topology, topology);
        assert_eq!(state.samples, samples);

        let before = state.describe();
        read_subobject(&mut state, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE(99), 0u32);
        assert_eq!(state.describe(), before);
    }

    #[test]
    fn shader_bytecode_is_copied_by_length() {
        assert!(code_bytes(&D3D12_SHADER_BYTECODE::default()).is_empty());
        let copy = {
            let bytes = [8u8, 0, 32, 91];
            let mut desc = D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: 0,
            };
            assert!(code_bytes(&desc).is_empty());
            desc.BytecodeLength = 3;
            code_bytes(&desc)
        };
        assert_eq!(copy, [8, 0, 32]);
    }
}
