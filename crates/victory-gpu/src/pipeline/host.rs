//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{CONST_VIEW_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
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
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
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
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
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
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
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
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
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

    /// Everything the host pipeline description holds, as text: part of a
    /// pipeline's name in the host library so cached PSOs always match the
    /// exact state passed to D3D12.
    fn host_describe(&self, use_depth_bounds: bool) -> String {
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
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {} {inputs:?} {} {} {:?} {} {:?}",
            host_blend(self),
            self.sample_mask,
            self.rasterizer,
            self.depth,
            use_depth_bounds,
            self.strip_cut.0,
            host_topology(self.topology).0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// Everything the state holds, as text: part of the key a title pipeline
    /// is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.depth_bounds,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out, aligning
    /// `D3D12_APPEND_ALIGNED_ELEMENT` to `min(4, element_bytes)` as D3D12 does.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                let align = bytes.clamp(1, 4);
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    let rem = next[slot] % align;
                    if rem == 0 {
                        next[slot]
                    } else {
                        next[slot].saturating_add(align - rem)
                    }
                } else {
                    e.AlignedByteOffset
                };
                next[slot] = offset.saturating_add(bytes);
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
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

/// How a stage's user-data block is bound in the host root signature:
/// inline 32-bit root constants normally, or a 2-DWORD root CBV when inline
/// constants would exceed D3D12's 64-DWORD root signature limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UserDataBinding {
    Constants { param: u32, dwords: u32 },
    RootCbv { param: u32, dwords: u32 },
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and whether it is inline or spilled.
    pub(crate) user_data: Option<UserDataBinding>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// Root parameters of 64-bit addresses carried in adjacent root constants:
    /// (title root parameter slot, low constant dword, our root parameter).
    pub(crate) const_cbvs: Vec<(u8, u32, u32)>,
    pub(crate) const_srvs: Vec<(u8, u32, u32)>,
    pub(crate) const_uavs: Vec<(u8, u32, u32)>,
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
        // An NGG geometry stage (`HwStage::Gs`) translates as a mesh stage (`ms_6_6`).
        let ngg_gs = find("geometry")
            .filter(|g| g.translated().is_some_and(|t| t.host.stage == victory_shader::HwStage::Gs));
        if compute.is_none()
            && shaders.iter().any(|s| {
                !matches!(s.stage, "vertex" | "pixel" | "mesh")
                    && !(s.stage == "geometry" && ngg_gs.is_some())
            })
        {
            return refuse(
                "only vertex, mesh, NGG geometry, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh").or(ngg_gs);
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
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)", shaders);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // Decide whether any stage's user-data root constants must spill to a
        // 2-DWORD root CBV to stay within D3D12's 64-DWORD root signature cap.
        const MAX_ROOT_DWORDS: u32 = 64;
        let stage_dwords: Vec<u32> = stages
            .iter()
            .map(|(s, _)| s.translated().unwrap().host.user_data_dwords())
            .collect();
        let fixed_dwords: u32 = stages
            .iter()
            .map(|(s, _)| {
                let t = &s.translated().unwrap().host;
                2 * (t.root_cbvs.len()
                    + t.root_srvs.len()
                    + t.root_uavs.len()
                    + t.const_cbvs.len()
                    + t.const_srvs.len()
                    + t.const_uavs.len()) as u32
            })
            .sum();
        let mut spilled = vec![false; stages.len()];
        let mut total_dwords = fixed_dwords + stage_dwords.iter().sum::<u32>();
        if total_dwords > MAX_ROOT_DWORDS {
            let mut order: Vec<usize> = (0..stages.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(stage_dwords[i]));
            for i in order {
                if total_dwords <= MAX_ROOT_DWORDS {
                    break;
                }
                if stage_dwords[i] > 2 {
                    spilled[i] = true;
                    total_dwords -= stage_dwords[i] - 2;
                }
            }
        }

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for (stage_index, &(shader, visibility)) in stages.iter().enumerate() {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                const_cbvs: Vec::new(),
                const_srvs: Vec::new(),
                const_uavs: Vec::new(),
            };
            let dwords = stage_dwords[stage_index];
            if dwords > 0 {
                let param = params.len() as u32;
                if spilled[stage_index] {
                    stage.user_data = Some(UserDataBinding::RootCbv { param, dwords });
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: 0,
                                RegisterSpace: USER_DATA_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                } else {
                    stage.user_data = Some(UserDataBinding::Constants { param, dwords });
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
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
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
            for (views, kind, out) in [
                (
                    &t.const_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.const_cbvs,
                ),
                (
                    &t.const_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.const_srvs,
                ),
                (
                    &t.const_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.const_uavs,
                ),
            ] {
                for &cv in views {
                    out.push((cv.slot, cv.dword, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: cv.register,
                                RegisterSpace: CONST_VIEW_SPACE,
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
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, compute.is_some()) {
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
                    let wave_size = stages
                        .first()
                        .and_then(|(s, _)| s.translated())
                        .and_then(|t| t.host.wave_size);
                    tracing::warn!(
                        error = %e,
                        ?shaders,
                        ?wave_size,
                        max_wave = host.wave_lane_count_max(),
                        "host pipeline refused"
                    );
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
        let use_depth_bounds = g.depth_bounds && host.depth_bounds();
        let state = g.host_describe(use_depth_bounds);
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso = match mesh_pipeline(
                &host.device,
                &root,
                code(0),
                code(1),
                g,
                use_depth_bounds,
                &name,
            ) {
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

        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1))],
            &state,
        );
        let made = if use_depth_bounds {
            graphics_pipeline_depth_bounds(&host.device, &root, code(0), code(1), g, &name)
        } else {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                VS: code(0),
                PS: code(1),
                BlendState: host_blend(g),
                SampleMask: g.sample_mask,
                RasterizerState: g.rasterizer,
                DepthStencilState: g.depth,
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: g.inputs.as_ptr(),
                    NumElements: g.inputs.len() as u32,
                },
                IBStripCutValue: g.strip_cut,
                // The console's rectangle and quad types draw as triangles on
                // the host (see `draw`).
                PrimitiveTopologyType: host_topology(g.topology),
                NumRenderTargets: g.targets.NumRenderTargets,
                RTVFormats: g.targets.RTFormats,
                DSVFormat: g.depth_format,
                SampleDesc: g.samples,
                ..Default::default()
            };
            super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
            )
        };
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
            slots: g.slots(),
            inputs: g.elements(),
        })
    }
}

/// Whether `format` is an integer (`*_UINT` or `*_SINT`) color format that
/// D3D12 forbids blending on.
fn is_integer_format(format: DXGI_FORMAT) -> bool {
    matches!(
        format.0,
        3 | 4 | 7 | 8 | 12 | 14 | 17 | 18 | 25 | 30 | 32 | 36 | 38 | 42 | 43 | 50 | 52 | 57 | 59
            | 62 | 64
    )
}

/// Host blend state for `g`: D3D12 rejects pipelines with `BlendEnable` set on
/// an integer render target, even when the console ignores blending on them.
/// When any active integer target would blend, expand target 0 into all eight
/// targets if `IndependentBlendEnable` is off and turn `BlendEnable` off on
/// each active integer target.
fn host_blend(g: &GraphicsState) -> D3D12_BLEND_DESC {
    let count = g.targets.NumRenderTargets.min(8) as usize;
    let independent = g.blend.IndependentBlendEnable.as_bool();
    let needs_fix = (0..count).any(|i| {
        is_integer_format(g.targets.RTFormats[i])
            && g.blend.RenderTarget[if independent { i } else { 0 }]
                .BlendEnable
                .as_bool()
    });
    if !needs_fix {
        return g.blend;
    }
    let mut blend = g.blend;
    if !independent {
        blend.RenderTarget = [blend.RenderTarget[0]; 8];
        blend.IndependentBlendEnable = true.into();
    }
    for (target, &format) in blend.RenderTarget[..count]
        .iter_mut()
        .zip(&g.targets.RTFormats[..count])
    {
        if is_integer_format(format) {
            target.BlendEnable = false.into();
        }
    }
    blend
}

fn host_topology(topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match topology.0 {
        RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => topology,
    }
}

/// A stream subobject: its type, then its description, pointer-aligned.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

fn depth_stencil1(g: &GraphicsState, enabled: bool) -> D3D12_DEPTH_STENCIL_DESC1 {
    D3D12_DEPTH_STENCIL_DESC1 {
        DepthEnable: g.depth.DepthEnable,
        DepthWriteMask: g.depth.DepthWriteMask,
        DepthFunc: g.depth.DepthFunc,
        StencilEnable: g.depth.StencilEnable,
        StencilReadMask: g.depth.StencilReadMask,
        StencilWriteMask: g.depth.StencilWriteMask,
        FrontFace: g.depth.FrontFace,
        BackFace: g.depth.BackFace,
        DepthBoundsTestEnable: enabled.into(),
    }
}

/// A graphics pipeline in stream form when the title enables depth bounds.
fn graphics_pipeline_depth_bounds(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    vs: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        vs: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC1>,
        layout: Sub<D3D12_INPUT_LAYOUT_DESC>,
        strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let topology = host_topology(g.topology);
    let mut stream = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(g)),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
            depth_stencil1(g, true),
        ),
        layout: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
            D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
        ),
        strip_cut: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, g.strip_cut),
        topology: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, topology),
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
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

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout or topology).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    depth_bounds: bool,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    let device = device.cast::<ID3D12Device2>()?;
    if depth_bounds {
        #[repr(C)]
        struct Stream1 {
            root: Sub<*mut std::ffi::c_void>,
            ms: Sub<D3D12_SHADER_BYTECODE>,
            ps: Sub<D3D12_SHADER_BYTECODE>,
            blend: Sub<D3D12_BLEND_DESC>,
            mask: Sub<u32>,
            raster: Sub<D3D12_RASTERIZER_DESC>,
            depth: Sub<D3D12_DEPTH_STENCIL_DESC1>,
            targets: Sub<D3D12_RT_FORMAT_ARRAY>,
            dsv: Sub<DXGI_FORMAT>,
            samples: Sub<DXGI_SAMPLE_DESC>,
        }
        let mut stream = Stream1 {
            root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
            ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(g)),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
            raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
            depth: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
                depth_stencil1(g, true),
            ),
            targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
            dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
            samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<Stream1>(),
            pPipelineStateSubobjectStream: (&raw mut stream).cast(),
        };
        return super::library::pipeline(
            name,
            // SAFETY: the stream holds whole subobjects pointing at live data.
            |library, name| unsafe { library.LoadPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { device.CreatePipelineState(&desc) },
        );
    }
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
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(g)),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
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
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }.to_vec()
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
    let direct_flags = input
        | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    let make_desc = |flags| D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: flags,
            },
        },
    };
    let serialize = |flags| -> windows::core::Result<(ID3DBlob, Option<String>)> {
        let desc = make_desc(flags);
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        // SAFETY: the description points at live arrays; any error blob is
        // copied to a `String` before release.
        let result =
            unsafe { D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) };
        let message = error.map(|error| {
            // SAFETY: a valid blob returned by `D3D12SerializeVersionedRootSignature`.
            let text = unsafe {
                std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                )
            };
            String::from_utf8_lossy(text).into_owned()
        });
        match result {
            Ok(()) => Ok((blob.ok_or_else(windows::core::Error::empty)?, message)),
            Err(e) => {
                if let Some(msg) = message {
                    tracing::warn!("root signature: {msg}");
                }
                Err(e)
            }
        }
    };
    // Older Windows 10 inbox `d3d12core.dll` (pre-Agility SDK / SM 6.6)
    // rejects the direct heap-indexing flags (`0xc00`). Retry without them
    // so pipelines that do not index the heaps directly can still be built.
    let (blob, _) = match serialize(direct_flags) {
        Ok(ok) => ok,
        Err(_) => {
            static FALLBACK: std::sync::Once = std::sync::Once::new();
            FALLBACK.call_once(|| {
                tracing::warn!(
                    "root signature: retrying without HEAP_DIRECTLY_INDEXED flags \
                     (older Windows 10 D3D12 runtime)"
                );
            });
            serialize(input)?
        }
    };
    // SAFETY: `blob` is valid for its reported size; the bytes are read in
    // full before `blob` is released.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(format: u32, offset: u32) -> D3D12_INPUT_ELEMENT_DESC {
        D3D12_INPUT_ELEMENT_DESC {
            Format: DXGI_FORMAT(format as i32),
            AlignedByteOffset: offset,
            ..Default::default()
        }
    }

    #[test]
    fn appended_elements_are_aligned() {
        // R8G8_UNORM (49, 2B) @ 0, then R32G32_FLOAT (16, 8B) appended -> aligned to 4.
        let g = GraphicsState {
            inputs: vec![
                elem(49, D3D12_APPEND_ALIGNED_ELEMENT),
                elem(16, D3D12_APPEND_ALIGNED_ELEMENT),
            ],
            ..Default::default()
        };
        let offsets: Vec<u32> = g.elements().into_iter().map(|e| e.offset).collect();
        assert_eq!(offsets, vec![0, 4]);

        // R8_UNORM (61, 1B), R16_FLOAT (54, 2B), R8_UNORM (61, 1B) -> 0, 2, 4.
        let g2 = GraphicsState {
            inputs: vec![
                elem(61, D3D12_APPEND_ALIGNED_ELEMENT),
                elem(54, D3D12_APPEND_ALIGNED_ELEMENT),
                elem(61, D3D12_APPEND_ALIGNED_ELEMENT),
            ],
            ..Default::default()
        };
        let offsets2: Vec<u32> = g2.elements().into_iter().map(|e| e.offset).collect();
        assert_eq!(offsets2, vec![0, 2, 4]);
    }

    #[test]
    fn huge_explicit_offset_does_not_overflow() {
        let g = GraphicsState {
            inputs: vec![
                elem(16, u32::MAX - 2),
                elem(16, D3D12_APPEND_ALIGNED_ELEMENT),
            ],
            ..Default::default()
        };
        let elems = g.elements();
        assert_eq!(elems[0].offset, u32::MAX - 2);
        assert_eq!(elems[1].offset, u32::MAX);
    }

    #[test]
    fn integer_targets_do_not_blend() {
        // Non-independent blend with target 0 enabled, mixed UNORM + UINT + SINT targets:
        // target 0 must expand across targets, IndependentBlendEnable turns on,
        // and only active integer targets lose BlendEnable.
        let mut g = GraphicsState::default();
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.blend.RenderTarget[0].RenderTargetWriteMask = 0x7;
        g.targets.NumRenderTargets = 3;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.targets.RTFormats[1] = DXGI_FORMAT_R32_UINT;
        g.targets.RTFormats[2] = DXGI_FORMAT_R16G16_SINT;

        let fixed = host_blend(&g);
        assert!(fixed.IndependentBlendEnable.as_bool());
        assert!(fixed.RenderTarget[0].BlendEnable.as_bool());
        assert!(!fixed.RenderTarget[1].BlendEnable.as_bool());
        assert!(!fixed.RenderTarget[2].BlendEnable.as_bool());
        assert_eq!(fixed.RenderTarget[1].RenderTargetWriteMask, 0x7);
        assert_eq!(fixed.RenderTarget[2].RenderTargetWriteMask, 0x7);

        // Independent blend enabled: only integer active targets lose BlendEnable.
        let mut g2 = GraphicsState::default();
        g2.blend.IndependentBlendEnable = true.into();
        g2.blend.RenderTarget[0].BlendEnable = true.into();
        g2.blend.RenderTarget[1].BlendEnable = true.into();
        g2.targets.NumRenderTargets = 2;
        g2.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UINT;
        g2.targets.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;

        let fixed2 = host_blend(&g2);
        assert!(fixed2.IndependentBlendEnable.as_bool());
        assert!(!fixed2.RenderTarget[0].BlendEnable.as_bool());
        assert!(fixed2.RenderTarget[1].BlendEnable.as_bool());
    }

    #[test]
    fn blend_states_without_integer_blending_are_kept() {
        let mut g = GraphicsState::default();
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // Inactive slot holds an integer format — must not alter the blend state.
        g.targets.RTFormats[1] = DXGI_FORMAT_R32_UINT;

        let kept = host_blend(&g);
        assert!(!kept.IndependentBlendEnable.as_bool());
        assert!(kept.RenderTarget[0].BlendEnable.as_bool());

        // Active integer target with blending already off is left untouched.
        let mut g2 = GraphicsState::default();
        g2.blend.IndependentBlendEnable = false.into();
        g2.blend.RenderTarget[0].BlendEnable = false.into();
        g2.targets.NumRenderTargets = 1;
        g2.targets.RTFormats[0] = DXGI_FORMAT_R32_SINT;

        let kept2 = host_blend(&g2);
        assert!(!kept2.IndependentBlendEnable.as_bool());
        assert!(!kept2.RenderTarget[0].BlendEnable.as_bool());
    }
}
