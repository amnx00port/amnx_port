//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint do_compact; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer compact : register(u2);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
groupshared uint gs_keep[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    bool keep = false;
    uint at = 0;
    if (id.x < records) {
        at = id.x * stride;
        uint cost;
        if (kind <= 1) {
            // Draws: vertex or index count, then instances.
            uint count = min(args.Load(at), 1u << 24);
            uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
            cost = count * instances;
            args.Store2(at, uint2(count, instances));
        } else if (kind == 2) {
            uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
            // x * y fits 32 bits; the whole count may not.
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64;
            args.Store3(at, g);
        } else {
            // Mesh dispatches: the console's are one-dimensional.
            uint3 g = args.Load3(at);
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128;
            args.Store3(at, g);
        }
        uint before;
        budget.InterlockedAdd(0, cost, before);
        if (before >= BUDGET || before + cost > BUDGET) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0);
        } else if (do_compact != 0 && cost > 0) {
            keep = true;
        }
    }
    bool write = false;
    uint out_idx = 0u;
    if (do_compact != 0 && records <= 64u) {
        // Single-group batches preserve argument order in shared memory and avoid global atomic contention.
        gs_keep[gtid.x] = keep ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x == 0u) {
            uint total = 0u;
            [loop]
            for (uint k = 0u; k < records; k++) {
                total += gs_keep[k];
            }
            compact.Store(0, total);
        }
        if (keep) {
            [loop]
            for (uint k = 0u; k < gtid.x; k++) {
                out_idx += gs_keep[k];
            }
            write = true;
        }
    } else if (keep) {
        compact.InterlockedAdd(0, 1u, out_idx);
        write = true;
    }
    if (write) {
        uint dst_at = 16u + out_idx * stride;
        [loop]
        for (uint i = 0; i < stride; i += 4) {
            compact.Store(dst_at + i, args.Load(at + i));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

pub(crate) struct Compacted {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) args_offset: u64,
    pub(crate) count_offset: u64,
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, optionally compacting surviving commands into shared host scratch,
/// and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    enable_compaction: bool,
) -> Option<Option<Compacted>> {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let compaction = if enable_compaction {
        let needed: u64 = 16 + u64::from(records) * u64::from(stride);
        list.compaction_scratch(needed)
    } else {
        None
    };
    let (compact_res, compact_gpu, do_compact, compacted) = match compaction {
        Some((res, cg_gpu, offset, ptr)) => {
            // SAFETY: ptr is a non-null mapped shared UMA buffer allocation aligned to 256 bytes.
            unsafe {
                (ptr as *mut u32).write_volatile(0);
            }
            let info = Compacted {
                buffer: res.clone(),
                args_offset: offset + 16,
                count_offset: offset,
            };
            (Some(res), cg_gpu, 1u32, Some(info))
        }
        None => (None, budget, 0u32, None),
    };
    let l = &list.list;
    let constants: [u32; 4] = [kind, stride, records, do_compact];
    // SAFETY: an open list; our root signature, pipeline and live buffers the caller's list holds.
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, compact_gpu);
        l.Dispatch(records.div_ceil(64), 1, 1);
        let mut barriers: [D3D12_RESOURCE_BARRIER; 3] = [
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ];
        let barrier_slice: &[D3D12_RESOURCE_BARRIER] = if let Some(c_res) = &compact_res {
            barriers[2] = transition(
                c_res,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            );
            &barriers[..3]
        } else {
            &barriers[..2]
        };
        l.ResourceBarrier(barrier_slice);
    }
    Some(compacted)
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource, compacted: Option<&ID3D12Resource>) {
    // SAFETY: an open list and live buffers.
    unsafe {
        if let Some(c) = compacted {
            list.list.ResourceBarrier(&[
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                transition(
                    c,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        } else {
            list.list.ResourceBarrier(&[transition(
                buffer,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}
