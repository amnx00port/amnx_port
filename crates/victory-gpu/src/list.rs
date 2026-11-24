//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Vec<u8>>>,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    _pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    _pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, iid, out)
}

fn new_list(kind: u32, allocator: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: crate::host::host().and_then(|h| match h.new_list() {
            Ok(list) => Some(list),
            Err(e) => {
                tracing::warn!(error = %e, "no host command list");
                None
            }
        }),
        #[cfg(windows)]
        state: Default::default(),
    });
    object::hand_out(LIST, list, iid, &LIST_IIDS, out)
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // Every submission of the list has run (submissions are waited for).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).clear();
    #[cfg(windows)]
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    #[cfg(not(windows))]
    let _ = (this, pipeline);
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: host memory copied at submission, or GPU-to-GPU when an
/// earlier tracked GPU write in this list covers the source range (#785).
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        if let Some(src_shadow) = list.latest_gpu_write_shadow(from as usize, len as usize)
            && let Some(dst_shadow) = crate::shadow::find(to as u64, len)
        {
            dst_shadow.refresh(list, to as u64, len);
            list.ensure_buffer(&src_shadow, COPY_SOURCE);
            list.ensure_buffer(&dst_shadow, COPY_DEST);
            // SAFETY: an open list; both shadow buffers cover the ranges.
            unsafe {
                list.list.CopyBufferRegion(
                    &dst_shadow.buffer,
                    dst_shadow.offset(to as u64),
                    &src_shadow.buffer,
                    src_shadow.offset(from as u64),
                    len,
                );
            }
            dst_shadow.gpu_writes(list, to as u64, len);
            return;
        }
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_written(dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("texture to buffer copies not implemented"));
            return;
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            const PLACEMENT_ALIGN: u64 = 512;
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            if let Some(shadow) = list.latest_gpu_write_shadow(data as usize, bytes as usize) {
                use windows::Win32::Graphics::Direct3D12::{
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                };
                // A tracked same-list GPU write covers this footprint: read
                // directly from the host shadow buffer instead of snapshotting
                // stale CPU bytes, rebasing unaligned offsets via a GPU copy (#785).
                let src_offset = shadow.offset(data as u64);
                list.ensure_buffer(&shadow, COPY_SOURCE);
                let src_buf = if src_offset.is_multiple_of(PLACEMENT_ALIGN) {
                    from.resource = shadow.buffer.as_raw().cast_const().cast();
                    from.footprint.offset = src_offset;
                    list.keep(&shadow.buffer);
                    shadow.buffer.clone()
                } else {
                    let Some(rebased) =
                        host.default_buffer(bytes, D3D12_RESOURCE_STATE_COPY_DEST)
                    else {
                        return;
                    };
                    // SAFETY: an open list; both buffers cover `bytes` at the given offsets.
                    unsafe {
                        list.list
                            .CopyBufferRegion(&rebased, 0, &shadow.buffer, src_offset, bytes);
                        list.list.ResourceBarrier(&[crate::host::transition(
                            &rebased,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                    }
                    list.keep(&rebased);
                    from.resource = rebased.as_raw().cast_const().cast();
                    from.footprint.offset = 0;
                    rebased
                };
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live buffer.
                    read = Some((
                        unsafe { src_buf.GetGPUVirtualAddress() } + from.footprint.offset,
                        bytes,
                    ));
                }
                _upload = src_buf;
            } else {
                let Some((buffer, staging)) = list.upload(host, bytes) else {
                    return;
                };
                // The list owns this mapped D3D12 upload buffer. Wine can report
                // its address as MEM_FREE, so use the host-upload path.
                list.defer_fill(staging, data, bytes as usize);
                from.resource = buffer.as_raw().cast_const().cast();
                from.footprint.offset = 0;
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live buffer.
                    read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
                }
                _upload = buffer;
            }
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_written(dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_written(dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            let dst_mip = dst_sub % u32::from(a.MipLevels.max(1));
            let src_mip = src_sub % u32::from(b.MipLevels.max(1));
            let (aw, ah) = ((a.Width >> dst_mip).max(1), (a.Height >> dst_mip).max(1));
            let (bw, bh) = ((b.Width >> src_mip).max(1), (b.Height >> src_mip).max(1));
            if (aw, ah) == (bw, bh) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: aw.min(bw) as i32,
                    bottom: ah.min(bh) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

#[cfg(windows)]
fn prepare_texture_uav_clear(
    list: &crate::host::HostList,
    state: &mut crate::draw::State,
    gpu_handle: u64,
    cpu_handle: u64,
    resource: This,
    has_rects: bool,
) -> Option<(
    windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE,
    windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE,
    &'static windows::Win32::Graphics::Direct3D12::ID3D12Resource,
)> {
    const UNORDERED_ACCESS: u32 = 0x8;
    let host = crate::host::host()?;
    let crate::descriptor::View::Unordered(view_resource, _counter, raw_desc) =
        crate::descriptor::view(cpu_handle).or_else(|| crate::descriptor::view(gpu_handle))?
    else {
        return None;
    };
    let resource_ptr = if !resource.is_null() {
        resource
    } else {
        view_resource as This
    };
    let texture = crate::resource::host_resource(resource_ptr)?;
    let desc = crate::descriptor::host::texture_uav_desc(resource_ptr, texture, raw_desc.as_ref())?;
    let (heap, index) = crate::descriptor::shader_slot(gpu_handle)?;
    let host_heap = heap.host()?;
    let heap_ptr = std::ptr::from_ref(heap) as usize;
    let slot = if index < host_heap.count() {
        index
    } else {
        let slot = host_heap.dynamic(1).or_else(|| {
            crate::host::reclaim_dynamic_slots(heap_ptr);
            host_heap.dynamic(1)
        })?;
        list.record_dynamic(heap_ptr, slot, 1);
        slot
    };
    host_heap.uav_at(slot, texture, desc.as_ref());
    let gpu_desc = host_heap.gpu(slot);
    let cpu_desc = host.uav_cpu(texture, desc.as_ref());
    state.ensure_view_heap(list, heap_ptr);
    if has_rects && crate::texture::title_written(texture) {
        crate::texture::upload(list, resource_ptr, texture);
    }
    ensure_state(list, resource_ptr, UNORDERED_ACCESS);
    crate::texture::host_written(resource_ptr);
    Some((gpu_desc, cpu_desc, texture))
}

/// `ClearUnorderedAccessViewUint(gpu_handle, cpu_handle, resource, values, count, rects)`
/// for resource-backed textures (#771).
#[allow(clippy::too_many_arguments)] // the D3D12 method's own arguments
extern "system" fn clear_uav_uint(
    this: This,
    gpu_handle: u64,
    cpu_handle: u64,
    resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        // SAFETY: the title passes four u32 values (null is tolerated).
        let Some(values) = (unsafe { values.as_ref() }) else {
            return;
        };
        let rects = if count > 0 && !rects.is_null() {
            // SAFETY: the title passes `count` rectangles laid out as `RECT`.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        } else {
            &[]
        };
        with_state(this, |list, state| {
            let Some((gpu_desc, cpu_desc, texture)) = prepare_texture_uav_clear(
                list,
                state,
                gpu_handle,
                cpu_handle,
                resource,
                !rects.is_empty(),
            ) else {
                return;
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("clear uav uint {resource:?} {values:?} rects {}", rects.len()));
            // SAFETY: an open list, matching shader-visible and CPU-only UAV
            // descriptors, and a live texture in UNORDERED_ACCESS state.
            unsafe {
                list.list
                    .ClearUnorderedAccessViewUint(gpu_desc, cpu_desc, texture, values, rects);
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, gpu_handle, cpu_handle, resource, values, count, rects);
}

/// `ClearUnorderedAccessViewFloat(gpu_handle, cpu_handle, resource, values, count, rects)`
/// for resource-backed textures (#771).
#[allow(clippy::too_many_arguments)] // the D3D12 method's own arguments
extern "system" fn clear_uav_float(
    this: This,
    gpu_handle: u64,
    cpu_handle: u64,
    resource: This,
    values: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        // SAFETY: the title passes four f32 values (null is tolerated).
        let Some(values) = (unsafe { values.as_ref() }) else {
            return;
        };
        let rects = if count > 0 && !rects.is_null() {
            // SAFETY: the title passes `count` rectangles laid out as `RECT`.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        } else {
            &[]
        };
        with_state(this, |list, state| {
            let Some((gpu_desc, cpu_desc, texture)) = prepare_texture_uav_clear(
                list,
                state,
                gpu_handle,
                cpu_handle,
                resource,
                !rects.is_empty(),
            ) else {
                return;
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("clear uav float {resource:?} {values:?} rects {}", rects.len()));
            // SAFETY: an open list, matching shader-visible and CPU-only UAV
            // descriptors, and a live texture in UNORDERED_ACCESS state.
            unsafe {
                list.list
                    .ClearUnorderedAccessViewFloat(gpu_desc, cpu_desc, texture, values, rects);
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, gpu_handle, cpu_handle, resource, values, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery(heap, type, index)`: forwards to the host `ID3D12QueryHeap`
/// when available.
extern "system" fn begin_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some((host_heap, count)) = crate::query::host_heap(heap)
        && let Some((host_kind, _)) = crate::query::query_type_info(kind)
        && index < count
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
        // SAFETY: open host command list and live query heap with in-bounds index.
        unsafe {
            list.list
                .BeginQuery(&host_heap, D3D12_QUERY_TYPE(host_kind), index)
        };
        list.keep(&host_heap);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `EndQuery(heap, type, index)`: forwards to the host `ID3D12QueryHeap`
/// when available.
extern "system" fn end_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some((host_heap, count)) = crate::query::host_heap(heap)
        && let Some((host_kind, _)) = crate::query::query_type_info(kind)
        && index < count
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
        // SAFETY: open host command list and live query heap with in-bounds index.
        unsafe {
            list.list
                .EndQuery(&host_heap, D3D12_QUERY_TYPE(host_kind), index)
        };
        list.keep(&host_heap);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: resolves real
/// GPU query results from the host `ID3D12QueryHeap` into a readback buffer
/// and copies them into title memory once the submission completes, falling
/// back to plausible synthetic values when headless or for unsupported query
/// types.
extern "system" fn resolve_queries(
    this: This,
    heap: This,
    kind: u32,
    first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    if count == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(host) = crate::host::host()
        && let Some((host_heap, heap_count)) = crate::query::host_heap(heap)
        && let Some((host_kind, stride)) = crate::query::query_type_info(kind)
        && first.checked_add(count).is_some_and(|end| end <= heap_count)
        && let Some(total) = u64::from(count).checked_mul(stride)
        && offset.checked_add(total).is_some_and(|end| end <= size)
        && let Some((rb_buf, rb_ptr)) = host.readback_buffer(total)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
        // SAFETY: open list, live query heap, and readback buffer large enough for `total` bytes.
        unsafe {
            list.list.ResolveQueryData(
                &host_heap,
                D3D12_QUERY_TYPE(host_kind),
                first,
                count,
                &rb_buf,
                0,
            );
        }
        list.keep(&host_heap);
        list.keep(&rb_buf);
        let dst = data as usize + offset as usize;
        let src = rb_ptr as usize;
        let len = total as usize;
        list.defer_after(move || {
            crate::watch::writing(dst, len);
            crate::host::write_mapped(dst as *mut u8, src as *const u8, len);
        });
        return;
    }
    #[cfg(not(windows))]
    let _ = (heap, first);
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset + u64::from(count) * bytes > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(_: This, _flags: u32, _control: u32, _address: u64, _size: u64) {}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// List-owned upload memory: `size` zeroed bytes aligned to `alignment` (a
/// power of two, or `0` for the default 16-byte alignment), kept alive until
/// the list is reset; null on invalid alignment, overflow, or OOM (#837).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    if this.is_null() {
        return std::ptr::null_mut();
    }
    let align = if alignment == 0 {
        16usize
    } else {
        let Ok(raw) = usize::try_from(alignment) else {
            return std::ptr::null_mut();
        };
        if !raw.is_power_of_two() {
            return std::ptr::null_mut();
        }
        raw.max(16)
    };
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    let mut block = Vec::new();
    if block.try_reserve_exact(len).is_err() {
        return std::ptr::null_mut();
    }
    block.resize(len, 0);
    let start = block.as_mut_ptr();
    let offset = start.align_offset(align);
    if offset >= align {
        return std::ptr::null_mut();
    }
    // SAFETY: `offset < align`, so `size` bytes remain inside `block`.
    let at = unsafe { start.add(offset) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    // Moving the vector keeps its heap block where it is.
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// `ID3D12GraphicsCommandList::ClearState(pPipelineState)` (slot 11 / 0x58, #848).
extern "system" fn clear_state(this: This, pipeline: This) {
    if !pipeline.is_null() {
        crate::pipeline::bound(pipeline);
    }
    #[cfg(windows)]
    with_state(this, |list, state| {
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("clear state {pipeline:?}"));
        state.clear(list, pipeline);
    });
    #[cfg(not(windows))]
    let _ = (this, pipeline);
}

/// ABI-compatible layout of `D3D12_DISCARD_REGION`.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct DiscardRegion {
    num_rects: u32,
    rects: *const windows::Win32::Foundation::RECT,
    first_subresource: u32,
    num_subresources: u32,
}

#[cfg(windows)]
const _: () = {
    use windows::Win32::Graphics::Direct3D12::D3D12_DISCARD_REGION;
    assert!(size_of::<DiscardRegion>() == size_of::<D3D12_DISCARD_REGION>());
    assert!(align_of::<DiscardRegion>() == align_of::<D3D12_DISCARD_REGION>());
};

/// Validates a `DiscardRegion` against a host texture description (#835).
#[cfg(windows)]
fn valid_discard_region(
    desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    planes: u32,
    region: &DiscardRegion,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    };
    if region.num_rects > 0 && region.rects.is_null() {
        return false;
    }
    let mips = u32::from(desc.MipLevels).max(1);
    let array = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize).max(1)
    };
    let Some(total) = mips
        .checked_mul(array)
        .and_then(|n| n.checked_mul(planes.max(1)))
    else {
        return false;
    };
    if region.num_subresources > 0 {
        let Some(end) = region.first_subresource.checked_add(region.num_subresources) else {
            return false;
        };
        if end > total {
            return false;
        }
    }
    if region.num_rects > 0 {
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            return false;
        }
        // D3D12 requires all subresources in a rect-based discard to share the
        // same mip level (so rect extents match).
        if region.num_subresources > 1 && mips > 1 {
            let mip0 = region.first_subresource % mips;
            for s in region.first_subresource + 1..region.first_subresource + region.num_subresources {
                if s % mips != mip0 {
                    return false;
                }
            }
        }
        // SAFETY: `region.num_rects > 0` and `region.rects` is non-null.
        let rects = unsafe { std::slice::from_raw_parts(region.rects, region.num_rects as usize) };
        if rects.iter().any(|r| r.left > r.right || r.top > r.bottom) {
            return false;
        }
    }
    true
}

/// Whether `state` is a D3D12-legal discard state matching `flags` (#835).
#[cfg(windows)]
fn discard_state_legal(state: u32, flags: u32) -> bool {
    const RENDER_TARGET: u32 = 0x4;
    const UNORDERED_ACCESS: u32 = 0x8;
    const DEPTH_WRITE: u32 = 0x10;
    const FLAG_RT: u32 = 1;
    const FLAG_DS: u32 = 2;
    const FLAG_UAV: u32 = 4;
    (state == RENDER_TARGET && flags & FLAG_RT != 0)
        || (state == UNORDERED_ACCESS && flags & FLAG_UAV != 0)
        || (state == DEPTH_WRITE && flags & FLAG_DS != 0)
}

/// `ID3D12GraphicsCommandList::DiscardResource(pResource, pRegion)` (slot 55 / 0x1b8, #835).
extern "system" fn discard_resource(this: This, resource: This, region: *const std::ffi::c_void) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_DISCARD_REGION, D3D12_FEATURE_DATA_FORMAT_INFO, D3D12_FEATURE_FORMAT_INFO,
        };
        let (Some(list), Some(host), Some(texture)) = (
            host_list(this),
            crate::host::host(),
            crate::resource::host_resource(resource),
        ) else {
            return;
        };
        let Some(state) = list.tracked_texture_state(resource) else {
            return;
        };
        // SAFETY: a live host texture.
        let desc = unsafe { texture.GetDesc() };
        if !discard_state_legal(state, desc.Flags.0 as u32) {
            return;
        }
        let region_ptr = if region.is_null() {
            None
        } else {
            // SAFETY: caller passes a `D3D12_DISCARD_REGION` pointer.
            let r = unsafe { &*region.cast::<DiscardRegion>() };
            let mut fmt_info = D3D12_FEATURE_DATA_FORMAT_INFO {
                Format: desc.Format,
                PlaneCount: 0,
            };
            // SAFETY: plain device capability query.
            let _ = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_INFO,
                    std::ptr::from_mut(&mut fmt_info).cast(),
                    size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>() as u32,
                )
            };
            if !valid_discard_region(&desc, u32::from(fmt_info.PlaneCount).max(1), r) {
                return;
            }
            Some(region.cast::<D3D12_DISCARD_REGION>())
        };
        list.keep(texture);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("discard resource {resource:?}"));
        // SAFETY: open list, live texture in verified discard-legal state, validated region.
        unsafe { list.list.DiscardResource(texture, region_ptr) };
    }
    #[cfg(not(windows))]
    let _ = (this, resource, region);
}

/// `ID3D12GraphicsCommandList::SetPredication(pBuffer, AlignedBufferOffset, Operation)` (slot 31 / 0xf8, #859).
extern "system" fn set_predication(this: This, buffer: This, offset: u64, operation: u32) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_PREDICATION_OP, D3D12_PREDICATION_OP_EQUAL_ZERO, D3D12_RESOURCE_STATE_PREDICATION,
            ID3D12Resource,
        };
        let Some(list) = host_list(this) else {
            return;
        };
        if buffer.is_null() {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| "set predication off".to_string());
            // SAFETY: open list; null disables predication.
            unsafe {
                list.list
                    .SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
            }
            return;
        }
        if operation > 1 || !offset.is_multiple_of(8) {
            return;
        }
        let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
            return;
        };
        if offset.checked_add(8).is_none_or(|end| end > size) {
            return;
        }
        let addr = data as u64 + offset;
        let op = D3D12_PREDICATION_OP(operation as i32);
        if let Some(shadow) = list.latest_gpu_write_shadow(addr as usize, 8) {
            let off = shadow.offset(addr);
            if !off.is_multiple_of(8) {
                return;
            }
            list.ensure_buffer(&shadow, D3D12_RESOURCE_STATE_PREDICATION.0 as u32);
            list.keep(&shadow.buffer);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("set predication {buffer:?}+{offset:#x} op {operation} (gpu)"));
            // SAFETY: open list, live shadow buffer in PREDICATION state, 8-byte aligned offset.
            unsafe { list.list.SetPredication(&shadow.buffer, off, op) };
        } else if let Some(shadow) = crate::shadow::find(addr, 8) {
            let off = shadow.offset(addr);
            if !off.is_multiple_of(8) {
                return;
            }
            shadow.refresh(list, addr, 8);
            list.ensure_buffer(&shadow, D3D12_RESOURCE_STATE_PREDICATION.0 as u32);
            list.keep(&shadow.buffer);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("set predication {buffer:?}+{offset:#x} op {operation}"));
            // SAFETY: open list, live shadow buffer in PREDICATION state, 8-byte aligned offset.
            unsafe { list.list.SetPredication(&shadow.buffer, off, op) };
        } else if let Some(host) = crate::host::host()
            && let Some((upload, ptr)) = list.upload(host, 8)
        {
            list.defer_fill(ptr, addr as *const u8, 8);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("set predication {buffer:?}+{offset:#x} op {operation} (upload)"));
            // SAFETY: open list, list-owned upload buffer (GENERIC_READ includes PREDICATION).
            unsafe { list.list.SetPredication(&upload, 0, op) };
        }
    }
    #[cfg(not(windows))]
    let _ = (this, buffer, offset, operation);
}

/// `ID3D12GraphicsCommandList::SetMarker(Metadata, pData, Size)` (slot 56 / 0x1c0, #859).
extern "system" fn set_marker(this: This, metadata: u32, data: *const std::ffi::c_void, size: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        let (ptr, len) = if data.is_null() || size == 0 {
            (None, 0)
        } else {
            (Some(data), size)
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("set marker meta={metadata} size={len}"));
        // SAFETY: open list; null/zero payload normalized.
        unsafe { list.list.SetMarker(metadata, ptr, len) };
    }
    #[cfg(not(windows))]
    let _ = (this, metadata, data, size);
}

/// `ID3D12GraphicsCommandList::BeginEvent(Metadata, pData, Size)` (slot 57 / 0x1c8, #859).
extern "system" fn begin_event(
    this: This,
    metadata: u32,
    data: *const std::ffi::c_void,
    size: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        let (ptr, len) = if data.is_null() || size == 0 {
            (None, 0)
        } else {
            (Some(data), size)
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("begin event meta={metadata} size={len}"));
        // SAFETY: open list; null/zero payload normalized.
        unsafe { list.list.BeginEvent(metadata, ptr, len) };
    }
    #[cfg(not(windows))]
    let _ = (this, metadata, data, size);
}

/// `ID3D12GraphicsCommandList::EndEvent()` (slot 58 / 0x1d0, #859).
extern "system" fn end_event(this: This) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        #[cfg(feature = "diagnostics")]
        list.note_op(|| "end event".to_string());
        // SAFETY: open list.
        unsafe { list.list.EndEvent() };
    }
    #[cfg(not(windows))]
    let _ = this;
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, host_list, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        if let Some(list) = host_list(this) {
            crate::draw::viewports(list, items(viewports, count));
        }
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        if let Some(list) = host_list(this) {
            crate::draw::scissors(list, items(rects, count));
        }
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        if let Some(list) = host_list(this) {
            // SAFETY: the title passes four floats or null.
            crate::draw::blend_factor(list, unsafe { factor.as_ref() });
        }
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        if let Some(list) = host_list(this) {
            crate::draw::stencil_ref(list, reference);
        }
    }
    pub(super) extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
        if let Some(list) = host_list(this) {
            crate::draw::depth_bounds(list, min, max);
        }
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// Collects and validates render target handles for `OMSetRenderTargets`:
    /// caps at D3D12's 8 simultaneous targets and guards null / zero / overflow (#796).
    pub(super) fn collect_target_handles(
        count: u32,
        handles: *const u64,
        one_range: i32,
    ) -> Vec<u64> {
        const MAX_RENDER_TARGETS: u32 = 8;
        let count = count.min(MAX_RENDER_TARGETS);
        if count == 0 || handles.is_null() {
            Vec::new()
        } else if one_range != 0 {
            // SAFETY: `count > 0` and `handles` is non-null.
            let first = unsafe { *handles };
            if first == 0 {
                Vec::new()
            } else {
                (0..u64::from(count))
                    .map_while(|i| first.checked_add(128 * i))
                    .collect()
            }
        } else {
            items(handles, count).to_vec()
        }
    }

    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles = collect_target_handles(count, handles, one_range);
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as depth_bounds;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// A GPU wait for a 64-bit memory value `(address, comparison, reference)`:
/// recorded on the list and awaited at submission time before the list's
/// host work runs, so cross-queue and CPU producers can satisfy it first.
extern "system" fn wait_value(this: This, address: u64, comparison: u32, reference: u64) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_wait(address, comparison, reference);
    }
    #[cfg(not(windows))]
    let _ = (this, address, comparison, reference);
}

#[cfg(windows)]
#[repr(C)]
struct BarrierGroup {
    kind: u32,
    count: u32,
    barriers: *const std::ffi::c_void,
}

#[cfg(windows)]
#[repr(C)]
struct TextureBarrier {
    sync_before: u32,
    sync_after: u32,
    access_before: u32,
    access_after: u32,
    layout_before: i32,
    layout_after: i32,
    resource: This,
    subresources: [u32; 6],
    flags: u32,
}

/// Maps an enhanced-barrier `D3D12_BARRIER_LAYOUT` to the equivalent legacy
/// host `D3D12_RESOURCE_STATES` mask (`None` for `UNDEFINED` or unknown).
#[cfg(any(windows, test))]
fn enhanced_layout_state(layout: i32) -> Option<u32> {
    Some(match layout {
        0 | 18 | 24 => 0,            // COMMON / PRESENT
        1 | 19 | 25 => 0x8C0,        // GENERIC_READ (SRV + COPY_SOURCE)
        2 => 0x4,                    // RENDER_TARGET
        3 | 20 | 26 => 0x8,          // UNORDERED_ACCESS
        4 => 0x10,                   // DEPTH_STENCIL_WRITE
        5 => 0x20,                   // DEPTH_STENCIL_READ
        6 | 21 | 27 => 0xC0,         // SHADER_RESOURCE (PS | NON_PS)
        7 | 22 | 28 => 0x800,        // COPY_SOURCE
        8 | 23 | 29 => 0x400,        // COPY_DEST
        9 => 0x2000,                 // RESOLVE_SOURCE
        10 => 0x1000,                // RESOLVE_DEST
        11 => 0x0100_0000,           // SHADING_RATE_SOURCE
        _ => return None,
    })
}

/// The enhanced barriers `(group count, groups)`: translates texture layout
/// transitions through [`ensure_state`] so host textures stay in the right
/// state for direct-indexed reads and cross-list `prelude` tracking, and emits
/// a global UAV barrier for global/buffer/UAV groups.
extern "system" fn barrier_groups(this: This, count: u32, groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        const GLOBAL: u32 = 0;
        const TEXTURE: u32 = 1;
        const BUFFER: u32 = 2;
        const ACCESS_UAV: u32 = 0x80;
        let mut need_uav = groups.is_null();
        if !groups.is_null() {
            // SAFETY: the title passes `count` contiguous `D3D12_BARRIER_GROUP` entries.
            let slice = unsafe { std::slice::from_raw_parts(groups.cast::<BarrierGroup>(), count as usize) };
            for group in slice {
                match group.kind {
                    TEXTURE if !group.barriers.is_null() && group.count > 0 => {
                        // SAFETY: the group points at `group.count` texture barriers.
                        let textures = unsafe {
                            std::slice::from_raw_parts(
                                group.barriers.cast::<TextureBarrier>(),
                                group.count as usize,
                            )
                        };
                        for tb in textures {
                            if !tb.resource.is_null()
                                && let Some(state) = enhanced_layout_state(tb.layout_after)
                            {
                                ensure_state(list, tb.resource, state);
                            }
                            if (tb.access_before | tb.access_after) & ACCESS_UAV != 0
                                || matches!(tb.layout_before, 3 | 20 | 26)
                                || matches!(tb.layout_after, 3 | 20 | 26)
                            {
                                need_uav = true;
                            }
                        }
                    }
                    GLOBAL | BUFFER => need_uav = true,
                    _ => need_uav = true,
                }
            }
        }
        if need_uav {
            // SAFETY: the same layout; no resource named.
            let barrier =
                unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
            // SAFETY: an open list.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
    }
    #[cfg(not(windows))]
    let _ = (this, count, groups);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 55] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (11, clear_state as *const ()),               // 0x58 ClearState
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (53, clear_uav_uint as *const ()),            // 0x1a8
    (54, clear_uav_float as *const ()),           // 0x1b0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, state::depth_bounds as *const ()),       // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait_value as *const ()),                // 0x238, 32 bits
    (72, wait_value as *const ()),                // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 14] = [
    (28, begin_query as *const ()),            // 0xe0 BeginQuery
    (29, end_query as *const ()),              // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0 ResolveQueryData
    (31, set_predication as *const ()),        // 0xf8 SetPredication
    (55, discard_resource as *const ()),       // 0x1b8 DiscardResource
    (56, set_marker as *const ()),             // 0x1c0 SetMarker
    (57, begin_event as *const ()),            // 0x1c8 BeginEvent
    (58, end_event as *const ()),              // 0x1d0 EndEvent
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    #[test]
    fn enhanced_barrier_layouts_map_to_legacy_states() {
        assert_eq!(enhanced_layout_state(0), Some(0)); // COMMON
        assert_eq!(enhanced_layout_state(2), Some(0x4)); // RENDER_TARGET
        assert_eq!(enhanced_layout_state(3), Some(0x8)); // UNORDERED_ACCESS
        assert_eq!(enhanced_layout_state(4), Some(0x10)); // DEPTH_STENCIL_WRITE
        assert_eq!(enhanced_layout_state(5), Some(0x20)); // DEPTH_STENCIL_READ
        assert_eq!(enhanced_layout_state(6), Some(0xC0)); // SHADER_RESOURCE
        assert_eq!(enhanced_layout_state(7), Some(0x800)); // COPY_SOURCE
        assert_eq!(enhanced_layout_state(8), Some(0x400)); // COPY_DEST
        assert_eq!(enhanced_layout_state(21), Some(0xC0)); // DIRECT_QUEUE_SHADER_RESOURCE
        assert_eq!(enhanced_layout_state(26), Some(0x8)); // COMPUTE_QUEUE_UNORDERED_ACCESS
        assert_eq!(enhanced_layout_state(-1), None); // UNDEFINED
    }

    #[cfg(windows)]
    #[test]
    fn clear_uav_uint_and_float_whole_and_rect_on_warp() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_TEXTURE_COPY_LOCATION,
        };
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let resource_iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let heap_iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);

        // Create allocator & command list.
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );

        // Create one shader-visible view heap (16 slots) and one CPU-only view heap (16 slots).
        #[repr(C)]
        struct ExtendedHeap([u32; 4], [u64; 2]);
        let mut gpu_heap = null;
        let gpu_desc = ExtendedHeap([0, 16, 1, 0], [0, 0]);
        assert_eq!(
            crate::descriptor::create_heap(null, (&raw const gpu_desc).cast(), &heap_iid, &mut gpu_heap),
            S_OK
        );
        let mut cpu_heap = null;
        let cpu_desc = ExtendedHeap([0, 16, 0, 0], [0, 0]);
        assert_eq!(
            crate::descriptor::create_heap(null, (&raw const cpu_desc).cast(), &heap_iid, &mut cpu_heap),
            S_OK
        );
        let gpu_base = crate::descriptor::heap(gpu_heap).start;
        let cpu_base = crate::descriptor::heap(cpu_heap).start;

        // Create 4x4 R32_UINT (42) and 4x4 R32_FLOAT (41) UAV textures.
        #[repr(C)]
        struct RawDesc {
            dimension: u32,
            alignment: u64,
            width: u64,
            height: u32,
            depth_or_array_size: u16,
            mip_levels: u16,
            format: u32,
            sample_count: u32,
            sample_quality: u32,
            layout: u32,
            flags: u32,
        }
        let hp: [u32; 5] = [1, 0, 0, 1, 1]; // DEFAULT heap
        let make_tex = |fmt: u32| -> This {
            let desc = RawDesc {
                dimension: 3, // TEXTURE2D
                alignment: 0,
                width: 4,
                height: 4,
                depth_or_array_size: 1,
                mip_levels: 1,
                format: fmt,
                sample_count: 1,
                sample_quality: 0,
                layout: 0,
                flags: 4, // ALLOW_UNORDERED_ACCESS
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const hp).cast(),
                    0,
                    (&raw const desc).cast(),
                    0,
                    std::ptr::null(),
                    &resource_iid,
                    &mut tex,
                ),
                S_OK
            );
            tex
        };
        let tex_u32 = make_tex(42);
        let tex_f32 = make_tex(41);

        // Create UAVs (null desc on typed R32_UINT/R32_FLOAT resources, matching host typeless synthesis).
        crate::descriptor::create_uav(null, tex_u32, null, std::ptr::null(), cpu_base);
        crate::descriptor::create_uav(null, tex_u32, null, std::ptr::null(), gpu_base);
        crate::descriptor::create_uav(null, tex_f32, null, std::ptr::null(), cpu_base + 32);
        crate::descriptor::create_uav(null, tex_f32, null, std::ptr::null(), gpu_base + 32);

        // Bind shader-visible view heap on list (slot 32 / 0x100).
        slot::<extern "system" fn(This, u32, *const This)>(list, 32 * 8)(list, 1, &gpu_heap);

        let center = Rect {
            left: 1,
            top: 1,
            right: 3,
            bottom: 3,
        };
        // Slot 53 (0x1a8): ClearUnorderedAccessViewUint — whole clear to 7, then centered 2x2 to 99.
        type ClearUint =
            extern "system" fn(This, u64, u64, This, *const [u32; 4], u32, *const Rect);
        let cu = slot::<ClearUint>(list, 53 * 8);
        let u_bg = [7u32, 0, 0, 0];
        let u_fg = [99u32, 0, 0, 0];
        cu(list, gpu_base, cpu_base, tex_u32, &u_bg, 0, std::ptr::null());
        barrier(list, 1, &UAV_BARRIER);
        cu(list, gpu_base, cpu_base, tex_u32, &u_fg, 1, &center);

        // Slot 54 (0x1b0): ClearUnorderedAccessViewFloat — whole clear to 1.5, then centered 2x2 to 3.25.
        type ClearFloat =
            extern "system" fn(This, u64, u64, This, *const [f32; 4], u32, *const Rect);
        let cf = slot::<ClearFloat>(list, 54 * 8);
        let f_bg = [1.5f32, 0.0, 0.0, 0.0];
        let f_fg = [3.25f32, 0.0, 0.0, 0.0];
        cf(list, gpu_base + 32, cpu_base + 32, tex_f32, &f_bg, 0, std::ptr::null());
        barrier(list, 1, &UAV_BARRIER);
        cf(list, gpu_base + 32, cpu_base + 32, tex_f32, &f_fg, 1, &center);

        // Copy both textures to readback buffers inside the same list and execute.
        let (rb_u32, ptr_u32) = host.readback_buffer(1024).expect("readback u32");
        let (rb_f32, ptr_f32) = host.readback_buffer(1024).expect("readback f32");
        let hl = host_list(list).expect("host list");
        for (tex, rb, fmt) in [(tex_u32, &rb_u32, 42u32), (tex_f32, &rb_f32, 41u32)] {
            ensure_state(hl, tex, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            let htex = crate::resource::host_resource(tex).expect("host texture");
            let dst = CopyLocation {
                resource: rb.as_raw().cast_const().cast(),
                kind: 1,
                footprint: crate::resource::PlacedFootprint {
                    offset: 0,
                    format: fmt,
                    width: 4,
                    height: 4,
                    depth: 1,
                    row_pitch: 256,
                },
            };
            let src = CopyLocation {
                resource: htex.as_raw().cast_const().cast(),
                kind: 0,
                footprint: crate::resource::PlacedFootprint {
                    offset: 0,
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            };
            // SAFETY: open list, live resources, verified identical layout to D3D12_TEXTURE_COPY_LOCATION.
            unsafe {
                hl.list.CopyTextureRegion(
                    (&raw const dst).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                    0,
                    0,
                    0,
                    (&raw const src).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                    None,
                );
            }
        }
        assert_eq!(close(list), S_OK);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None, "no D3D12 debug-layer errors");

        // Verify all 16 pixels of both textures.
        for y in 0..4usize {
            for x in 0..4usize {
                let in_center = (1..3).contains(&x) && (1..3).contains(&y);
                // SAFETY: readback buffers hold 4 rows at 256-byte pitch.
                let (got_u, got_f) = unsafe {
                    (
                        ptr_u32.add(y * 256 + x * 4).cast::<u32>().read(),
                        ptr_f32.add(y * 256 + x * 4).cast::<f32>().read(),
                    )
                };
                assert_eq!(got_u, if in_center { 99 } else { 7 }, "u32 ({x},{y})");
                assert_eq!(got_f, if in_center { 3.25 } else { 1.5 }, "f32 ({x},{y})");
            }
        }

        let _ = (
            release(tex_u32),
            release(tex_f32),
            release(gpu_heap),
            release(cpu_heap),
            release(list),
            release(allocator),
        );
    }

    #[cfg(windows)]
    #[test]
    fn copy_texture_region_reads_same_list_gpu_written_buffer_aligned_and_unaligned() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_TEXTURE_COPY_LOCATION,
        };
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let resource_iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);

        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );

        #[repr(C)]
        struct RawDesc {
            dimension: u32,
            alignment: u64,
            width: u64,
            height: u32,
            depth_or_array_size: u16,
            mip_levels: u16,
            format: u32,
            sample_count: u32,
            sample_quality: u32,
            layout: u32,
            flags: u32,
        }
        let hp: [u32; 5] = [1, 0, 0, 1, 1];
        let make_tex = || -> This {
            let desc = RawDesc {
                dimension: 3,
                alignment: 0,
                width: 4,
                height: 4,
                depth_or_array_size: 1,
                mip_levels: 1,
                format: 42, // R32_UINT
                sample_count: 1,
                sample_quality: 0,
                layout: 0,
                flags: 0,
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const hp).cast(),
                    0,
                    (&raw const desc).cast(),
                    0,
                    std::ptr::null(),
                    &resource_iid,
                    &mut tex,
                ),
                S_OK
            );
            tex
        };
        let tex_aligned = make_tex();
        let tex_unaligned = make_tex();

        // Guest buffer (4096 bytes), filled in CPU memory with stale 0xDEAD_BEEF.
        let buf_desc = RawDesc {
            dimension: 1, // BUFFER
            alignment: 0,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1, // ROW_MAJOR
            flags: 0,
        };
        let mut guest_buf = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const hp).cast(),
                0,
                (&raw const buf_desc).cast(),
                0,
                std::ptr::null(),
                &resource_iid,
                &mut guest_buf,
            ),
            S_OK
        );
        let (buf_ptr, buf_len) = crate::resource::buffer_bytes(guest_buf).expect("guest buffer");
        assert!(buf_len >= 4096);
        // SAFETY: guest buffer holds 4096 writable bytes.
        unsafe {
            for w in std::slice::from_raw_parts_mut(buf_ptr.cast::<u32>(), 1024) {
                *w = 0xDEAD_BEEF;
            }
        }

        // Simulate a same-list GPU write into the guest buffer's host shadow:
        // write 0x1111_0000+i at offset 0 (512-byte aligned, bytes 0..784) and
        // 0x2222_0000+i at offset 1028 (NOT 512-byte aligned, bytes 1028..1812)
        // via GPU CopyBufferRegion while CPU title memory remains stale 0xDEAD_BEEF.
        const UNALIGNED_OFF: usize = 1028;
        let hl = host_list(list).expect("host list");
        let shadow = crate::shadow::find(buf_ptr as u64, buf_len).expect("shadow");
        let (upload_buf, upload_ptr) = hl.upload(host, 4096).expect("upload");
        // SAFETY: mapped 4096-byte upload buffer.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(upload_ptr, 4096);
            slice.fill(0);
            for y in 0..4usize {
                for x in 0..4usize {
                    let v1 = 0x1111_0000u32 + (y * 4 + x) as u32;
                    let v2 = 0x2222_0000u32 + (y * 4 + x) as u32;
                    slice[y * 256 + x * 4..y * 256 + x * 4 + 4].copy_from_slice(&v1.to_le_bytes());
                    let off2 = UNALIGNED_OFF + y * 256 + x * 4;
                    slice[off2..off2 + 4].copy_from_slice(&v2.to_le_bytes());
                }
            }
        }
        shadow.refresh(hl, buf_ptr as u64, 4096);
        hl.ensure_buffer(&shadow, COPY_DEST);
        // SAFETY: open list; upload buffer and shadow both cover 4096 bytes.
        unsafe {
            hl.list.CopyBufferRegion(&shadow.buffer, 0, &upload_buf, 0, 4096);
        }
        shadow.gpu_writes(hl, buf_ptr as u64, 4096);

        // Now record guest CopyTextureRegion (slot 16) from BOTH aligned offset 0
        // and unaligned offset 1028 in the SAME command list.
        let copy_tex = |dst_tex: This, src_offset: u64| {
            let dst = CopyLocation {
                resource: dst_tex,
                kind: 0,
                footprint: crate::resource::PlacedFootprint {
                    offset: 0,
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            };
            let src = CopyLocation {
                resource: guest_buf,
                kind: 1,
                footprint: crate::resource::PlacedFootprint {
                    offset: src_offset,
                    format: 42,
                    width: 4,
                    height: 4,
                    depth: 1,
                    row_pitch: 256,
                },
            };
            copy_texture_region(list, &dst, 0, 0, 0, &src, std::ptr::null());
        };
        copy_tex(tex_aligned, 0);
        copy_tex(tex_unaligned, UNALIGNED_OFF as u64);

        // Read both destination textures back to verify GPU-written bytes landed.
        let (rb_a, ptr_a) = host.readback_buffer(1024).expect("readback aligned");
        let (rb_u, ptr_u) = host.readback_buffer(1024).expect("readback unaligned");
        for (tex, rb) in [(tex_aligned, &rb_a), (tex_unaligned, &rb_u)] {
            ensure_state(hl, tex, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            let htex = crate::resource::host_resource(tex).expect("host texture");
            let dst = CopyLocation {
                resource: rb.as_raw().cast_const().cast(),
                kind: 1,
                footprint: crate::resource::PlacedFootprint {
                    offset: 0,
                    format: 42,
                    width: 4,
                    height: 4,
                    depth: 1,
                    row_pitch: 256,
                },
            };
            let src = CopyLocation {
                resource: htex.as_raw().cast_const().cast(),
                kind: 0,
                footprint: crate::resource::PlacedFootprint {
                    offset: 0,
                    format: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                    row_pitch: 0,
                },
            };
            // SAFETY: open list, live resources, verified layout.
            unsafe {
                hl.list.CopyTextureRegion(
                    (&raw const dst).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                    0,
                    0,
                    0,
                    (&raw const src).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                    None,
                );
            }
        }
        assert_eq!(close(list), S_OK);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None, "no D3D12 debug-layer errors");

        for y in 0..4usize {
            for x in 0..4usize {
                let idx = (y * 4 + x) as u32;
                // SAFETY: 4 rows at 256-byte row pitch.
                let (got_a, got_u) = unsafe {
                    (
                        ptr_a.add(y * 256 + x * 4).cast::<u32>().read(),
                        ptr_u.add(y * 256 + x * 4).cast::<u32>().read(),
                    )
                };
                assert_eq!(got_a, 0x1111_0000 + idx, "aligned ({x},{y})");
                assert_eq!(got_u, 0x2222_0000 + idx, "unaligned 1028 ({x},{y})");
            }
        }

        let _ = (
            release(tex_aligned),
            release(tex_unaligned),
            release(guest_buf),
            release(list),
            release(allocator),
        );
    }

    #[test]
    fn allocate_upload_validates_alignment_and_fails_fallibly() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        // Non-power-of-two alignments below 16 must be rejected (#837).
        assert!(allocate_upload(list, 16, 3).is_null());
        assert!(allocate_upload(list, 16, 6).is_null());
        assert!(allocate_upload(list, 16, 12).is_null());
        assert!(allocate_upload(list, 16, 24).is_null());
        // Alignment 0 defaults to 16; powers of two <= 16 are promoted to 16.
        let p0 = allocate_upload(list, 32, 0);
        assert!(!p0.is_null() && (p0 as usize).is_multiple_of(16));
        let p4 = allocate_upload(list, 32, 4);
        assert!(!p4.is_null() && (p4 as usize).is_multiple_of(16));
        // Unsatisfiable sizes return null cleanly without aborting.
        assert!(allocate_upload(list, u64::MAX, 16).is_null());
        assert!(allocate_upload(list, usize::MAX as u64 / 2, 16).is_null());
        assert!(allocate_upload(null, 16, 16).is_null());

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    #[test]
    fn clear_state_discard_predication_and_marker_slots_match_both_tables() {
        for table in [&LIST_FUNCTIONS.0, &NEWER_LIST_FUNCTIONS.0] {
            assert_eq!(table[11], clear_state as *const ());
            assert_eq!(table[31], set_predication as *const ());
            assert_eq!(table[55], discard_resource as *const ());
            assert_eq!(table[56], set_marker as *const ());
            assert_eq!(table[57], begin_event as *const ());
            assert_eq!(table[58], end_event as *const ());
        }
    }

    #[cfg(windows)]
    #[test]
    fn clear_state_discard_predication_markers_and_targets_on_warp() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_TEXTURE_COPY_LOCATION,
        };
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let resource_iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let heap_iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);

        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );

        // Edge-case OMSetRenderTargets handle collection: count=0, null, first=0,
        // cap at 8, and overflow stop without wrapping (#796).
        let zero_handle = 0u64;
        let max_handle = u64::MAX - 64;
        let valid_first = 0x1000u64;
        assert!(state::collect_target_handles(0, &max_handle, 1).is_empty());
        assert!(state::collect_target_handles(4, std::ptr::null(), 1).is_empty());
        assert!(state::collect_target_handles(4, &zero_handle, 1).is_empty());
        assert_eq!(
            state::collect_target_handles(16, &valid_first, 1).len(),
            8
        );
        assert_eq!(
            state::collect_target_handles(4, &max_handle, 1),
            vec![max_handle]
        );
        state::targets(list, 0, std::ptr::null(), 1, std::ptr::null());
        state::targets(list, 4, &zero_handle, 1, std::ptr::null());

        // Create a 4x1 R8G8B8A8_UNORM render target texture + RTV heap.
        #[repr(C)]
        struct HeapDesc {
            kind: u32,
            count: u32,
            flags: u32,
            mask: u32,
        }
        let mut rtv_heap = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                std::ptr::from_ref(&HeapDesc {
                    kind: 2, // RTV
                    count: 4,
                    flags: 0,
                    mask: 0,
                })
                .cast(),
                &heap_iid,
                &mut rtv_heap,
            ),
            S_OK
        );
        let rtv_handle = slot::<extern "system" fn(This) -> u64>(rtv_heap, 0x48)(rtv_heap);

        #[repr(C)]
        struct RawDesc {
            dimension: u32,
            alignment: u64,
            width: u64,
            height: u32,
            depth_or_array_size: u16,
            mip_levels: u16,
            format: u32,
            sample_count: u32,
            sample_quality: u32,
            layout: u32,
            flags: u32,
        }
        let hp: [u32; 5] = [1, 0, 0, 1, 1];
        let tex_desc = RawDesc {
            dimension: 3, // TEXTURE2D
            alignment: 0,
            width: 4,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28, // R8G8B8A8_UNORM
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 1, // ALLOW_RENDER_TARGET
        };
        let mut tex = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const hp).cast(),
                0,
                (&raw const tex_desc).cast(),
                0x4, // RENDER_TARGET
                null.cast(),
                &resource_iid,
                &mut tex,
            ),
            S_OK
        );
        crate::descriptor::create_rtv(null, tex, null.cast(), rtv_handle);

        // Create a 64-byte predicate buffer with [0u64, 1u64].
        let buf_desc = RawDesc {
            dimension: 1, // BUFFER
            alignment: 0,
            width: 64,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        };
        let mut pred_buf = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const hp).cast(),
                0,
                (&raw const buf_desc).cast(),
                0,
                null.cast(),
                &resource_iid,
                &mut pred_buf,
            ),
            S_OK
        );
        let (pred_ptr, _) = crate::resource::buffer_bytes(pred_buf).expect("predicate bytes");
        // SAFETY: 64-byte guest buffer; write 0 at +0 and 1 at +8.
        unsafe {
            pred_ptr.cast::<u64>().write(0);
            pred_ptr.add(8).cast::<u64>().write(1);
        }

        // Markers/events (including null payload with non-zero size) must succeed cleanly.
        let label = b"wave7\0";
        begin_event(list, 1, label.as_ptr().cast(), label.len() as u32);
        set_marker(list, 1, std::ptr::null(), 16);

        // Put texture into RENDER_TARGET state via DiscardResource + ClearRenderTargetView.
        let hl = host_list(list).expect("host list");
        ensure_state(hl, tex, 0x4);
        discard_resource(list, tex, std::ptr::null());
        // Invalid inverted rect discard must be cleanly ignored.
        let bad_rect = RECT {
            left: 2,
            top: 0,
            right: 1,
            bottom: 1,
        };
        let bad_region = DiscardRegion {
            num_rects: 1,
            rects: &bad_rect,
            first_subresource: 0,
            num_subresources: 1,
        };
        discard_resource(list, tex, std::ptr::from_ref(&bad_region).cast());

        // Clear whole 4x1 texture to black, then pixel 0 to red.
        clear_rtv(list, rtv_handle, &[0.0, 0.0, 0.0, 1.0], 0, std::ptr::null());
        let r0 = [0i32, 0, 1, 1];
        clear_rtv(list, rtv_handle, &[1.0, 0.0, 0.0, 1.0], 1, r0.as_ptr().cast());

        // ClearState resets pipeline/state while preserving earlier recorded work and tracked states (#848).
        clear_state(list, null);
        let r1 = [1i32, 0, 2, 1];
        clear_rtv(list, rtv_handle, &[0.0, 1.0, 0.0, 1.0], 1, r1.as_ptr().cast());

        // Predication EQUAL_ZERO on value 0 (+0) -> SKIP clear of pixel 2 (stays black).
        set_predication(list, pred_buf, 0, 0);
        let r2 = [2i32, 0, 3, 1];
        clear_rtv(list, rtv_handle, &[0.0, 0.0, 1.0, 1.0], 1, r2.as_ptr().cast());

        // Predication EQUAL_ZERO on value 1 (+8) -> RUN clear of pixel 3 (becomes blue).
        set_predication(list, pred_buf, 8, 0);
        let r3 = [3i32, 0, 4, 1];
        clear_rtv(list, rtv_handle, &[0.0, 0.0, 1.0, 1.0], 1, r3.as_ptr().cast());

        // Disable predication and close event scope.
        set_predication(list, null, 0, 0);
        end_event(list);

        // Read back 4 pixels.
        let (rb, rb_ptr) = host.readback_buffer(256).expect("readback");
        ensure_state(hl, tex, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let htex = crate::resource::host_resource(tex).expect("host texture");
        let dst = CopyLocation {
            resource: rb.as_raw().cast_const().cast(),
            kind: 1,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 28,
                width: 4,
                height: 1,
                depth: 1,
                row_pitch: 256,
            },
        };
        let src = CopyLocation {
            resource: htex.as_raw().cast_const().cast(),
            kind: 0,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                row_pitch: 0,
            },
        };
        // SAFETY: open list, live resources, verified layout.
        unsafe {
            hl.list.CopyTextureRegion(
                (&raw const dst).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                0,
                0,
                0,
                (&raw const src).cast::<D3D12_TEXTURE_COPY_LOCATION>(),
                None,
            );
        }
        assert_eq!(close(list), S_OK);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None, "no D3D12 debug-layer errors");

        // SAFETY: 4 RGBA8 pixels (16 bytes) inside 256-byte readback buffer.
        let got: [u8; 16] = unsafe { *(rb_ptr.cast::<[u8; 16]>()) };
        assert_eq!(
            got,
            [
                255, 0, 0, 255, // pixel 0: red before ClearState
                0, 255, 0, 255, // pixel 1: green after ClearState
                0, 0, 0, 255, // pixel 2: skipped by EQUAL_ZERO predication on 0
                0, 0, 255, 255, // pixel 3: executed by EQUAL_ZERO predication on 1
            ]
        );

        let _ = (
            release(pred_buf),
            release(tex),
            release(rtv_heap),
            release(list),
            release(allocator),
        );
    }
}
