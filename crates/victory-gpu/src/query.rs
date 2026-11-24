//! Query heaps. On Windows, each title query heap creates a matching host
//! `ID3D12QueryHeap` so `BeginQuery`/`EndQuery`/`ResolveQueryData` measure
//! real host GPU occlusion, timestamps, and pipeline statistics (falling back
//! cleanly when headless or for console-only query heap types).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUERY_HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    pub(crate) kind: u32,
    pub(crate) count: u32,
    pub(crate) nodes: u32,
}

struct Heap {
    _object: Object,
    desc: Desc,
    #[cfg(windows)]
    host: Option<windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap>,
}

/// Maps a console/D3D12 query heap type (`D3D12_QUERY_HEAP_TYPE`) to a host
/// D3D12 query heap type value (`0=OCCLUSION`, `1=TIMESTAMP`,
/// `2=PIPELINE_STATISTICS`, `3=SO_STATISTICS`, `5=COPY_QUEUE_TIMESTAMP`).
pub(crate) fn host_heap_type(kind: u32) -> Option<i32> {
    match kind {
        0..=3 | 5 => Some(kind as i32),
        _ => None,
    }
}

/// Maps a console/D3D12 `D3D12_QUERY_TYPE` to `(host_query_type, result_bytes)`.
/// Returns `None` for unknown/console-only query types.
pub(crate) fn query_type_info(kind: u32) -> Option<(i32, u64)> {
    Some(match kind {
        0 => (0, 8),         // OCCLUSION
        1 => (1, 8),         // BINARY_OCCLUSION
        2 => (2, 8),         // TIMESTAMP
        3 => (3, 88),        // PIPELINE_STATISTICS (11 * u64)
        4..=7 => (kind as i32, 16), // SO_STATISTICS_STREAM0..3 (2 * u64)
        _ => return None,
    })
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    #[cfg(windows)]
    let host = create_host_heap(desc);
    let heap = Box::new(Heap {
        _object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host,
    });
    object::hand_out(QUERY_HEAP, heap, iid, &QUERY_HEAP_IIDS, out)
}

#[cfg(windows)]
fn create_host_heap(desc: Desc) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap> {
    use windows::Win32::Graphics::Direct3D12::{D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE};
    let host = crate::host::host()?;
    let host_kind = host_heap_type(desc.kind)?;
    if desc.count == 0 {
        return None;
    }
    let host_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE(host_kind),
        Count: desc.count,
        NodeMask: 0,
    };
    let mut heap = None;
    // SAFETY: plain D3D12 query heap creation on the host device.
    match unsafe { host.device.CreateQueryHeap(&host_desc, &mut heap) } {
        Ok(()) => heap,
        Err(e) => {
            tracing::warn!(error = %e, ?desc, "no host query heap");
            None
        }
    }
}

/// Returns the host `ID3D12QueryHeap` and slot count for a title query heap.
#[cfg(windows)]
pub(crate) fn host_heap(
    this: This,
) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap, u32)> {
    if this.is_null() {
        return None;
    }
    // SAFETY: title passes a query heap object created by `create`.
    let heap = unsafe { &*this.cast::<Heap>() };
    Some((heap.host.clone()?, heap.desc.count))
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &QUERY_HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

static FUNCTIONS: Table<8> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 8>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const ();
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_standard_query_heap_and_query_types() {
        assert_eq!(host_heap_type(0), Some(0)); // OCCLUSION
        assert_eq!(host_heap_type(1), Some(1)); // TIMESTAMP
        assert_eq!(host_heap_type(2), Some(2)); // PIPELINE_STATISTICS
        assert_eq!(host_heap_type(3), Some(3)); // SO_STATISTICS
        assert_eq!(host_heap_type(5), Some(5)); // COPY_QUEUE_TIMESTAMP
        assert_eq!(host_heap_type(99), None);

        assert_eq!(query_type_info(0), Some((0, 8)));
        assert_eq!(query_type_info(1), Some((1, 8)));
        assert_eq!(query_type_info(2), Some((2, 8)));
        assert_eq!(query_type_info(3), Some((3, 88)));
        assert_eq!(query_type_info(4), Some((4, 16)));
        assert_eq!(query_type_info(7), Some((7, 16)));
        assert_eq!(query_type_info(8), None);
    }
}
