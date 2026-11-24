//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    if unsafe { desc.Anonymous.Buffer.NumElements } == 0 {
        once!(
            address = format_args!("{address:#x}"),
            "buffer shader view of zero elements"
        );
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    buffer.FirstElement += offset / element;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    if unsafe { desc.Anonymous.Buffer.NumElements } == 0 {
        once!(
            address = format_args!("{address:#x}"),
            "unordered buffer view of zero elements"
        );
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "unordered buffer view not on an element of its host copy"
        );
        return None;
    }
    buffer.FirstElement += offset / element;
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    null_for_on(&host.device, view, at);
}

fn null_for_on(device: &ID3D12Device, view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view_on(device, at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => {
            return null_buffer_view_on(device, at);
        }
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like_on(device, desc, at),
        View::Placed(false, _, desc) => {
            return null_like_on(device, desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view_on(device, at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view_on(device: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like_on(device: &ID3D12Device, words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view_on(device, at);
    }
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    null_view_on(&host.device, at);
}

fn null_view_on(device: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    fn placed_buffer_view(unordered: bool, address: u64, len: u64, elements: u32) -> View {
        let mut resource = [0u32; 14];
        resource[0] = 1; // D3D12_RESOURCE_DIMENSION_BUFFER
        resource[4] = len as u32;
        resource[5] = (len >> 32) as u32;
        let words: [u32; 10] = if unordered {
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: elements,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            // SAFETY: 40-byte POD view description read as 10 u32 words.
            unsafe {
                std::ptr::from_ref(&uav)
                    .cast::<[u32; 10]>()
                    .read_unaligned()
            }
        } else {
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: elements,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            // SAFETY: 40-byte POD view description read as 10 u32 words.
            unsafe {
                std::ptr::from_ref(&srv)
                    .cast::<[u32; 10]>()
                    .read_unaligned()
            }
        };
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&words);
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        View::Placed(unordered, resource, desc)
    }

    /// Zero-element buffer SRVs and UAVs become null buffer descriptors
    /// instead of removing the host D3D12 device (`0x887A0001`).
    #[test]
    fn buffer_views_of_no_elements_are_null() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 8) else {
            return;
        };
        let backing = vec![0u32; 1024];
        let addr = backing.as_ptr() as u64;
        let res_id = backing.as_ptr() as usize;
        crate::shadow::register(res_id, addr, 4096);

        let slot0 = heap.dynamic(4).unwrap();
        heap.make_at(slot0, &placed_buffer_view(false, addr, 256, 1));
        heap.make_at(slot0 + 1, &placed_buffer_view(true, addr, 256, 1));
        heap.make_at(slot0 + 2, &placed_buffer_view(false, addr, 256, 0));
        heap.make_at(slot0 + 3, &placed_buffer_view(true, addr, 256, 0));

        {
            let kept = heap.kept.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                kept[slot0 as usize].is_some(),
                "1-element SRV must keep shadow"
            );
            assert!(
                kept[(slot0 + 1) as usize].is_some(),
                "1-element UAV must keep shadow"
            );
            assert!(
                kept[(slot0 + 2) as usize].is_none(),
                "0-element SRV must be null"
            );
            assert!(
                kept[(slot0 + 3) as usize].is_none(),
                "0-element UAV must be null"
            );
        }

        heap.release(slot0, 4);
        crate::shadow::unregister(res_id, addr);
        // SAFETY: plain D3D12 device status query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Runs a synthetic compute dispatch reading and writing zero-element and
    /// non-zero-element buffer views under the D3D12 debug layer across every
    /// SM 6.6 hardware adapter and WARP.
    #[test]
    fn zero_element_buffer_views_dispatch_cleanly_across_adapters() {
        const CHILD_ENV: &str = "VICTORY_ZERO_BUF_DEBUG_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let exe = std::env::current_exe().expect("current_exe");
            let out = std::process::Command::new(exe)
                .env(CHILD_ENV, "1")
                .args([
                    "--exact",
                    "descriptor::host::tests::zero_element_buffer_views_dispatch_cleanly_across_adapters",
                    "--nocapture",
                ])
                .output()
                .expect("spawn debug-layer child");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprint!("{stdout}{stderr}");
            assert!(
                out.status.success(),
                "debug-layer child failed ({}):\n{stdout}\n{stderr}",
                out.status
            );
            return;
        }

        use victory_shader::native::{
            ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo,
            UsedResource,
        };
        use victory_shader::{dxbc, emit, io, ir, resource, structure};
        use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
            IDXGIAdapter, IDXGIFactory4,
        };
        use windows::core::Interface;

        // Enable D3D12 debug layer in this fresh child process before creating any device.
        let mut debug_enabled = false;
        // SAFETY: called before any D3D12 device exists in this child process.
        unsafe {
            let mut dbg: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut dbg).is_ok()
                && let Some(d) = dbg
            {
                d.EnableDebugLayer();
                debug_enabled = true;
            }
        }
        eprintln!("D3D12 debug layer enabled in child: {debug_enabled}");

        // Verify `HostHeap::make_at` on `host()` under the debug layer as well.
        if let Some(h) = host()
            && let Some(heap) = HostHeap::new(false, 4)
        {
            let backing = vec![0u32; 1024];
            let addr = backing.as_ptr() as u64;
            let res_id = backing.as_ptr() as usize;
            crate::shadow::register(res_id, addr, 4096);
            let iq = h.device.cast::<ID3D12InfoQueue>().ok();
            if let Some(iq) = &iq {
                // SAFETY: clearing info queue messages on our device.
                unsafe { iq.ClearStoredMessages() };
            }
            let s = heap.dynamic(4).unwrap();
            heap.make_at(s, &placed_buffer_view(false, addr, 256, 0));
            heap.make_at(s + 1, &placed_buffer_view(true, addr, 256, 0));
            heap.make_at(s + 2, &placed_buffer_view(false, addr, 256, 2));
            heap.make_at(s + 3, &placed_buffer_view(true, addr, 256, 2));
            heap.release(s, 4);
            crate::shadow::unregister(res_id, addr);
            // SAFETY: plain device status query.
            assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
            if let Some(iq) = &iq {
                assert_no_d3d12_errors("HostHeap(default)", iq);
            }
        }

        // Build synthetic GFX10 compute shader:
        //   slot 0: 0-element raw SRV -> reads 0 into v1
        //   slot 1: 2-element raw SRV -> reads 0x1234_5678 into v2
        //   slot 2: 0-element raw UAV -> drops store of 0xDEAD_BEEF
        //   slot 3: 4-element raw UAV -> stores [v1, v2, 0xCAFE_BABE]
        let s_mov = |dst: u8, src: u8| 0xBE80_0300u32 | (dst as u32) << 16 | src as u32;
        let s_add = |dst: u8, s0: u8, s1: u8| {
            0x8000_0000u32 | (dst as u32) << 16 | (s1 as u32) << 8 | s0 as u32
        };
        let s_load4 = |dst: u8, base: u8| {
            [
                0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25,
            ]
        };
        let buf_load = |vdst: u8, srsrc: u8, off: u16| {
            [
                0x38 << 26 | 0x0C << 18 | (off as u32 & 0xFFF),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8,
            ]
        };
        let buf_store = |vsrc: u8, srsrc: u8, off: u16| {
            [
                0x38 << 26 | 0x1C << 18 | (off as u32 & 0xFFF),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vsrc as u32) << 8,
            ]
        };
        let mut words = vec![s_mov(3, 0x80), s_mov(2, 0)];
        words.extend_from_slice(&s_load4(4, 2)); // slot 0 (offset 0)
        words.push(s_add(2, 0, 128 + 32));
        words.extend_from_slice(&s_load4(8, 2)); // slot 1 (offset 32)
        words.push(s_add(2, 0, 128 + 64));
        words.extend_from_slice(&s_load4(12, 2)); // slot 2 (offset 64)
        words.push(s_add(2, 2, 128 + 32));
        words.extend_from_slice(&s_load4(16, 2)); // slot 3 (offset 96)
        words.extend_from_slice(&buf_load(1, 4, 0));
        words.extend_from_slice(&buf_load(2, 8, 0));
        words.extend_from_slice(&[0x7E06_02FF, 0xDEAD_BEEF]); // v_mov_b32 v3, 0xDEAD_BEEF
        words.extend_from_slice(&buf_store(3, 12, 0));
        words.extend_from_slice(&[0x7E08_02FF, 0xCAFE_BABE]); // v_mov_b32 v4, 0xCAFE_BABE
        words.extend_from_slice(&buf_store(1, 16, 0));
        words.extend_from_slice(&buf_store(2, 16, 4));
        words.extend_from_slice(&buf_store(4, 16, 8));
        words.push(0xBF81_0000); // s_endpgm

        let used_slot = |kind, off_dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: off_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                used_slot(ResourceType::Srv, 0),
                used_slot(ResourceType::Srv, 8),
                used_slot(ResourceType::Uav, 16),
                used_slot(ResourceType::Uav, 24),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut empty = vec![0u8; 32];
        empty[..4].copy_from_slice(b"DXBC");
        empty[20..22].copy_from_slice(&1u16.to_le_bytes());
        empty[24..28].copy_from_slice(&32u32.to_le_bytes());
        let container = dxbc::Container::parse(&empty).expect("empty container");
        let prog = ir::build(&native.code, false, native.vgprs).expect("ir");
        let st = structure::structure(&prog).expect("structure");
        let res = resource::track(&prog, &native.layout, native.stage, Some(&st)).expect("track");
        let raised = io::raise(&prog, &native).expect("io");
        let cs = emit::emit(&container, &native, &prog, &st, &raised, &res).expect("emit");

        // Enumerate all SM 6.6 hardware adapters + WARP.
        let mut adapters: Vec<(String, ID3D12Device)> = Vec::new();
        // SAFETY: plain DXGI/D3D12 enumeration and feature queries.
        unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            let check_dev = |dev: ID3D12Device| -> Option<ID3D12Device> {
                let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
                    HighestShaderModel: D3D_SHADER_MODEL_6_6,
                };
                (dev.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut sm).cast(),
                    size_of_val(&sm) as u32,
                )
                .is_ok()
                    && sm.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0)
                    .then_some(dev)
            };
            let mut i = 0;
            while let Ok(ad) = factory.EnumAdapters1(i) {
                i += 1;
                let Ok(desc) = ad.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..len]);
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&ad, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(d) = dev.and_then(check_dev)
                {
                    adapters.push((name, d));
                }
            }
            if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() {
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(d) = dev.and_then(check_dev)
                {
                    adapters.push(("WARP".into(), d));
                }
            }
        }
        assert!(!adapters.is_empty(), "no SM 6.6 adapter available");

        for (label, dev) in &adapters {
            // SAFETY: valid D3D12 dispatch, resource, heap, and info-queue calls.
            unsafe {
                let iq = dev.cast::<ID3D12InfoQueue>().ok();
                if let Some(iq) = &iq {
                    iq.ClearStoredMessages();
                }

                let param = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: emit::USER_DATA_SPACE,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
                let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                    Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                    Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                        Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                            NumParameters: 1,
                            pParameters: &param,
                            Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                            ..Default::default()
                        },
                    },
                };
                let mut blob: Option<ID3DBlob> = None;
                D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, None)
                    .expect("serialize root sig");
                let blob = blob.unwrap();
                let root: ID3D12RootSignature = dev
                    .CreateRootSignature(
                        0,
                        std::slice::from_raw_parts(
                            blob.GetBufferPointer().cast::<u8>(),
                            blob.GetBufferSize(),
                        ),
                    )
                    .expect("CreateRootSignature");
                let pso: ID3D12PipelineState = dev
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.dxil.as_ptr().cast(),
                            BytecodeLength: cs.dxil.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap_or_else(|e| panic!("{label}: CreateComputePipelineState: {e}"));

                let make_buf = |heap_ty, width, flags, state| -> ID3D12Resource {
                    let mut out: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: heap_ty,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: width,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            Flags: flags,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut out,
                    )
                    .expect("CreateCommittedResource");
                    out.unwrap()
                };

                let srv_buf = make_buf(
                    D3D12_HEAP_TYPE_UPLOAD,
                    16,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                );
                let mut p = std::ptr::null_mut();
                srv_buf.Map(0, None, Some(&mut p)).expect("Map srv_buf");
                p.cast::<[u32; 2]>().write([0x1234_5678, 0x9ABC_DEF0]);
                srv_buf.Unmap(0, None);

                let uav_buf = make_buf(
                    D3D12_HEAP_TYPE_DEFAULT,
                    16,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let rb_buf = make_buf(
                    D3D12_HEAP_TYPE_READBACK,
                    16,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );

                let heap: ID3D12DescriptorHeap = dev
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 4,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        ..Default::default()
                    })
                    .expect("CreateDescriptorHeap");
                let step = dev
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                let base = heap.GetCPUDescriptorHandleForHeapStart();
                let slot = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: base.ptr + i * step,
                };

                // Slots 0 & 2: zero-element buffer SRV & UAV routed through `null_for_on`.
                null_for_on(dev, &placed_buffer_view(false, 0x1000, 16, 0), slot(0));
                null_for_on(dev, &placed_buffer_view(true, 0x1000, 16, 0), slot(2));

                // Slots 1 & 3: non-zero-element buffer SRV (`NumElements = 2`) & UAV (`NumElements = 4`).
                dev.CreateShaderResourceView(
                    &srv_buf,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                NumElements: 2,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                                ..Default::default()
                            },
                        },
                    }),
                    slot(1),
                );
                dev.CreateUnorderedAccessView(
                    &uav_buf,
                    None::<&ID3D12Resource>,
                    Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                NumElements: 4,
                                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                ..Default::default()
                            },
                        },
                    }),
                    slot(3),
                );

                let queue: ID3D12CommandQueue = dev
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .expect("CreateCommandQueue");
                let alloc: ID3D12CommandAllocator = dev
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .expect("CreateCommandAllocator");
                let list: ID3D12GraphicsCommandList = dev
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
                    .expect("CreateCommandList");

                list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.SetComputeRootSignature(&root);
                let ud = [0u32; 4];
                list.SetComputeRoot32BitConstants(0, 4, ud.as_ptr().cast(), 0);
                list.Dispatch(1, 1, 1);
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(&uav_buf),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                            },
                        ),
                    },
                    ..Default::default()
                }]);
                list.CopyResource(&rb_buf, &uav_buf);
                list.Close().expect("Close");
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                while fence.GetCompletedValue() < 1 {
                    std::thread::yield_now();
                }

                assert!(
                    dev.GetDeviceRemovedReason().is_ok(),
                    "{label}: device was removed"
                );
                if let Some(iq) = &iq {
                    assert_no_d3d12_errors(label, iq);
                }

                let mut rp = std::ptr::null_mut();
                rb_buf.Map(0, None, Some(&mut rp)).expect("Map rb_buf");
                let got = rp.cast::<[u32; 3]>().read();
                rb_buf.Unmap(0, None);
                eprintln!(
                    "zero-element buffer view dispatch on {label} (debug_layer={}, info_queue={}): got={got:#x?}",
                    debug_enabled,
                    iq.is_some()
                );
                assert_eq!(got, [0, 0x1234_5678, 0xCAFE_BABE], "{label}: readback");
            }
        }
    }

    fn assert_no_d3d12_errors(label: &str, iq: &ID3D12InfoQueue) {
        // SAFETY: standard two-call D3D12 `GetMessage` pattern with exact buffer size.
        unsafe {
            let n = iq.GetNumStoredMessages();
            let mut bad = Vec::new();
            for i in 0..n {
                let mut len = 0usize;
                let _ = iq.GetMessage(i, None, &mut len);
                if len == 0 {
                    continue;
                }
                let mut buf = vec![0u8; len];
                let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if iq.GetMessage(i, Some(msg), &mut len).is_ok() {
                    let sev = (*msg).Severity;
                    if matches!(
                        sev,
                        D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
                    ) {
                        let desc = std::slice::from_raw_parts(
                            (*msg).pDescription,
                            (*msg).DescriptionByteLength.saturating_sub(1),
                        );
                        bad.push(String::from_utf8_lossy(desc).into_owned());
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{label}: D3D12 debug layer reported errors: {bad:#?}"
            );
        }
    }
}
