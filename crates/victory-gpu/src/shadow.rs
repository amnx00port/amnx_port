//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    pub(crate) len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Option<(usize, Vec<u8>)>>,
    /// The buffer's current host D3D12 resource state (`0`: `COMMON`).
    pub(crate) state: AtomicU32,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Drop for Shadow {
    fn drop(&mut self) {
        // Defer releasing the underlying GPU buffer across subsequent frames
        // so directly-indexed descriptors referencing it cannot page-fault.
        crate::host::retire_resource(self.buffer.clone());
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            state: AtomicU32::new(0),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, address.saturating_add(len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        let base = self.start as usize;
        let range = (base + from, base + to);
        let prior_busy: Vec<(usize, usize)> = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let prior = pending
                .iter()
                .copied()
                .filter(|&(a, e)| a < range.1 && e > range.0)
                .collect();
            pending.push(range);
            prior
        };
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &mut *before
            && (*at, *at + bytes.len()) == (start, end)
        {
            // Refresh any uncovered gaps inside [from, to) that an intervening
            // `update()` may have written since the snapshot was first taken (#906).
            refresh_snapshot_gaps(*at, bytes, from, to, base, &prior_busy, host);
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                let old_end = at + bytes.len();
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(old_end, end));
                give_back(bytes);
                let overlap_from = from.max(at);
                let overlap_to = to.min(old_end);
                if overlap_from < overlap_to {
                    refresh_snapshot_gaps(
                        start,
                        &mut grown,
                        overlap_from,
                        overlap_to,
                        base,
                        &prior_busy,
                        host,
                    );
                }
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before_all)) = &mut *snapshot
            && *taken <= from
            && to <= *taken + before_all.len()
            && from < to
        {
            let taken = *taken;
            let before = &mut before_all[from - taken..to - taken];
            // By 64 KiB piece: stage each piece out of write-combined GPU
            // memory into cached RAM once using reusable thread-local scratch
            // buffers, and coalesce contiguous modified pieces into a single
            // `watch::writing` + `write_mapped` run (#918).
            const PIECE: usize = 64 << 10;
            thread_local! {
                static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>, Vec<u8>)> = const {
                    std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new()))
                };
            }
            let mut synced = Vec::new();
            SCRATCH.with(|cell| {
                let mut triple = cell.borrow_mut();
                if triple.0.len() < PIECE {
                    triple.0.resize(PIECE, 0);
                }
                if triple.1.len() < PIECE {
                    triple.1.resize(PIECE, 0);
                }
                let (title_buf, host_buf, run_buf) = &mut *triple;
                run_buf.clear();
                let mut run_start = 0usize;
                let flush_run = |run_start: usize, run_buf: &mut Vec<u8>, written: &mut usize| {
                    if !run_buf.is_empty() {
                        crate::watch::writing(run_start, run_buf.len());
                        crate::host::write_mapped(
                            run_start as *mut u8,
                            run_buf.as_ptr(),
                            run_buf.len(),
                        );
                        *written += run_buf.len();
                        run_buf.clear();
                    }
                };
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the mapping; only the submitting thread
                    // accesses it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let old = &mut before[at - from..at - from + n];
                    let host_ram = &mut host_buf[..n];
                    host_ram.copy_from_slice(host);
                    let title = &mut title_buf[..n];
                    if host_ram != old && crate::host::read_into((start + at) as *const u8, title) {
                        // Title memory with the GPU's bytes put in; write back
                        // to the host shadow buffer only if title had newer
                        // bytes the GPU did not overwrite.
                        let mut title_changed = false;
                        let mut host_changed = false;
                        for ((t, h), o) in title.iter_mut().zip(host_ram.iter_mut()).zip(old.iter())
                        {
                            if *h != *o && *t != *h {
                                *t = *h;
                                title_changed = true;
                            } else if *h != *t {
                                *h = *t;
                                host_changed = true;
                            }
                        }
                        if title_changed {
                            if !run_buf.is_empty() && run_start + run_buf.len() != start + at {
                                flush_run(run_start, run_buf, &mut written);
                            }
                            if run_buf.is_empty() {
                                run_start = start + at;
                            }
                            run_buf.extend_from_slice(title);
                        } else {
                            flush_run(run_start, run_buf, &mut written);
                        }
                        if host_changed {
                            host.copy_from_slice(host_ram);
                        }
                        // Keep snapshot in sync with the post-merge bytes so
                        // subsequent overlapping copy_backs in the same
                        // submission do not re-apply stale diffs (#906).
                        old.copy_from_slice(host_ram);
                        synced.push((start + at) / PAGE);
                    } else {
                        flush_run(run_start, run_buf, &mut written);
                    }
                    at += n;
                }
                flush_run(run_start, run_buf, &mut written);
            });
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Remove one matching pending entry; drop the snapshot once the last
        // pending range finishes (#906).
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let key = (start + from, start + to);
        if let Some(pos) = pending.iter().position(|&r| r == key) {
            pending.swap_remove(pos);
        }
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// Refreshes any subranges of `[from, to)` inside snapshot `bytes` (starting at
/// offset `snap_start`) that were not already covered by `prior_busy`, so
/// intervening CPU updates in snapshot gaps are not later diffed as GPU writes
/// (#906).
fn refresh_snapshot_gaps<'a, F>(
    snap_start: usize,
    bytes: &mut [u8],
    from: usize,
    to: usize,
    base: usize,
    prior_busy: &[(usize, usize)],
    host: F,
) where
    F: Fn(usize, usize) -> &'a [u8],
{
    let mut busy: Vec<(usize, usize)> = prior_busy
        .iter()
        .filter_map(|&(a, e)| {
            let a = a.saturating_sub(base).max(from);
            let e = e.saturating_sub(base).min(to);
            (a < e).then_some((a, e))
        })
        .collect();
    busy.sort_unstable();
    let mut at = from;
    for (a, e) in busy {
        if a > at {
            bytes[at - snap_start..a - snap_start].copy_from_slice(host(at, a));
        }
        at = at.max(e);
    }
    if at < to {
        bytes[at - snap_start..to - snap_start].copy_from_slice(host(at, to));
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let Some(end) = start.checked_add(len) else {
        return;
    };
    buffers().insert((start, resource), Entry { end, shadow: None });
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .filter(|(_, e)| e.end >= end)
        .min_by_key(|(k, e)| e.end - k.0)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Like [`find`], but when `address` lies outside registered buffers (raw
/// memory root SRVs, e.g. large mesh/geometry streams), extends the loose copy
/// up to `reach` bytes across contiguous committed memory so multi-megabyte
/// root SRV reads do not fault past a 4 MiB window boundary (#951).
pub(crate) fn find_wide(address: u64, reach: u64) -> Option<Arc<Shadow>> {
    let min_end = address.checked_add(1)?;
    {
        let map = buffers();
        if map
            .range(..=(address, usize::MAX))
            .rev()
            .any(|(_, e)| e.end >= min_end)
        {
            drop(map);
            return find(address, 1);
        }
    }
    let span = committed_span(address, reach.max(1)).max(1);
    let end = address.checked_add(span)?;
    loose(address, end).or_else(|| find(address, 1))
}

/// Contiguous committed virtual bytes starting at `address`, capped at `max`.
fn committed_span(address: u64, max: u64) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut cur = address as usize;
    let mut total = 0u64;
    while total < max {
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer of the exact `MEMORY_BASIC_INFORMATION` size.
        let ok = unsafe {
            VirtualQuery(
                cur as *const _,
                &mut info,
                std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if ok == 0 || info.State != MEM_COMMIT || info.RegionSize == 0 {
            break;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if region_end <= cur {
            break;
        }
        let step = ((region_end - cur) as u64).min(max - total);
        total += step;
        let Some(next) = cur.checked_add(step as usize) else {
            break;
        };
        cur = next;
    }
    total
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)?.checked_sub(start)?;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    // Reuse any already-created wider loose copy covering [address, end).
    if let Some(shadow) = map
        .values()
        .find(|s| s.start <= address && s.start.saturating_add(s.len) >= end)
    {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_snapshot_gaps_updates_only_uncovered_subranges() {
        let base = 0x1000usize;
        let mut snap = vec![0xAAu8; 32];
        let live = [0xBBu8; 32];
        // Existing busy ranges cover [4..10) and [18..24) in buffer offsets.
        let prior = [(base + 4, base + 10), (base + 18, base + 24)];
        refresh_snapshot_gaps(0, &mut snap, 2, 28, base, &prior, |a, b| &live[a..b]);
        // Untouched outside [2..28):
        assert_eq!(&snap[0..2], &[0xAA; 2]);
        assert_eq!(&snap[28..32], &[0xAA; 4]);
        // Refreshed in gaps [2..4), [10..18), [24..28):
        assert_eq!(&snap[2..4], &[0xBB; 2]);
        assert_eq!(&snap[10..18], &[0xBB; 8]);
        assert_eq!(&snap[24..28], &[0xBB; 4]);
        // Preserved inside already-busy ranges [4..10) and [18..24):
        assert_eq!(&snap[4..10], &[0xAA; 6]);
        assert_eq!(&snap[18..24], &[0xAA; 6]);
    }

    #[test]
    fn register_and_loose_reject_overflowing_ranges() {
        register(0xDEAD_BEEF, u64::MAX - 8, 64);
        assert!(buffers().get(&(u64::MAX - 8, 0xDEAD_BEEF)).is_none());
        assert!(loose(u64::MAX - 16, u64::MAX).is_none());
    }

    #[test]
    fn copy_back_coalesces_contiguous_pieces_and_merges_gpu_writes() {
        let Some(host) = host() else { return };
        // Three 64 KiB pages of title memory so copy_back spans 3 contiguous pieces.
        let size = 3 * (64 << 10);
        let mut title = vec![0x11u8; size];
        let addr = title.as_mut_ptr() as u64;
        let Some((buffer, data)) = host.shared_buffer(size as u64) else {
            return;
        };
        // SAFETY: live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, addr, size as u64);
        // Copy initial title bytes into host shadow buffer.
        // SAFETY: `data` covers `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr(), data, size) };
        shadow.writing(addr as usize, size);
        // Simulate GPU writing bytes in every 64 KiB piece (including head and
        // tail regardless of buffer alignment) while title concurrently
        // updates an unrelated byte in page 1.
        // SAFETY: offsets inside `size`.
        unsafe {
            *data.add(0) = 0x55;
            *data.add(64 << 10) = 0x66;
            *data.add(128 << 10) = 0x77;
            *data.add(size - 1) = 0x88;
        }
        title[(64 << 10) + 100] = 0x99;
        let written = shadow.copy_back(addr as usize, size);
        assert_eq!(written, size);
        assert_eq!(title[0], 0x55);
        assert_eq!(title[64 << 10], 0x66);
        assert_eq!(title[128 << 10], 0x77);
        assert_eq!(title[size - 1], 0x88);
        // Concurrent title write must be preserved and synced back to host shadow.
        assert_eq!(title[(64 << 10) + 100], 0x99);
        // SAFETY: offset inside `size`.
        assert_eq!(unsafe { *data.add((64 << 10) + 100) }, 0x99);
    }
}
