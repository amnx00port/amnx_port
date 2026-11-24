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
use std::sync::atomic::{AtomicU8, Ordering};
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
    len: u64,
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
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            address.saturating_add(len).clamp(from, end) - start,
        )
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
        let end = address.saturating_add(len);
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if pending.contains(&(address, end)) {
                return;
            }
            pending.push((address, end));
        }
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, bytes)) = &mut *before
            && *at <= from
            && to <= *at + bytes.len()
        {
            // Snapshot union already covers `[from, to)`; refresh only this
            // newly declared range in place so a refresh since the surrounding
            // snapshot was taken cannot leave a stale baseline, without
            // reallocating or recopying the entire union.
            bytes[from - *at..to - *at].copy_from_slice(host(from, to));
            return;
        }
        let (start, finish) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(finish - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), finish));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, finish)),
        }
        // The range just declared is snapshotted as it is now: a refresh
        // since the surrounding snapshot was taken must not leave a stale
        // baseline (a gap turned into a write would replay the old bytes).
        grown[from - start..to - start].copy_from_slice(host(from, to));
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
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
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
        // The copy-back leaves the copy equal to title memory over the range;
        // the snapshot takes those bytes so an overlapping copy-back does not
        // replay a title edit the GPU did not make.
        if let Some((taken, before)) = snapshot.as_mut()
            && *taken <= from
            && to <= *taken + before.len()
            && from < to
        {
            let (s, e) = (from - *taken, to - *taken);
            // SAFETY: inside the mapping.
            let host = unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) };
            before[s..e].copy_from_slice(host);
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address.saturating_add(len)));
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

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

struct Buffers {
    by_start: BTreeMap<(u64, usize), Entry>,
    /// Longest registered buffer length; bounds reverse scans in [`find`].
    max_len: u64,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    by_start: BTreeMap::new(),
    max_len: 0,
});

fn buffers() -> std::sync::MutexGuard<'static, Buffers> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let Some(end) = start.checked_add(len) else {
        return;
    };
    let mut state = buffers();
    let replaced = state
        .by_start
        .insert((start, resource), Entry { end, shadow: None });
    if len >= state.max_len {
        state.max_len = len;
    } else if replaced.is_some_and(|old| old.end - start == state.max_len) {
        state.max_len = state
            .by_start
            .iter()
            .map(|(&(s, _), e)| e.end - s)
            .max()
            .unwrap_or(0);
    }
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let mut state = buffers();
    if let Some(removed) = state.by_start.remove(&(start, resource))
        && removed.end - start == state.max_len
    {
        state.max_len = state
            .by_start
            .iter()
            .map(|(&(s, _), e)| e.end - s)
            .max()
            .unwrap_or(0);
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let required = len.max(1);
    let end = address.checked_add(required)?;
    let mut state = buffers();
    // No registered buffer can cover `[address, end)` if `required > max_len`,
    // or if its start is below `end - max_len`.
    let mut key: Option<((u64, usize), u64)> = None;
    if required <= state.max_len {
        let min_start = end.saturating_sub(state.max_len);
        // The innermost buffer holding the range: the largest start at or below
        // `address` whose end reaches `end`; of equal starts, the smallest end.
        // Same-start entries are contiguous, so the first covering block found
        // in start-descending order is the innermost.
        for (&k, e) in state
            .by_start
            .range((min_start, 0)..=(address, usize::MAX))
            .rev()
        {
            if let Some((best, bend)) = &key {
                if best.0 != k.0 {
                    break;
                }
                if e.end >= end && e.end < *bend {
                    key = Some((k, e.end));
                }
            } else if e.end >= end {
                key = Some((k, e.end));
            }
        }
    }
    let Some((key, _)) = key else {
        drop(state);
        return loose(address, end);
    };
    let entry = state.by_start.get_mut(&key)?;
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
        .by_start
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
    fn overflowing_registrations_do_not_enter_the_map() {
        let start = u64::MAX - 8;
        register(0, start, 9);
        assert!(!buffers().by_start.contains_key(&(start, 0)));
    }

    #[test]
    fn overflowing_loose_window_is_rejected() {
        assert!(find(u64::MAX, 1).is_none());
        assert!(find(u64::MAX - 1, 1).is_none());
    }

    #[test]
    fn max_len_tracks_register_replace_and_unregister() {
        // Use a distinctive large length above any concurrent test's buffers.
        let base = 0x7000_0000_0000u64;
        let big = 0x10_0000_0000u64;
        let mid = 0x08_0000_0000u64;
        register(9001, base, mid);
        assert!(buffers().max_len >= mid);
        register(9002, base + 0x1000, big);
        assert_eq!(buffers().max_len, big);
        // Shrinking the current longest entry recomputes max_len.
        register(9002, base + 0x1000, 64);
        assert_eq!(buffers().max_len, mid);
        // Unregistering the remaining longest entry recomputes max_len again.
        unregister(9001, base);
        assert!(buffers().max_len < mid);
        unregister(9002, base + 0x1000);
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod warp_tests {
    use super::*;

    fn warp() -> Option<&'static crate::host::Host> {
        if std::env::var_os("VICTORY_D3D12_WARP").is_none() {
            eprintln!("skipping live D3D12 test: VICTORY_D3D12_WARP is unset");
            return None;
        }
        Some(host().expect("create the requested WARP device"))
    }

    fn shadow(device: &crate::host::Host, title: &mut [u8]) -> Shadow {
        let (buffer, data) = device
            .shared_buffer(PAGE as u64)
            .expect("map a WARP shadow buffer");
        // SAFETY: the buffer is live and mapped for at least title.len() bytes.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Shadow::new(
            buffer,
            data,
            gpu,
            title.as_mut_ptr() as u64,
            title.len() as u64,
        )
    }

    struct Registration(usize, u64);

    impl Registration {
        fn new(resource: usize, start: u64, len: u64) -> Self {
            register(resource, start, len);
            Self(resource, start)
        }
    }

    impl Drop for Registration {
        fn drop(&mut self) {
            unregister(self.0, self.1);
        }
    }

    #[test]
    fn lookup_checks_past_noncovering_buffers() {
        let Some(_device) = warp() else { return };
        let mut title = vec![0u8; 160];
        let start = title.as_mut_ptr() as u64;
        let _outer = Registration::new(0, start, 160);
        let _short: Vec<_> = (1..=70)
            .map(|i| Registration::new(0, start + i, 1))
            .collect();
        let copy = find(start + 90, 4).unwrap();
        assert_eq!(copy.offset(start + 90), 90);
    }

    #[test]
    fn equal_start_lookup_chooses_the_smallest_covering_buffer() {
        let Some(_device) = warp() else { return };
        let mut title = vec![0u8; 64];
        let start = title.as_mut_ptr() as u64;
        let _short = Registration::new(1, start, 8);
        let _long = Registration::new(2, start, 64);
        assert_eq!(find(start + 4, 4).unwrap().left(start + 4), 4);
        assert_eq!(find(start + 4, 5).unwrap().left(start + 4), 60);
    }

    #[test]
    fn oversized_updates_and_writes_clip_to_the_buffer() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start - 2, usize::MAX), title.len());
        copy.writing(start + 14, usize::MAX);
        // SAFETY: within the live mapping; the title and host copy are distinct.
        unsafe { copy.data.add(15).write(9) };
        assert_eq!(copy.copy_back(start + 14, usize::MAX), 2);
        assert_eq!(title[15], 9);
    }

    #[test]
    fn refresh_past_the_end_does_not_underflow() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = Arc::new(shadow(device, &mut title));
        let list = device.new_list().unwrap();
        copy.refresh(&list, copy.start + copy.len + 1, 1);
    }

    #[test]
    fn a_newly_declared_gap_takes_its_own_baseline() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start, title.len()), title.len());
        copy.writing(start, 2);
        copy.writing(start + 12, 2);
        title[6] = 2;
        assert_eq!(copy.update(start + 6, 1), 1);
        copy.writing(start + 6, 1);
        title[6] = 3;
        assert_eq!(copy.copy_back(start + 6, 1), 0);
        assert_eq!(title[6], 3);
        assert_eq!(copy.copy_back(start, 2), 0);
        assert_eq!(copy.copy_back(start + 12, 2), 0);
    }

    #[test]
    fn overlapping_copy_backs_do_not_replay_an_earlier_cpu_edit() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start, title.len()), title.len());
        copy.writing(start, 8);
        copy.writing(start + 4, 8);
        title[5] = 2;
        // SAFETY: within the live mapping; simulates a GPU write before copy-back.
        unsafe { copy.data.write(9) };
        assert_eq!(copy.copy_back(start, 8), 8);
        assert_eq!(title[0], 9);
        title[5] = 3;
        assert_eq!(copy.copy_back(start + 4, 8), 0);
        assert_eq!(title[5], 3);
    }

    #[test]
    fn covered_subrange_refreshes_in_place_without_reallocating_snapshot() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 32];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start, title.len()), title.len());
        // Establish a snapshot union covering [0, 32) while leaving [10, 14) undeclared.
        copy.writing(start, 4);
        copy.writing(start + 28, 4);
        let ptr_before = {
            let guard = copy.before.lock().unwrap();
            guard.as_ref().unwrap().1.as_ptr()
        };
        // Update the gap inside the already-covered union and declare it.
        title[10] = 7;
        assert_eq!(copy.update(start + 10, 4), 4);
        copy.writing(start + 10, 4);
        assert_eq!(title[10], 7);
        let (ptr_after, byte10) = {
            let guard = copy.before.lock().unwrap();
            let (_, bytes) = guard.as_ref().unwrap();
            (bytes.as_ptr(), bytes[10])
        };
        assert_eq!(ptr_before, ptr_after, "covered subrange must not reallocate");
        assert_eq!(byte10, 7, "newly declared subrange baseline must refresh");
    }
}
