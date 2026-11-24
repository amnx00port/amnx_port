//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture when that does not hold it yet: once per
//! change, before the submission's lists run, not in every list that
//! samples the texture ([`Staging::take_upload`]).
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Whether the memory was write-watched at the last refresh.
    watched: std::sync::atomic::AtomicBool,
    /// The title resource, for notes (diagnostics).
    #[cfg(feature = "diagnostics")]
    resource: usize,
    last: Mutex<Last>,
    /// Copies recorded into the host texture (tests).
    #[cfg(test)]
    uploads: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into; and whether the host texture has it.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
    copies: Copies,
}

/// The staging buffer's contents by number (each detiling makes new ones)
/// and the number last copied into the host texture. Nothing else writes a
/// host texture whose contents come from title memory (it is no render
/// target, depth target or unordered view; a host copy or resolve into it
/// ends its uploads), so it needs a copy only when the numbers differ.
#[derive(Debug, Default)]
struct Copies {
    detiled: u64,
    uploaded: Option<u64>,
}

impl Copies {
    /// The staging buffer has new contents.
    fn detiled(&mut self) {
        self.detiled += 1;
    }

    /// Whether the host texture lacks the staging buffer's contents, which
    /// it holds from now on (the caller copies them). The first time it
    /// lacks them even if nothing was detiled yet (memory not committed).
    fn take(&mut self) -> bool {
        let due = self.uploaded != Some(self.detiled);
        self.uploaded = Some(self.detiled);
        due
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Notes, once per list, that the list reads `resource`'s host copy, which
/// must then hold what its title memory holds: at submission the memory is
/// detiled into the staging buffer if it changed, and the staging buffer is
/// copied into the host copy if that does not hold it yet, before any list
/// of the submission runs (`Host::execute`). The list itself records
/// nothing, so a texture whose memory does not change is not copied again,
/// and stays in the state the lists that read it leave it in.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (
        staging(resource, texture),
        crate::resource::host_copy(resource),
    ) else {
        return;
    };
    // Checked once per submission however many of its lists sample it; the
    // list holds both until it is reset.
    list.defer_texture(staging, copy);
}

impl Staging {
    /// The staging buffer's bytes: what an upload copies.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// At submission, after [`Staging::refresh`]: whether the host texture
    /// lacks what the staging buffer holds. It holds it from now on: the
    /// caller records [`Staging::record_upload`] before the lists run.
    pub(crate) fn take_upload(&self) -> bool {
        self.last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .copies
            .take()
    }

    /// Records the copy of the staging buffer into every subresource of
    /// `texture` (the host copy, in the copy destination state) in `list`.
    /// The staging buffer must live until the list has run.
    pub(crate) fn record_upload(&self, list: &HostList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            #[cfg(feature = "diagnostics")]
            {
                let f = footprint.Footprint;
                // SAFETY: live resources.
                let (gpu, d) = unsafe { (self.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
                let len = u64::from(f.RowPitch) * u64::from(self.rows[i]) * u64::from(f.Depth);
                list.note_read(
                    || {
                        format!(
                            "texture upload {:#x} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                            self.resource,
                            d.Width,
                            d.Height,
                            d.DepthOrArraySize,
                            d.Format.0,
                            d.MipLevels,
                            self.footprints.len(),
                            self.size,
                            footprint.Offset,
                            f.Width,
                            f.Height,
                            f.Depth,
                            f.RowPitch
                        )
                    },
                    (gpu + footprint.Offset, len),
                );
            }
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        #[cfg(test)]
        self.uploads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// About what refreshing the texture costs, in bytes to go through: the
    /// whole memory if it is hashed (not write-watched, or not known yet),
    /// next to nothing if it is watched and unchanged.
    fn work(&self) -> u64 {
        if self.watched.load(std::sync::atomic::Ordering::Relaxed) {
            0
        } else {
            self.layout.size
        }
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all). Returns the bytes of
    /// title memory read.
    fn refresh(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let watched = crate::watch::writes(start, start + size, false);
        self.watched
            .store(watched.is_some(), std::sync::atomic::Ordering::Relaxed);
        if let Some((now, written)) = watched {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            last.copies.detiled();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        last.copies.detiled();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Brings the staging buffers of a submission's textures up to date
/// (distinct ones, sampled since the last title write), on several threads
/// when there is enough to go through: each texture's hashing and
/// detiling reads its own memory and writes its own staging buffer (in
/// physical-page views, which cannot be watched, every refresh hashes the
/// whole memory).
pub(crate) fn refresh(textures: &[&Staging]) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let started = std::time::Instant::now();
    let read = AtomicU64::new(0);
    let work = textures.iter().map(|s| s.work()).sum();
    each_on_threads(textures, threads(work, textures.len()), |s| {
        read.fetch_add(s.refresh() as u64, Relaxed);
    });
    let preparing = &crate::host::PREPARING;
    preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    preparing[3].fetch_add(read.into_inner(), Relaxed);
}

/// Threads to refresh `count` textures on, going through `work` bytes.
fn threads(work: u64, count: usize) -> usize {
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cores = *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    threads_for(work, count, cores)
}

/// One thread per 4 MiB to go through (hashing that takes about a
/// millisecond, starting a thread tens of microseconds), at most one per
/// texture, per core and eight: memory bandwidth runs out before cores do.
fn threads_for(work: u64, count: usize, cores: usize) -> usize {
    const PER_THREAD: u64 = 4 << 20;
    const MOST: usize = 8;
    let most = count.min(cores).clamp(1, MOST);
    usize::try_from(work / PER_THREAD)
        .unwrap_or(usize::MAX)
        .clamp(1, most)
}

/// Calls `each` on every item, on `threads` threads (the calling one among
/// them), each taking the next item when it is done with one.
fn each_on_threads<T: Sync>(items: &[T], threads: usize, each: impl Fn(&T) + Sync) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let next = AtomicUsize::new(0);
    let work = || {
        while let Some(item) = items.get(next.fetch_add(1, Relaxed)) {
            each(item);
        }
    };
    if threads <= 1 {
        return work();
    }
    std::thread::scope(|scope| {
        for _ in 1..threads {
            // One that does not start leaves its share to the others.
            let _ = std::thread::Builder::new()
                .name("victory-textures".into())
                .spawn_scoped(scope, work);
        }
        work();
    });
}

/// How many times `resource`'s host copy was given its staging buffer
/// (tests).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|m| m.get(&(resource as usize)))
        .map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        watched: Default::default(),
        #[cfg(feature = "diagnostics")]
        resource: resource as usize,
        last: Mutex::default(),
        #[cfg(test)]
        uploads: Default::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::{Copies, each_on_threads, threads_for};

    #[test]
    fn a_host_texture_is_copied_once_per_new_contents() {
        let mut copies = Copies::default();
        // The first check copies, detiled or not (memory not committed) ...
        assert!(copies.take());
        assert!(!copies.take());
        // ... later ones only after the staging buffer has new contents,
        copies.detiled();
        assert!(copies.take());
        assert!(!copies.take());
        // once however often it was detiled since.
        copies.detiled();
        copies.detiled();
        assert!(copies.take());
        assert!(!copies.take());
    }

    #[test]
    fn every_texture_is_refreshed_once_on_any_number_of_threads() {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
        for (items, threads) in [(0, 1), (0, 4), (1, 8), (5, 1), (1000, 3), (1000, 8)] {
            let seen: Vec<AtomicU32> = (0..items).map(|_| AtomicU32::new(0)).collect();
            let indices: Vec<usize> = (0..items).collect();
            each_on_threads(&indices, threads, |&i| {
                seen[i].fetch_add(1, Relaxed);
            });
            let once = seen.iter().all(|n| n.load(Relaxed) == 1);
            assert!(once, "{items} items on {threads} threads");
        }
    }

    #[test]
    fn threads_go_where_there_is_hashing_to_do() {
        const MIB: u64 = 1 << 20;
        // Little to go through (watched memory, small textures): this one.
        assert_eq!(threads_for(0, 100, 16), 1);
        assert_eq!(threads_for(7 * MIB, 100, 16), 1);
        // One per 4 MiB ...
        assert_eq!(threads_for(12 * MIB, 100, 16), 3);
        // ... and at most one per texture, per core and eight.
        assert_eq!(threads_for(1 << 30, 2, 16), 2);
        assert_eq!(threads_for(1 << 30, 100, 4), 4);
        assert_eq!(threads_for(1 << 30, 100, 64), 8);
        assert_eq!(threads_for(1 << 30, 0, 0), 1);
    }

    /// What a submission's checks of textures in physical-page memory cost
    /// (each hashed whole): one after another (before), and on the threads
    /// a refresh of that much takes (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_hashing`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_hashing_on_threads() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        let textures: Vec<Vec<u8>> = (0..TEXTURES)
            .map(|t| (0..SIZE).map(|i| (i * 31 + t) as u8).collect())
            .collect();
        let threads = super::threads((TEXTURES * SIZE) as u64, TEXTURES);
        let hash_all = |threads: usize| {
            let started = std::time::Instant::now();
            each_on_threads(&textures, threads, |t| {
                std::hint::black_box(blake3::hash(t));
            });
            started.elapsed()
        };
        hash_all(threads); // warm up
        let before = hash_all(1);
        let after = hash_all(threads);
        println!(
            "{TEXTURES} textures of {} MiB hashed: one thread {before:?}, {threads} threads \
             {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}
