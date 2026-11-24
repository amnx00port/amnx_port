//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
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
    /// Its bytes (`GetCopyableFootprints` total).
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the 128-bit digest of the memory
/// detiled (memory not watched) along with the physical-pool write epoch at
/// that digest, or the write-watch asking it was detiled after, with room to
/// read the memory into.
#[derive(Default)]
struct Last {
    digest: Option<[u64; 2]>,
    epoch: u64,
    asked: u64,
    /// Cached `true` once `GetWriteWatch` failed on committed physical-page
    /// memory so later submissions skip the global journal lock and kernel
    /// `GetWriteWatch` call (#916).
    not_watched: bool,
    read: Vec<u8>,
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

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
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
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
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
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let Some(end) = start.checked_add(size) else {
            return 0;
        };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if !last.not_watched
            && let Some((now, written)) = crate::watch::writes(start, end, false)
        {
            let prev = last.asked;
            if prev != 0 && written.iter().all(|&(_, w)| w <= prev) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            // If we already detiled once, only re-detile subresources whose
            // tiled byte extents overlap pages written since `prev` (#915).
            let dirty: Option<Vec<(u64, u64)>> = (prev != 0).then(|| {
                written
                    .iter()
                    .filter(|&&(_, w)| w > prev)
                    .map(|&(page, _)| {
                        let lo = page.saturating_sub(start) as u64;
                        let hi = page.saturating_add(0x1000).saturating_sub(start).min(size) as u64;
                        (lo, hi)
                    })
                    .collect()
            });
            self.detile_ranges(&last.read, dirty.as_deref());
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // Fast path: if no physical-page writes or remappings occurred since
        // this texture was last digested, its contents cannot have changed.
        let epoch = crate::watch::physical_epoch();
        if last.digest.is_some() && last.epoch == epoch {
            return 0;
        }
        // Streamed textures mapped via physical-page section views may be only
        // partially resident (for example, mip tail resident while top mips
        // are still `MEM_RESERVE`). Read every committed region safely into
        // `last.read` (zeroing uncommitted gaps) so resident mips upload
        // instead of leaving the whole texture black and eliminating TOCTOU
        // faults against concurrent unmaps.
        last.read.resize(size, 0);
        if !read_resident(self.memory, &mut last.read) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory has no committed pages"
            );
            return 0;
        }
        // Memory is committed and not write-watched (physical-page view): skip
        // future `GetWriteWatch` kernel calls on this texture (#916).
        last.not_watched = true;
        let digest = fast_digest_128(&last.read);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        last.epoch = epoch;
        if last.digest == Some(digest) {
            return size;
        }
        last.digest = Some(digest);
        self.detile(&last.read);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.detile_ranges(tiled, None);
    }

    /// Detiles subresources whose tiled extents intersect `dirty` (or all
    /// subresources when `dirty` is `None`).
    fn detile_ranges(&self, tiled: &[u8], dirty: Option<&[(u64, u64)]>) {
        #[cfg(feature = "diagnostics")]
        if dirty.is_none() && tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            if let Some(ranges) = dirty
                && let Ok((lo, hi)) = self.layout.extents(mip, slice)
                && !ranges.iter().any(|&(a, b)| a < hi && lo < b)
            {
                continue;
            }
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let Some(len) = footprint_slice_len(self.size, footprint.Offset, f.RowPitch, rows, f.Depth)
            else {
                tracing::warn!(mip, slice, offset = footprint.Offset, size = self.size, "staging footprint out of bounds");
                return;
            };
            // SAFETY: `footprint.Offset + len <= self.size` inside the mapped staging buffer.
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

/// Byte length of one placed subresource slice inside a staging buffer of
/// `total` bytes. `GetCopyableFootprints` omits trailing alignment padding on
/// the final row, so `RowPitch * rows * depth` can exceed `total - offset`.
fn footprint_slice_len(
    total: u64,
    offset: u64,
    row_pitch: u32,
    rows: u32,
    depth: u32,
) -> Option<usize> {
    let avail = usize::try_from(total.checked_sub(offset)?).ok()?;
    let full = (row_pitch as usize)
        .checked_mul(rows as usize)?
        .checked_mul(depth as usize)?;
    Some(full.min(avail))
}

/// Fast non-cryptographic 128-bit digest for change detection on streamed
/// texture pools (#916): 4-lane unrolled 64-bit mixing runs at memory
/// bandwidth instead of full cryptographic blake3.
fn fast_digest_128(data: &[u8]) -> [u64; 2] {
    const K0: u64 = 0x9E37_79B9_7F4A_7C15;
    const K1: u64 = 0xBF58_476D_1CE4_E5B9;
    const K2: u64 = 0x94D0_49BB_1331_11EB;
    const K3: u64 = 0xD6E8_FEB8_6659_FD93;
    let mut s0 = K0 ^ (data.len() as u64);
    let mut s1 = K1;
    let mut s2 = K2;
    let mut s3 = K3;
    let (chunks, rem) = data.as_chunks::<32>();
    for c in chunks {
        let a = u64::from_le_bytes(c[0..8].try_into().unwrap());
        let b = u64::from_le_bytes(c[8..16].try_into().unwrap());
        let x = u64::from_le_bytes(c[16..24].try_into().unwrap());
        let y = u64::from_le_bytes(c[24..32].try_into().unwrap());
        s0 = (s0 ^ a).wrapping_mul(K1).rotate_left(27);
        s1 = (s1 ^ b).wrapping_mul(K2).rotate_left(31);
        s2 = (s2 ^ x).wrapping_mul(K3).rotate_left(23);
        s3 = (s3 ^ y).wrapping_mul(K0).rotate_left(29);
    }
    let (tail, last_bytes) = rem.as_chunks::<8>();
    for t in tail {
        let w = u64::from_le_bytes(*t);
        s0 = (s0 ^ w).wrapping_mul(K1).rotate_left(27);
        std::mem::swap(&mut s0, &mut s1);
    }
    let mut last = 0u64;
    for (i, &b) in last_bytes.iter().enumerate() {
        last |= u64::from(b) << (i * 8);
    }
    if !last_bytes.is_empty() {
        s2 = (s2 ^ last).wrapping_mul(K2).rotate_left(23);
    }
    let mix = |mut x: u64| {
        x = (x ^ (x >> 30)).wrapping_mul(K1);
        x = (x ^ (x >> 27)).wrapping_mul(K2);
        x ^ (x >> 31)
    };
    let lo = mix(s0 ^ s2.rotate_left(17) ^ (data.len() as u64));
    let hi = mix(s1 ^ s3.rotate_left(13) ^ lo);
    [lo, hi]
}

/// Copies all committed regions of `[at, at + dst.len())` into `dst`, zeroing
/// uncommitted gaps. Returns `true` if at least one byte was committed.
fn read_resident(at: *const u8, dst: &mut [u8]) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let base = at as usize;
    let total = dst.len();
    let Some(end) = base.checked_add(total) else {
        dst.fill(0);
        return false;
    };
    let mut p = base;
    let mut any = false;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 {
            dst[p - base..].fill(0);
            break;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        let len = region_end.saturating_sub(p);
        if len == 0 {
            break;
        }
        let off = p - base;
        if info.State == MEM_COMMIT {
            any = true;
            // `copy_mapped` tolerates a page unmapped concurrently between
            // `VirtualQuery` and the copy, zero-filling any faulting page.
            crate::host::copy_mapped(
                dst[off..off + len].as_mut_ptr(),
                p as *const u8,
                len,
                false,
            );
        } else {
            dst[off..off + len].fill(0);
        }
        p = region_end;
    }
    any
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive). Rejects overflowing spans (#935).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT || info.RegionSize == 0 {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            return true;
        };
        if next <= p {
            return false;
        }
        p = next;
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
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::{committed, fast_digest_128, footprint_slice_len};

    #[test]
    fn footprint_slice_is_capped_to_staging_total() {
        // 4x4 RGBA8: RowPitch is 256, 4 rows, total = 3*256 + 16 = 784 < 1024.
        assert_eq!(footprint_slice_len(784, 0, 256, 4, 1), Some(784));
        // Subresource at offset 512 in a 1000-byte buffer.
        assert_eq!(footprint_slice_len(1000, 512, 256, 4, 1), Some(488));
        // Offset past the buffer end is rejected.
        assert_eq!(footprint_slice_len(784, 800, 256, 4, 1), None);
    }

    #[test]
    fn committed_rejects_overflowing_spans() {
        let buf = [0u8; 64];
        assert!(committed(buf.as_ptr(), buf.len()));
        assert!(!committed(usize::MAX as *const u8, 16));
        assert!(!committed((usize::MAX - 8) as *const u8, 64));
    }

    #[test]
    fn fast_digest_128_detects_single_bit_and_length_changes() {
        let mut a = vec![0x5Au8; 100];
        let d0 = fast_digest_128(&a);
        assert_eq!(d0, fast_digest_128(&a));
        // Flip one bit in the middle and one bit in the tail.
        a[41] ^= 1;
        let d1 = fast_digest_128(&a);
        assert_ne!(d0, d1);
        a[41] ^= 1;
        a[99] ^= 0x80;
        let d2 = fast_digest_128(&a);
        assert_ne!(d0, d2);
        // Different length with same prefix must differ.
        assert_ne!(d0, fast_digest_128(&a[..99]));
    }
}
