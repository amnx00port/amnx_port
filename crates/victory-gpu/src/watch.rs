//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// Pages already asked of the kernel during a submission's preparation on
/// this thread. While `Host::execute` prepares a submission, every title
/// write the submission must see has already happened (except steps on this
/// thread that write title memory, which call [`clear_sweep`]); the journal
/// still holds what the earlier asking found, so nested and overlapping
/// ranges reuse it without repeating `GetWriteWatch`.
#[derive(Default)]
struct SweepState {
    active: bool,
    /// Sorted, disjoint, non-adjacent half-open page intervals `[first, last)`.
    swept: Vec<(usize, usize)>,
}

thread_local! {
    static SWEEP: RefCell<SweepState> = const {
        RefCell::new(SweepState {
            active: false,
            swept: Vec::new(),
        })
    };
}

/// Guard keeping a per-submission write-watch sweep active on this thread
/// until dropped.
pub(crate) struct Sweep {
    was_active: bool,
    prev: Vec<(usize, usize)>,
}

/// Begins a write-watch sweep on this thread until the returned guard drops.
pub(crate) fn sweep() -> Sweep {
    SWEEP.with(|cell| {
        let mut s = cell.borrow_mut();
        let was_active = s.active;
        let prev = if was_active {
            std::mem::take(&mut s.swept)
        } else {
            s.swept.clear();
            Vec::new()
        };
        s.active = true;
        Sweep { was_active, prev }
    })
}

/// Clears what the current thread's sweep has asked so far, so later askings
/// see writes made during submission preparation.
pub(crate) fn clear_sweep() {
    SWEEP.with(|cell| {
        cell.borrow_mut().swept.clear();
    });
}

impl Drop for Sweep {
    fn drop(&mut self) {
        SWEEP.with(|cell| {
            let mut s = cell.borrow_mut();
            s.active = self.was_active;
            if self.was_active {
                s.swept = std::mem::take(&mut self.prev);
            } else {
                s.swept.clear();
            }
        });
    }
}

/// Sub-intervals of `[first, last)` not covered by sorted disjoint `swept`.
fn uncovered(swept: &[(usize, usize)], first: usize, last: usize) -> Vec<(usize, usize)> {
    let mut gaps = Vec::new();
    let mut at = first;
    for &(a, b) in swept {
        if b <= at {
            continue;
        }
        if a >= last {
            break;
        }
        if a > at {
            gaps.push((at, a));
        }
        at = at.max(b);
        if at >= last {
            break;
        }
    }
    if at < last {
        gaps.push((at, last));
    }
    gaps
}

/// Inserts `[first, last)` into sorted disjoint `swept`, merging overlaps
/// and adjacent intervals.
fn mark_swept(swept: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    if first >= last {
        return;
    }
    let pos = swept.partition_point(|&(_, b)| b < first);
    let mut end_idx = pos;
    let (mut merged_a, mut merged_b) = (first, last);
    while end_idx < swept.len() && swept[end_idx].0 <= merged_b {
        merged_a = merged_a.min(swept[end_idx].0);
        merged_b = merged_b.max(swept[end_idx].1);
        end_idx += 1;
    }
    swept.splice(pos..end_idx, [(merged_a, merged_b)]);
}

/// Queries the kernel (or view guards if `guarded`) for one uncovered page
/// interval `[first, last)`. Appends dirty page numbers to `dirty`.
fn query_gap(
    j: &mut Journal,
    first: usize,
    last: usize,
    guarded: bool,
    dirty: &mut Vec<usize>,
) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    #[cfg(test)]
    tests::count_kernel_query();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        if !guarded {
            return false;
        }
        let Some(pages) = guard(first * PAGE, last * PAGE) else {
            return false;
        };
        dirty.extend(pages);
    } else {
        dirty.extend(j.found[..count].iter().map(|&at| at / PAGE));
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    true
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let gaps = if first >= last {
        vec![(first, last)]
    } else {
        SWEEP.with(|cell| {
            let s = cell.borrow();
            if s.active {
                uncovered(&s.swept, first, last)
            } else {
                vec![(first, last)]
            }
        })
    };
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut dirty = Vec::new();
    let mut swept_gaps = Vec::new();
    let mut failed_any = false;
    for &(g_first, g_last) in &gaps {
        if query_gap(j, g_first, g_last, guarded, &mut dirty) {
            swept_gaps.push((g_first, g_last));
        } else {
            failed_any = true;
            break;
        }
    }
    if !swept_gaps.is_empty() {
        SWEEP.with(|cell| {
            let mut s = cell.borrow_mut();
            if s.active {
                for (a, b) in swept_gaps {
                    mark_swept(&mut s.swept, a, b);
                }
            }
        });
    }
    // Commit any pages whose kernel dirty bit was already reset before a later
    // gap failed so subsequent queries of the watched prefix never lose them.
    if !dirty.is_empty() || !failed_any {
        j.asked += 1;
        let now = j.asked;
        dirty.sort_unstable();
        dirty.dedup();
        for page in dirty {
            j.written.insert(page, now);
        }
        if failed_any {
            return None;
        }
        return Some((
            now,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    None
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    thread_local! {
        static KERNEL_QUERIES: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn count_kernel_query() {
        KERNEL_QUERIES.with(|c| c.set(c.get() + 1));
    }

    fn kernel_queries() -> usize {
        KERNEL_QUERIES.with(|c| c.get())
    }

    fn reset_kernel_queries() {
        KERNEL_QUERIES.with(|c| c.set(0));
    }

    /// Drops journal entries for `[start, end)` so virtual addresses reused by
    /// Windows across `VirtualAlloc`/`VirtualFree` do not collide across tests.
    fn forget_pages(start: usize, end: usize) {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<usize> = j.written.range(first..last).map(|(&p, _)| p).collect();
        for p in keys {
            j.written.remove(&p);
        }
    }

    struct WatchedAlloc {
        addr: usize,
        len: usize,
    }

    impl WatchedAlloc {
        fn new(pages: usize) -> Self {
            let len = pages * PAGE;
            // SAFETY: fresh write-watched allocation freed on drop.
            let addr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(addr, 0);
            forget_pages(addr, addr + len);
            Self { addr, len }
        }
    }

    impl Drop for WatchedAlloc {
        fn drop(&mut self) {
            forget_pages(self.addr, self.addr + self.len);
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.addr as *mut _, 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    #[test]
    fn sweep_interval_helpers() {
        let mut swept = Vec::new();
        assert_eq!(uncovered(&swept, 10, 20), [(10, 20)]);
        mark_swept(&mut swept, 12, 15);
        assert_eq!(swept, [(12, 15)]);
        assert_eq!(uncovered(&swept, 10, 20), [(10, 12), (15, 20)]);
        assert_eq!(uncovered(&swept, 12, 15), []);

        mark_swept(&mut swept, 18, 22);
        mark_swept(&mut swept, 15, 18);
        assert_eq!(swept, [(12, 22)]);
        assert_eq!(uncovered(&swept, 10, 24), [(10, 12), (22, 24)]);

        mark_swept(&mut swept, 5, 12);
        assert_eq!(swept, [(5, 22)]);
        assert_eq!(uncovered(&swept, 6, 22), []);
    }

    #[test]
    fn nested_and_overlapping_ranges_reuse_sweep_without_extra_kernel_queries() {
        let alloc = WatchedAlloc::new(4);
        let p = alloc.addr;
        let base = p / PAGE;
        let (_, before) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert!(before.is_empty());

        // Write pages 1 and 2 before entering the submission sweep.
        // SAFETY: committed pages of ours.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write(1);
            ((p + 2 * PAGE + 16) as *mut u8).write(2);
        }

        let guard = sweep();
        reset_kernel_queries();

        // Outer buffer queries the kernel once.
        let (t1, outer) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 1);
        assert_eq!(outer, [(base + 1, t1), (base + 2, t1)]);

        // Nested and overlapping buffers reuse the sweep and journal without
        // repeating GetWriteWatch, while advancing asking timestamps.
        let (t2, inner) = writes(p + PAGE, p + 3 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 1);
        assert!(t2 > t1);
        assert_eq!(inner, [(base + 1, t1), (base + 2, t1)]);

        let (t3, overlap) = writes(p + 2 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 1);
        assert!(t3 > t2);
        assert_eq!(overlap, [(base + 2, t1)]);
        drop(guard);

        // New sweep where an inner range is asked first, then the outer range:
        // only the two uncovered gaps query the kernel, and a re-query hits 0.
        // SAFETY: committed pages of ours.
        unsafe {
            ((p + 4) as *mut u8).write(3);
            ((p + 3 * PAGE + 4) as *mut u8).write(4);
        }
        let guard = sweep();
        reset_kernel_queries();
        let (t4, mid) = writes(p + PAGE, p + 3 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 1);
        assert_eq!(mid, [(base + 1, t1), (base + 2, t1)]);

        let (t5, full) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 3);
        assert!(t5 > t4);
        assert_eq!(
            full,
            [(base, t5), (base + 1, t1), (base + 2, t1), (base + 3, t5)]
        );

        let (t6, left_half) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 3);
        assert!(t6 > t5);
        assert_eq!(left_half, [(base, t5), (base + 1, t1)]);
        drop(guard);
    }

    #[test]
    fn sweep_reset_sees_subsequent_writes_and_reports_to_all_askers() {
        let alloc = WatchedAlloc::new(2);
        let p = alloc.addr;
        let base = p / PAGE;
        assert!(
            writes(p, p + 2 * PAGE, false)
                .expect("watched")
                .1
                .is_empty()
        );

        // SAFETY: committed page of ours.
        unsafe { ((p + 8) as *mut u8).write(1) };

        let guard = sweep();
        reset_kernel_queries();
        let (t1, first) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 1);
        assert_eq!(first, [(base, t1)]);

        // A step that writes title memory clears the sweep so later askings
        // in the same submission see the write.
        // SAFETY: committed page of ours.
        unsafe { ((p + 16) as *mut u8).write(2) };
        clear_sweep();

        let (t2, second) = writes(p, p + PAGE, false).expect("watched");
        assert_eq!(kernel_queries(), 2);
        assert!(t2 > t1);
        assert_eq!(second, [(base, t2)]);
        drop(guard);

        // A later asker outside the sweep still observes the latest write.
        let (t3, outside) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(t3 > t2);
        assert_eq!(outside, [(base, t2)]);
    }

    #[test]
    fn partial_sweep_preserves_writes_when_a_later_gap_fails() {
        // Reserve 3 * PAGE, release, then place [2 * PAGE watched | 1 * PAGE plain]
        // contiguously so a single range spans watched and unwatched pages.
        // SAFETY: test allocations freed before returning.
        let (watched, plain) = unsafe {
            let mut pair = None;
            for _ in 0..16 {
                let probe = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE);
                if probe.is_null() {
                    continue;
                }
                VirtualFree(probe, 0, MEM_RELEASE);
                let w = VirtualAlloc(
                    probe,
                    2 * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                );
                if w.is_null() {
                    continue;
                }
                let u = VirtualAlloc(
                    (w as usize + 2 * PAGE) as *const _,
                    PAGE,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                );
                if !u.is_null() {
                    pair = Some((w as usize, u as usize));
                    break;
                }
                VirtualFree(w, 0, MEM_RELEASE);
            }
            pair.expect("contiguous watched and plain allocations")
        };
        let watched_alloc = WatchedAlloc {
            addr: watched,
            len: 2 * PAGE,
        };
        forget_pages(watched, plain + PAGE);
        assert_eq!(plain, watched + 2 * PAGE);
        let page_a = watched / PAGE;

        assert!(
            writes(watched, watched + 2 * PAGE, false)
                .expect("watched")
                .1
                .is_empty()
        );

        let guard = sweep();
        // Sweep page B first so the mixed query splits into [A, B) and [C, C+1).
        assert!(
            writes(watched + PAGE, watched + 2 * PAGE, false)
                .expect("watched B")
                .1
                .is_empty()
        );
        // Write page A, then ask A..=C (with both guarded=false and guarded=true).
        // SAFETY: committed page of ours.
        unsafe { ((watched + 4) as *mut u8).write(9) };
        assert!(writes(watched, plain + PAGE, false).is_none());
        assert!(writes(watched, plain + PAGE, true).is_none());
        drop(guard);

        // Even though the mixed query returned None after resetting A's kernel bit,
        // the write to A was journaled and remains visible to later askers.
        let (now, recovered) = writes(watched, watched + PAGE, false).expect("watched A");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, page_a);
        assert!(recovered[0].1 <= now);

        drop(watched_alloc);
        // SAFETY: our plain allocation.
        unsafe { VirtualFree(plain as *mut _, 0, MEM_RELEASE) };
    }
}
