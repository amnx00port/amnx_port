//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
}
const COUNTERS: usize = 22;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

fn parse_mode(value: Option<&str>) -> u8 {
    match value {
        Some("plain") => 1,
        Some("none") => 2,
        _ => 0,
    }
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("VICTORY_COPY_MODE").ok().as_deref()))
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let frames = if frames.is_finite() && frames > 0.0 {
        frames
    } else {
        1.0
    };
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    );
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn counter_enum_matches_array_length() {
        assert_eq!(C::Submissions as usize, 0);
        assert_eq!(C::TexChanged as usize + 1, COUNTERS);
    }

    #[test]
    fn parse_mode_maps_known_and_unknown_values() {
        assert_eq!(parse_mode(None), 0);
        assert_eq!(parse_mode(Some("")), 0);
        assert_eq!(parse_mode(Some("guarded")), 0);
        assert_eq!(parse_mode(Some("plain")), 1);
        assert_eq!(parse_mode(Some("none")), 2);
    }

    #[test]
    fn range_accumulates_same_key_and_separates_distinct_ranges() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();

        range(0x1000, 64, 10, 0x20000);
        range(0x1000, 64, 25, 0x20000);
        range(0x2000, 128, 7, 0x40000);

        let map = RANGES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("ranges recorded");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&(0x1000, 64)), Some(&(2, 128, 35, 0x20000)));
        assert_eq!(map.get(&(0x2000, 128)), Some(&(1, 128, 7, 0x40000)));
    }

    #[test]
    fn add_and_log_drain_counters_and_ranges() {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for a in &COUNT {
            a.store(0, Relaxed);
        }
        RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();

        add(C::Submissions, 4);
        add(C::Copies, 12);
        add(C::CopyBytes, 4096);
        add(C::TexChanged, 512);
        let probe = 0u64;
        range((&raw const probe) as usize, size_of::<u64>(), 3, 0);

        // Zero and non-finite frame counts must not panic or divide by zero.
        log(0.0);
        for a in &COUNT {
            assert_eq!(a.load(Relaxed), 0);
        }
        assert!(RANGES.lock().unwrap_or_else(|e| e.into_inner()).is_none());

        add(C::Submissions, 2);
        log(f64::NAN);
        assert_eq!(COUNT[C::Submissions as usize].load(Relaxed), 0);
    }

    #[test]
    fn page_faults_queries_current_process() {
        // Current process always has at least one page fault since startup.
        assert!(page_faults() > 0);
    }
}
