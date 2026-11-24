//! The process's own memory, sampled with every summary.

/// Working set, its peak, private (committed) bytes and page faults.
#[cfg(windows)]
pub(crate) fn sample(out: &mut Vec<(&'static str, u64)>) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process; the extended structure starts with the
    // plain one and `cb` gives its size.
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    };
    if ok != 0 {
        out.push(("process.working_set", c.WorkingSetSize as u64));
        out.push(("process.peak_working_set", c.PeakWorkingSetSize as u64));
        out.push(("process.private", c.PrivateUsage as u64));
        out.push(("process.page_faults", u64::from(c.PageFaultCount)));
    }
}

#[cfg(not(windows))]
pub(crate) fn sample(_: &mut Vec<(&'static str, u64)>) {}
