//! Frame pacing: the console's display interrupt, which titles schedule
//! against and wait on. Driven here from a steady monotonic clock so the
//! title paces its loop at the rate it asked for, using a Windows
//! high-resolution waitable timer (`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`)
//! plus a short spin-tail to the exact deadline so release builds do not
//! suffer 15.6 ms OS timer-quantum drift.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::object::{E_INVALIDARG, HRESULT, S_OK, This};

const TIMEOUT: HRESULT = 0x8007_0102_u32 as i32; // WAIT_TIMEOUT

/// One 60 Hz frame, the default interval before the title sets one.
const SIXTIETH: Duration = Duration::from_nanos(16_666_667);

/// Spin-tail window before the exact tick deadline: sleeping stops this far
/// ahead of the target `Instant` and yields/spins out the final sub-millisecond
/// to absorb OS wakeup jitter.
const SPIN_MARGIN: Duration = Duration::from_micros(500);

struct Clock {
    /// Anchor time for `first`.
    start: Instant,
    interval: Duration,
    /// Tick number at `start`, monotonic across interval changes.
    first: u64,
    /// Optional per-event scheduling offset in microseconds (`schedule`).
    offsets_us: [u32; 4],
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
/// Last tick delivered, per event kind (origin, present, ...).
static LAST: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Frames begun so far (frame waits that ended).
static FRAME: AtomicU64 = AtomicU64::new(0);

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

fn clock() -> MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

fn ensure_clock(slot: &mut Option<Clock>) -> &mut Clock {
    slot.get_or_insert_with(|| Clock {
        start: Instant::now(),
        interval: SIXTIETH,
        first: 0,
        offsets_us: [0; 4],
    })
}

/// `SetFrameIntervalX(output, period_us, count, flags)`.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    period_us: u32,
    count: u32,
    _flags: u32,
) -> HRESULT {
    let interval = Duration::from_micros(u64::from(period_us) * u64::from(count));
    if interval.is_zero() {
        return E_INVALIDARG;
    }
    tracing::debug!(?interval, "frame interval");
    let now = Instant::now();
    let mut guard = clock();
    let mut offsets_us = [0; 4];
    let first = match guard.as_ref() {
        Some(c) => {
            offsets_us = c.offsets_us;
            let elapsed = now.saturating_duration_since(c.start).as_nanos();
            c.first + (elapsed / c.interval.as_nanos().max(1)) as u64
        }
        None => 0,
    };
    *guard = Some(Clock {
        start: now,
        interval,
        first,
        offsets_us,
    });
    S_OK
}

/// `ScheduleFrameEventX(kind, offset_us, flags)`: records the per-event offset
/// within the frame interval (clamped below one full interval).
pub(crate) extern "system" fn schedule(
    _: This,
    kind: u32,
    offset_us: u32,
    _flags: u32,
) -> HRESULT {
    let mut guard = clock();
    let c = ensure_clock(&mut guard);
    let slot = (kind as usize).min(c.offsets_us.len() - 1);
    let max_offset = (c.interval.as_micros() as u32).saturating_sub(1);
    c.offsets_us[slot] = offset_us.min(max_offset);
    S_OK
}

/// Pure deadline calculation for the next tick strictly after `last_tick`.
/// Returns `(tick, target_instant)` without accumulated drift across frames.
fn compute_next_deadline(
    start: Instant,
    interval: Duration,
    first: u64,
    offset: Duration,
    last_tick: u64,
    now: Instant,
) -> (u64, Instant) {
    let interval_ns = interval.as_nanos().max(1);
    let offset_ns = offset.as_nanos().min(interval_ns.saturating_sub(1));
    let elapsed_ns = now.saturating_duration_since(start).as_nanos();
    // Solve `first + k` such that `k * interval_ns + offset_ns > elapsed_ns`.
    let k_from_clock = if elapsed_ns < offset_ns {
        0u64
    } else {
        ((elapsed_ns - offset_ns) / interval_ns + 1) as u64
    };
    let tick = (first + k_from_clock).max(last_tick + 1);
    let target_ns = u128::from(tick - first) * interval_ns + offset_ns;
    // Split nanoseconds cleanly into Duration (safe up to >500 years).
    let secs = (target_ns / 1_000_000_000) as u64;
    let nanos = (target_ns % 1_000_000_000) as u32;
    let target = start + Duration::new(secs, nanos);
    (tick, target)
}

/// The next tick for `kind` and its exact target `Instant`.
fn next_tick(kind: u32, now: Instant) -> (u64, Instant) {
    let mut guard = clock();
    let c = ensure_clock(&mut guard);
    let slot = (kind as usize).min(c.offsets_us.len() - 1);
    let offset = Duration::from_micros(u64::from(c.offsets_us[slot]));
    let prev = last(kind).load(Ordering::Relaxed);
    compute_next_deadline(c.start, c.interval, c.first, offset, prev, now)
}

/// Sleeps accurately until `deadline` using a Windows high-resolution waitable
/// timer plus a sub-millisecond spin-tail so frame pacing does not drift on
/// Windows' default 15.6 ms scheduler tick.
fn sleep_until(deadline: Instant) {
    let now = Instant::now();
    if now >= deadline {
        return;
    }
    let remaining = deadline - now;
    if remaining > SPIN_MARGIN {
        let coarse = remaining - SPIN_MARGIN;
        sleep_precise(coarse);
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(windows)]
fn sleep_precise(duration: Duration) {
    use std::cell::Cell;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, INFINITE,
        SetWaitableTimer, TIMER_ALL_ACCESS, WaitForSingleObject,
    };

    struct Timer(HANDLE);
    impl Drop for Timer {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: handle created by `CreateWaitableTimerExW` for this thread.
                unsafe { CloseHandle(self.0) };
            }
        }
    }

    thread_local! {
        static HI_RES_TIMER: Cell<Option<Timer>> = const { Cell::new(None) };
    }

    let waited = HI_RES_TIMER.with(|cell| {
        let mut timer = cell.take();
        if timer.is_none() {
            // SAFETY: creating a thread-local unnamed high-resolution waitable timer.
            let handle = unsafe {
                CreateWaitableTimerExW(
                    std::ptr::null(),
                    std::ptr::null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS,
                )
            };
            if !handle.is_null() {
                timer = Some(Timer(handle));
            }
        }
        let ok = if let Some(ref t) = timer {
            // Relative negative due time in 100-nanosecond intervals.
            let ticks_100ns = (duration.as_nanos() / 100).max(1);
            let due = -(i64::try_from(ticks_100ns).unwrap_or(i64::MAX));
            // SAFETY: valid waitable timer handle and due time pointer.
            unsafe {
                SetWaitableTimer(t.0, &due, 0, None, std::ptr::null(), 0) != 0
                    && WaitForSingleObject(t.0, INFINITE) == 0
            }
        } else {
            false
        };
        cell.set(timer);
        ok
    });
    if !waited {
        std::thread::sleep(duration);
    }
}

#[cfg(not(windows))]
fn sleep_precise(duration: Duration) {
    std::thread::sleep(duration);
}

/// `WaitFrameEventX(kind, timeout_ms, event, token)`: sleeps until the next
/// tick of `kind` (or `timeout_ms`) and writes the token. A zero timeout
/// returns immediately without yielding or sleeping.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _event: *mut u8,
    token: *mut u64,
) -> HRESULT {
    let now = Instant::now();
    let (tick, target) = next_tick(kind, now);
    let wait = target.saturating_duration_since(now);
    let limit = Duration::from_millis(u64::from(timeout_ms));
    let hr = if wait > limit {
        if !limit.is_zero() {
            sleep_until(now + limit);
        }
        TIMEOUT
    } else {
        if !wait.is_zero() {
            sleep_until(target);
        }
        last(kind).fetch_max(tick, Ordering::Relaxed);
        FRAME.fetch_add(1, Ordering::Relaxed);
        S_OK
    };
    if !token.is_null() {
        // SAFETY: optional out parameter.
        unsafe { token.write(if hr == S_OK { tick } else { 0 }) };
    }
    hr
}

/// `GetFrameStatisticsX(count, stats)`: zeroed counts and timestamps for now.
pub(crate) extern "system" fn statistics(_: This, count: u32, stats: *mut u8) -> HRESULT {
    if stats.is_null() {
        return E_INVALIDARG;
    }
    const ENTRY: usize = 0x58;
    // SAFETY: the title passes room for `count` `D3D12XBOX_FRAME_PIPELINE_STATISTICS`.
    unsafe { stats.write_bytes(0, count as usize * ENTRY) };
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_next_deadline_has_no_cumulative_drift() {
        let start = Instant::now();
        let interval = Duration::from_micros(16_667);
        // Simulate 120 frames with small random jitter around each deadline.
        let mut last_tick = 0;
        for frame in 1..=120u64 {
            // Arrive 2 ms early before tick `frame`.
            let now = start + interval * (frame as u32) - Duration::from_millis(2);
            let (tick, target) =
                compute_next_deadline(start, interval, 0, Duration::ZERO, last_tick, now);
            assert_eq!(tick, frame);
            assert_eq!(target, start + interval * (frame as u32));
            last_tick = tick;
        }
    }

    #[test]
    fn compute_next_deadline_honors_event_offset() {
        let start = Instant::now();
        let interval = Duration::from_millis(16);
        let offset = Duration::from_millis(4);
        let (tick, target) = compute_next_deadline(start, interval, 10, offset, 9, start);
        assert_eq!(tick, 10);
        assert_eq!(target, start + offset);

        let (tick2, target2) = compute_next_deadline(
            start,
            interval,
            10,
            offset,
            10,
            start + Duration::from_millis(5),
        );
        assert_eq!(tick2, 11);
        assert_eq!(target2, start + interval + offset);
    }

    #[test]
    fn set_interval_rejects_zero() {
        assert_eq!(
            set_interval(std::ptr::null_mut(), std::ptr::null_mut(), 0, 1, 0),
            E_INVALIDARG
        );
    }
}
