//! Diagnostics for pipeline-worker occupancy and queued work.
//!
//! These counters answer a question that shader wait-time totals alone cannot:
//! was work still queued while workers were waiting for a shared translation?
//! They do not prove that every queued pipeline was independently runnable;
//! its own shader dependencies are not inspected here.

#[cfg(all(windows, feature = "diagnostics"))]
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[cfg(all(windows, feature = "diagnostics"))]
thread_local! {
    /// True only while the current thread is executing the pipeline worker loop.
    static PIPELINE_WORKER: Cell<bool> = const { Cell::new(false) };
}

struct State {
    last_event: Instant,
    registered_workers: u64,
    active_jobs: u64,
    queued_jobs: u64,
    dependency_waiters: u64,
    queued_with_dependency_wait_us: u64,
    queued_with_dependency_wait_periods: u64,
    all_registered_workers_waiting_with_queue_us: u64,
    all_registered_workers_waiting_with_queue_periods: u64,
    peak_active_jobs: u64,
    peak_queued_jobs: u64,
    peak_dependency_waiters: u64,
}

impl State {
    fn new(now: Instant) -> Self {
        Self {
            last_event: now,
            registered_workers: 0,
            active_jobs: 0,
            queued_jobs: 0,
            dependency_waiters: 0,
            queued_with_dependency_wait_us: 0,
            queued_with_dependency_wait_periods: 0,
            all_registered_workers_waiting_with_queue_us: 0,
            all_registered_workers_waiting_with_queue_periods: 0,
            peak_active_jobs: 0,
            peak_queued_jobs: 0,
            peak_dependency_waiters: 0,
        }
    }

    fn has_queued_dependency_wait(&self) -> bool {
        self.queued_jobs > 0 && self.dependency_waiters > 0
    }

    fn has_all_registered_workers_waiting_with_queue(&self) -> bool {
        self.queued_jobs > 0
            && self.registered_workers > 0
            && self.dependency_waiters >= self.registered_workers
    }
}

struct Snapshot {
    registered_workers: u64,
    active_jobs: u64,
    queued_jobs: u64,
    dependency_waiters: u64,
    queued_with_dependency_wait_us: u64,
    queued_with_dependency_wait_periods: u64,
    all_registered_workers_waiting_with_queue_us: u64,
    all_registered_workers_waiting_with_queue_periods: u64,
    peak_active_jobs: u64,
    peak_queued_jobs: u64,
    peak_dependency_waiters: u64,
}

struct Tracker {
    state: Mutex<State>,
}

impl Tracker {
    #[cfg(all(windows, feature = "diagnostics"))]
    fn new() -> Self {
        Self::new_at(Instant::now())
    }

    fn new_at(now: Instant) -> Self {
        Self {
            state: Mutex::new(State::new(now)),
        }
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    fn update(&self, change: impl FnOnce(&mut State)) -> bool {
        self.update_at(Instant::now(), change)
    }

    fn update_at(&self, now: Instant, change: impl FnOnce(&mut State)) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = now.max(state.last_event);
        let elapsed_us = now.duration_since(state.last_event).as_micros() as u64;

        // Attribute elapsed time to the state that held during that interval.
        if state.has_queued_dependency_wait() {
            state.queued_with_dependency_wait_us = state
                .queued_with_dependency_wait_us
                .saturating_add(elapsed_us);
        }
        if state.has_all_registered_workers_waiting_with_queue() {
            state.all_registered_workers_waiting_with_queue_us = state
                .all_registered_workers_waiting_with_queue_us
                .saturating_add(elapsed_us);
        }

        let had_queued_dependency_wait = state.has_queued_dependency_wait();
        let had_all_registered_workers_waiting_with_queue =
            state.has_all_registered_workers_waiting_with_queue();

        state.last_event = now;
        change(&mut state);

        if !had_queued_dependency_wait && state.has_queued_dependency_wait() {
            state.queued_with_dependency_wait_periods =
                state.queued_with_dependency_wait_periods.saturating_add(1);
        }
        let entered_all_registered_workers_backlog =
            !had_all_registered_workers_waiting_with_queue
                && state.has_all_registered_workers_waiting_with_queue();
        if entered_all_registered_workers_backlog {
            state.all_registered_workers_waiting_with_queue_periods = state
                .all_registered_workers_waiting_with_queue_periods
                .saturating_add(1);
        }

        state.peak_active_jobs = state.peak_active_jobs.max(state.active_jobs);
        state.peak_queued_jobs = state.peak_queued_jobs.max(state.queued_jobs);
        state.peak_dependency_waiters =
            state.peak_dependency_waiters.max(state.dependency_waiters);

        entered_all_registered_workers_backlog
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    fn snapshot(&self) -> Snapshot {
        self.snapshot_at(Instant::now())
    }

    fn snapshot_at(&self, now: Instant) -> Snapshot {
        self.update_at(now, |_| {});
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        Snapshot {
            registered_workers: state.registered_workers,
            active_jobs: state.active_jobs,
            queued_jobs: state.queued_jobs,
            dependency_waiters: state.dependency_waiters,
            queued_with_dependency_wait_us: state.queued_with_dependency_wait_us,
            queued_with_dependency_wait_periods: state.queued_with_dependency_wait_periods,
            all_registered_workers_waiting_with_queue_us: state.all_registered_workers_waiting_with_queue_us,
            all_registered_workers_waiting_with_queue_periods: state
                .all_registered_workers_waiting_with_queue_periods,
            peak_active_jobs: state.peak_active_jobs,
            peak_queued_jobs: state.peak_queued_jobs,
            peak_dependency_waiters: state.peak_dependency_waiters,
        }
    }
}

#[cfg(all(windows, feature = "diagnostics"))]
static TRACKER: OnceLock<Tracker> = OnceLock::new();

#[cfg(all(windows, feature = "diagnostics"))]
fn tracker() -> &'static Tracker {
    TRACKER.get_or_init(Tracker::new)
}

/// Applies a state change and reports immediately when every registered
/// worker becomes dependency-blocked while at least one pipeline remains queued.
#[cfg(all(windows, feature = "diagnostics"))]
fn update(change: impl FnOnce(&mut State)) {
    if tracker().update(change) {
        report();
    }
}

/// Marks a pipeline worker thread for the lifetime of its receive loop.
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) struct WorkerGuard;

#[cfg(all(windows, feature = "diagnostics"))]
pub(super) fn worker() -> WorkerGuard {
    PIPELINE_WORKER.with(|worker| worker.set(true));
    update(|state| state.registered_workers = state.registered_workers.saturating_add(1));
    WorkerGuard
}

#[cfg(all(windows, feature = "diagnostics"))]
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        PIPELINE_WORKER.with(|worker| worker.set(false));
        update(|state| {
            state.registered_workers = state.registered_workers.saturating_sub(1);
        });
    }
}

/// Records one pipeline entering the worker queue.
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) fn job_queued() {
    update(|state| state.queued_jobs = state.queued_jobs.saturating_add(1));
}

/// Moves one queued pipeline to the active-worker count until its guard drops.
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) struct JobGuard;

#[cfg(all(windows, feature = "diagnostics"))]
pub(super) fn job_started() -> JobGuard {
    update(|state| {
        state.queued_jobs = state.queued_jobs.saturating_sub(1);
        state.active_jobs = state.active_jobs.saturating_add(1);
    });
    JobGuard
}

#[cfg(all(windows, feature = "diagnostics"))]
impl Drop for JobGuard {
    fn drop(&mut self) {
        update(|state| {
            state.active_jobs = state.active_jobs.saturating_sub(1);
        });
    }
}

/// Counts only a worker blocked in the shader translation OnceLock.
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) struct DependencyWaitGuard;

#[cfg(all(windows, feature = "diagnostics"))]
pub(super) fn dependency_wait() -> Option<DependencyWaitGuard> {
    if !PIPELINE_WORKER.with(Cell::get) {
        return None;
    }
    update(|state| {
        state.dependency_waiters = state.dependency_waiters.saturating_add(1);
    });
    Some(DependencyWaitGuard)
}

#[cfg(all(windows, feature = "diagnostics"))]
impl Drop for DependencyWaitGuard {
    fn drop(&mut self) {
        let tracker = tracker();
        update(|state| {
            state.dependency_waiters = state.dependency_waiters.saturating_sub(1);
        });

        // A queue that remains backed up cannot reach drained(). Report when
        // the last dependency waiter leaves so the measured interval is still
        // visible even when other pipeline jobs are pending.
        let snapshot = tracker.snapshot();
        if snapshot.dependency_waiters == 0 && snapshot.queued_jobs > 0 {
            report();
        }
    }
}

/// Logs cumulative intervals during which shader dependencies were contested
/// while other pipeline jobs remained queued.
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) fn report() {
    let s = tracker().snapshot();
    tracing::info!(
        registered_workers = s.registered_workers,
        active_jobs = s.active_jobs,
        queued_jobs = s.queued_jobs,
        dependency_waiters = s.dependency_waiters,
        queued_with_dependency_wait_us = s.queued_with_dependency_wait_us,
        queued_with_dependency_wait_periods = s.queued_with_dependency_wait_periods,
        all_registered_workers_waiting_with_queue_us = s.all_registered_workers_waiting_with_queue_us,
        all_registered_workers_waiting_with_queue_periods = s.all_registered_workers_waiting_with_queue_periods,
        peak_active_jobs = s.peak_active_jobs,
        peak_queued_jobs = s.peak_queued_jobs,
        peak_dependency_waiters = s.peak_dependency_waiters,
        "pipeline worker backlog diagnostics"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn times_only_backlog_that_overlaps_shared_shader_waits() {
        let start = Instant::now();
        let tracker = Tracker::new_at(start);

        tracker.update_at(start, |s| {
            s.registered_workers = 2;
        });
        tracker.update_at(start + Duration::from_millis(10), |s| {
            s.active_jobs = 2;
            s.dependency_waiters = 2;
        });
        tracker.update_at(start + Duration::from_millis(20), |s| {
            s.queued_jobs = 3;
        });
        tracker.update_at(start + Duration::from_millis(90), |s| {
            s.dependency_waiters = 1;
        });
        tracker.update_at(start + Duration::from_millis(120), |s| {
            s.queued_jobs = 0;
        });

        let s = tracker.snapshot_at(start + Duration::from_millis(150));
        assert_eq!(s.queued_with_dependency_wait_us, 100_000);
        assert_eq!(s.queued_with_dependency_wait_periods, 1);
        assert_eq!(s.all_registered_workers_waiting_with_queue_us, 70_000);
        assert_eq!(s.all_registered_workers_waiting_with_queue_periods, 1);
        assert_eq!(s.active_jobs, 2);
        assert_eq!(s.dependency_waiters, 1);
        assert_eq!(s.peak_active_jobs, 2);
        assert_eq!(s.peak_queued_jobs, 3);
        assert_eq!(s.peak_dependency_waiters, 2);
        assert_eq!(s.registered_workers, 2);
        assert_eq!(s.queued_jobs, 0);
    }

    #[test]
    fn reports_only_new_all_registered_worker_backlog_intervals() {
        let start = Instant::now();
        let tracker = Tracker::new_at(start);

        assert!(!tracker.update_at(start, |s| {
            s.registered_workers = 2;
        }));
        assert!(!tracker.update_at(start + Duration::from_millis(10), |s| {
            s.queued_jobs = 1;
        }));
        assert!(!tracker.update_at(start + Duration::from_millis(20), |s| {
            s.dependency_waiters = 1;
        }));

        // The second waiter blocks the last registered worker while work is still queued.
        assert!(tracker.update_at(start + Duration::from_millis(30), |s| {
            s.dependency_waiters = 2;
        }));

        // More queued work must not emit another "entered full backlog" event.
        assert!(!tracker.update_at(start + Duration::from_millis(40), |s| {
            s.queued_jobs = 2;
        }));

        // A registered worker becoming available ends the interval; blocking it again
        // starts a distinct interval that should be reported once.
        assert!(!tracker.update_at(start + Duration::from_millis(50), |s| {
            s.dependency_waiters = 1;
        }));
        assert!(tracker.update_at(start + Duration::from_millis(60), |s| {
            s.dependency_waiters = 2;
        }));

        let s = tracker.snapshot_at(start + Duration::from_millis(70));
        assert_eq!(s.all_registered_workers_waiting_with_queue_periods, 2);
        assert_eq!(s.queued_jobs, 2);
        assert_eq!(s.dependency_waiters, 2);
        assert_eq!(s.registered_workers, 2);
    }

    #[test]
    fn separate_backlog_intervals_are_counted_separately() {
        let start = Instant::now();
        let tracker = Tracker::new_at(start);

        tracker.update_at(start, |s| {
            s.registered_workers = 1;
        });
        tracker.update_at(start + Duration::from_millis(5), |s| {
            s.dependency_waiters = 1;
            s.queued_jobs = 1;
        });
        tracker.update_at(start + Duration::from_millis(15), |s| {
            s.dependency_waiters = 0;
            s.queued_jobs = 0;
        });
        tracker.update_at(start + Duration::from_millis(25), |s| {
            s.dependency_waiters = 1;
            s.queued_jobs = 2;
        });
        tracker.update_at(start + Duration::from_millis(45), |s| {
            s.dependency_waiters = 0;
            s.queued_jobs = 0;
        });

        let s = tracker.snapshot_at(start + Duration::from_millis(50));
        assert_eq!(s.queued_with_dependency_wait_us, 30_000);
        assert_eq!(s.queued_with_dependency_wait_periods, 2);
        assert_eq!(s.all_registered_workers_waiting_with_queue_us, 30_000);
        assert_eq!(s.all_registered_workers_waiting_with_queue_periods, 2);
    }
}
