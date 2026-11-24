//! Opt-in profiling: where a run's time and memory go.
//!
//! Off unless `VICTORY_PROFILE` is set when the runtime starts
//! ([`init`]); then every few seconds a summary is logged (frame times
//! with percentiles and hitches, the work per frame, memory, caches, the
//! timed scopes that took longest, what was loaded), and with
//! `VICTORY_PROFILE=<dir>` every frame is a row of `<dir>/frames.csv` and
//! scopes, loads and counters are events in `<dir>/trace.json` ([`out`]).
//!
//! Off, every hook is one relaxed load of a flag. On, a hook is a few
//! relaxed atomic adds on a [`Stat`] its call site looked up once.
//!
//! `VICTORY_PROFILE_OVERLAY=1` (or `VICTORY_PROFILE=overlay`) turns it on
//! with an on-screen panel as well, which F3 shows and hides; see
//! [`overlay`].
//!
//! ```ignore
//! victory_profile::scope!("hle", "ReadFile", len);      // times the block
//! victory_profile::count!("gpu", "upload buffer", size); // counts bytes
//! victory_profile::cache!("shader", "disk", Hit);
//! victory_profile::resident!("gpu", "memory", size);     // -size on free
//! victory_profile::load("dstorage", || name.into(), bytes, took);
//! ```

mod memory;
mod out;
pub mod overlay;
pub mod stats;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use stats::{Ring, Summary, Top, fps};

static ON: AtomicBool = AtomicBool::new(false);
/// Whether `trace.json` is being written.
static TRACE: AtomicBool = AtomicBool::new(false);
static START: OnceLock<Instant> = OnceLock::new();
static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Whether profiling is on.
#[inline]
pub fn on() -> bool {
    ON.load(Relaxed)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the variables ask for.
#[derive(Debug, Clone, PartialEq)]
struct Settings {
    /// `VICTORY_PROFILE=<dir>` (anything but `1` or `overlay`): where the
    /// files go.
    dir: Option<PathBuf>,
    /// `VICTORY_PROFILE_INTERVAL=<seconds>` between summaries (default 5).
    interval: Duration,
    /// `VICTORY_PROFILE_HITCH_MS`: a frame interval above this is a hitch
    /// (default 50, a frame and a half at the title's 30 fps).
    hitch_ms: f64,
    /// `VICTORY_PROFILE_TRACE_US`: scopes shorter than this are counted
    /// but not traced (default 100).
    trace_min_ns: u64,
    /// `VICTORY_PROFILE_DUMP=<seconds>,...`: full reports at those times.
    dumps: Vec<Duration>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            dir: None,
            interval: Duration::from_secs(5),
            hitch_ms: 50.0,
            trace_min_ns: 100_000,
            dumps: Vec::new(),
        }
    }
}

impl Settings {
    fn parse(
        profile: &str,
        interval: Option<&str>,
        hitch_ms: Option<&str>,
        trace_us: Option<&str>,
        dumps: Option<&str>,
    ) -> Settings {
        let d = Settings::default();
        let number = |v: Option<&str>| v.and_then(|v| v.trim().parse::<f64>().ok());
        let positive = |v: Option<&str>| number(v).filter(|v| v.is_finite() && *v > 0.0);
        let profile = profile.trim();
        let mut dumps: Vec<Duration> = dumps
            .unwrap_or("")
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .filter(|t| t.is_finite() && *t >= 0.0)
            .map(Duration::from_secs_f64)
            .collect();
        dumps.sort();
        Settings {
            dir: (!matches!(profile, "" | "1" | "overlay")).then(|| PathBuf::from(profile)),
            interval: positive(interval)
                .map_or(d.interval, |s| Duration::from_secs_f64(s.max(0.5))),
            hitch_ms: positive(hitch_ms).unwrap_or(d.hitch_ms),
            trace_min_ns: number(trace_us)
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map_or(d.trace_min_ns, |us| (us * 1000.0) as u64),
            dumps,
        }
    }
}

fn settings() -> &'static Settings {
    static DEFAULT: OnceLock<Settings> = OnceLock::new();
    SETTINGS
        .get()
        .unwrap_or_else(|| DEFAULT.get_or_init(Settings::default))
}

/// Turns profiling on if `VICTORY_PROFILE` is set: `1` for the logged
/// summaries, a folder for the files as well, `overlay` for the summaries
/// and the on-screen panel (as `VICTORY_PROFILE_OVERLAY=1` alone does).
/// Call once, early.
pub fn init() {
    let var = |name: &str| std::env::var(name).ok();
    let profile = var("VICTORY_PROFILE");
    let overlay = overlay::wanted(
        profile.as_deref(),
        var("VICTORY_PROFILE_OVERLAY").as_deref(),
    );
    let Some(profile) = profile.or_else(|| overlay.then(|| "1".to_owned())) else {
        return;
    };
    let s = Settings::parse(
        &profile,
        var("VICTORY_PROFILE_INTERVAL").as_deref(),
        var("VICTORY_PROFILE_HITCH_MS").as_deref(),
        var("VICTORY_PROFILE_TRACE_US").as_deref(),
        var("VICTORY_PROFILE_DUMP").as_deref(),
    );
    if SETTINGS.set(s).is_err() {
        return;
    }
    let s = settings();
    START.get_or_init(Instant::now);
    if let Some(dir) = &s.dir {
        let opened = victory_core::ensure_outside_repo(dir)
            .map_err(|e| e.to_string())
            .and_then(|()| out::open(dir).map_err(|e| e.to_string()));
        match opened {
            Ok(()) => {
                TRACE.store(true, Relaxed);
                out::write(
                    r#"{"name":"thread_name","ph":"M","pid":1,"tid":0,"args":{"name":"frames"}}"#,
                );
                tracing::info!(target: "victory_profile", dir = %dir.display(), "profiling to frames.csv and trace.json");
            }
            Err(e) => {
                tracing::warn!(target: "victory_profile", dir = %dir.display(), error = %e, "profile files not written");
            }
        }
    }
    ON.store(true, Relaxed);
    if overlay {
        overlay::enable();
        tracing::info!(target: "victory_profile", "profile overlay on; F3 shows and hides it");
    }
    tracing::info!(
        target: "victory_profile",
        interval_s = s.interval.as_secs_f64(),
        hitch_ms = s.hitch_ms,
        "profiling on"
    );
    let _ = std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(reporter);
}

/// Microseconds since [`init`], for the trace.
fn us(at: Instant) -> f64 {
    let start = *START.get_or_init(Instant::now);
    at.saturating_duration_since(start).as_secs_f64() * 1e6
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

// Named statistics.

/// What a [`Stat`] measures, which says what its four numbers are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Calls, an amount (bytes, items), nanoseconds, the longest call.
    Timer,
    /// Events and an amount; no time.
    Counter,
    /// Hits, misses, evictions.
    Cache,
    /// Additions, bytes added, removals, bytes removed; plus the level
    /// now and its peak.
    Resident,
    /// Loads, bytes, nanoseconds, the longest load.
    Load,
}

/// A cache lookup's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEvent {
    Hit = 0,
    Miss = 1,
    Evict = 2,
}

/// One named statistic: since the last summary and since the start.
#[derive(Debug)]
pub struct Stat {
    pub kind: Kind,
    pub group: &'static str,
    pub name: &'static str,
    now: [AtomicU64; 4],
    all: [AtomicU64; 4],
    level: AtomicI64,
    peak: AtomicI64,
}

impl Stat {
    const fn new(kind: Kind, group: &'static str, name: &'static str) -> Stat {
        Stat {
            kind,
            group,
            name,
            now: [const { AtomicU64::new(0) }; 4],
            all: [const { AtomicU64::new(0) }; 4],
            level: AtomicI64::new(0),
            peak: AtomicI64::new(0),
        }
    }

    fn bump(&self, i: usize, v: u64) {
        self.now[i].fetch_add(v, Relaxed);
        self.all[i].fetch_add(v, Relaxed);
    }

    fn longest(&self, v: u64) {
        self.now[3].fetch_max(v, Relaxed);
        self.all[3].fetch_max(v, Relaxed);
    }

    /// A call (or load) of `amount` that took `took`.
    pub fn record(&self, amount: u64, took: Duration) {
        let ns = nanos(took);
        self.bump(0, 1);
        self.bump(1, amount);
        self.bump(2, ns);
        self.longest(ns);
    }

    /// An event of `amount`.
    pub fn add(&self, amount: u64) {
        self.bump(0, 1);
        self.bump(1, amount);
    }

    pub fn cache(&self, event: CacheEvent) {
        self.bump(event as usize, 1);
    }

    /// `delta` bytes more (or fewer) resident.
    pub fn resident(&self, delta: i64) {
        if delta >= 0 {
            self.bump(0, 1);
            self.bump(1, delta as u64);
        } else {
            self.bump(2, 1);
            self.bump(3, delta.unsigned_abs());
        }
        let level = self.level.fetch_add(delta, Relaxed) + delta;
        self.peak.fetch_max(level, Relaxed);
    }

    fn take(&self) -> [u64; 4] {
        self.now.each_ref().map(|a| a.swap(0, Relaxed))
    }

    fn totals(&self) -> [u64; 4] {
        self.all.each_ref().map(|a| a.load(Relaxed))
    }

    /// Bytes resident now and at most (never below 0: what was resident
    /// before profiling began is not known).
    fn levels(&self) -> (u64, u64) {
        let clamp = |v: i64| v.max(0) as u64;
        (
            clamp(self.level.load(Relaxed)),
            clamp(self.peak.load(Relaxed)),
        )
    }

    fn label(&self) -> String {
        format!("{}/{}", self.group, self.name)
    }
}

static STATS: Mutex<Vec<&'static Stat>> = Mutex::new(Vec::new());

/// The statistic of that kind and name, made on first use. Call sites
/// keep it ([`stat!`]), so this runs once per site.
pub fn stat(kind: Kind, group: &'static str, name: &'static str) -> &'static Stat {
    let mut stats = lock(&STATS);
    if let Some(s) = stats
        .iter()
        .find(|s| s.kind == kind && s.group == group && s.name == name)
        .copied()
    {
        return s;
    }
    let s: &'static Stat = Box::leak(Box::new(Stat::new(kind, group, name)));
    stats.push(s);
    s
}

fn all_stats() -> Vec<&'static Stat> {
    lock(&STATS).clone()
}

/// The call site's [`Stat`], looked up once.
#[macro_export]
macro_rules! stat {
    ($kind:ident, $group:expr, $name:expr) => {{
        static AT: ::std::sync::OnceLock<&'static $crate::Stat> = ::std::sync::OnceLock::new();
        *AT.get_or_init(|| $crate::stat($crate::Kind::$kind, $group, $name))
    }};
}

/// Times the rest of the enclosing block as `group/name`, optionally with
/// an amount (bytes, items) evaluated only when profiling is on.
#[macro_export]
macro_rules! scope {
    ($group:expr, $name:expr) => {
        $crate::scope!($group, $name, 0u64)
    };
    ($group:expr, $name:expr, $amount:expr) => {
        let _profile_scope = if $crate::on() {
            $crate::Scope::start($crate::stat!(Timer, $group, $name), ($amount) as u64)
        } else {
            $crate::Scope::off()
        };
    };
}

/// A call of `amount` already timed as `took` (a `Duration`).
#[macro_export]
macro_rules! record {
    ($group:expr, $name:expr, $amount:expr, $took:expr) => {
        if $crate::on() {
            $crate::stat!(Timer, $group, $name).record(($amount) as u64, $took);
        }
    };
}

/// An event of `amount` (bytes, items).
#[macro_export]
macro_rules! count {
    ($group:expr, $name:expr, $amount:expr) => {
        if $crate::on() {
            $crate::stat!(Counter, $group, $name).add(($amount) as u64);
        }
    };
}

/// A cache lookup: `Hit`, `Miss` or `Evict`.
#[macro_export]
macro_rules! cache {
    ($group:expr, $name:expr, $event:ident) => {
        if $crate::on() {
            $crate::stat!(Cache, $group, $name).cache($crate::CacheEvent::$event);
        }
    };
}

/// Bytes now resident in `group/name`: positive when allocated, negative
/// when freed.
#[macro_export]
macro_rules! resident {
    ($group:expr, $name:expr, $delta:expr) => {
        if $crate::on() {
            $crate::stat!(Resident, $group, $name).resident(($delta) as i64);
        }
    };
}

/// A timed scope; see [`scope!`].
#[must_use]
pub struct Scope {
    at: Option<(&'static Stat, Instant, u64)>,
}

impl Scope {
    #[inline]
    pub fn off() -> Scope {
        Scope { at: None }
    }

    pub fn start(stat: &'static Stat, amount: u64) -> Scope {
        Scope {
            at: Some((stat, Instant::now(), amount)),
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some((stat, started, amount)) = self.at else {
            return;
        };
        let took = started.elapsed();
        stat.record(amount, took);
        if TRACE.load(Relaxed) && nanos(took) >= settings().trace_min_ns {
            let tid = out::thread_id();
            let args = if amount > 0 {
                format!(r#""amount":{amount}"#)
            } else {
                String::new()
            };
            out::span(
                stat.group,
                stat.name,
                tid,
                us(started),
                took.as_secs_f64() * 1e6,
                &args,
            );
        }
    }
}

// Frames.

/// Work of the frame being made, summed until it is presented.
static WORK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const SUBMISSIONS: usize = 0;
const LISTS: usize = 1;
const PREPARE_NS: usize = 2;
const GPU_WAIT_NS: usize = 3;

/// A host submission of `lists` command lists: `prepare` making it ready
/// on the CPU, `gpu_wait` waiting for its fence.
pub fn submission(lists: usize, prepare: Duration, gpu_wait: Duration) {
    if !on() {
        return;
    }
    WORK[SUBMISSIONS].fetch_add(1, Relaxed);
    WORK[LISTS].fetch_add(lists as u64, Relaxed);
    WORK[PREPARE_NS].fetch_add(nanos(prepare), Relaxed);
    WORK[GPU_WAIT_NS].fetch_add(nanos(gpu_wait), Relaxed);
}

/// A presented frame, as the presenter knows it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Frame {
    /// The present count, as `VICTORY_CAPTURE` numbers frames.
    pub number: u64,
    pub width: u32,
    pub height: u32,
    /// The presented texture's DXGI format.
    pub format: u32,
    /// Draws and dispatches run on the host since the last frame.
    pub draws: u64,
    pub dispatches: u64,
    /// The present itself: the copy to the swap chain and `Present`.
    pub present: Duration,
}

/// One frame's row of `frames.csv`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct FrameRow {
    pub(crate) number: u64,
    pub(crate) time_s: f64,
    /// Since the previous present (0 for the first).
    pub(crate) interval_ms: f64,
    pub(crate) present_ms: f64,
    pub(crate) prepare_ms: f64,
    pub(crate) gpu_wait_ms: f64,
    pub(crate) submissions: u64,
    pub(crate) lists: u64,
    pub(crate) draws: u64,
    pub(crate) dispatches: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) hitch: bool,
}

struct Frames {
    last: Option<Instant>,
    period: Vec<FrameRow>,
    recent: Ring<FrameRow>,
    total: u64,
    hitches: u64,
}

/// Frames kept for a dump.
const RECENT: usize = 300;

static FRAMES: Mutex<Option<Frames>> = Mutex::new(None);

fn frames() -> MutexGuard<'static, Option<Frames>> {
    lock(&FRAMES)
}

/// Counts a presented frame: its row, its trace events, the work since
/// the previous one.
pub fn frame(f: Frame) {
    if !on() {
        return;
    }
    let now = Instant::now();
    let work = WORK.each_ref().map(|a| a.swap(0, Relaxed));
    let per_ms = |ns: u64| ns as f64 / 1e6;
    let mut guard = frames();
    let state = guard.get_or_insert_with(|| Frames {
        last: None,
        period: Vec::new(),
        recent: Ring::new(RECENT),
        total: 0,
        hitches: 0,
    });
    let interval_ms = state.last.map_or(0.0, |last| ms(now - last));
    state.last = Some(now);
    let row = FrameRow {
        number: f.number,
        time_s: us(now) / 1e6,
        interval_ms,
        present_ms: ms(f.present),
        prepare_ms: per_ms(work[PREPARE_NS]),
        gpu_wait_ms: per_ms(work[GPU_WAIT_NS]),
        submissions: work[SUBMISSIONS],
        lists: work[LISTS],
        draws: f.draws,
        dispatches: f.dispatches,
        width: f.width,
        height: f.height,
        format: f.format,
        hitch: interval_ms > settings().hitch_ms,
    };
    state.total += 1;
    state.hitches += u64::from(row.hitch);
    state.period.push(row);
    state.recent.push(row);
    drop(guard);
    if !TRACE.load(Relaxed) {
        return;
    }
    out::frame_row(&row);
    let ts = us(now);
    out::span(
        "frame",
        &format!("frame {}", row.number),
        out::FRAMES_TID,
        ts - row.interval_ms * 1e3,
        row.interval_ms * 1e3,
        &format!(
            r#""gpu_wait_ms":{:.3},"prepare_ms":{:.3},"draws":{},"dispatches":{},"submissions":{}"#,
            row.gpu_wait_ms, row.prepare_ms, row.draws, row.dispatches, row.submissions
        ),
    );
    out::counter(
        "frame ms",
        ts,
        &[
            ("interval", row.interval_ms),
            ("gpu wait", row.gpu_wait_ms),
            ("prepare", row.prepare_ms),
        ],
    );
    if row.hitch {
        out::instant(
            "hitch",
            ts,
            &format!(
                r#""frame":{},"interval_ms":{:.3}"#,
                row.number, row.interval_ms
            ),
        );
    }
}

/// The frame-time lines of a set of rows.
fn frame_lines(rows: &[FrameRow], seconds: f64) -> Vec<String> {
    let intervals: Vec<f64> = rows
        .iter()
        .map(|r| r.interval_ms)
        .filter(|&v| v > 0.0)
        .collect();
    let Some(s) = Summary::of(&intervals) else {
        return vec![format!("{} frames", rows.len())];
    };
    let hitches = rows.iter().filter(|r| r.hitch).count();
    let stutters = stats::stutters(&intervals, 2.0);
    let n = rows.len().max(1) as f64;
    let mean = |f: fn(&FrameRow) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let mut lines = vec![format!(
        "{} frames in {seconds:.1} s: {:.1} fps (1% low {:.1}), frame ms avg {:.1} p50 {:.1} \
         p95 {:.1} p99 {:.1} max {:.1}; {hitches} hitches over {:.0} ms, {stutters} stutters \
         (over twice the median)",
        rows.len(),
        fps(s.avg),
        fps(s.low_1),
        s.avg,
        s.p50,
        s.p95,
        s.p99,
        s.max,
        settings().hitch_ms,
    )];
    let last = rows.last().copied().unwrap_or_default();
    lines.push(format!(
        "per frame: {:.2} ms preparing, {:.2} ms waiting for the GPU, {:.2} ms \
         presenting; {:.1} submissions, {:.1} lists, {:.0} draws, {:.0} dispatches; \
         last frame {} at {}x{} format {}",
        mean(|r| r.prepare_ms),
        mean(|r| r.gpu_wait_ms),
        mean(|r| r.present_ms),
        mean(|r| r.submissions as f64),
        mean(|r| r.lists as f64),
        mean(|r| r.draws as f64),
        mean(|r| r.dispatches as f64),
        last.number,
        last.width,
        last.height,
        last.format,
    ));
    lines
}

// Loads.

#[derive(Debug, Clone)]
struct LoadItem {
    kind: &'static str,
    name: String,
    bytes: u64,
    ns: u64,
}

struct Loads {
    biggest: Top<LoadItem>,
    slowest: Top<LoadItem>,
    all_biggest: Top<LoadItem>,
    all_slowest: Top<LoadItem>,
}

/// The biggest and slowest loads listed.
const TOP_LOADS: usize = 5;

static LOADS: Mutex<Option<Loads>> = Mutex::new(None);

/// Something the title loaded (a file read, a storage request, a
/// decoded stream): of `kind`, named by `name` (asked only when on),
/// `bytes` big, taking `took`.
pub fn load(kind: &'static str, name: impl FnOnce() -> String, bytes: u64, took: Duration) {
    if !on() {
        return;
    }
    stat(Kind::Load, "load", kind).record(bytes, took);
    let item = LoadItem {
        kind,
        name: name(),
        bytes,
        ns: nanos(took),
    };
    let now = Instant::now();
    if TRACE.load(Relaxed) {
        let tid = out::thread_id();
        out::span(
            &format!("load {kind}"),
            &item.name,
            tid,
            us(now) - took.as_secs_f64() * 1e6,
            took.as_secs_f64() * 1e6,
            &format!(r#""bytes":{bytes}"#),
        );
    }
    let mut loads = lock(&LOADS);
    let loads = loads.get_or_insert_with(|| Loads {
        biggest: Top::new(TOP_LOADS),
        slowest: Top::new(TOP_LOADS),
        all_biggest: Top::new(TOP_LOADS),
        all_slowest: Top::new(TOP_LOADS),
    });
    loads.biggest.push(item.bytes, item.clone());
    loads.slowest.push(item.ns, item.clone());
    loads.all_biggest.push(item.bytes, item.clone());
    loads.all_slowest.push(item.ns, item);
}

fn load_lines(all: bool) -> Vec<String> {
    let mut guard = lock(&LOADS);
    let Some(loads) = guard.as_mut() else {
        return Vec::new();
    };
    let (biggest, slowest) = if all {
        (&mut loads.all_biggest, &mut loads.all_slowest)
    } else {
        (&mut loads.biggest, &mut loads.slowest)
    };
    let item = |i: &LoadItem| {
        format!(
            "{} {} {} in {:.1} ms",
            i.kind,
            i.name,
            bytes_text(i.bytes as f64),
            i.ns as f64 / 1e6
        )
    };
    let mut lines = Vec::new();
    if !biggest.items().is_empty() {
        let list: Vec<String> = biggest.items().iter().map(|(_, i)| item(i)).collect();
        lines.push(format!("biggest loads: {}", list.join("; ")));
        let list: Vec<String> = slowest.items().iter().map(|(_, i)| item(i)).collect();
        lines.push(format!("slowest loads: {}", list.join("; ")));
    }
    if !all {
        biggest.clear();
        slowest.clear();
    }
    lines
}

// Notes, samplers and the summaries.

static NOTES: Mutex<BTreeMap<&'static str, String>> = Mutex::new(BTreeMap::new());

/// Context for a dump (the last pipeline, the device's state), kept
/// under `key`; `value` is asked only when on.
pub fn note(key: &'static str, value: impl FnOnce() -> String) {
    if on() {
        lock(&NOTES).insert(key, value());
    }
}

/// A source of named byte gauges (video memory and the like), asked with
/// every summary.
pub type Sampler = fn(&mut Vec<(&'static str, u64)>);

static SAMPLERS: Mutex<Vec<Sampler>> = Mutex::new(Vec::new());

/// Adds a sampler (once per function).
pub fn sampler(f: Sampler) {
    let mut samplers = lock(&SAMPLERS);
    if !samplers.iter().any(|&g| std::ptr::fn_addr_eq(g, f)) {
        samplers.push(f);
    }
}

fn samples() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    memory::sample(&mut out);
    let samplers = lock(&SAMPLERS).clone();
    for f in samplers {
        f(&mut out);
    }
    out
}

/// Bytes as MB (10^6, as the frame-rate line) or GB.
fn bytes_text(b: f64) -> String {
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.1} MB", b / 1e6)
    }
}

fn memory_lines(stats: &[&'static Stat]) -> Vec<String> {
    let sampled = samples();
    let mut lines = Vec::new();
    if !sampled.is_empty() {
        let list: Vec<String> = sampled
            .iter()
            .map(|&(name, v)| {
                if name.ends_with("faults") {
                    format!("{name} {v}")
                } else {
                    format!("{name} {}", bytes_text(v as f64))
                }
            })
            .collect();
        lines.push(format!("memory: {}", list.join(", ")));
    }
    let resident: Vec<String> = stats
        .iter()
        .filter(|s| s.kind == Kind::Resident)
        .map(|s| {
            let (now, peak) = s.levels();
            format!(
                "{} {} (peak {})",
                s.label(),
                bytes_text(now as f64),
                bytes_text(peak as f64)
            )
        })
        .collect();
    if !resident.is_empty() {
        lines.push(format!("resident: {}", resident.join(", ")));
    }
    if TRACE.load(Relaxed) {
        let mut series: Vec<(String, f64)> = sampled
            .iter()
            .filter(|(n, _)| !n.ends_with("faults"))
            .map(|&(n, v)| (n.to_owned(), v as f64 / 1e6))
            .collect();
        series.extend(
            stats
                .iter()
                .filter(|s| s.kind == Kind::Resident)
                .map(|s| (s.label(), s.levels().0 as f64 / 1e6)),
        );
        let series: Vec<(&str, f64)> = series.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        out::counter("memory MB", us(Instant::now()), &series);
    }
    lines
}

/// The caches, timers, counters and loads lines, of the period (taking
/// its numbers) or, with `all`, since the start.
fn stat_lines(stats: &[&'static Stat], all: bool) -> Vec<String> {
    let numbers: Vec<(&'static Stat, [u64; 4])> = stats
        .iter()
        .map(|&s| (s, if all { s.totals() } else { s.take() }))
        .collect();
    let mut lines = Vec::new();
    let caches: Vec<String> = numbers
        .iter()
        .filter(|(s, n)| s.kind == Kind::Cache && n[..3].iter().any(|&v| v > 0))
        .map(|(s, [hits, misses, evictions, _])| {
            let rate = 100.0 * *hits as f64 / (*hits + *misses).max(1) as f64;
            format!(
                "{} {hits} hits {misses} misses ({rate:.1}%) {evictions} evictions",
                s.label()
            )
        })
        .collect();
    if !caches.is_empty() {
        lines.push(format!("caches: {}", caches.join(", ")));
    }
    let mut timers: Vec<_> = numbers
        .iter()
        .filter(|(s, n)| s.kind == Kind::Timer && n[0] > 0)
        .collect();
    timers.sort_by_key(|(_, n)| std::cmp::Reverse(n[2]));
    for (s, [calls, amount, ns, longest]) in timers.into_iter().take(12) {
        let mut line = format!(
            "  {:<32} {calls:>9} calls {:>10.1} ms, avg {:.1} us, max {:.1} ms",
            s.label(),
            *ns as f64 / 1e6,
            *ns as f64 / 1e3 / *calls as f64,
            *longest as f64 / 1e6
        );
        if *amount > 0 {
            let _ = write!(line, ", {}", bytes_text(*amount as f64));
        }
        lines.push(line);
    }
    let counters: Vec<String> = numbers
        .iter()
        .filter(|(s, n)| s.kind == Kind::Counter && n[0] > 0)
        .map(|(s, [events, amount, ..])| format!("{} {events}x {amount}", s.label()))
        .collect();
    if !counters.is_empty() {
        lines.push(format!(
            "counters (events x amount): {}",
            counters.join(", ")
        ));
    }
    for (s, [loads, bytes, ns, longest]) in numbers
        .iter()
        .filter(|(s, n)| s.kind == Kind::Load && n[0] > 0)
    {
        let seconds = *ns as f64 / 1e9;
        lines.push(format!(
            "loads {}: {loads} {} in {seconds:.2} s busy ({:.0} MB/s), longest {:.1} ms",
            s.name,
            bytes_text(*bytes as f64),
            *bytes as f64 / 1e6 / seconds.max(1e-9),
            *longest as f64 / 1e6
        ));
    }
    lines.extend(load_lines(all));
    lines
}

fn log(lines: &[String]) {
    for line in lines {
        tracing::info!(target: "victory_profile", "{line}");
    }
}

/// The periodic summary, over the last `period`.
fn summary(period: Duration) {
    let rows = frames()
        .as_mut()
        .map(|f| std::mem::take(&mut f.period))
        .unwrap_or_default();
    let stats = all_stats();
    let mut lines = frame_lines(&rows, period.as_secs_f64());
    lines.extend(memory_lines(&stats));
    lines.extend(stat_lines(&stats, false));
    log(&lines);
    out::flush();
}

static EXTRAS: Mutex<Vec<fn()>> = Mutex::new(Vec::new());

/// Something more for the scheduled dumps (`VICTORY_PROFILE_DUMP`), such
/// as every thread's stack. Not run by [`dump`] itself: a run failing may
/// not survive suspending its threads.
pub fn with_dump(f: fn()) {
    let mut extras = lock(&EXTRAS);
    if !extras.iter().any(|&g| std::ptr::fn_addr_eq(g, f)) {
        extras.push(f);
    }
}

/// Logs everything known now, since the start: the frames so far and the
/// last few hundred, memory, every statistic, the notes. For a run about
/// to end badly (the host GPU failing), or on `VICTORY_PROFILE_DUMP`.
pub fn dump(reason: &str) {
    if !on() {
        return;
    }
    let (recent, total, hitches) = frames().as_ref().map_or((Vec::new(), 0, 0), |f| {
        (
            f.recent.iter().copied().collect::<Vec<_>>(),
            f.total,
            f.hitches,
        )
    });
    let start = *START.get_or_init(Instant::now);
    let mut lines = vec![format!(
        "profile dump ({reason}) after {:.1} s: {total} frames, {hitches} hitches",
        start.elapsed().as_secs_f64()
    )];
    if let (Some(first), Some(last)) = (recent.first(), recent.last()) {
        lines.push(format!(
            "frames {}-{} (the last {}), last presented {:.2} s ago:",
            first.number,
            last.number,
            recent.len(),
            (start.elapsed().as_secs_f64() - last.time_s).max(0.0)
        ));
        let seconds = last.time_s - first.time_s;
        lines.extend(frame_lines(&recent, seconds));
    }
    let stats = all_stats();
    lines.extend(memory_lines(&stats));
    lines.extend(stat_lines(&stats, true));
    for (key, value) in lock(&NOTES).iter() {
        lines.push(format!("{key}: {value}"));
    }
    log(&lines);
    if TRACE.load(Relaxed) {
        out::instant(
            "dump",
            us(Instant::now()),
            &format!(r#""reason":"{}""#, out::escape(reason)),
        );
    }
    out::flush();
}

/// The reporting thread: summaries every interval, dumps when asked.
fn reporter() {
    let s = settings();
    let start = *START.get_or_init(Instant::now);
    let mut dumps = s.dumps.iter().copied().peekable();
    let mut last = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(200));
        if let Some(&at) = dumps.peek()
            && start.elapsed() >= at
        {
            dumps.next();
            dump(&format!(
                "VICTORY_PROFILE_DUMP at {:.0} s",
                at.as_secs_f64()
            ));
            let extras = lock(&EXTRAS).clone();
            for f in extras {
                f();
            }
        }
        let period = last.elapsed();
        if period >= s.interval {
            last = Instant::now();
            summary(period);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_from_the_variables() {
        let s = Settings::parse("1", None, None, None, None);
        assert_eq!(s, Settings::default());
        let s = Settings::parse(
            r"C:\perf",
            Some("2"),
            Some("40"),
            Some("0"),
            Some("90, 30,x,-1"),
        );
        assert_eq!(s.dir, Some(PathBuf::from(r"C:\perf")));
        assert_eq!(s.interval, Duration::from_secs(2));
        assert_eq!(s.hitch_ms, 40.0);
        assert_eq!(s.trace_min_ns, 0);
        assert_eq!(s.dumps, [Duration::from_secs(30), Duration::from_secs(90)]);
        // Nonsense falls back; a tiny interval is held at half a second.
        let s = Settings::parse(" ", Some("0.01"), Some("-3"), Some("nan"), None);
        assert_eq!(s.dir, None);
        assert_eq!(s.interval, Duration::from_millis(500));
        assert_eq!(s.hitch_ms, 50.0);
        assert_eq!(s.trace_min_ns, 100_000);
        assert_eq!(Settings::parse("overlay", None, None, None, None).dir, None);
    }

    #[test]
    fn stats_keep_the_period_and_the_total() {
        let s = Stat::new(Kind::Timer, "test", "timer");
        s.record(10, Duration::from_micros(5));
        s.record(20, Duration::from_micros(15));
        assert_eq!(s.take(), [2, 30, 20_000, 15_000]);
        assert_eq!(s.take(), [0; 4]);
        s.record(1, Duration::from_micros(1));
        assert_eq!(s.totals(), [3, 31, 21_000, 15_000]);
    }

    #[test]
    fn resident_levels_and_peaks() {
        let s = Stat::new(Kind::Resident, "test", "resident");
        s.resident(100);
        s.resident(50);
        s.resident(-120);
        assert_eq!(s.levels(), (30, 150));
        assert_eq!(s.totals(), [2, 150, 1, 120]);
        let freed_first = Stat::new(Kind::Resident, "test", "before");
        freed_first.resident(-10);
        assert_eq!(freed_first.levels(), (0, 0));
    }

    #[test]
    fn caches_count_by_event() {
        let s = Stat::new(Kind::Cache, "test", "cache");
        s.cache(CacheEvent::Hit);
        s.cache(CacheEvent::Hit);
        s.cache(CacheEvent::Miss);
        s.cache(CacheEvent::Evict);
        assert_eq!(s.take(), [2, 1, 1, 0]);
    }

    #[test]
    fn one_stat_per_name() {
        let a = stat(Kind::Counter, "test", "same");
        let b = stat(Kind::Counter, "test", "same");
        let c = stat(Kind::Timer, "test", "same");
        assert!(std::ptr::eq(a, b));
        assert!(!std::ptr::eq(a, c));
    }

    #[test]
    fn frame_lines_sum_up_rows() {
        let row = |number, interval_ms, hitch| FrameRow {
            number,
            interval_ms,
            hitch,
            draws: 100,
            gpu_wait_ms: 10.0,
            width: 1920,
            height: 1080,
            ..Default::default()
        };
        let mut rows = vec![row(1, 0.0, false)];
        rows.extend((2..=10).map(|n| row(n, 33.0, false)));
        rows.push(row(11, 99.0, true));
        let lines = frame_lines(&rows, 1.0);
        assert!(lines[0].starts_with("11 frames in 1.0 s: "), "{}", lines[0]);
        assert!(lines[0].contains("max 99.0; 1 hitches"), "{}", lines[0]);
        assert!(lines[0].contains("1 stutters"), "{}", lines[0]);
        assert!(
            lines[1].contains("10.00 ms waiting for the GPU"),
            "{}",
            lines[1]
        );
        assert!(
            lines[1].contains("last frame 11 at 1920x1080"),
            "{}",
            lines[1]
        );
        assert_eq!(frame_lines(&[row(1, 0.0, false)], 1.0), ["1 frames"]);
    }

    #[test]
    fn bytes_read_as_mb_or_gb() {
        assert_eq!(bytes_text(1.5e6), "1.5 MB");
        assert_eq!(bytes_text(2.25e9), "2.25 GB");
    }
}
