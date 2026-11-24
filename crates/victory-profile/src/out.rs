//! The files under `VICTORY_PROFILE=<dir>`: `frames.csv`, one row per
//! presented frame, and `trace.json`, Chrome trace events (open it in
//! <https://ui.perfetto.dev> or `chrome://tracing`).
//!
//! The trace is a JSON array written as it goes and never closed, which
//! the trace event format allows, so a run that crashes or is killed
//! still leaves a readable file.

use std::cell::Cell;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::FrameRow;

/// Events past this are dropped (about 1-2 GB of trace), with one line.
const MAX_EVENTS: u64 = 10_000_000;

struct Files {
    frames: Option<BufWriter<File>>,
    trace: Option<BufWriter<File>>,
    events: u64,
}

static FILES: Mutex<Files> = Mutex::new(Files {
    frames: None,
    trace: None,
    events: 0,
});

fn files() -> std::sync::MutexGuard<'static, Files> {
    FILES.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) const CSV_HEADER: &str = "frame,time_s,interval_ms,fps,present_ms,prepare_ms,\
gpu_wait_ms,submissions,lists,draws,dispatches,width,height,format,hitch\n";

/// Creates both files in `dir`.
pub(crate) fn open(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut frames = BufWriter::new(File::create(dir.join("frames.csv"))?);
    frames.write_all(CSV_HEADER.as_bytes())?;
    let mut trace = BufWriter::new(File::create(dir.join("trace.json"))?);
    trace.write_all(b"[\n")?;
    let mut f = files();
    f.frames = Some(frames);
    f.trace = Some(trace);
    Ok(())
}

pub(crate) fn csv_row(r: &FrameRow) -> String {
    format!(
        "{},{:.4},{:.3},{:.2},{:.3},{:.3},{:.3},{},{},{},{},{},{},{},{}\n",
        r.number,
        r.time_s,
        r.interval_ms,
        crate::stats::fps(r.interval_ms),
        r.present_ms,
        r.prepare_ms,
        r.gpu_wait_ms,
        r.submissions,
        r.lists,
        r.draws,
        r.dispatches,
        r.width,
        r.height,
        r.format,
        u8::from(r.hitch),
    )
}

pub(crate) fn frame_row(r: &FrameRow) {
    let row = csv_row(r);
    if let Some(w) = files().frames.as_mut() {
        let _ = w.write_all(row.as_bytes());
    }
}

/// `s` as the inside of a JSON string.
pub(crate) fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// A small id for the calling thread; its first event names it.
pub(crate) fn thread_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static ID: Cell<u64> = const { Cell::new(0) };
    }
    ID.with(|id| {
        if id.get() == 0 {
            let n = NEXT.fetch_add(1, Relaxed);
            id.set(n);
            let current = std::thread::current();
            let name = current
                .name()
                .map_or_else(|| format!("thread {n}"), str::to_owned);
            write(&format!(
                r#"{{"name":"thread_name","ph":"M","pid":1,"tid":{n},"args":{{"name":"{}"}}}}"#,
                escape(&name)
            ));
        }
        id.get()
    })
}

/// The id of the frames' own row in the trace.
pub(crate) const FRAMES_TID: u64 = 0;

/// One event, a JSON object.
pub(crate) fn write(event: &str) {
    let mut f = files();
    let f = &mut *f;
    let Some(w) = f.trace.as_mut() else { return };
    f.events += 1;
    if f.events == MAX_EVENTS {
        tracing::warn!(target: "victory_profile", "trace.json is full; later events are dropped");
    }
    if f.events >= MAX_EVENTS {
        return;
    }
    let _ = w.write_all(event.as_bytes());
    let _ = w.write_all(b",\n");
}

/// A span: `name` in category `cat`, from `ts_us` for `dur_us`, with JSON
/// `args` (an object's inside, may be empty).
pub(crate) fn span(cat: &str, name: &str, tid: u64, ts_us: f64, dur_us: f64, args: &str) {
    write(&format!(
        r#"{{"name":"{}","cat":"{}","ph":"X","pid":1,"tid":{tid},"ts":{ts_us:.1},"dur":{dur_us:.1},"args":{{{args}}}}}"#,
        escape(name),
        escape(cat),
    ));
}

/// A counter track `name` with numeric series.
pub(crate) fn counter(name: &str, ts_us: f64, series: &[(&str, f64)]) {
    let mut args = String::new();
    for (i, (k, v)) in series.iter().enumerate() {
        if i > 0 {
            args.push(',');
        }
        let v = if v.is_finite() { *v } else { 0.0 };
        let _ = write!(args, r#""{}":{v}"#, escape(k));
    }
    write(&format!(
        r#"{{"name":"{}","ph":"C","pid":1,"ts":{ts_us:.1},"args":{{{args}}}}}"#,
        escape(name)
    ));
}

/// A moment worth seeing on the frames' row (a hitch, a dump).
pub(crate) fn instant(name: &str, ts_us: f64, args: &str) {
    write(&format!(
        r#"{{"name":"{}","ph":"i","s":"g","pid":1,"tid":{FRAMES_TID},"ts":{ts_us:.1},"args":{{{args}}}}}"#,
        escape(name)
    ));
}

#[cfg(test)]
fn close() {
    let mut f = files();
    f.frames = None;
    f.trace = None;
}

pub(crate) fn flush() {
    let mut f = files();
    if let Some(w) = f.frames.as_mut() {
        let _ = w.flush();
    }
    if let Some(w) = f.trace.as_mut() {
        let _ = w.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(escape(r#"a "b" \c"#), r#"a \"b\" \\c"#);
        assert_eq!(escape("x\ny\u{1}"), "x\\ny\\u0001");
        assert_eq!(escape("é"), "é");
    }

    #[test]
    fn a_csv_row_matches_the_header() {
        let row = FrameRow {
            number: 42,
            time_s: 1.5,
            interval_ms: 40.0,
            present_ms: 1.0,
            prepare_ms: 2.0,
            gpu_wait_ms: 30.0,
            submissions: 3,
            lists: 7,
            draws: 900,
            dispatches: 12,
            width: 1920,
            height: 1080,
            format: 28,
            hitch: false,
        };
        let line = csv_row(&row);
        assert_eq!(
            line,
            "42,1.5000,40.000,25.00,1.000,2.000,30.000,3,7,900,12,1920,1080,28,0\n"
        );
        assert_eq!(line.split(',').count(), CSV_HEADER.split(',').count());
    }

    #[test]
    fn files_are_created_with_their_headers() {
        let dir = tempfile::tempdir().unwrap();
        open(dir.path()).unwrap();
        counter("memory", 1.0, &[("bytes", 2.0), ("bad", f64::NAN)]);
        flush();
        let csv = std::fs::read_to_string(dir.path().join("frames.csv")).unwrap();
        assert_eq!(csv, CSV_HEADER);
        let trace = std::fs::read_to_string(dir.path().join("trace.json")).unwrap();
        assert!(trace.starts_with("[\n"));
        assert!(trace.contains(r#""args":{"bytes":2,"bad":0}"#), "{trace}");
        close();
    }
}
