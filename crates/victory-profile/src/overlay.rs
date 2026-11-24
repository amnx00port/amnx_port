//! The on-screen panel (`VICTORY_PROFILE_OVERLAY=1`, or
//! `VICTORY_PROFILE=overlay`): frame rate and times with a graph of the
//! last frames, GPU waits, draws, memory, video memory, cache hit rates and
//! load rates, drawn by the presenter into a corner of the back buffer.
//! F3 shows and hides it.
//!
//! Everything here is plain data: the text ([`lines`]), the image in
//! palette indices ([`rasterize`]) with our own 5x7 font ([`FONT`]), and
//! its pixels in the back buffer's format ([`write_rows`]). The GPU side
//! only copies the rows into the back buffer.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::stats::{Summary, fps};
use crate::{Kind, bytes_text, lock};

static ENABLED: AtomicBool = AtomicBool::new(false);
static VISIBLE: AtomicBool = AtomicBool::new(false);

/// Whether the variables ask for the panel: `VICTORY_PROFILE=overlay` or
/// `VICTORY_PROFILE_OVERLAY=1`.
pub(crate) fn wanted(profile: Option<&str>, overlay: Option<&str>) -> bool {
    profile.is_some_and(|p| p.trim().eq_ignore_ascii_case("overlay"))
        || overlay.is_some_and(|v| matches!(v.trim(), "1" | "on" | "true"))
}

pub(crate) fn enable() {
    ENABLED.store(true, Relaxed);
    VISIBLE.store(true, Relaxed);
}

/// Whether the panel was asked for (shown or hidden); one relaxed load.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// Shows or hides the panel (the presenter's F3).
pub fn toggle() {
    VISIBLE.fetch_xor(true, Relaxed);
}

/// The panel at most this often redrawn.
const REDRAW: Duration = Duration::from_millis(250);
/// Memory and video memory at most this often asked; load rates over this.
const RESAMPLE: Duration = Duration::from_secs(1);

struct Cached {
    at: Instant,
    scale: u32,
    image: Arc<Image>,
}

static CACHED: Mutex<Option<Cached>> = Mutex::new(None);

/// The panel for a back buffer `height` pixels high, if it is shown: the
/// same image (the same `Arc`) until it is redrawn.
pub fn image(height: u32) -> Option<Arc<Image>> {
    if !enabled() || !VISIBLE.load(Relaxed) {
        return None;
    }
    let scale = (height / 720).clamp(1, 4);
    let mut cached = lock(&CACHED);
    if let Some(c) = cached.as_ref()
        && c.scale == scale
        && c.at.elapsed() < REDRAW
    {
        return Some(c.image.clone());
    }
    let input = gather();
    let image = Arc::new(rasterize(
        &lines(&input),
        &input.intervals,
        input.hitch_ms,
        scale,
    ));
    *cached = Some(Cached {
        at: Instant::now(),
        scale,
        image: image.clone(),
    });
    Some(image)
}

/// What the panel shows, gathered from the profiler.
#[derive(Debug, Clone, Default)]
pub(crate) struct PanelInput {
    pub(crate) frame: u64,
    /// The last frames' intervals in milliseconds, oldest first.
    pub(crate) intervals: Vec<f64>,
    pub(crate) gpu_wait_ms: f64,
    pub(crate) draws: f64,
    pub(crate) submissions: f64,
    /// Memory gauges by name, as the samplers give them.
    pub(crate) samples: Vec<(&'static str, u64)>,
    /// Caches: name, hits, misses (since the start).
    pub(crate) caches: Vec<(String, u64, u64)>,
    /// MB/s over the last second.
    pub(crate) file_mbps: f64,
    pub(crate) storage_mbps: f64,
    pub(crate) hitch_ms: f64,
}

/// Frames in the graph (and in the panel's averages).
pub(crate) const GRAPH_FRAMES: usize = 120;

struct Rates {
    at: Instant,
    bytes: [u64; 2],
    mbps: [f64; 2],
}

/// When the cache counters were last read, and what they were.
type Sampled = (Instant, Vec<(&'static str, u64)>);

static SAMPLED: Mutex<Option<Sampled>> = Mutex::new(None);
static RATES: Mutex<Option<Rates>> = Mutex::new(None);

fn gather() -> PanelInput {
    let rows: Vec<crate::FrameRow> = crate::frames().as_ref().map_or_else(Vec::new, |f| {
        let skip = f.recent.len().saturating_sub(GRAPH_FRAMES);
        f.recent.iter().skip(skip).copied().collect()
    });
    let n = rows.len().max(1) as f64;
    let mean = |f: fn(&crate::FrameRow) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let samples = {
        let mut sampled = lock(&SAMPLED);
        if sampled
            .as_ref()
            .is_none_or(|(at, _)| at.elapsed() >= RESAMPLE)
        {
            *sampled = Some((Instant::now(), crate::samples()));
        }
        sampled.as_ref().map(|(_, s)| s.clone()).unwrap_or_default()
    };
    let stats = crate::all_stats();
    let caches = stats
        .iter()
        .filter(|s| s.kind == Kind::Cache)
        .map(|s| {
            let [hits, misses, ..] = s.totals();
            (s.label(), hits, misses)
        })
        .filter(|&(_, hits, misses)| hits + misses > 0)
        .collect();
    // Bytes read since the start: every `ReadFile` (the file loads are
    // only the big or slow ones), then the DirectStorage requests.
    let mut bytes = [0u64; 2];
    for s in &stats {
        let i = match (s.kind, s.group, s.name) {
            (Kind::Timer, "hle", "ReadFile") => 0,
            (Kind::Load, _, name) if name.starts_with("dstorage") => 1,
            _ => continue,
        };
        bytes[i] += s.totals()[1];
    }
    let mbps = {
        let mut rates = lock(&RATES);
        let r = rates.get_or_insert_with(|| Rates {
            at: Instant::now(),
            bytes,
            mbps: [0.0; 2],
        });
        let seconds = r.at.elapsed().as_secs_f64();
        if seconds >= RESAMPLE.as_secs_f64() {
            r.mbps = [0, 1].map(|i| bytes[i].saturating_sub(r.bytes[i]) as f64 / 1e6 / seconds);
            r.bytes = bytes;
            r.at = Instant::now();
        }
        r.mbps
    };
    PanelInput {
        frame: rows.last().map_or(0, |r| r.number),
        intervals: rows
            .iter()
            .map(|r| r.interval_ms)
            .filter(|&v| v > 0.0)
            .collect(),
        gpu_wait_ms: mean(|r| r.gpu_wait_ms),
        draws: mean(|r| r.draws as f64),
        submissions: mean(|r| r.submissions as f64),
        samples,
        caches,
        file_mbps: mbps[0],
        storage_mbps: mbps[1],
        hitch_ms: crate::settings().hitch_ms,
    }
}

/// Longest line shown, in characters.
pub(crate) const MAX_COLUMNS: usize = 52;

/// The panel's text, upper case, each line at most [`MAX_COLUMNS`].
pub(crate) fn lines(i: &PanelInput) -> Vec<String> {
    let mut out = vec![format!("VICTORY PROFILE  FRAME {}  F3 HIDES", i.frame)];
    match Summary::of(&i.intervals) {
        Some(s) => out.push(format!(
            "{:.1} FPS  {:.1} MS AVG  1% LOW {:.1} FPS  MAX {:.1} MS",
            fps(s.avg),
            s.avg,
            fps(s.low_1),
            s.max
        )),
        None => out.push("WAITING FOR FRAMES".to_owned()),
    }
    out.push(format!(
        "GPU WAIT {:.1} MS  DRAWS {:.0}  SUBMITS {:.1}",
        i.gpu_wait_ms, i.draws, i.submissions
    ));
    let get = |name: &str| {
        i.samples
            .iter()
            .find(|(k, _)| *k == name)
            .map(|&(_, v)| bytes_text(v as f64))
    };
    let na = || "N/A".to_owned();
    out.push(format!(
        "RAM {} WORKING SET  {} COMMIT",
        get("process.working_set").unwrap_or_else(na),
        get("process.private").unwrap_or_else(na)
    ));
    out.push(match (get("vram.local"), get("vram.local_budget")) {
        (Some(used), Some(budget)) => format!(
            "VRAM {used} OF {budget}  SHARED {}",
            get("vram.shared").unwrap_or_else(na)
        ),
        _ => "VRAM N/A".to_owned(),
    });
    let caches: Vec<String> = i
        .caches
        .iter()
        .map(|(name, hits, misses)| {
            let rate = 100.0 * *hits as f64 / (*hits + *misses).max(1) as f64;
            format!("{name} {rate:.1}%")
        })
        .collect();
    for pair in caches.chunks(2) {
        out.push(pair.join("  "));
    }
    out.push(format!(
        "LOADS FILE {:.1} MB/S  STORAGE {:.1} MB/S",
        i.file_mbps, i.storage_mbps
    ));
    out.into_iter()
        .map(|l| l.to_ascii_uppercase().chars().take(MAX_COLUMNS).collect())
        .collect()
}

/// Our own 5x7 font: seven rows of five, top to bottom, `#` set. Characters
/// it lacks show as `?`; text is upper case.
pub(crate) const FONT: &[(char, &str)] = &[
    (' ', "..... ..... ..... ..... ..... ..... ....."),
    ('A', ".###. #...# #...# ##### #...# #...# #...#"),
    ('B', "####. #...# #...# ####. #...# #...# ####."),
    ('C', ".###. #...# #.... #.... #.... #...# .###."),
    ('D', "####. #...# #...# #...# #...# #...# ####."),
    ('E', "##### #.... #.... ####. #.... #.... #####"),
    ('F', "##### #.... #.... ####. #.... #.... #...."),
    ('G', ".###. #...# #.... #.### #...# #...# .####"),
    ('H', "#...# #...# #...# ##### #...# #...# #...#"),
    ('I', ".###. ..#.. ..#.. ..#.. ..#.. ..#.. .###."),
    ('J', "..### ...#. ...#. ...#. ...#. #..#. .##.."),
    ('K', "#...# #..#. #.#.. ##... #.#.. #..#. #...#"),
    ('L', "#.... #.... #.... #.... #.... #.... #####"),
    ('M', "#...# ##.## #.#.# #.#.# #...# #...# #...#"),
    ('N', "#...# #...# ##..# #.#.# #..## #...# #...#"),
    ('O', ".###. #...# #...# #...# #...# #...# .###."),
    ('P', "####. #...# #...# ####. #.... #.... #...."),
    ('Q', ".###. #...# #...# #...# #.#.# #..#. .##.#"),
    ('R', "####. #...# #...# ####. #.#.. #..#. #...#"),
    ('S', ".#### #.... #.... .###. ....# ....# ####."),
    ('T', "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#.."),
    ('U', "#...# #...# #...# #...# #...# #...# .###."),
    ('V', "#...# #...# #...# #...# #...# .#.#. ..#.."),
    ('W', "#...# #...# #...# #.#.# #.#.# #.#.# .#.#."),
    ('X', "#...# #...# .#.#. ..#.. .#.#. #...# #...#"),
    ('Y', "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#.."),
    ('Z', "##### ....# ...#. ..#.. .#... #.... #####"),
    ('0', ".###. #...# #..## #.#.# ##..# #...# .###."),
    ('1', "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###."),
    ('2', ".###. #...# ....# ...#. ..#.. .#... #####"),
    ('3', "####. ....# ....# .###. ....# ....# ####."),
    ('4', "...#. ..##. .#.#. #..#. ##### ...#. ...#."),
    ('5', "##### #.... ####. ....# ....# #...# .###."),
    ('6', "..##. .#... #.... ####. #...# #...# .###."),
    ('7', "##### ....# ...#. ..#.. .#... .#... .#..."),
    ('8', ".###. #...# #...# .###. #...# #...# .###."),
    ('9', ".###. #...# #...# .#### ....# ...#. .##.."),
    ('.', "..... ..... ..... ..... ..... .##.. .##.."),
    (',', "..... ..... ..... ..... .##.. ..#.. .#..."),
    (':', "..... .##.. .##.. ..... .##.. .##.. ....."),
    ('/', "....# ....# ...#. ..#.. .#... #.... #...."),
    ('%', "##..# ##..# ...#. ..#.. .#... #..## #..##"),
    ('(', "...#. ..#.. .#... .#... .#... ..#.. ...#."),
    (')', ".#... ..#.. ...#. ...#. ...#. ..#.. .#..."),
    ('-', "..... ..... ..... ##### ..... ..... ....."),
    ('+', "..... ..#.. ..#.. ##### ..#.. ..#.. ....."),
    ('=', "..... ..... ##### ..... ##### ..... ....."),
    ('_', "..... ..... ..... ..... ..... ..... #####"),
    ('|', "..#.. ..#.. ..#.. ..#.. ..#.. ..#.. ..#.."),
    ('<', "...#. ..#.. .#... #.... .#... ..#.. ...#."),
    ('>', ".#... ..#.. ...#. ....# ...#. ..#.. .#..."),
    ('!', "..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#.."),
    ('?', ".###. #...# ....# ...#. ..#.. ..... ..#.."),
];

/// The rows of `c`'s glyph, top to bottom.
fn glyph(c: char) -> std::str::SplitAsciiWhitespace<'static> {
    let find = |c: char| FONT.iter().find(|(g, _)| *g == c).map(|&(_, rows)| rows);
    find(c.to_ascii_uppercase())
        .or_else(|| find('?'))
        .expect("the font has '?'")
        .split_ascii_whitespace()
}

/// Palette indices of [`Image`] pixels.
pub const BACKGROUND: u8 = 0;
pub const TEXT: u8 = 1;
pub const BAR: u8 = 2;
pub const HITCH: u8 = 3;
pub const GUIDE: u8 = 4;

/// Linear RGB of each palette index.
pub const PALETTE: [[f32; 3]; 5] = [
    [0.02, 0.02, 0.03],
    [0.95, 0.95, 0.95],
    [0.25, 0.80, 0.35],
    [0.95, 0.30, 0.25],
    [0.40, 0.40, 0.45],
];

/// A character cell: the 5x7 glyph and a pixel of spacing around it.
const CELL_W: u32 = 6;
const CELL_H: u32 = 9;
/// Space around everything.
const PAD: u32 = 4;
/// Between the text and the graph.
const GAP: u32 = 4;
const BAR_W: u32 = 2;
const GRAPH_H: u32 = 36;

/// The panel in palette indices, row by row, `scale` times the font's size.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    fn at(&self, x: u32, y: u32) -> u8 {
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Lays out `lines` and, below them, a bar per frame interval in
/// `intervals` (newest at the right; a full bar is twice `hitch_ms`, bars
/// above `hitch_ms` in red, a guide line at `hitch_ms`).
pub(crate) fn rasterize(lines: &[String], intervals: &[f64], hitch_ms: f64, scale: u32) -> Image {
    let columns = lines
        .iter()
        .map(|l| l.chars().count().min(MAX_COLUMNS))
        .max()
        .unwrap_or(0) as u32;
    let width = 2 * PAD + (columns * CELL_W).max(GRAPH_FRAMES as u32 * BAR_W);
    let text_h = lines.len() as u32 * CELL_H;
    let height = 2 * PAD + text_h + GAP + GRAPH_H;
    let mut pixels = vec![BACKGROUND; (width * height) as usize];
    let mut set = |x: u32, y: u32, v: u8| {
        if x < width && y < height {
            pixels[(y * width + x) as usize] = v;
        }
    };
    for (row, line) in lines.iter().enumerate() {
        for (col, c) in line.chars().take(MAX_COLUMNS).enumerate() {
            let (x0, y0) = (PAD + col as u32 * CELL_W, PAD + row as u32 * CELL_H + 1);
            for (gy, bits) in glyph(c).enumerate() {
                for (gx, b) in bits.bytes().enumerate() {
                    if b == b'#' {
                        set(x0 + gx as u32, y0 + gy as u32, TEXT);
                    }
                }
            }
        }
    }
    let bottom = height - PAD; // one past the graph's last row
    let top_ms = (2.0 * hitch_ms).max(1.0);
    let guide = bottom - (GRAPH_H / 2);
    for x in PAD..width - PAD {
        set(x, guide, GUIDE);
    }
    let shown = &intervals[intervals.len().saturating_sub(GRAPH_FRAMES)..];
    let right = width - PAD;
    for (i, &ms) in shown.iter().rev().enumerate() {
        let x = right - (i as u32 + 1) * BAR_W;
        let bar = ((ms / top_ms).clamp(0.0, 1.0) * f64::from(GRAPH_H)) as u32;
        let color = if ms > hitch_ms { HITCH } else { BAR };
        for y in bottom - bar.max(1)..bottom {
            for dx in 0..BAR_W - 1 {
                set(x + dx, y, color);
            }
        }
    }
    let image = Image {
        width,
        height,
        scale: 1,
        pixels,
    };
    scaled(image, scale.max(1))
}

/// Each pixel `scale` times in both directions.
fn scaled(image: Image, scale: u32) -> Image {
    if scale == 1 {
        return image;
    }
    let (width, height) = (image.width * scale, image.height * scale);
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(image.at(x / scale, y / scale));
        }
    }
    Image {
        width,
        height,
        scale,
        pixels,
    }
}

/// Back buffer formats the panel can be written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgb10a2,
    Rgba16f,
}

impl PixelFormat {
    /// From a DXGI format number (the swap chain formats the presenter
    /// uses).
    pub fn from_dxgi(format: u32) -> Option<PixelFormat> {
        Some(match format {
            28 => PixelFormat::Rgba8,
            87 => PixelFormat::Bgra8,
            24 => PixelFormat::Rgb10a2,
            10 => PixelFormat::Rgba16f,
            _ => return None,
        })
    }

    pub fn bytes(self) -> usize {
        match self {
            PixelFormat::Rgba16f => 8,
            _ => 4,
        }
    }

    /// An opaque color in this format, little-endian.
    fn encode(self, [r, g, b]: [f32; 3]) -> Vec<u8> {
        let unorm = |v: f32, max: f32| (v.clamp(0.0, 1.0) * max).round() as u32;
        match self {
            PixelFormat::Rgba8 => [r, g, b]
                .map(|v| unorm(v, 255.0) as u8)
                .into_iter()
                .chain([255])
                .collect(),
            PixelFormat::Bgra8 => [b, g, r]
                .map(|v| unorm(v, 255.0) as u8)
                .into_iter()
                .chain([255])
                .collect(),
            PixelFormat::Rgb10a2 => {
                let word =
                    unorm(r, 1023.0) | unorm(g, 1023.0) << 10 | unorm(b, 1023.0) << 20 | 3 << 30;
                word.to_le_bytes().to_vec()
            }
            PixelFormat::Rgba16f => [r, g, b, 1.0]
                .iter()
                .flat_map(|&v| half(v).to_le_bytes())
                .collect(),
        }
    }
}

/// `v` (0 to 65504) as a half float, rounded down; zero below the
/// smallest normal half.
fn half(v: f32) -> u16 {
    let v = v.clamp(0.0, 65504.0);
    if v < 6.104e-5 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) + 15 - 127;
    ((exponent << 10) | ((bits >> 13) & 0x3ff)) as u16
}

/// Writes `image` in `format` into `out`, rows `row_pitch` bytes apart
/// (rows that do not fit are left out).
pub fn write_rows(image: &Image, format: PixelFormat, row_pitch: usize, out: &mut [u8]) {
    let colors: Vec<Vec<u8>> = PALETTE.iter().map(|&c| format.encode(c)).collect();
    let bpp = format.bytes();
    for y in 0..image.height as usize {
        let start = y * row_pitch;
        let Some(row) = out.get_mut(start..start + image.width as usize * bpp) else {
            return;
        };
        let pixels = &image.pixels[y * image.width as usize..][..image.width as usize];
        for (dst, &p) in row.chunks_exact_mut(bpp).zip(pixels) {
            dst.copy_from_slice(&colors[usize::from(p).min(colors.len() - 1)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PanelInput {
        let mut intervals = vec![20.0; 99];
        intervals.push(60.0);
        PanelInput {
            frame: 1234,
            intervals,
            gpu_wait_ms: 12.3,
            draws: 812.0,
            submissions: 3.0,
            samples: vec![
                ("process.working_set", 6_120_000_000),
                ("process.private", 7_200_000_000),
                ("vram.local", 3_100_000_000),
                ("vram.local_budget", 7_500_000_000),
                ("vram.shared", 250_000_000),
            ],
            caches: vec![
                ("shader/disk cache".to_owned(), 990, 10),
                ("gpu/host pipelines".to_owned(), 3, 1),
                ("gpu/detiled textures".to_owned(), 0, 5),
            ],
            file_mbps: 12.34,
            storage_mbps: 140.0,
            hitch_ms: 50.0,
        }
    }

    #[test]
    fn the_panel_reads_as_expected() {
        let l = lines(&sample());
        assert_eq!(l[0], "VICTORY PROFILE  FRAME 1234  F3 HIDES");
        assert_eq!(l[1], "49.0 FPS  20.4 MS AVG  1% LOW 16.7 FPS  MAX 60.0 MS");
        assert_eq!(l[2], "GPU WAIT 12.3 MS  DRAWS 812  SUBMITS 3.0");
        assert_eq!(l[3], "RAM 6.12 GB WORKING SET  7.20 GB COMMIT");
        assert_eq!(l[4], "VRAM 3.10 GB OF 7.50 GB  SHARED 250.0 MB");
        assert_eq!(l[5], "SHADER/DISK CACHE 99.0%  GPU/HOST PIPELINES 75.0%");
        assert_eq!(l[6], "GPU/DETILED TEXTURES 0.0%");
        assert_eq!(l[7], "LOADS FILE 12.3 MB/S  STORAGE 140.0 MB/S");
        assert!(l.iter().all(|line| line.chars().count() <= MAX_COLUMNS));
        let empty = lines(&PanelInput::default());
        assert_eq!(empty[1], "WAITING FOR FRAMES");
        assert_eq!(empty[3], "RAM N/A WORKING SET  N/A COMMIT");
        assert_eq!(empty[4], "VRAM N/A");
    }

    #[test]
    fn the_font_is_well_formed_and_covers_the_panel() {
        for (i, (c, rows)) in FONT.iter().enumerate() {
            let rows: Vec<&str> = rows.split_ascii_whitespace().collect();
            assert_eq!(rows.len(), 7, "{c:?}");
            for row in rows {
                assert!(
                    row.len() == 5 && row.bytes().all(|b| b == b'.' || b == b'#'),
                    "{c:?}"
                );
            }
            assert!(FONT[..i].iter().all(|(d, _)| d != c), "{c:?} twice");
        }
        for line in lines(&sample()) {
            for c in line.chars() {
                assert!(FONT.iter().any(|(g, _)| *g == c), "no glyph for {c:?}");
            }
        }
        assert!(glyph('a').eq(glyph('A')));
        assert!(glyph('~').eq(glyph('?')));
    }

    #[test]
    fn glyphs_land_in_their_cells() {
        let image = rasterize(&["I".to_owned()], &[], 50.0, 1);
        assert_eq!(image.width, 2 * PAD + GRAPH_FRAMES as u32 * BAR_W);
        assert_eq!(image.height, 2 * PAD + CELL_H + GAP + GRAPH_H);
        // "I" starts ".###." one pixel below the cell's top.
        assert_eq!(image.at(PAD, PAD + 1), BACKGROUND);
        assert_eq!(image.at(PAD + 1, PAD + 1), TEXT);
        assert_eq!(image.at(PAD + 2, PAD + 2), TEXT);
        assert_eq!(image.at(PAD + 1, PAD + 2), BACKGROUND);
        // A long line widens the panel.
        let wide = rasterize(&["W".repeat(80)], &[], 50.0, 1);
        assert_eq!(wide.width, 2 * PAD + MAX_COLUMNS as u32 * CELL_W);
    }

    #[test]
    fn bars_show_the_newest_frames_at_the_right() {
        let image = rasterize(&[], &[100.0, 10.0], 50.0, 1);
        let bottom = image.height - PAD - 1;
        let right = image.width - PAD;
        // 10 ms of a 100 ms graph: 3 of 36 rows, green.
        assert_eq!(image.at(right - BAR_W, bottom), BAR);
        assert_eq!(image.at(right - BAR_W, bottom - 2), BAR);
        assert_eq!(image.at(right - BAR_W, bottom - 3), BACKGROUND);
        // The hitch before it fills the graph, red.
        assert_eq!(image.at(right - 2 * BAR_W, bottom), HITCH);
        assert_eq!(image.at(right - 2 * BAR_W, bottom + 1 - GRAPH_H), HITCH);
        // The guide at the hitch threshold, where no bar is.
        assert_eq!(image.at(PAD, bottom + 1 - GRAPH_H / 2), GUIDE);
    }

    #[test]
    fn scaling_repeats_pixels() {
        let one = rasterize(&["I".to_owned()], &[30.0], 50.0, 1);
        let two = rasterize(&["I".to_owned()], &[30.0], 50.0, 2);
        assert_eq!(
            (two.width, two.height, two.scale),
            (2 * one.width, 2 * one.height, 2)
        );
        for (x, y) in [
            (PAD + 1, PAD + 1),
            (PAD, PAD),
            (one.width - PAD - 2, one.height - PAD - 1),
        ] {
            let v = one.at(x, y);
            assert_eq!([two.at(2 * x, 2 * y), two.at(2 * x + 1, 2 * y + 1)], [v, v]);
        }
    }

    #[test]
    fn colors_in_every_format() {
        assert_eq!(
            PixelFormat::Rgba8.encode([1.0, 0.0, 0.5]),
            [255, 0, 128, 255]
        );
        assert_eq!(
            PixelFormat::Bgra8.encode([1.0, 0.0, 0.5]),
            [128, 0, 255, 255]
        );
        assert_eq!(
            PixelFormat::Rgb10a2.encode([1.0, 0.0, 0.0]),
            [0xff, 0x03, 0x00, 0xc0]
        );
        assert_eq!(
            PixelFormat::Rgba16f.encode([1.0, 0.5, 0.0]),
            [0x00, 0x3c, 0x00, 0x38, 0x00, 0x00, 0x00, 0x3c]
        );
        assert_eq!(PixelFormat::from_dxgi(87), Some(PixelFormat::Bgra8));
        assert_eq!(PixelFormat::from_dxgi(2), None);
        assert_eq!(half(2.0), 0x4000);
    }

    #[test]
    fn rows_go_at_their_pitch() {
        let image = Image {
            width: 2,
            height: 2,
            scale: 1,
            pixels: vec![TEXT, BACKGROUND, BACKGROUND, TEXT],
        };
        let mut out = vec![7u8; 256 + 8];
        write_rows(&image, PixelFormat::Rgba8, 256, &mut out);
        let text = PixelFormat::Rgba8.encode(PALETTE[TEXT as usize]);
        let back = PixelFormat::Rgba8.encode(PALETTE[BACKGROUND as usize]);
        assert_eq!(out[..4], text[..]);
        assert_eq!(out[4..8], back[..]);
        assert_eq!(out[8], 7, "padding untouched");
        assert_eq!(out[256..260], back[..]);
        assert_eq!(out[260..264], text[..]);
        // A buffer too small for a row stops there.
        let mut short = vec![0u8; 8];
        write_rows(&image, PixelFormat::Rgba8, 256, &mut short);
        assert_eq!(short[..4], text[..]);
    }

    #[test]
    fn the_variables_ask_for_the_panel() {
        assert!(wanted(Some("overlay"), None));
        assert!(wanted(None, Some("1")));
        assert!(wanted(Some(r"C:\perf"), Some("on")));
        assert!(!wanted(Some("1"), None));
        assert!(!wanted(None, Some("0")));
        assert!(!wanted(None, None));
    }
}
