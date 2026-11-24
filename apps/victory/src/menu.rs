//! What opening `victory.exe` from Explorer shows: a small menu, where the
//! command-line help used to flash in a window that closed at once (#527).
//!
//! Run a game (its executable dragged onto the window, or typed), run the
//! last one again, check what this PC has for VIctory (Windows 10's shader
//! model, AVX2, the GPU's shader model, mesh shaders and wave size, the
//! configured paths), or open the data folder (saves, `crashes.log`). A
//! game runs in a child `victory run`, so a crash or an exit of the title
//! leaves the menu up: the window stays open until Enter, with how the run
//! ended, so its last lines can be read.
//!
//! After a run that ended badly the menu names the known problems its
//! output and crash log show (`crate::doctor`), with where each is tracked.
//!
//! S sets the game folder in `victory.toml`, K shows the keyboard's
//! stand-in pad, and M switches the console model runs use (remembered).
//! Those three follow the launcher in #1330.
//!
//! The colours are a sunset gradient of VIctory's own (violet, pink,
//! peach), nothing of the title's. `NO_COLOR` turns them off, and so does a
//! console without virtual-terminal sequences.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Whether `victory` was started with no arguments in a console of its own
/// (a double-click), or `VICTORY_MENU=1` asks for the menu anyway.
pub fn wanted() -> bool {
    if std::env::args_os().len() != 1 {
        return false;
    }
    std::env::var("VICTORY_MENU").is_ok_and(|v| v == "1") || own_console()
}

/// The console has no other process attached: Explorer made it for us.
#[cfg(windows)]
fn own_console() -> bool {
    let mut ids = [0u32; 2];
    // SAFETY: the buffer holds `ids.len()` process ids.
    let n = unsafe {
        windows_sys::Win32::System::Console::GetConsoleProcessList(
            ids.as_mut_ptr(),
            ids.len() as u32,
        )
    };
    n == 1
}

#[cfg(not(windows))]
fn own_console() -> bool {
    false
}

/// A 24-bit colour.
type Rgb = (u8, u8, u8);

/// The sunset, top to bottom: dusk blue, violet, orchid, pink, peach, amber.
const SUNSET: [Rgb; 6] = [
    (86, 98, 196),
    (134, 96, 204),
    (198, 104, 184),
    (236, 128, 164),
    (246, 160, 124),
    (250, 190, 110),
];
const CREAM: Rgb = (255, 244, 230);
const DIM: Rgb = (176, 150, 190);
const GOOD: Rgb = (120, 220, 200);
const WARN: Rgb = (250, 190, 110);
const BAD: Rgb = (240, 100, 140);

/// Writes text in colour, or plain.
#[derive(Clone, Copy)]
struct Paint {
    color: bool,
}

impl Paint {
    fn rgb(self, text: &str, (r, g, b): Rgb) -> String {
        if self.color {
            format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    fn bold(self, text: &str, c: Rgb) -> String {
        if self.color {
            format!("\x1b[1m{}", self.rgb(text, c))
        } else {
            text.to_owned()
        }
    }

    /// Each character of `text` at its place along `stops`.
    fn gradient(self, text: &str, stops: &[Rgb]) -> String {
        if !self.color {
            return text.to_owned();
        }
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len().max(2) - 1;
        let mut out = String::from("\x1b[1m");
        for (i, ch) in chars.iter().enumerate() {
            let (r, g, b) = along(stops, i as f32 / n as f32);
            out.push_str(&format!("\x1b[38;2;{r};{g};{b}m{ch}"));
        }
        out.push_str("\x1b[0m");
        out
    }
}

/// The colour at `t` (0..=1) along evenly spaced `stops`.
fn along(stops: &[Rgb], t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
    let i = (t.floor() as usize).min(stops.len() - 2);
    let f = t - i as f32;
    let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * f).round() as u8;
    let (a, b) = (stops[i], stops[i + 1]);
    (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

/// Turns on virtual-terminal sequences; false where the console has none.
/// `NO_COLOR` turns colour off, `CLICOLOR_FORCE=1` on (output to a file).
#[cfg(windows)]
fn enable_color() -> bool {
    use windows_sys::Win32::System::Console::{
        ENABLE_VIRTUAL_TERMINAL_PROCESSING, GetConsoleMode, GetStdHandle, STD_OUTPUT_HANDLE,
        SetConsoleMode,
    };
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("CLICOLOR_FORCE").is_ok_and(|v| v == "1") {
        return true;
    }
    // SAFETY: plain console calls on our own output handle.
    unsafe {
        let out = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0;
        GetConsoleMode(out, &mut mode) != 0
            && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
fn enable_color() -> bool {
    std::env::var_os("NO_COLOR").is_none()
}

/// What a menu answer asks for.
#[derive(Debug, PartialEq, Eq)]
enum Choice {
    Run,
    Again,
    Check,
    Data,
    Folder,
    Keys,
    Model,
    Quit,
    Unknown,
}

fn choice(input: &str) -> Choice {
    // (A byte-order mark leads piped input from some shells.)
    match input
        .trim_start_matches('\u{feff}')
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" => Choice::Run,
        "2" | "r" => Choice::Again,
        "3" | "c" => Choice::Check,
        "4" | "d" => Choice::Data,
        "s" => Choice::Folder,
        "k" => Choice::Keys,
        "m" => Choice::Model,
        "q" | "0" | "x" | "exit" | "quit" => Choice::Quit,
        _ => Choice::Unknown,
    }
}

/// A path as a console gives it when a file is dragged onto the window or
/// pasted: spaces around it, quotes around it, PowerShell's `& '...'`.
fn clean_path(input: &str) -> PathBuf {
    let mut s = input.trim_start_matches('\u{feff}').trim();
    if let Some(rest) = s.strip_prefix('&') {
        s = rest.trim();
    }
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return PathBuf::from(&s[1..s.len() - 1]);
    }
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        // PowerShell doubles an apostrophe inside single quotes.
        return PathBuf::from(s[1..s.len() - 1].replace("''", "'"));
    }
    PathBuf::from(s)
}

/// Where the last game run from the menu is remembered.
fn last_file(data_dir: &Path) -> PathBuf {
    data_dir.join("menu-last-game.txt")
}

fn load_last(data_dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(last_file(data_dir)).ok()?;
    let path = PathBuf::from(text.trim());
    path.is_file().then_some(path)
}

fn save_last(data_dir: &Path, exe: &Path) {
    let _ = std::fs::create_dir_all(data_dir);
    let _ = std::fs::write(last_file(data_dir), exe.display().to_string());
}

/// The console model runs from the menu use (`victory run --console`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Model {
    SeriesX,
    SeriesS,
}

impl Model {
    fn arg(self) -> &'static str {
        match self {
            Model::SeriesX => "series-x",
            Model::SeriesS => "series-s",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Model::SeriesX => "Series X",
            Model::SeriesS => "Series S",
        }
    }

    fn other(self) -> Self {
        match self {
            Model::SeriesX => Model::SeriesS,
            Model::SeriesS => Model::SeriesX,
        }
    }
}

/// Where the menu's console model is remembered.
fn model_file(data_dir: &Path) -> PathBuf {
    data_dir.join("menu-console.txt")
}

fn load_model(data_dir: &Path) -> Model {
    match std::fs::read_to_string(model_file(data_dir)) {
        Ok(t) if t.trim() == Model::SeriesS.arg() => Model::SeriesS,
        _ => Model::SeriesX,
    }
}

fn save_model(data_dir: &Path, model: Model) {
    let _ = std::fs::create_dir_all(data_dir);
    let _ = std::fs::write(model_file(data_dir), model.arg());
}

/// The config file the tools read: `VICTORY_CONFIG`, else `victory.toml`
/// in the current folder (beside `victory.exe` when it was double-clicked).
fn config_path() -> PathBuf {
    std::env::var_os("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map_or_else(|| PathBuf::from("victory.toml"), PathBuf::from)
}

/// `text` (a config file, possibly empty) with `[paths].game_dir` set to
/// `dir`: the active `game_dir` line replaced, else one added under
/// `[paths]`, else a `[paths]` table appended. Comments and every other line
/// stay as they were.
fn set_game_dir(text: &str, dir: &Path) -> String {
    let dir = dir.display().to_string();
    // A literal string keeps backslashes as they are; it cannot hold an
    // apostrophe, which a basic string with escapes can.
    let value = if dir.contains('\'') {
        format!("\"{}\"", dir.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        format!("'{dir}'")
    };
    let line = format!("game_dir = {value}");
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let header = |l: &str| l.trim_start().starts_with('[');
    match lines.iter().position(|l| l.trim() == "[paths]") {
        Some(at) => {
            let end = lines[at + 1..]
                .iter()
                .position(|l| header(l))
                .map_or(lines.len(), |n| at + 1 + n);
            let active = lines[at + 1..end].iter().position(|l| {
                let l = l.trim_start();
                l.strip_prefix("game_dir")
                    .is_some_and(|rest| rest.trim_start().starts_with('='))
            });
            match active {
                Some(n) => lines[at + 1 + n] = line,
                None => lines.insert(at + 1, line),
            }
        }
        None => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push(String::new());
            }
            lines.push("[paths]".to_owned());
            lines.push(line);
        }
    }
    let mut out = lines.join(newline);
    out.push_str(newline);
    out
}

/// The keyboard's stand-in pad (`gameinput.rs`, as the README lists it).
fn keys_text(p: Paint) -> String {
    let rows = [
        ("W A S D / I J K L", "left / right stick"),
        ("arrows", "D-pad"),
        ("Enter, Backspace, X, Y", "A, B, X, Y"),
        ("Q / E", "left / right shoulder"),
        ("Z / C", "left / right trigger"),
        ("F / H", "left / right stick click"),
        ("Esc, Tab", "Menu, View"),
    ];
    let mut out = format!("\n  {}\n\n", p.bold("Keyboard controls", SUNSET[2]));
    for (keys, pad) in rows {
        out.push_str(&format!(
            "  {:<24}{}\n",
            p.rgb(keys, CREAM),
            p.rgb(pad, DIM)
        ));
    }
    out.push_str(&format!(
        "\n  {}\n",
        p.rgb(
            "With no XInput pad connected, the keyboard is a pad while the game's window has focus.",
            DIM
        )
    ));
    out
}

/// One line of the setup check.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Good,
    Warn,
    Bad,
}

/// What [`check`] found, as facts.
#[derive(Debug, Clone, Default)]
struct Pc {
    /// Windows build number, if read.
    windows_build: Option<u32>,
    avx2: bool,
    /// The GPU device made with shader model 6.6, and if so its mesh
    /// shader support and widest wave.
    sm66: Option<bool>,
    mesh_shaders: Option<bool>,
    max_lanes: Option<u32>,
    /// The Agility runtime folder beside this executable.
    agility_beside: bool,
    /// The default adapter's name and user-mode driver version, as DXGI
    /// gives them (also without shader model 6.6).
    gpu_name: Option<String>,
    driver: Option<String>,
}

/// The setup check's lines: status, what, and what to do.
fn verdicts(pc: &Pc) -> Vec<(Status, String, Option<String>)> {
    let mut out = Vec::new();
    match pc.windows_build {
        Some(b) if b >= 22000 => out.push((Status::Good, format!("Windows 11 (build {b})"), None)),
        Some(b) if pc.sm66 == Some(true) => {
            out.push((Status::Good, format!("Windows 10 (build {b})"), None))
        }
        Some(b) => out.push((
            Status::Warn,
            format!("Windows 10 (build {b})"),
            Some(if pc.agility_beside {
                "Its built-in D3D12 stops at shader model 6.5. A D3D12\\ folder is beside victory.exe; this build must also be made to load it (#175).".to_owned()
            } else {
                "Its built-in D3D12 stops at shader model 6.5. VIctory needs 6.6 from the D3D12 Agility SDK runtime (#84, #175).".to_owned()
            }),
        )),
        None => out.push((Status::Warn, "Windows version unknown".to_owned(), None)),
    }
    out.push(if pc.avx2 {
        (Status::Good, "CPU has AVX2".to_owned(), None)
    } else {
        (
            Status::Bad,
            "CPU has no AVX2".to_owned(),
            Some("The title needs it.".to_owned()),
        )
    });
    if let Some(name) = &pc.gpu_name {
        let what = match &pc.driver {
            Some(d) => format!("GPU: {name} (driver {d})"),
            None => format!("GPU: {name}"),
        };
        out.push((Status::Good, what, None));
    }
    match pc.sm66 {
        Some(true) => out.push((Status::Good, "GPU: shader model 6.6".to_owned(), None)),
        Some(false) => out.push((
            Status::Bad,
            "GPU: no shader model 6.6 device".to_owned(),
            Some("Translated shaders need it. On Windows 10, see above.".to_owned()),
        )),
        None => out.push((Status::Warn, "GPU: not checked".to_owned(), None)),
    }
    if pc.sm66 == Some(true) {
        match pc.mesh_shaders {
            Some(true) => out.push((Status::Good, "GPU: mesh shaders".to_owned(), None)),
            Some(false) => out.push((
                Status::Warn,
                "GPU: no mesh shaders".to_owned(),
                Some("Menus can run; the world's mesh draws are refused (#3).".to_owned()),
            )),
            None => {}
        }
        if let Some(lanes) = pc.max_lanes {
            out.push(if lanes >= 64 {
                (Status::Good, format!("GPU: waves up to {lanes} lanes"), None)
            } else {
                (
                    Status::Warn,
                    format!("GPU: waves of {lanes} lanes"),
                    Some("The title's programs are written for 64; some run in halves, some are refused (#578, #778).".to_owned()),
                )
            });
        }
    }
    out
}

/// The configured paths' lines, as [`verdicts`] gives them.
fn path_verdicts(cfg: &victory_core::Config) -> Vec<(Status, String, Option<String>)> {
    let mut out = Vec::new();
    match &cfg.game_dir {
        Some(d) if d.is_dir() => {
            out.push((Status::Good, format!("Game folder: {}", d.display()), None));
        }
        Some(d) => out.push((
            Status::Warn,
            format!("Game folder: {} (not there)", d.display()),
            Some(
                "Fix game_dir in victory.toml. Running from this menu does not need it.".to_owned(),
            ),
        )),
        None => out.push((
            Status::Warn,
            "Game folder: not set".to_owned(),
            Some(
                "S in the menu sets it (game_dir in victory.toml); 1 asks for the .exe.".to_owned(),
            ),
        )),
    }
    for (what, dir) in [
        ("Data folder", &cfg.data_dir),
        ("Cache folder", &cfg.cache_dir),
    ] {
        match victory_core::ensure_outside_repo(dir) {
            Ok(()) => out.push((Status::Good, format!("{what}: {}", dir.display()), None)),
            Err(e) => out.push((
                Status::Bad,
                format!("{what}: {}", dir.display()),
                Some(e.to_string()),
            )),
        }
    }
    out
}

/// The facts of this PC.
fn check() -> Pc {
    let mut pc = Pc {
        windows_build: windows_build(),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        agility_beside: std::env::current_exe()
            .ok()
            .and_then(|e| {
                e.parent()
                    .map(|d| d.join("D3D12").join("D3D12Core.dll").is_file())
            })
            .unwrap_or(false),
        ..Pc::default()
    };
    adapter(&mut pc);
    gpu(&mut pc);
    pc
}

/// A driver version as DXGI packs it (four 16-bit parts, high first).
fn driver_version(v: u64) -> String {
    format!(
        "{}.{}.{}.{}",
        v >> 48,
        (v >> 32) & 0xFFFF,
        (v >> 16) & 0xFFFF,
        v & 0xFFFF
    )
}

/// The default adapter's name and driver, from DXGI.
#[cfg(windows)]
fn adapter(pc: &mut Pc) {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIDevice, IDXGIFactory1};
    use windows::core::Interface;
    // SAFETY: plain DXGI queries; the descriptor is a local copy.
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else {
            return;
        };
        let Ok(adapter) = factory.EnumAdapters1(0) else {
            return;
        };
        if let Ok(desc) = adapter.GetDesc1() {
            let name = &desc.Description;
            let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            let name = String::from_utf16_lossy(&name[..len]).trim().to_owned();
            if !name.is_empty() {
                pc.gpu_name = Some(name);
            }
        }
        if let Ok(v) = adapter.CheckInterfaceSupport(&IDXGIDevice::IID) {
            pc.driver = Some(driver_version(v as u64));
        }
    }
}

#[cfg(not(windows))]
fn adapter(_: &mut Pc) {}

#[cfg(windows)]
fn windows_build() -> Option<u32> {
    use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let (key, name) = (
        wide("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion"),
        wide("CurrentBuildNumber"),
    );
    let mut buf = [0u16; 32];
    let mut len = std::mem::size_of_val(&buf) as u32;
    // SAFETY: NUL-terminated names and a buffer of `len` bytes.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            buf.as_mut_ptr().cast(),
            &mut len,
        )
    };
    if status != 0 {
        return None;
    }
    String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)])
        .trim()
        .parse()
        .ok()
}

#[cfg(not(windows))]
fn windows_build() -> Option<u32> {
    None
}

/// The default adapter's shader model, mesh shaders and wave size, asked
/// in this process: what a run from this executable would get.
#[cfg(windows)]
fn gpu(pc: &mut Pc) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_D3D12_OPTIONS7,
        D3D12_FEATURE_DATA_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
        D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
    };
    let Ok(host) = victory_shader::emit::host::Host::new(false) else {
        pc.sm66 = Some(false);
        return;
    };
    pc.sm66 = Some(true);
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: each query writes its own structure of the given size.
    unsafe {
        if host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut o7).cast(),
                size_of_val(&o7) as u32,
            )
            .is_ok()
        {
            pc.mesh_shaders = Some(o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED);
        }
        if host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut o1).cast(),
                size_of_val(&o1) as u32,
            )
            .is_ok()
        {
            pc.max_lanes = Some(o1.WaveLaneCountMax);
        }
    }
}

#[cfg(not(windows))]
fn gpu(_: &mut Pc) {}

/// The title line and the horizon under it.
fn banner(p: Paint) -> String {
    let title = p.gradient("V I C T O R Y", &SUNSET);
    let horizon = if p.color {
        p.gradient(&"\u{2500}".repeat(40), &SUNSET)
    } else {
        "-".repeat(40)
    };
    format!(
        "\n  {title}\n  {}\n  {horizon}\n",
        p.rgb("your own copy of the title, on this PC", DIM)
    )
}

fn menu_text(p: Paint, last: Option<&Path>, game_dir: Option<&Path>, model: Model) -> String {
    let key = |k: &str| p.bold(k, SUNSET[3]);
    let mut out = banner(p);
    out.push('\n');
    out.push_str(&format!("  {}  Run a game\n", key("1")));
    match last {
        Some(path) => out.push_str(&format!(
            "  {}  Run again   {}\n",
            key("2"),
            p.rgb(&path.display().to_string(), DIM)
        )),
        None => out.push_str(&p.rgb("  2  Run again   (nothing run yet)\n", DIM)),
    }
    out.push_str(&format!("  {}  Check this PC\n", key("3")));
    out.push_str(&format!(
        "  {}  Open the data folder   {}\n",
        key("4"),
        p.rgb("saves, crashes.log", DIM)
    ));
    let folder = game_dir.map_or_else(|| "(not set)".to_owned(), |d| d.display().to_string());
    out.push('\n');
    out.push_str(&format!(
        "  {}  Set the game folder   {}\n",
        key("S"),
        p.rgb(&folder, DIM)
    ));
    out.push_str(&format!("  {}  Keyboard controls\n", key("K")));
    out.push_str(&format!(
        "  {}  Console: {}   {}\n",
        key("M"),
        model.name(),
        p.rgb(&format!("(M switches to {})", model.other().name()), DIM)
    ));
    out.push_str(&format!("  {}  Quit\n\n", key("Q")));
    out.push_str(&format!("  {} ", p.rgb("Choose:", CREAM)));
    out
}

/// Hints start this far in, under the line they explain.
const HINT_INDENT: usize = 5;

/// `text` in lines of at most `width` characters, broken at spaces (a word
/// longer than that gets a line of its own).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(20);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let len = line.chars().count();
        if len > 0 && len + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The console window's width in columns (80 where it cannot be read).
#[cfg(windows)]
fn console_columns() -> usize {
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_OUTPUT_HANDLE,
    };
    // SAFETY: a plain query of our own output handle into a local structure.
    unsafe {
        let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
        if GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) == 0 {
            return 80;
        }
        usize::try_from(info.srWindow.Right - info.srWindow.Left + 1).unwrap_or(80)
    }
}

#[cfg(not(windows))]
fn console_columns() -> usize {
    80
}

fn check_text(p: Paint, pc: &Pc, cfg: Option<&victory_core::Config>, columns: usize) -> String {
    let mut out = String::new();
    let mut section = |title: &str, lines: Vec<(Status, String, Option<String>)>| {
        out.push_str(&format!("\n  {}\n\n", p.bold(title, SUNSET[2])));
        for (status, what, hint) in lines {
            let mark = match status {
                Status::Good => p.rgb("\u{2713}", GOOD),
                Status::Warn => p.rgb("!", WARN),
                Status::Bad => p.rgb("\u{2717}", BAD),
            };
            out.push_str(&format!("  {mark}  {what}\n"));
            if let Some(h) = hint {
                // One column spare: a line that fills the window wraps early
                // in some consoles.
                for line in wrap(&h, columns.saturating_sub(HINT_INDENT + 1)) {
                    out.push_str(&format!("{:HINT_INDENT$}{}\n", "", p.rgb(&line, DIM)));
                }
            }
        }
    };
    section("This PC", verdicts(pc));
    if let Some(cfg) = cfg {
        section("Paths", path_verdicts(cfg));
    }
    out
}

/// The check as plain text for a GitHub report: no colour, no wrapping,
/// the user's profile folder written as `%USERPROFILE%`.
fn report_text(check: &str, home: Option<&str>) -> String {
    let mut text = format!(
        "VIctory {}, Check this PC\n{check}",
        env!("CARGO_PKG_VERSION")
    );
    if let Some(home) = home.filter(|h| h.len() > 3) {
        text = text.replace(home, "%USERPROFILE%");
    }
    text
}

/// Puts `text` on the clipboard as Unicode text (CRLF line ends, as
/// Windows programs paste them).
#[cfg(windows)]
fn copy_to_clipboard(text: &str) -> bool {
    use windows_sys::Win32::Foundation::GlobalFree;
    use windows_sys::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData,
    };
    use windows_sys::Win32::System::Memory::{
        GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock,
    };
    /// `CF_UNICODETEXT`.
    const UNICODE_TEXT: u32 = 13;
    let wide: Vec<u16> = text
        .replace("\r\n", "\n")
        .replace('\n', "\r\n")
        .encode_utf16()
        .chain([0])
        .collect();
    // SAFETY: the clipboard is opened and closed here; the memory handed to
    // it is a movable block holding `wide`, which the clipboard owns once
    // SetClipboardData succeeds (and is freed here otherwise).
    unsafe {
        let mem = GlobalAlloc(GMEM_MOVEABLE, wide.len() * 2);
        if mem.is_null() {
            return false;
        }
        let at = GlobalLock(mem).cast::<u16>();
        if at.is_null() {
            GlobalFree(mem);
            return false;
        }
        std::ptr::copy_nonoverlapping(wide.as_ptr(), at, wide.len());
        GlobalUnlock(mem);
        if OpenClipboard(std::ptr::null_mut()) == 0 {
            GlobalFree(mem);
            return false;
        }
        EmptyClipboard();
        let set = !SetClipboardData(UNICODE_TEXT, mem).is_null();
        CloseClipboard();
        if !set {
            GlobalFree(mem);
        }
        set
    }
}

#[cfg(not(windows))]
fn copy_to_clipboard(_: &str) -> bool {
    false
}

fn read_line(input: &mut impl BufRead) -> Option<String> {
    let mut line = String::new();
    match input.read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

fn pause(p: Paint, input: &mut impl BufRead) {
    print!("\n  {} ", p.rgb("Press Enter to go back to the menu.", DIM));
    let _ = std::io::stdout().flush();
    let _ = read_line(input);
}

/// The command that runs `game` the way the menu does: `victory run <game>
/// --execute --console <model>` from this same executable, in this console.
fn run_command(victory: &Path, game: &Path, model: Model) -> std::process::Command {
    let mut c = std::process::Command::new(victory);
    c.arg("run")
        .arg(game)
        .arg("--execute")
        .arg("--console")
        .arg(model.arg());
    c
}

/// How a run ended, for the line after it: the exit code, and in hex where
/// it reads as an NTSTATUS (a crash such as 0xC0000005).
fn ended(code: Option<i32>) -> (Status, String) {
    match code {
        Some(0) => (Status::Good, "The run ended.".to_owned()),
        Some(c) if c < 0 => (
            Status::Bad,
            format!("The run stopped: exit code {:#010X}.", c as u32),
        ),
        Some(c) => (Status::Bad, format!("The run stopped: exit code {c}.")),
        None => (Status::Bad, "The run was stopped.".to_owned()),
    }
}

/// Runs `f` with Ctrl+C and Ctrl+Break left to the child process sharing
/// the console: the run stops, the menu stays.
#[cfg(windows)]
fn sparing_the_menu<T>(f: impl FnOnce() -> T) -> T {
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    unsafe extern "system" fn handled(ctrl: u32) -> windows_sys::core::BOOL {
        i32::from(ctrl == CTRL_C_EVENT || ctrl == CTRL_BREAK_EVENT)
    }
    // SAFETY: a handler of the expected signature, removed again below.
    unsafe { SetConsoleCtrlHandler(Some(handled), 1) };
    let out = f();
    // SAFETY: as above.
    unsafe { SetConsoleCtrlHandler(Some(handled), 0) };
    out
}

#[cfg(not(windows))]
fn sparing_the_menu<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// Runs `exe` in a child `victory run <exe> --execute`, then waits for
/// Enter whatever happened to it.
fn run_game(p: Paint, exe: &Path, data_dir: &Path, model: Model, input: &mut impl BufRead) {
    if !exe.is_file() {
        println!(
            "\n  {} {}",
            p.rgb("\u{2717}", BAD),
            p.rgb("No file there.", CREAM)
        );
        pause(p, input);
        return;
    }
    save_last(data_dir, exe);
    println!(
        "\n  {} {}   {}\n",
        p.rgb("\u{25b6}", SUNSET[4]),
        exe.display(),
        p.rgb(model.name(), DIM)
    );
    let before = crate::doctor::crash_logs(data_dir);
    let (status, said) = match std::env::current_exe() {
        Ok(me) => sparing_the_menu(|| run_captured(run_command(&me, exe, model))),
        Err(e) => (Err(e), String::new()),
    };
    let (mark, text) = match status {
        Ok(s) => {
            let (status, text) = ended(s.code());
            let mark = match status {
                Status::Good => p.rgb("\u{2713}", GOOD),
                _ => p.rgb("\u{2717}", BAD),
            };
            (mark, p.rgb(&text, CREAM))
        }
        Err(e) => (
            p.rgb("\u{2717}", BAD),
            p.rgb(&format!("The run could not start: {e}"), CREAM),
        ),
    };
    println!("\n  {mark} {text}");
    let crashed = crate::doctor::crash_text_since(data_dir, &before);
    let found = crate::doctor::diagnose(&format!("{said}\n{crashed}"));
    if !found.is_empty() {
        print!("{}", findings_text(p, &found, console_columns()));
    }
    pause(p, input);
}

/// Most of a run's error output the menu keeps for the doctor (its end,
/// where a run says why it stopped).
const KEPT_OUTPUT: usize = 4 << 20;

/// Runs `command` with its error output (the log) shown as it comes and
/// kept: how it ended, and the end of that output.
fn run_captured(
    mut command: std::process::Command,
) -> (std::io::Result<std::process::ExitStatus>, String) {
    use std::io::Read;
    command.stderr(std::process::Stdio::piped());
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return (Err(e), String::new()),
    };
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut kept = Vec::new();
        let Some(mut from) = stderr else {
            return kept;
        };
        let mut out = std::io::stderr();
        let mut buf = [0u8; 8192];
        while let Ok(n) = from.read(&mut buf) {
            if n == 0 {
                break;
            }
            // Shown as it comes, colours and all.
            let _ = out.write_all(&buf[..n]);
            kept.extend_from_slice(&buf[..n]);
            if kept.len() > KEPT_OUTPUT {
                kept.drain(..kept.len() - KEPT_OUTPUT);
            }
        }
        kept
    });
    let status = child.wait();
    let kept = reader.join().unwrap_or_default();
    (status, String::from_utf8_lossy(&kept).into_owned())
}

/// The doctor's findings after a run, as the menu shows them.
fn findings_text(p: Paint, found: &[&crate::doctor::Known], columns: usize) -> String {
    let mut out = format!("\n  {}\n", p.bold("What happened", SUNSET[2]));
    let width = columns.saturating_sub(HINT_INDENT + 1);
    for k in found {
        out.push_str(&format!(
            "\n  {}  {}\n",
            p.rgb("!", WARN),
            p.rgb(k.title, CREAM)
        ));
        for line in wrap(k.explain, width) {
            out.push_str(&format!("{:HINT_INDENT$}{}\n", "", p.rgb(&line, DIM)));
        }
        for &n in k.links {
            let at = crate::doctor::link(n);
            out.push_str(&format!("{:HINT_INDENT$}{}\n", "", p.rgb(&at, SUNSET[1])));
        }
    }
    out
}

/// Saves `dropped` (a folder, or a file in it) as the game folder in the
/// config file; what happened, as a line for the menu.
fn set_folder(dropped: &Path) -> (Status, String) {
    let dir = if dropped.is_file() {
        dropped.parent().map(Path::to_path_buf)
    } else {
        Some(dropped.to_path_buf())
    };
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return (Status::Bad, "No folder there.".to_owned());
    };
    if let Err(e) = victory_core::ensure_outside_repo(&dir) {
        return (Status::Bad, e.to_string());
    }
    let path = config_path();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    if let Err(e) = std::fs::write(&path, set_game_dir(&text, &dir)) {
        return (
            Status::Bad,
            format!("Could not write {}: {e}", path.display()),
        );
    }
    let mut said = format!("Game folder set in {}.", path.display());
    if std::env::var_os("VICTORY_GAME_DIR").is_some_and(|v| !v.is_empty()) {
        said.push_str(" VICTORY_GAME_DIR is set, and it wins over the file.");
    }
    (Status::Good, said)
}

/// Names the console window (it shows the executable's path otherwise).
#[cfg(windows)]
fn set_title(title: &str) {
    let wide: Vec<u16> = title.encode_utf16().chain([0]).collect();
    // SAFETY: a NUL-terminated string that outlives the call.
    unsafe { windows_sys::Win32::System::Console::SetConsoleTitleW(wide.as_ptr()) };
}

#[cfg(not(windows))]
fn set_title(_: &str) {}

pub fn run() -> Result<()> {
    let p = Paint {
        color: enable_color(),
    };
    set_title("VIctory");
    let data_dir = victory_core::Config::load()
        .map(|c| c.data_dir)
        .unwrap_or_else(|_| std::env::temp_dir().join("victory"));
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        let last = load_last(&data_dir);
        let model = load_model(&data_dir);
        let game_dir = victory_core::Config::load().ok().and_then(|c| c.game_dir);
        print!(
            "{}",
            menu_text(p, last.as_deref(), game_dir.as_deref(), model)
        );
        let _ = std::io::stdout().flush();
        let Some(line) = read_line(&mut input) else {
            return Ok(());
        };
        match choice(&line) {
            Choice::Run => {
                print!(
                    "\n  {}\n  {} ",
                    p.rgb("Drag the game's .exe onto this window, or type its path, then press Enter.", CREAM),
                    p.rgb(">", SUNSET[3])
                );
                let _ = std::io::stdout().flush();
                let Some(path) = read_line(&mut input) else {
                    return Ok(());
                };
                if !path.trim().is_empty() {
                    run_game(p, &clean_path(&path), &data_dir, model, &mut input);
                }
            }
            Choice::Again => match last {
                Some(exe) => run_game(p, &exe, &data_dir, model, &mut input),
                None => {
                    println!("\n  {}", p.rgb("Nothing run yet: choose 1 first.", DIM));
                    pause(p, &mut input);
                }
            },
            Choice::Check => {
                // Making the GPU device can take a moment.
                print!("\n  {}", p.rgb("Checking this PC...", DIM));
                let _ = std::io::stdout().flush();
                let pc = check();
                println!();
                let cfg = victory_core::Config::load().ok();
                print!("{}", check_text(p, &pc, cfg.as_ref(), console_columns()));
                print!(
                    "\n  {} ",
                    p.rgb(
                        "Enter: back to the menu.  C, then Enter: copy this for a GitHub report.",
                        DIM
                    )
                );
                let _ = std::io::stdout().flush();
                let Some(answer) = read_line(&mut input) else {
                    return Ok(());
                };
                if choice(&answer) == Choice::Check {
                    let plain = check_text(Paint { color: false }, &pc, cfg.as_ref(), usize::MAX);
                    let home = std::env::var("USERPROFILE").ok();
                    let text = report_text(&plain, home.as_deref());
                    let said = if copy_to_clipboard(&text) {
                        "Copied. Paste it into the report form on GitHub."
                    } else {
                        "Could not reach the clipboard."
                    };
                    println!("\n  {}", p.rgb(said, GOOD));
                    pause(p, &mut input);
                }
            }
            Choice::Data => {
                let _ = std::fs::create_dir_all(&data_dir);
                let opened = std::process::Command::new("explorer")
                    .arg(&data_dir)
                    .spawn()
                    .is_ok();
                let text = if opened {
                    format!("Opened {}", data_dir.display())
                } else {
                    format!("Could not open {}", data_dir.display())
                };
                println!("\n  {}", p.rgb(&text, DIM));
            }
            Choice::Folder => {
                print!(
                    "\n  {}\n  {} ",
                    p.rgb(
                        "Drag the game's folder (or its .exe) onto this window, or type its path, then press Enter.",
                        CREAM
                    ),
                    p.rgb(">", SUNSET[3])
                );
                let _ = std::io::stdout().flush();
                let Some(answer) = read_line(&mut input) else {
                    return Ok(());
                };
                if !answer.trim().is_empty() {
                    let said = set_folder(&clean_path(&answer));
                    let (mark, colour) = match said.0 {
                        Status::Good => ("\u{2713}", GOOD),
                        _ => ("\u{2717}", BAD),
                    };
                    println!("\n  {} {}", p.rgb(mark, colour), p.rgb(&said.1, CREAM));
                    pause(p, &mut input);
                }
            }
            Choice::Keys => {
                print!("{}", keys_text(p));
                pause(p, &mut input);
            }
            Choice::Model => save_model(&data_dir, model.other()),
            Choice::Quit => return Ok(()),
            Choice::Unknown if line.trim().is_empty() => {}
            Choice::Unknown => {
                println!(
                    "\n  {}",
                    p.rgb("Type 1 to 4, S, K or M and press Enter (Q to quit).", WARN)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragged_and_pasted_paths_come_clean() {
        let want = PathBuf::from(r"D:\Games\Some Title\title.exe");
        for input in [
            "D:\\Games\\Some Title\\title.exe\r\n",
            "  \"D:\\Games\\Some Title\\title.exe\"  \n",
            "& 'D:\\Games\\Some Title\\title.exe'\n",
            "'D:\\Games\\Some Title\\title.exe'",
        ] {
            assert_eq!(clean_path(input), want, "{input:?}");
        }
        // PowerShell's drag of a path with an apostrophe doubles it.
        let tom = PathBuf::from(r"D:\Games\Tom's Game\title.exe");
        assert_eq!(clean_path("& 'D:\\Games\\Tom''s Game\\title.exe'\r\n"), tom);
        // In double quotes (cmd), an apostrophe is kept as it is.
        assert_eq!(clean_path("\"D:\\Games\\Tom's Game\\title.exe\""), tom);
    }

    #[test]
    fn answers_map_to_choices() {
        assert_eq!(choice("1\n"), Choice::Run);
        assert_eq!(choice(" R "), Choice::Again);
        assert_eq!(choice("3"), Choice::Check);
        assert_eq!(choice("d"), Choice::Data);
        assert_eq!(choice("Q"), Choice::Quit);
        assert_eq!(choice("exit"), Choice::Quit);
        assert_eq!(choice("9"), Choice::Unknown);
        assert_eq!(choice("\u{feff}3\r\n"), Choice::Check);
    }

    #[test]
    fn the_last_game_is_remembered_while_it_exists() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_last(dir.path()), None);
        let exe = dir.path().join("title.exe");
        std::fs::write(&exe, b"").unwrap();
        save_last(dir.path(), &exe);
        assert_eq!(load_last(dir.path()), Some(exe.clone()));
        std::fs::remove_file(&exe).unwrap();
        assert_eq!(load_last(dir.path()), None, "a moved game is not offered");
    }

    #[test]
    fn the_check_says_what_matters_on_windows_10_and_nvidia() {
        let pc = Pc {
            windows_build: Some(19045),
            avx2: true,
            sm66: Some(true),
            mesh_shaders: Some(false),
            max_lanes: Some(32),
            agility_beside: false,
            gpu_name: Some("NVIDIA GeForce RTX 3060".to_owned()),
            driver: Some("32.0.15.6094".to_owned()),
        };
        let v = verdicts(&pc);
        let find = |what: &str| v.iter().find(|l| l.1.starts_with(what)).unwrap();
        assert_eq!(find("Windows 10").0, Status::Good);
        assert!(find("Windows 10").2.is_none());
        assert_eq!(find("CPU has AVX2").0, Status::Good);
        assert_eq!(
            find("GPU: NVIDIA").1,
            "GPU: NVIDIA GeForce RTX 3060 (driver 32.0.15.6094)"
        );
        assert_eq!(find("GPU: no mesh shaders").0, Status::Warn);
        assert_eq!(find("GPU: waves of 32").0, Status::Warn);
        let without_sm66 = verdicts(&Pc {
            sm66: Some(false),
            ..pc.clone()
        });
        assert_eq!(without_sm66[0].0, Status::Warn);
        assert!(without_sm66[0].2.as_deref().unwrap().contains("Agility"));
        // Windows 11, no SM 6.6: nothing about mesh or waves (not asked).
        let pc = Pc {
            windows_build: Some(26100),
            sm66: Some(false),
            ..pc
        };
        let v = verdicts(&pc);
        assert_eq!(v[0].0, Status::Good);
        assert!(v.iter().any(|l| l.0 == Status::Bad && l.1.contains("6.6")));
        assert!(
            !v.iter()
                .any(|l| l.1.contains("mesh") || l.1.contains("waves"))
        );
    }

    #[test]
    fn colours_follow_the_sunset_and_stay_out_when_off() {
        assert_eq!(along(&SUNSET, 0.0), SUNSET[0]);
        assert_eq!(along(&SUNSET, 1.0), SUNSET[5]);
        let mid = along(&[(0, 0, 0), (200, 100, 50)], 0.5);
        assert_eq!(mid, (100, 50, 25));
        let plain = Paint { color: false };
        assert_eq!(plain.gradient("VICTORY", &SUNSET), "VICTORY");
        assert!(!menu_text(plain, None, None, Model::SeriesX).contains('\x1b'));
        let colored = Paint { color: true };
        assert!(
            colored
                .gradient("VI", &SUNSET)
                .contains("\x1b[38;2;86;98;196mV")
        );
    }

    #[test]
    fn hints_wrap_to_the_window_under_their_line() {
        let hint = "Its built-in D3D12 stops at shader model 6.5. VIctory needs 6.6 from the D3D12 Agility SDK runtime (#84, #175).";
        let lines = wrap(hint, 40);
        assert!(lines.len() > 1);
        assert!(lines.iter().all(|l| l.chars().count() <= 40), "{lines:?}");
        assert_eq!(lines.join(" "), hint);
        // A word longer than the width keeps its own line, unbroken.
        let long = wrap(
            r"see C:\a\very\long\path\that\does\not\fit\at\all.exe now",
            20,
        );
        assert_eq!(long[0], "see");
        assert!(long[1].ends_with("all.exe"));
        assert_eq!(long[2], "now");
        // In the check, hint lines stay inside the window, indented.
        let pc = Pc {
            windows_build: Some(19045),
            sm66: Some(false),
            ..Pc::default()
        };
        let text = check_text(Paint { color: false }, &pc, None, 50);
        for line in text.lines() {
            assert!(line.chars().count() <= 50, "{line:?}");
        }
        let hint_lines: Vec<_> = text.lines().filter(|l| l.starts_with("     ")).collect();
        assert!(hint_lines.len() > 2, "{text}");
        assert!(text.contains("\n     Its built-in D3D12"));
    }

    #[test]
    fn a_game_runs_in_a_child_and_its_end_is_told() {
        let c = run_command(
            Path::new(r"C:\VIctory\victory.exe"),
            Path::new(r"D:\Games\title.exe"),
            Model::SeriesS,
        );
        assert_eq!(c.get_program(), r"C:\VIctory\victory.exe");
        let args: Vec<_> = c.get_args().collect();
        assert_eq!(
            args,
            [
                "run",
                r"D:\Games\title.exe",
                "--execute",
                "--console",
                "series-s"
            ]
        );
        assert_eq!(ended(Some(0)), (Status::Good, "The run ended.".to_owned()));
        assert_eq!(
            ended(Some(0xC000_0005_u32 as i32)).1,
            "The run stopped: exit code 0xC0000005."
        );
        assert_eq!(ended(Some(1)).1, "The run stopped: exit code 1.");
        assert_eq!(ended(None).0, Status::Bad);
    }

    #[test]
    fn the_check_copies_as_a_plain_report_without_the_home_folder() {
        assert_eq!(driver_version(0x0020_0000_0F00_17CE), "32.0.3840.6094");
        let pc = Pc {
            windows_build: Some(19045),
            sm66: Some(false),
            gpu_name: Some("NVIDIA GeForce GTX 1060 6GB".to_owned()),
            ..Pc::default()
        };
        let long = "x ".repeat(200);
        let plain = check_text(Paint { color: false }, &pc, None, usize::MAX);
        let report = report_text(
            &format!(
                "{plain}  Data folder: C:\\Users\\someone\\AppData\\Local\\victory\\data\n{long}"
            ),
            Some(r"C:\Users\someone"),
        );
        assert!(report.starts_with("VIctory "));
        assert!(report.contains("GPU: NVIDIA GeForce GTX 1060 6GB"));
        assert!(report.contains(r"%USERPROFILE%\AppData\Local\victory\data"));
        assert!(!report.contains("someone"));
        assert!(!report.contains('\x1b'));
        // Not wrapped: a hint stays on one line.
        assert!(
            report
                .lines()
                .any(|l| l.contains("shader model 6.5") && l.contains("Agility"))
        );
    }

    /// `[paths].game_dir` of a config file, as the tools parse it.
    fn game_dir_of(text: &str) -> Option<String> {
        let t: toml::Table = toml::from_str(text).expect("valid TOML");
        Some(t.get("paths")?.get("game_dir")?.as_str()?.to_owned())
    }

    #[test]
    fn the_game_folder_lands_in_the_config_and_leaves_the_rest() {
        let dir = Path::new(r"D:\Games\Title 1.0");
        // No file yet.
        let new = set_game_dir("", dir);
        assert_eq!(game_dir_of(&new).as_deref(), Some(r"D:\Games\Title 1.0"));
        // The example file: its comments and other keys stay, the active
        // line changes.
        let example = "# Copy to victory.toml\n[paths]\n# Your own dump.\ngame_dir = 'D:\\victory-data\\game'\n# cache_dir = 'x'\ntest_titles = []\n";
        let set = set_game_dir(example, dir);
        assert_eq!(game_dir_of(&set).as_deref(), Some(r"D:\Games\Title 1.0"));
        assert!(set.contains("# Your own dump.") && set.contains("# cache_dir = 'x'"));
        assert!(set.contains("test_titles = []"));
        assert_eq!(set.matches("game_dir").count(), 1);
        // A [paths] table without the key, before another table.
        let other = "[paths]\ncache_dir = 'C:\\c'\n\n[later]\nx = 1\n";
        let set = set_game_dir(other, dir);
        assert_eq!(game_dir_of(&set).as_deref(), Some(r"D:\Games\Title 1.0"));
        assert!(set.contains("[later]\nx = 1"));
        // No [paths] table at all.
        let set = set_game_dir("[later]\nx = 1", dir);
        assert_eq!(game_dir_of(&set).as_deref(), Some(r"D:\Games\Title 1.0"));
        // An apostrophe needs a basic string.
        let tom = Path::new(r"D:\Games\Tom's");
        assert_eq!(
            game_dir_of(&set_game_dir("", tom)).as_deref(),
            Some(r"D:\Games\Tom's")
        );
        // Windows line ends stay Windows line ends.
        assert!(set_game_dir("[paths]\r\ntest_titles = []\r\n", dir).contains("'\r\n"));
    }

    #[test]
    fn the_console_model_switches_and_is_remembered() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_model(dir.path()), Model::SeriesX);
        save_model(dir.path(), Model::SeriesX.other());
        assert_eq!(load_model(dir.path()), Model::SeriesS);
        assert_eq!(Model::SeriesS.other(), Model::SeriesX);
        let p = Paint { color: false };
        let text = menu_text(p, None, Some(Path::new(r"D:\G")), Model::SeriesS);
        assert!(text.contains("Console: Series S   (M switches to Series X)"));
        assert!(text.contains(r"Set the game folder   D:\G"));
        assert_eq!(choice("s"), Choice::Folder);
        assert_eq!(choice("K\r\n"), Choice::Keys);
        assert_eq!(choice("m"), Choice::Model);
    }

    #[test]
    fn the_keys_are_the_stand_in_pad() {
        let text = keys_text(Paint { color: false });
        for row in [
            "W A S D / I J K L",
            "Esc, Tab",
            "Z / C",
            "left / right trigger",
        ] {
            assert!(text.contains(row), "{row}");
        }
    }

    #[test]
    fn after_a_run_the_known_problems_are_named_with_their_links() {
        let found = crate::doctor::diagnose(
            "root signature flags c01\nVIctory: exception 0x80000003 at victory-image-x.dll+0x5149aec",
        );
        let text = findings_text(Paint { color: false }, &found, 60);
        assert!(text.contains("What happened"));
        assert!(text.contains("!  Windows 10's built-in D3D12 is too old"));
        assert!(text.contains("github.com/Marvisak/VIctory/issues/175"));
        assert!(text.contains("instance pool is full"));
        assert!(text.lines().all(|l| l.chars().count() <= 60), "{text}");
    }

    #[test]
    fn a_run_shows_and_keeps_its_error_output() {
        // A child that writes to its error output and fails (cmd's `echo`).
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "echo root signature flags c01 1>&2 & exit 3"]);
        let (status, said) = run_captured(c);
        assert_eq!(status.unwrap().code(), Some(3));
        assert!(said.contains("root signature flags c01"), "{said:?}");
    }

    #[test]
    fn the_menu_offers_run_again_only_with_a_last_game() {
        let p = Paint { color: false };
        assert!(menu_text(p, None, None, Model::SeriesX).contains("(nothing run yet)"));
        let text = menu_text(
            p,
            Some(Path::new(r"D:\Games\title.exe")),
            None,
            Model::SeriesX,
        );
        assert!(text.contains(r"Run again   D:\Games\title.exe"));
    }
}
