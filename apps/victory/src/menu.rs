//! Windows interactive launcher menu and PC readiness check (#910).
//!
//! When a user double-clicks `victory.exe` on Windows (no command-line
//! arguments), this module keeps the console open, inspects the host PC's
//! DirectX 12 / Shader Model 6.6 / Mesh Shader readiness, lists any titles
//! found in the configured directories, and lets the user launch a game
//! directly (or drag-and-drop any `.exe` onto the console window).

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::{Console, Unimplemented, inspect, run, show_config, translate};

/// Prints a full PC readiness & configuration report (`victory doctor`).
pub(crate) fn doctor() -> Result<()> {
    println!("VIctory {} — Windows PC Readiness Check", env!("CARGO_PKG_VERSION"));
    println!("====================================================");
    println!(
        "Platform    : {} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    #[cfg(windows)]
    match victory_shader::emit::host::probe_host() {
        Ok(gpu) => {
            let kind = if gpu.software { "software (WARP)" } else { "hardware" };
            let sm_hi = (gpu.highest_shader_model >> 4) & 0xF;
            let sm_lo = gpu.highest_shader_model & 0xF;
            println!("GPU         : {} [{kind}]", gpu.name);
            println!("VRAM        : {} MiB", gpu.vram_mb);
            println!(
                "Shader Model: {}.{}  [{}]",
                sm_hi,
                sm_lo,
                if gpu.shader_model_6_6 { "OK — SM 6.6+" } else { "NEEDS SM 6.6" }
            );
            println!(
                "Mesh Shaders: {}",
                if gpu.mesh_shaders { "OK (supported)" } else { "NOT SUPPORTED (mesh titles will skip)" }
            );
            println!(
                "Wave Ops    : {} (lanes {}..={})",
                if gpu.wave_ops { "OK" } else { "missing" },
                gpu.wave_lanes_min,
                gpu.wave_lanes_max
            );
            println!(
                "Binding Tier: Tier {}  [{}]",
                gpu.resource_binding_tier,
                if gpu.resource_binding_tier >= 3 { "OK" } else { "LOW" }
            );
            if gpu.software {
                println!("Note        : Running on Microsoft WARP software renderer (no hardware GPU detected).");
            }
        }
        Err(e) => {
            println!("GPU         : UNAVAILABLE ({e})");
            println!("              Install current DirectX 12 GPU drivers on Windows 10/11 x64.");
        }
    }
    #[cfg(not(windows))]
    {
        println!("GPU         : Host D3D12 execution requires Windows 10/11 x64.");
    }
    println!();
    show_config()?;

    let cfg = victory_core::Config::load()?;
    let games = discover_executables(&cfg);
    println!();
    if games.is_empty() {
        println!("Titles found: none (set `game_dir` in victory.toml or pass an .exe path)");
    } else {
        println!("Titles found ({}):", games.len());
        for (i, p) in games.iter().enumerate() {
            println!("  [{}] {}", i + 1, p.display());
        }
    }
    Ok(())
}

/// Interactive launcher shown when `victory.exe` is started with no arguments.
pub(crate) fn interactive() -> Result<()> {
    let mut console = Console::SeriesX;
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        println!();
        println!("====================================================");
        println!("  VIctory {} — Xbox Series Native Runtime", env!("CARGO_PKG_VERSION"));
        println!("====================================================");
        print_compact_readiness();

        let cfg = victory_core::Config::load().ok();
        let games = cfg
            .as_ref()
            .map(discover_executables)
            .unwrap_or_default();

        println!();
        println!(
            "Console profile: {}",
            match console {
                Console::SeriesX => "Xbox Series X",
                Console::SeriesS => "Xbox Series S",
            }
        );
        if !games.is_empty() {
            println!();
            println!("Discovered titles:");
            for (i, path) in games.iter().enumerate() {
                let label = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("title.exe");
                println!("  [{}] Launch {label} ({})", i + 1, path.display());
            }
        }
        println!();
        println!("Actions:");
        println!("  [L] Launch a title .exe (type or drag-and-drop path)");
        println!("  [S] Setup game folder (save victory.toml)");
        println!("  [K] Controls & hotkeys guide (Controller / Keyboard + Mouse)");
        println!("  [M] Toggle console model (Series X / Series S)");
        println!("  [I] Inspect a PE executable");
        println!("  [T] Translate shaders (dry-run check)");
        println!("  [D] Full PC readiness & configuration report");
        println!("  [Q] Quit");
        print!("\nChoice> ");
        let _ = io::stdout().flush();

        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            // EOF (non-interactive pipe): exit cleanly without spinning.
            return Ok(());
        }
        let choice = line.trim();
        if choice.is_empty() {
            continue;
        }
        if let Ok(idx) = choice.parse::<usize>()
            && idx >= 1
            && idx <= games.len()
        {
            let exe = &games[idx - 1];
            println!("Launching {}...", exe.display());
            if let Err(e) = run::run(exe, Unimplemented::Trap, false, true, console) {
                eprintln!("Error: {e:#}");
                pause(&stdin);
            }
            continue;
        }

        match choice.to_ascii_lowercase().as_str() {
            "l" => {
                if let Some(exe) =
                    prompt_path(&stdin, "Path to title .exe (or drag-and-drop here): ")?
                    && !exe.as_os_str().is_empty()
                    && let Err(e) = run::run(&exe, Unimplemented::Trap, false, true, console)
                {
                    eprintln!("Error: {e:#}");
                    pause(&stdin);
                }
            }
            "s" => {
                if let Err(e) = setup_config(&stdin) {
                    eprintln!("Error: {e:#}");
                }
                pause(&stdin);
            }
            "k" => {
                print_controls();
                pause(&stdin);
            }
            "m" => {
                console = match console {
                    Console::SeriesX => Console::SeriesS,
                    Console::SeriesS => Console::SeriesX,
                };
            }
            "i" => {
                if let Some(exe) = prompt_path(&stdin, "Path to PE file to inspect: ")?
                    && !exe.as_os_str().is_empty()
                {
                    if let Err(e) = inspect::run(&exe, false, false) {
                        eprintln!("Error: {e:#}");
                    }
                    pause(&stdin);
                }
            }
            "t" => {
                if let Some(path) = prompt_path(&stdin, "Path to shaders/game folder (blank = configured): ")? {
                    let paths = if path.as_os_str().is_empty() {
                        Vec::new()
                    } else {
                        vec![path]
                    };
                    if let Err(e) = translate::run(paths, false, false, false, None) {
                        eprintln!("Error: {e:#}");
                    }
                    pause(&stdin);
                }
            }
            "d" | "c" => {
                if let Err(e) = doctor() {
                    eprintln!("Error: {e:#}");
                }
                pause(&stdin);
            }
            "q" | "exit" => return Ok(()),
            other => {
                // If the user dragged-and-dropped an .exe directly at the main menu prompt!
                let dropped = clean_path_input(other);
                if dropped.is_file()
                    && dropped
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
                {
                    if let Err(e) = run::run(&dropped, Unimplemented::Trap, false, true, console) {
                        eprintln!("Error: {e:#}");
                        pause(&stdin);
                    }
                } else {
                    println!("Unknown choice '{other}'. Enter a menu key or Q to quit.");
                }
            }
        }
    }
}

fn setup_config(stdin: &io::Stdin) -> Result<()> {
    println!();
    println!("VIctory Game Folder Setup");
    println!("----------------------------------------------------");
    println!("Enter the folder containing your legally owned Xbox Series title dump.");
    println!("Tip: you can drag-and-drop the folder directly into this window.");
    let Some(dir) = prompt_path(stdin, "Game folder path (blank to cancel): ")? else {
        return Ok(());
    };
    if dir.as_os_str().is_empty() {
        println!("Cancelled.");
        return Ok(());
    }
    let dir = if dir.is_file() {
        dir.parent().map(Path::to_path_buf).unwrap_or(dir)
    } else {
        dir
    };
    let target = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("victory.toml")))
        .filter(|p| p.parent().is_some_and(|d| d.exists()))
        .unwrap_or_else(|| PathBuf::from("victory.toml"));
    let toml = format_config_toml(&dir);
    std::fs::write(&target, toml)?;
    println!("Saved configuration to {}", target.display());
    println!("Titles inside {} will now appear automatically in the launcher.", dir.display());
    Ok(())
}

/// Renders a minimal `victory.toml` pointing `[paths].game_dir` at `dir`.
pub(crate) fn format_config_toml(dir: &Path) -> String {
    let raw = dir.to_string_lossy();
    let escaped = raw.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "# Generated by VIctory launcher setup.\n\
         [paths]\n\
         game_dir = \"{escaped}\"\n"
    )
}

fn print_controls() {
    println!();
    println!("VIctory Controls & Window Hotkeys");
    println!("====================================================");
    println!("Window / Display:");
    println!("  F11 or Alt+Enter       Toggle borderless fullscreen");
    println!("  Alt+F4 / Close [X]     Clean title shutdown");
    println!("  Title bar              Live FPS, frame time, and draw count");
    println!();
    println!("Xbox Controller (recommended — plug-and-play XInput + Impulse Triggers):");
    println!("  Up to 4 controllers supported automatically; switches off keyboard");
    println!("  fallback on first real controller press.");
    println!();
    println!("PC Keyboard + Mouse Fallback (active when no gamepad is pressed):");
    println!("  W / A / S / D          Left Stick (move)");
    println!("  I / J / K / L          Right Stick (camera)");
    println!("  Right Click + Mouse    Aim (LT) + Mouse Camera Look");
    println!("  F9                     Toggle continuous Mouse Camera Look");
    println!("  Left Click / C         Right Trigger (RT / fire / accelerate)");
    println!("  Right Click / Z        Left Trigger (LT / aim / brake)");
    println!("  Middle Click / H       Right Stick Click (R3)");
    println!("  Space / Enter          A");
    println!("  Left Ctrl / Backspace  B");
    println!("  R / X                  X");
    println!("  Y                      Y");
    println!("  Q / E                  LB / RB");
    println!("  Left Shift / F         Left Stick Click (L3 / sprint)");
    println!("  Arrow Keys             D-Pad");
    println!("  Esc / Tab              Menu / View");
    println!();
    println!("GPU Selection (multi-GPU / laptops):");
    println!("  Automatic high-performance discrete GPU selection is enabled.");
    println!("  Override with env var: VICTORY_ADAPTER=0 (index) or VICTORY_ADAPTER=RTX");
}

fn print_compact_readiness() {
    #[cfg(windows)]
    match victory_shader::emit::host::probe_host() {
        Ok(gpu) => {
            let sm = if gpu.shader_model_6_6 { "SM6.6 OK" } else { "NO SM6.6" };
            let mesh = if gpu.mesh_shaders { "Mesh OK" } else { "No Mesh" };
            let wave = if gpu.wave_lanes_max > 0 {
                format!("Wave {}-{}", gpu.wave_lanes_min, gpu.wave_lanes_max)
            } else {
                "Wave ?".to_owned()
            };
            println!(
                "Host GPU: {} ({} MiB) | {sm} | {mesh} | {wave}",
                gpu.name, gpu.vram_mb
            );
        }
        Err(e) => println!("Host GPU: unavailable ({e})"),
    }
    #[cfg(not(windows))]
    println!("Host GPU: Windows x64 required for execution");
}

fn prompt_path(stdin: &io::Stdin, prompt: &str) -> Result<Option<PathBuf>> {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    if stdin.read_line(&mut buf)? == 0 {
        return Ok(None);
    }
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Ok(Some(PathBuf::new()));
    }
    Ok(Some(clean_path_input(trimmed)))
}

fn pause(stdin: &io::Stdin) {
    print!("\nPress Enter to continue...");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    let _ = stdin.read_line(&mut buf);
}

/// Cleans surrounding whitespace and Explorer drag-and-drop quotes from a path.
pub(crate) fn clean_path_input(raw: &str) -> PathBuf {
    let s = raw.trim();
    let s = s
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|t| t.strip_suffix('\'')))
        .unwrap_or(s);
    PathBuf::from(s.trim())
}

/// Discovers candidate title `.exe` files under configured `game_dir` and
/// `test_titles` directories (up to 2 levels deep).
pub(crate) fn discover_executables(cfg: &victory_core::Config) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for root in cfg.game_dir.iter().chain(&cfg.test_titles) {
        if root.is_file() && is_title_exe(root) {
            found.push(root.clone());
            continue;
        }
        if !root.is_dir() {
            continue;
        }
        for entry in walkdir::WalkDir::new(root)
            .max_depth(2)
            .into_iter()
            .filter_map(Result::ok)
        {
            let p = entry.path();
            if p.is_file() && is_title_exe(p) {
                found.push(p.to_path_buf());
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

fn is_title_exe(path: &Path) -> bool {
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
    {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    !name.starts_with("unins") && !name.contains("crash") && !name.contains("redist")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_path_strips_explorer_quotes_and_spaces() {
        assert_eq!(
            clean_path_input("  \"C:\\Games\\My Title\\game.exe\" \r\n"),
            PathBuf::from("C:\\Games\\My Title\\game.exe")
        );
        assert_eq!(
            clean_path_input("'D:/Xbox/title.exe'"),
            PathBuf::from("D:/Xbox/title.exe")
        );
    }

    #[test]
    fn is_title_exe_filters_non_game_binaries() {
        assert!(is_title_exe(Path::new("C:\\Games\\Title.EXE")));
        assert!(!is_title_exe(Path::new("C:\\Games\\unins000.exe")));
        assert!(!is_title_exe(Path::new("C:\\Games\\CrashHandler.exe")));
        assert!(!is_title_exe(Path::new("C:\\Games\\readme.txt")));
    }

    #[test]
    fn doctor_runs_cleanly_on_host() {
        assert!(doctor().is_ok());
    }

    #[test]
    fn format_config_toml_escapes_windows_backslashes() {
        let toml = format_config_toml(Path::new(r"D:\Xbox Games\Title"));
        assert!(toml.contains("[paths]"));
        assert!(toml.contains(r#"game_dir = "D:\\Xbox Games\\Title""#));
    }
}
