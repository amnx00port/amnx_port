//! `victory doctor`, and what the menu says after a run: the problems this
//! tracker already knows, found in a run's output or crash log, each with
//! what it means in plain words and where it is tracked.
//!
//! The table holds only signatures that VIctory, D3D12 or the title print
//! verbatim and that issues have traced to a cause. A run that matches none
//! gets no guess: the check's report (menu option 3) is the next step.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A known problem: any of `any` in a run's output (case-insensitive) is
/// its signature.
pub struct Known {
    pub any: &'static [&'static str],
    pub title: &'static str,
    pub explain: &'static str,
    /// Issues and pull requests in Marvisak/VIctory.
    pub links: &'static [u32],
}

/// Most specific first: a run can match several.
pub const KNOWN: &[Known] = &[
    Known {
        any: &["Unsupported bit-flag set", "root signature flags c01"],
        title: "Windows 10's built-in D3D12 is too old",
        explain: "VIctory needs shader model 6.6 and descriptor heaps that shaders index directly. Windows 10's built-in D3D12 runtime has neither, so every pipeline's root signature is refused and nothing draws. The D3D12 Agility SDK runtime beside victory.exe provides both.",
        links: &[84, 175, 969],
    },
    Known {
        any: &["+0x5149aec", "INSTANCE_LIST"],
        title: "The title stopped itself: its instance pool is full",
        explain: "The title asserts when its INSTANCE_LIST pool (45,000 entries) fills. In the reports this follows draws being dropped because some programs don't translate yet; fixing the culling shaders refused as 'a structured buffer of unknown stride' let runs go on.",
        links: &[698, 704],
    },
    Known {
        any: &["+0xb5a5f8"],
        title: "A known crash at Game Init",
        explain: "Reported on Windows 10 with main, where it went away with the Agility runtime. It has also been seen with that runtime present, so it can have other causes.",
        links: &[741, 998],
    },
    Known {
        any: &["has no mesh shaders"],
        title: "This GPU has no mesh shaders",
        explain: "The world's mesh pipelines are refused on GPUs without mesh shaders (for example GTX 10 series, RX 400 to 5000). A compute fallback that draws them is proposed.",
        links: &[3, 961],
    },
    Known {
        any: &["DXGI_ERROR_DEVICE_HUNG", "0x887A0006"],
        title: "The GPU hung (device removed)",
        explain: "Most such reports are on NVIDIA, whose waves are 32 lanes while the title's mesh programs are written for 64. Fixes are in progress; on other GPUs, a report with the DRED output helps.",
        links: &[778, 587, 578],
    },
    Known {
        any: &["wave lane count of 64"],
        title: "Some programs need 64-lane waves",
        explain: "This GPU runs 32-lane waves, and compute programs translated for 64 are refused until running them in 32-lane halves is supported.",
        links: &[233, 1107],
    },
    Known {
        any: &["E_OUTOFMEMORY", "0x8007000E"],
        title: "The GPU ran out of memory",
        explain: "The world needs more video memory than many 8 GB cards have. A lower texture quality switch is proposed; closing other programs helps a little.",
        links: &[737, 1086],
    },
    Known {
        any: &["os error 1455"],
        title: "Windows ran out of commit (the paging file)",
        explain: "Close other applications or make the Windows paging file larger, then try again (see Getting started in the README).",
        links: &[],
    },
    Known {
        any: &["malformed image"],
        title: "That file isn't a title's executable",
        explain: "VIctory runs the title's own .exe from your copy of the game. Pick that file.",
        links: &[],
    },
];

/// `text` without terminal colour sequences (`ESC [ ... letter`).
fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The known problems `text` (a run's output or crash log) shows, in table
/// order, each once.
pub fn diagnose(text: &str) -> Vec<&'static Known> {
    let text = plain(text).to_ascii_lowercase();
    KNOWN
        .iter()
        .filter(|k| k.any.iter().any(|s| text.contains(&s.to_ascii_lowercase())))
        .collect()
}

/// Where an issue or pull request is.
pub fn link(n: u32) -> String {
    format!("github.com/Marvisak/VIctory/issues/{n}")
}

/// The crash logs in the data folder (one per title), with their sizes.
pub fn crash_logs(data_dir: &Path) -> Vec<(PathBuf, u64)> {
    let Ok(dirs) = std::fs::read_dir(data_dir) else {
        return Vec::new();
    };
    dirs.flatten()
        .map(|d| d.path().join("crashes.log"))
        .filter_map(|p| Some((p.metadata().ok()?.len(), p)))
        .map(|(len, p)| (p, len))
        .collect()
}

/// What the crash logs gained since `before` (as [`crash_logs`] gave them).
pub fn crash_text_since(data_dir: &Path, before: &[(PathBuf, u64)]) -> String {
    let mut text = String::new();
    for (path, len) in crash_logs(data_dir) {
        let was = before.iter().find(|b| b.0 == path).map_or(0, |b| b.1);
        if len > was
            && let Ok(all) = std::fs::read(&path)
        {
            let from = usize::try_from(was).unwrap_or(0).min(all.len());
            text.push_str(&String::from_utf8_lossy(&all[from..]));
        }
    }
    text
}

/// `victory doctor`: the known problems in `files`, or in the crash logs
/// of the data folder when none is given.
pub fn run(files: Vec<PathBuf>) -> Result<()> {
    let files = if files.is_empty() {
        let cfg = victory_core::Config::load()?;
        let logs: Vec<PathBuf> = crash_logs(&cfg.data_dir).into_iter().map(|l| l.0).collect();
        anyhow::ensure!(
            !logs.is_empty(),
            "no crashes.log in {}; pass a saved log of the run instead",
            cfg.data_dir.display()
        );
        logs
    } else {
        files
    };
    let mut text = String::new();
    for f in &files {
        let bytes = std::fs::read(f).with_context(|| format!("reading {}", f.display()))?;
        text.push_str(&String::from_utf8_lossy(&bytes));
        text.push('\n');
    }
    let found = diagnose(&text);
    if found.is_empty() {
        println!(
            "No known problem found. To report it, `victory` (double-clicked) > 3 > C copies this PC's details."
        );
    }
    for k in found {
        println!("* {}\n  {}", k.title, k.explain);
        for &n in k.links {
            println!("  {}", link(n));
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn titles(text: &str) -> Vec<&'static str> {
        diagnose(text).iter().map(|k| k.title).collect()
    }

    #[test]
    fn each_known_problem_is_found_by_its_signature() {
        // Lines as VIctory, D3D12 and the title print them (from the issues).
        let cases = [
            (
                "WARN victory_gpu::pipeline::host: root signature: Unsupported bit-flag set (root signature flags c01).",
                "Windows 10's built-in D3D12 is too old",
            ),
            (
                "VIctory: unhandled exception 0x80000003 at victory-image-b99f876b74d4a77f.dll+0x5149aec",
                "The title stopped itself: its instance pool is full",
            ),
            (
                "VIctory: unhandled exception 0xc0000005 at victory-image-b99f876b74d4a77f.dll+0xb5a5f8 (read 0x68)",
                "A known crash at Game Init",
            ),
            (
                "WARN host pipeline refused why=the host GPU has no mesh shaders (or no shader model 6.6)",
                "This GPU has no mesh shaders",
            ),
            (
                "ERROR device removed reason=DXGI_ERROR_DEVICE_HUNG (0x887A0006)",
                "The GPU hung (device removed)",
            ),
            (
                "D3D12 ERROR: the shader requests a wave lane count of 64; the device maximum is 32",
                "Some programs need 64-lane waves",
            ),
            (
                "WARN victory_gpu::host: no shared buffer size=52428800 error=0x8007000E",
                "The GPU ran out of memory",
            ),
            (
                "VIctory: cannot commit 4096 MiB (os error 1455); increase the Windows paging file size",
                "Windows ran out of commit (the paging file)",
            ),
            (
                "Error: malformed image: Unknown MS-DOS file",
                "That file isn't a title's executable",
            ),
        ];
        for (line, title) in cases {
            assert_eq!(titles(line), [title], "{line}");
        }
    }

    #[test]
    fn colours_case_and_repeats_do_not_matter() {
        let coloured = "\x1b[33m WARN\x1b[0m \x1b[2mvictory_gpu\x1b[0m: \x1b[3merror\x1b[0m=Unsupported \x1b[1mbit-flag\x1b[0m set";
        // A colour sequence inside the signature is removed before matching.
        assert_eq!(titles(coloured), ["Windows 10's built-in D3D12 is too old"]);
        assert_eq!(titles("e_outofmemory"), ["The GPU ran out of memory"]);
        let twice = "+0x5149aec\n+0x5149aec\nINSTANCE_LIST 45000/45000";
        assert_eq!(titles(twice).len(), 1);
    }

    #[test]
    fn several_problems_come_in_table_order_and_clean_runs_get_none() {
        let both = "os error 1455 ... root signature flags c01";
        assert_eq!(
            titles(both),
            [
                "Windows 10's built-in D3D12 is too old",
                "Windows ran out of commit (the paging file)"
            ]
        );
        let clean = "INFO victory: title 7952987A\nINFO frame 120: drawn 340 / 340 draws";
        assert!(diagnose(clean).is_empty());
        for k in KNOWN {
            assert!(!k.any.is_empty() && !k.title.is_empty() && !k.explain.is_empty());
        }
    }

    #[test]
    fn only_what_crash_logs_gained_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let title = dir.path().join("7952987A");
        std::fs::create_dir_all(&title).unwrap();
        let log = title.join("crashes.log");
        std::fs::write(&log, "old: +0xb5a5f8\n").unwrap();
        let before = crash_logs(dir.path());
        assert_eq!(before.len(), 1);
        std::fs::write(&log, "old: +0xb5a5f8\nnew: +0x5149aec\n").unwrap();
        let since = crash_text_since(dir.path(), &before);
        assert_eq!(since, "new: +0x5149aec\n");
        assert_eq!(
            titles(&since),
            ["The title stopped itself: its instance pool is full"]
        );
        // A log that appears during the run is read whole.
        assert!(crash_text_since(dir.path(), &[]).contains("old:"));
    }
}
