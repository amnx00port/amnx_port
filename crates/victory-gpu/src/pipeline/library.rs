//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library { library, _blob: Vec::new(), added: 0 })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(&library.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

/// Writes the library (when available and modified) and the early-pipeline
/// key list (whenever new keys were noted, even without a host library) if
/// `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let Some(dir) = DIR.get() else { return };
    let library = library();
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        let dirty = USED_CHANGED.load(Ordering::Relaxed)
            || library.is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).added > 0);
        if !dirty {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    save_to(dir, library);
}

fn save_to(dir: &std::path::Path, library: Option<&Mutex<Library>>) {
    let early_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
    let blob = library.and_then(|library| {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 {
            return None;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                Some(bytes)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                None
            }
        }
    });
    if blob.is_none() && !early_changed {
        return;
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        if early_changed {
            USED_CHANGED.store(true, Ordering::Relaxed);
        }
        tracing::warn!(error = %e, "pipeline cache directory not created");
        return;
    }
    let mut used: Vec<String> = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .cloned()
        .collect();
    used.sort_unstable();
    let early_saved = if early_changed {
        match write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            Ok(()) => true,
            Err(e) => {
                USED_CHANGED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "early pipeline list not saved");
                false
            }
        }
    } else {
        false
    };
    if let Some(blob) = blob {
        match write_atomically(&dir.join(LIBRARY_FILE), &blob) {
            Ok(()) => tracing::info!(
                bytes = blob.len(),
                early = used.len(),
                made = MADE.load(Ordering::Relaxed),
                loaded = LOADED.load(Ordering::Relaxed),
                started_early = EARLY.load(Ordering::Relaxed),
                "host pipeline library saved"
            ),
            Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
        }
    } else if early_saved {
        tracing::info!(
            early = used.len(),
            started_early = EARLY.load(Ordering::Relaxed),
            "early pipeline list saved"
        );
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn read_early_list(dir: &std::path::Path) -> HashSet<String> {
    let listed = std::fs::read_to_string(dir.join(EARLY_FILE)).unwrap_or_default();
    listed
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let set = DIR.get().map_or_else(HashSet::new, |d| read_early_list(d));
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn early_pipeline_keys_persist_and_reload_without_a_host_library() {
        let temp = tempfile::tempdir().unwrap();
        let cache_dir = temp.path().join("pso-cache");
        use_pipeline_cache(cache_dir.clone());
        USED_CHANGED.store(false, Ordering::Relaxed);

        // No dirty flag set: saving without a host library creates nothing.
        save_to(&cache_dir, None);
        assert!(!cache_dir.join(EARLY_FILE).exists());
        assert!(!cache_dir.join(LIBRARY_FILE).exists());

        // Record two distinct keys (plus a duplicate) and persist with `library == None`.
        note_used("gfx:alpha_key");
        note_used("cs:beta_key");
        note_used("gfx:alpha_key");
        assert!(USED_CHANGED.load(Ordering::Relaxed));

        save_to(&cache_dir, None);
        assert!(cache_dir.join(EARLY_FILE).exists());
        assert!(!cache_dir.join(LIBRARY_FILE).exists());

        let on_disk = std::fs::read_to_string(cache_dir.join(EARLY_FILE)).unwrap();
        let lines: Vec<&str> = on_disk.lines().collect();
        assert!(lines.is_sorted());
        assert_eq!(lines.iter().filter(|&&l| l == "gfx:alpha_key").count(), 1);
        assert!(lines.contains(&"cs:beta_key"));

        // Simulate next launch: clear in-memory `USED` and verify `wanted_early` reloads
        // the saved keys from `EARLY_FILE` when `DIR` points at `cache_dir`.
        if DIR.get() == Some(&cache_dir) {
            *USED.lock().unwrap_or_else(|e| e.into_inner()) = None;
            assert!(wanted_early("gfx:alpha_key"));
            assert!(wanted_early("cs:beta_key"));
            assert!(!wanted_early("gfx:missing_key"));

            // Also exercise `save(true)` end-to-end when only a new early key is added.
            note_used("gfx:gamma_key");
            save(true);
        } else {
            let reloaded = read_early_list(&cache_dir);
            assert!(reloaded.contains("gfx:alpha_key"));
            assert!(reloaded.contains("cs:beta_key"));
            assert!(!reloaded.contains("gfx:missing_key"));
            note_used("gfx:gamma_key");
            save_to(&cache_dir, None);
        }

        let reloaded = read_early_list(&cache_dir);
        assert!(reloaded.contains("cs:beta_key"));
        assert!(reloaded.contains("gfx:alpha_key"));
        assert!(reloaded.contains("gfx:gamma_key"));
        assert!(!reloaded.contains("gfx:missing_key"));
    }
}
