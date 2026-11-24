//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).
//!
//! A container this build cannot translate is cached too, as its error
//! (marker, length, checksum, text: the length and checksum are what tell a
//! damaged entry from a real one): translation depends only on the container
//! and on what [`build`] hashes, so it would fail the same way again. Without
//! that, every run translated the same few hundred refused programs again
//! (seconds each for the large ones) while the title waited for them.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies this build: the BLAKE3 of our executable file and of
/// the translation settings it runs with (the loop limit, which a
/// diagnostics build can change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let exe = std::env::current_exe().and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        match exe {
            Ok(mut bytes) => {
                bytes.extend_from_slice(&limit.to_le_bytes());
                *blake3::hash(&bytes).as_bytes()
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let translate = |c: &[u8]| victory_shader::emit::translate(c).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => translate(container),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        Ok(match (self.translate)(&input[32..]) {
            Ok(shader) => encode(&shader),
            Err(error) => encode_failure(&error),
        })
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                // The cache is a convenience: a folder that cannot be
                // written must not cost the reason a container is refused
                // for. Answer from the translation itself.
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "shader cache entry not written; translating without it"
                );
                break;
            }
        };
        let entry = std::fs::read(&path)
            .ok()
            .and_then(|b| decode(&b).map(Ok).or_else(|| decode_failure(&b).map(Err)));
        if let Some(result) = entry {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            // The frame line counts shaders: a refusal, fresh or served
            // from a `VSF2` entry, is logged with ok=false but is neither
            // "translated" nor "from the cache".
            if result.is_ok() {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            tracing::debug!(path = %path.display(), ok = result.is_ok(), "shader {what}");
            return result;
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    let result = (converter.translate)(container);
    if result.is_ok() {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    result
}

const MAGIC: &[u8; 4] = b"VSH1";
/// A refusal's entry: this, then the text's length (little endian, 4 bytes),
/// then a checksum of the text (4 bytes), then the error's text (UTF-8). The
/// length tells a cut-short entry from a shorter error, and the checksum
/// tells a changed byte in an entry of the right length: either is damage,
/// and is translated again rather than served with a wrong reason.
const FAILED: &[u8; 4] = b"VSF2";
/// The bytes the payload's length and checksum take, after [`FAILED`].
const LENGTH: usize = 4;
const CHECK: usize = 4;

/// The first four bytes of the text's BLAKE3: enough to call a changed byte
/// damage (this is not against an adversary, only against a torn write).
fn checksum(text: &[u8]) -> [u8; 4] {
    let h = blake3::hash(text);
    h.as_bytes()[..CHECK].try_into().unwrap()
}

fn encode_failure(error: &str) -> Vec<u8> {
    let mut entry = Vec::with_capacity(FAILED.len() + LENGTH + CHECK + error.len());
    entry.extend_from_slice(FAILED);
    entry.extend_from_slice(&(error.len() as u32).to_le_bytes());
    entry.extend_from_slice(&checksum(error.as_bytes()));
    entry.extend_from_slice(error.as_bytes());
    entry
}

fn decode_failure(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(FAILED.as_slice())?;
    let (head, text) = rest.split_at_checked(LENGTH + CHECK)?;
    let length = u32::from_le_bytes(head[..LENGTH].try_into().ok()?) as usize;
    if text.len() != length || checksum(text) != head[LENGTH..] {
        return None;
    }
    String::from_utf8(text.to_vec()).ok()
}
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that call `cached`: the hit/miss counters are
    /// process-wide, so two of them at once would race on the deltas.
    static CACHE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let _serial = CACHE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn failures_come_back_from_the_cache_too() {
        let _serial = CACHE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a container that does not translate".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data is not supported".to_owned())
        };
        let error = Err("a scalar memory read as data is not supported".to_owned());
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build tries again, and may translate it.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[8; 32], container, ok), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // A damaged failure entry is a miss, tried again.
        for e in walk(dir.path()) {
            if std::fs::read(&e).unwrap().starts_with(FAILED) {
                std::fs::write(e, [FAILED.as_slice(), &[0xff, 0xfe]].concat()).unwrap();
            }
        }
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn failure_entries_round_trip_and_are_no_shader() {
        let bytes = encode_failure("an address not from a root parameter");
        assert_eq!(
            decode_failure(&bytes).as_deref(),
            Some("an address not from a root parameter")
        );
        assert_eq!(decode(&bytes), None);
        assert_eq!(decode_failure(&encode(&shader())), None);
    }

    #[test]
    fn a_damaged_failure_entry_is_no_entry() {
        let full = encode_failure("a scalar memory read as data");
        assert_eq!(
            decode_failure(&full).as_deref(),
            Some("a scalar memory read as data")
        );
        // Truncated after the marker, and after a valid UTF-8 prefix: both
        // are damaged, not an entry with a shorter error, so both are retried.
        for n in [FAILED.len(), FAILED.len() + 4, full.len() - 1] {
            assert_eq!(decode_failure(&full[..n]), None, "cut to {n} bytes");
        }
        // A byte changed in the text with the length left intact: damage,
        // not an entry with a different (still valid) reason.
        let mut flipped = full.clone();
        let at = FAILED.len() + LENGTH + CHECK + 1;
        assert!(flipped[at].is_ascii());
        flipped[at] = if flipped[at] == b'a' { b'b' } else { b'a' };
        assert_eq!(decode_failure(&flipped), None, "same length, changed byte");
        // Text that is not UTF-8, and a marker followed by text that is not
        // one (a shader entry's magic, say) are no entry either.
        assert_eq!(
            decode_failure(&[FAILED.as_slice(), &[0xff, 0xfe]].concat()),
            None
        );
        assert_eq!(decode_failure(b"VSH1a scalar memory read as data"), None);
        // An entry of the previous layout (no checksum) is no entry: it is
        // translated again rather than read with its text's first bytes as
        // a checksum.
        let old = [b"VSF1".as_slice(), &full[FAILED.len()..]].concat();
        assert_eq!(decode_failure(&old), None);
    }

    /// A cache whose entry cannot be written (here: the entry's path is a
    /// directory, so the rename onto it fails) must not turn a refusal into
    /// an OS error: the reason is the answer, the cache is only a shortcut.
    #[test]
    fn a_cache_that_cannot_be_written_still_says_why() {
        use victory_pipeline::CacheKey;
        let _serial = CACHE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a container that does not translate".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data".to_owned())
        };
        // Block the entry's own path with a directory before the first call.
        let input = [&build[..], container].concat();
        let key = CacheKey::new(
            &Translate {
                translate: fail,
                ran: AtomicBool::new(false),
            },
            &input,
        );
        std::fs::create_dir_all(cache.path_for(&key)).unwrap();
        assert_eq!(
            cached(&cache, &build, container, fail),
            Err("a scalar memory read as data".to_owned()),
            "the reason, not the OS error"
        );
        assert!(runs.load(Ordering::Relaxed) >= 1);
    }

    /// The frame line reports "shaders from the cache and translated so
    /// far": a refusal, whether fresh or served from a `VSF2` entry, is
    /// neither, and must not move either counter.
    #[test]
    fn refusals_are_not_counted_as_translated() {
        let _serial = CACHE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [9u8; 32];
        let container = b"yet another container that does not translate".as_slice();
        let fail = |_: &[u8]| Err::<HostShader, _>("a scalar memory read as data".to_owned());
        let error = Err("a scalar memory read as data".to_owned());

        let (h0, m0) = counts();
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error, "from disk");
        let (h1, m1) = counts();
        assert_eq!(
            (h1 - h0, m1 - m0),
            (0, 0),
            "a refusal, fresh or cached, is neither a hit nor a translation"
        );

        // A shader that does translate (another build, so the cached
        // refusal is not in the way) is counted, once fresh, once cached.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h2, m2) = counts();
        assert_eq!((h2 - h1, m2 - m1), (0, 1), "translated once");
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h3, m3) = counts();
        assert_eq!((h3 - h2, m3 - m2), (1, 0), "then from the cache");
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
