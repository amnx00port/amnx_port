//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again, and a
//! cache that cannot be written is skipped (it never fails a translation).
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// The build fingerprint in front of every cache input, from [`build`].
const BUILD_LEN: usize = 32;

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
fn build() -> &'static [u8; BUILD_LEN] {
    static BUILD: OnceLock<[u8; BUILD_LEN]> = OnceLock::new();
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
/// encoded translation out. `translated` keeps what it produced, so a
/// store that cannot publish the entry costs no second translation.
struct Translate<F> {
    translate: F,
    translated: OnceLock<HostShader>,
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
        let Some((_, container)) = input.split_at_checked(BUILD_LEN) else {
            anyhow::bail!("shader cache input lacks the build fingerprint");
        };
        let shader = (self.translate)(container).map_err(anyhow::Error::msg)?;
        let _ = self.translated.set(shader.clone());
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; BUILD_LEN],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        translated: OnceLock::new(),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            // The cache never fails a translation: a store it cannot publish
            // still holds the translation (a miss), and a translation that
            // failed inside it fails here.
            Err(e) => {
                let message = format!("{e:#}");
                tracing::debug!(error = %message, "shader cache unusable");
                return match converter.translated.get() {
                    Some(shader) => {
                        MISSES.fetch_add(1, Ordering::Relaxed);
                        Ok(shader.clone())
                    }
                    None => Err(message),
                };
            }
        };
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.translated.get().is_some() {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation that already happened (the store had no usable entry
    // afterwards) is used as is.
    if let Some(shader) = converter.translated.get() {
        return Ok(shader.clone());
    }
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
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

/// The tag of [`RootType::Other`]: a tag cannot hold every `u32`, so its
/// raw value follows it as its own word.
const OTHER_ROOT: u32 = 6;

fn root_tag(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(_) => OTHER_ROOT,
    }
}

/// Little-endian words and length-prefixed lists; a root type is its tag,
/// then the raw value for [`RootType::Other`].
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_tag(u.root)] {
            word(&mut out, v);
        }
        if let RootType::Other(raw) = u.root {
            word(&mut out, raw);
        }
        word(&mut out, u.dword);
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
        let index = r.word()?;
        let slot = u8::try_from(r.word()?).ok()?;
        let root = match r.word()? {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            2 => RootType::Table,
            3 => RootType::ImmediateSampler,
            4 => RootType::ResourceHeap,
            5 => RootType::SamplerHeap,
            OTHER_ROOT => RootType::Other(r.word()?),
            _ => return None,
        };
        user_data.push(UserData {
            index,
            slot,
            root,
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
    use victory_pipeline::CacheKey;

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

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
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

    /// A shader with every field at an unusual value.
    fn rich_shader() -> HostShader {
        HostShader {
            stage: HwStage::Gs,
            dxil: (0u8..=255).collect(),
            user_data: vec![
                UserData {
                    index: 0,
                    slot: 0,
                    root: RootType::Constants,
                    dword: 1,
                },
                UserData {
                    index: 7,
                    slot: 255,
                    root: RootType::Other(0xdead_beef),
                    dword: u32::MAX,
                },
                UserData {
                    index: u32::MAX,
                    slot: 2,
                    root: RootType::SamplerHeap,
                    dword: 3,
                },
            ],
            samplers: vec![[1, 2, 3, 4], [0, 0, 0, 0]],
            static_samplers: vec![
                StaticSampler::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0]),
                StaticSampler::from_words([0, 0, 0, 0]),
            ],
            root_cbvs: vec![0, 1, 255],
            root_srvs: Vec::new(),
        }
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }

    #[test]
    fn the_whole_shader_survives_the_round_trip() {
        assert_eq!(decode(&encode(&rich_shader())), Some(rich_shader()));
    }

    #[test]
    fn every_stage_round_trips() {
        let stages = [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Vs,
            HwStage::Ps,
            HwStage::Ms,
            HwStage::Cs,
        ];
        for stage in stages {
            let mut s = shader();
            s.stage = stage;
            assert_eq!(decode(&encode(&s)), Some(s.clone()), "{stage:?}");
        }
    }

    #[test]
    fn every_root_type_round_trips() {
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(1),
            RootType::Other(6),
            RootType::Other(u32::MAX),
        ];
        for root in roots {
            let mut s = shader();
            s.user_data[0].root = root;
            assert_eq!(decode(&encode(&s)), Some(s.clone()), "{root:?}");
        }
    }

    #[test]
    fn every_truncation_is_not_a_shader() {
        let bytes = encode(&rich_shader());
        for n in 0..bytes.len() {
            assert_eq!(decode(&bytes[..n]), None, "{n} of {} bytes", bytes.len());
        }
        assert_eq!(decode(&bytes), Some(rich_shader()));
    }

    #[test]
    fn impossible_lengths_are_not_shaders() {
        let s = rich_shader();
        let bytes = encode(&s);

        // The DXIL length (after the magic and the stage).
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "an oversized DXIL");

        // The user-data count (after the DXIL).
        let at = 12 + s.dxil.len();
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "too many user-data entries");

        // The last list's length: the bytes it promises are not there.
        let at = bytes.len() - s.root_srvs.len() - 4;
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "a root SRV list that is not there");
    }

    #[test]
    fn unknown_root_tags_are_not_shaders() {
        let s = shader();
        let bytes = encode(&s);
        // The first user-data entry: count, index, slot, then the root tag.
        let at = 12 + s.dxil.len() + 4 + 4 + 4;
        for tag in [7u32, 1000, u32::MAX] {
            let mut b = bytes.clone();
            b[at..at + 4].copy_from_slice(&tag.to_le_bytes());
            assert_eq!(decode(&b), None, "{tag}");
        }
    }

    #[test]
    fn damaged_entries_never_panic() {
        let bytes = encode(&rich_shader());
        for at in 0..bytes.len() {
            for xor in [0x01u8, 0x80, 0xff] {
                let mut b = bytes.clone();
                b[at] ^= xor;
                let _ = decode(&b);
            }
        }
        // Entries that are not one of ours: arbitrary bytes behind the magic.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for len in 0..96 {
            let mut b = b"VSH1".to_vec();
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b.push(state as u8);
            }
            let _ = decode(&b);
        }
    }

    #[test]
    fn each_container_and_build_keeps_its_own_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |c: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            let mut s = shader();
            s.dxil = c.to_vec();
            Ok(s)
        };
        let of = |c: &[u8]| {
            let mut s = shader();
            s.dxil = c.to_vec();
            s
        };
        let (a, b) = (b"first container".as_slice(), b"second container");
        assert_eq!(cached(&cache, &[7; 32], a, translate), Ok(of(a)));
        assert_eq!(cached(&cache, &[7; 32], b, translate), Ok(of(b)));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(walk(dir.path()).len(), 2, "one entry per container");

        assert_eq!(cached(&cache, &[7; 32], a, translate), Ok(of(a)));
        assert_eq!(cached(&cache, &[7; 32], b, translate), Ok(of(b)));
        assert_eq!(runs.load(Ordering::Relaxed), 2, "both came from the cache");

        // Another build's translation is a separate entry too.
        assert_eq!(cached(&cache, &[8; 32], b, translate), Ok(of(b)));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(walk(dir.path()).len(), 3);
    }

    #[test]
    fn a_cache_that_cannot_be_written_still_translates() {
        // A directory where the entry goes makes the atomic rename fail.
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7; 32];
        let container = b"a console container".as_slice();
        let translate = |_: &[u8]| -> Result<HostShader, String> { Ok(shader()) };
        let converter = Translate {
            translate,
            translated: OnceLock::new(),
        };
        let input = [&build[..], container].concat();
        let path = cache.path_for(&CacheKey::new(&converter, &input));
        std::fs::create_dir_all(&path).unwrap();

        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "no second translation");
        assert!(path.is_dir(), "the entry path is still unusable");
    }

    #[test]
    fn the_converter_needs_the_build_fingerprint() {
        let converter = Translate {
            translate: |c: &[u8]| -> Result<HostShader, String> {
                let mut s = shader();
                s.dxil = c.to_vec();
                Ok(s)
            },
            translated: OnceLock::new(),
        };
        assert!(converter.convert(b"short").is_err());
        let input = [&[0u8; 32][..], b"container"].concat();
        let mut expected = shader();
        expected.dxil = b"container".to_vec();
        assert_eq!(
            converter.convert(&input).ok().and_then(|b| decode(&b)),
            Some(expected)
        );
    }
}
