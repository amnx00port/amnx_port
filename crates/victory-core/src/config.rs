//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Clone)]
pub struct Config {
    /// The user's dump of Grand Theft Auto VI.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// Host controller and DualSense settings.
    pub input: InputConfig,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            game_dir: None,
            cache_dir: PathBuf::new(),
            data_dir: PathBuf::new(),
            test_titles: Vec::new(),
            input: InputConfig::default(),
            source: None,
        }
    }
}

/// Host controller and DualSense input configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct InputConfig {
    /// Whether to suppress duplicate virtual XInput pads (e.g. Steam Input `0x28DE`)
    /// while a physical/dongle DualSense HID device is active.
    pub suppress_virtual_xinput: VirtualXInputPolicy,
    /// Pause gamepad input and release rumble/adaptive triggers while unfocused.
    pub mute_unfocused: bool,
    /// PlayStation 5 DualSense / DualSense Edge settings.
    pub dualsense: DualSenseConfig,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            suppress_virtual_xinput: VirtualXInputPolicy::Auto,
            mute_unfocused: true,
            dualsense: DualSenseConfig::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VirtualXInputPolicy {
    #[default]
    Auto,
    Always,
    Never,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HapticsMode {
    #[default]
    Hid,
    #[serde(rename = "audio_4ch")]
    Audio4Ch,
    Hybrid,
    Off,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    #[default]
    Vibration,
    Hybrid,
    Off,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TouchpadClickMode {
    #[default]
    Split,
    View,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgePaddleMode {
    #[default]
    Hardware,
    Custom,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgePadBinding {
    None,
    A,
    B,
    X,
    Y,
    LeftShoulder,
    RightShoulder,
    LeftStick,
    RightStick,
    Up,
    Down,
    Left,
    Right,
    View,
    Menu,
    #[default]
    PaddleLeft1,
    PaddleRight1,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GyroMode {
    #[default]
    AdsOrTouch,
    Ads,
    Touch,
    Always,
    Off,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GyroSpace {
    #[default]
    PlayerSpace,
    Yaw,
    Roll,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedBrightness {
    High,
    #[default]
    Medium,
    Low,
    Off,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DualSenseConfig {
    pub enabled: bool,
    pub haptics_mode: HapticsMode,
    pub speaker_mirror: bool,
    pub adaptive_triggers: TriggerMode,
    pub trigger_strength: f32,
    pub touchpad_click: TouchpadClickMode,
    pub edge_paddles: EdgePaddleMode,
    pub edge_left_paddle: EdgePadBinding,
    pub edge_right_paddle: EdgePadBinding,
    pub gyro_mode: GyroMode,
    pub gyro_space: GyroSpace,
    pub gyro_sensitivity: f32,
    pub gyro_deadzone_floor: f32,
    pub lightbar_color: [u8; 3],
    pub led_brightness: LedBrightness,
}

impl Default for DualSenseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            haptics_mode: HapticsMode::Hid,
            speaker_mirror: false,
            adaptive_triggers: TriggerMode::Vibration,
            trigger_strength: 1.0,
            touchpad_click: TouchpadClickMode::Split,
            edge_paddles: EdgePaddleMode::Hardware,
            edge_left_paddle: EdgePadBinding::PaddleLeft1,
            edge_right_paddle: EdgePadBinding::PaddleRight1,
            gyro_mode: GyroMode::AdsOrTouch,
            gyro_space: GyroSpace::PlayerSpace,
            gyro_sensitivity: 1.0,
            gyro_deadzone_floor: 0.12,
            lightbar_color: [0x00, 0x40, 0xFF],
            led_brightness: LedBrightness::Medium,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
    #[serde(default)]
    input: FileInput,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileInput {
    suppress_virtual_xinput: Option<VirtualXInputPolicy>,
    mute_unfocused: Option<bool>,
    #[serde(default)]
    dualsense: FileDualSense,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDualSense {
    enabled: Option<bool>,
    haptics_mode: Option<HapticsMode>,
    speaker_mirror: Option<bool>,
    adaptive_triggers: Option<TriggerMode>,
    trigger_strength: Option<f32>,
    touchpad_click: Option<TouchpadClickMode>,
    edge_paddles: Option<EdgePaddleMode>,
    edge_left_paddle: Option<EdgePadBinding>,
    edge_right_paddle: Option<EdgePadBinding>,
    gyro_mode: Option<GyroMode>,
    gyro_space: Option<GyroSpace>,
    gyro_sensitivity: Option<f32>,
    gyro_deadzone_floor: Option<f32>,
    lightbar_color: Option<String>,
    led_brightness: Option<LedBrightness>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = std::env::var_os("VICTORY_CONFIG").map(PathBuf::from);
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FileConfig,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_str = |k: &str| {
            env(k)
                .and_then(|v| v.into_string().ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        };
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let paths = file.paths;
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => paths.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(paths.cache_dir)
            .map_or_else(|| default_dir("cache"), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(paths.data_dir)
            .map_or_else(|| default_dir("data"), Ok)?;
        let input = resolve_input(file.input, &env_str)?;
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(paths.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            input,
            source,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
fn default_dir(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
}

fn parse_bool_env(name: &str, raw: &str) -> Result<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("invalid boolean in {name}: {raw:?}"),
    }
}

fn parse_enum_env<T: for<'de> Deserialize<'de>>(name: &str, raw: &str) -> Result<T> {
    #[derive(Deserialize)]
    struct Wrap<V> {
        v: V,
    }
    let snippet = format!("v = {:?}", raw.to_ascii_lowercase());
    toml::from_str::<Wrap<T>>(&snippet)
        .map(|w| w.v)
        .with_context(|| format!("invalid value in {name}: {raw:?}"))
}

pub fn parse_hex_rgb(raw: &str) -> Result<[u8; 3]> {
    let hex = raw.trim().trim_start_matches('#');
    anyhow::ensure!(
        hex.len() == 6 && hex.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected 6-digit RGB hex color (e.g. \"0040FF\"), got {raw:?}"
    );
    Ok([
        u8::from_str_radix(&hex[0..2], 16)?,
        u8::from_str_radix(&hex[2..4], 16)?,
        u8::from_str_radix(&hex[4..6], 16)?,
    ])
}

fn resolve_input(
    file: FileInput,
    env_str: &impl Fn(&str) -> Option<String>,
) -> Result<InputConfig> {
    let defaults = InputConfig::default();
    let ds_defaults = defaults.dualsense;
    let ds = file.dualsense;

    let suppress_virtual_xinput = match env_str("VICTORY_SUPPRESS_VIRTUAL_XINPUT") {
        Some(v) => parse_enum_env("VICTORY_SUPPRESS_VIRTUAL_XINPUT", &v)?,
        None => file
            .suppress_virtual_xinput
            .unwrap_or(defaults.suppress_virtual_xinput),
    };
    let mute_unfocused = match env_str("VICTORY_MUTE_UNFOCUSED") {
        Some(v) => parse_bool_env("VICTORY_MUTE_UNFOCUSED", &v)?,
        None => file.mute_unfocused.unwrap_or(defaults.mute_unfocused),
    };
    let enabled = match env_str("VICTORY_DUALSENSE") {
        Some(v) => parse_bool_env("VICTORY_DUALSENSE", &v)?,
        None => ds.enabled.unwrap_or(ds_defaults.enabled),
    };
    let haptics_mode = match env_str("VICTORY_DUALSENSE_HAPTICS") {
        Some(v) => parse_enum_env("VICTORY_DUALSENSE_HAPTICS", &v)?,
        None => ds.haptics_mode.unwrap_or(ds_defaults.haptics_mode),
    };
    let speaker_mirror = match env_str("VICTORY_DUALSENSE_SPEAKER_MIRROR") {
        Some(v) => parse_bool_env("VICTORY_DUALSENSE_SPEAKER_MIRROR", &v)?,
        None => ds.speaker_mirror.unwrap_or(ds_defaults.speaker_mirror),
    };
    let adaptive_triggers = match env_str("VICTORY_DUALSENSE_TRIGGERS") {
        Some(v) => parse_enum_env("VICTORY_DUALSENSE_TRIGGERS", &v)?,
        None => ds
            .adaptive_triggers
            .unwrap_or(ds_defaults.adaptive_triggers),
    };
    let trigger_strength = match env_str("VICTORY_DUALSENSE_TRIGGER_STRENGTH") {
        Some(v) => v
            .parse::<f32>()
            .context("parsing VICTORY_DUALSENSE_TRIGGER_STRENGTH")?,
        None => ds.trigger_strength.unwrap_or(ds_defaults.trigger_strength),
    }
    .clamp(0.0, 1.0);
    let gyro_mode = match env_str("VICTORY_DUALSENSE_GYRO") {
        Some(v) => parse_enum_env("VICTORY_DUALSENSE_GYRO", &v)?,
        None => ds.gyro_mode.unwrap_or(ds_defaults.gyro_mode),
    };
    let gyro_space = match env_str("VICTORY_DUALSENSE_GYRO_SPACE") {
        Some(v) => parse_enum_env("VICTORY_DUALSENSE_GYRO_SPACE", &v)?,
        None => ds.gyro_space.unwrap_or(ds_defaults.gyro_space),
    };
    let gyro_sensitivity = match env_str("VICTORY_DUALSENSE_GYRO_SENSITIVITY") {
        Some(v) => v
            .parse::<f32>()
            .context("parsing VICTORY_DUALSENSE_GYRO_SENSITIVITY")?,
        None => ds.gyro_sensitivity.unwrap_or(ds_defaults.gyro_sensitivity),
    }
    .clamp(0.0, 10.0);
    let gyro_deadzone_floor = match env_str("VICTORY_DUALSENSE_GYRO_DEADZONE") {
        Some(v) => v
            .parse::<f32>()
            .context("parsing VICTORY_DUALSENSE_GYRO_DEADZONE")?,
        None => ds
            .gyro_deadzone_floor
            .unwrap_or(ds_defaults.gyro_deadzone_floor),
    }
    .clamp(0.0, 0.5);
    let lightbar_color = match env_str("VICTORY_DUALSENSE_COLOR").or(ds.lightbar_color) {
        Some(hex) => parse_hex_rgb(&hex)?,
        None => ds_defaults.lightbar_color,
    };

    Ok(InputConfig {
        suppress_virtual_xinput,
        mute_unfocused,
        dualsense: DualSenseConfig {
            enabled,
            haptics_mode,
            speaker_mirror,
            adaptive_triggers,
            trigger_strength,
            touchpad_click: ds.touchpad_click.unwrap_or(ds_defaults.touchpad_click),
            edge_paddles: ds.edge_paddles.unwrap_or(ds_defaults.edge_paddles),
            edge_left_paddle: ds.edge_left_paddle.unwrap_or(ds_defaults.edge_left_paddle),
            edge_right_paddle: ds
                .edge_right_paddle
                .unwrap_or(ds_defaults.edge_right_paddle),
            gyro_mode,
            gyro_space,
            gyro_sensitivity,
            gyro_deadzone_floor,
            lightbar_color,
            led_brightness: ds.led_brightness.unwrap_or(ds_defaults.led_brightness),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FileConfig {
            paths: FilePaths {
                game_dir: Some("from-file".into()),
                cache_dir: Some("cache-from-file".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FileConfig {
            paths: FilePaths {
                game_dir: Some("from-file".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(OsString::new);
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FileConfig {
            paths: FilePaths {
                test_titles: vec!["from-file".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(OsString::new);
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FileConfig::default();
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(|| value.clone());
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
        let cfg = Config::resolve(parsed, None, |_| None).unwrap();
        assert_eq!(cfg.input, InputConfig::default());
    }

    #[test]
    fn dualsense_config_and_env_overrides_resolve() {
        let text = r##"
[input]
suppress_virtual_xinput = "always"
mute_unfocused = false

[input.dualsense]
haptics_mode = "audio_4ch"
speaker_mirror = true
adaptive_triggers = "hybrid"
trigger_strength = 0.75
touchpad_click = "view"
edge_paddles = "custom"
edge_left_paddle = "a"
edge_right_paddle = "b"
gyro_mode = "ads"
gyro_space = "yaw"
gyro_sensitivity = 1.5
gyro_deadzone_floor = 0.15
lightbar_color = "#FF2080"
led_brightness = "high"
"##;
        let parsed: FileConfig = toml::from_str(text).unwrap();
        let env = |k: &str| match k {
            "VICTORY_DUALSENSE_TRIGGERS" => Some(OsString::from("vibration")),
            "VICTORY_DUALSENSE_GYRO_SPACE" => Some(OsString::from("player_space")),
            _ => None,
        };
        let cfg = Config::resolve(parsed, None, env).unwrap();
        assert_eq!(
            cfg.input.suppress_virtual_xinput,
            VirtualXInputPolicy::Always
        );
        assert!(!cfg.input.mute_unfocused);
        let ds = &cfg.input.dualsense;
        assert_eq!(ds.haptics_mode, HapticsMode::Audio4Ch);
        assert!(ds.speaker_mirror);
        assert_eq!(ds.adaptive_triggers, TriggerMode::Vibration);
        assert_eq!(ds.trigger_strength, 0.75);
        assert_eq!(ds.touchpad_click, TouchpadClickMode::View);
        assert_eq!(ds.edge_paddles, EdgePaddleMode::Custom);
        assert_eq!(ds.edge_left_paddle, EdgePadBinding::A);
        assert_eq!(ds.edge_right_paddle, EdgePadBinding::B);
        assert_eq!(ds.gyro_mode, GyroMode::Ads);
        assert_eq!(ds.gyro_space, GyroSpace::PlayerSpace);
        assert_eq!(ds.gyro_sensitivity, 1.5);
        assert_eq!(ds.gyro_deadzone_floor, 0.15);
        assert_eq!(ds.lightbar_color, [0xFF, 0x20, 0x80]);
        assert_eq!(ds.led_brightness, LedBrightness::High);
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }
}
