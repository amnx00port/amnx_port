//! A profile-specific, read-only replacement for a loose controls diagram.

use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;

struct Diagram {
    package_path: &'static str,
    profiles: PathBuf,
}

static DIAGRAM: OnceLock<Diagram> = OnceLock::new();

/// Configure the one virtual package file whose artwork the title can replace.
pub fn use_controller_diagram(package_path: &'static str, profiles: PathBuf) {
    let _ = DIAGRAM.set(Diagram {
        package_path,
        profiles,
    });
}

pub(crate) fn controller_diagram(name: &str) -> Option<File> {
    let diagram = DIAGRAM.get()?;
    diagram.open(name, super::super::gameinput::active_glyph_profile)
}

impl Diagram {
    fn open(&self, name: &str, profile: impl FnOnce() -> &'static str) -> Option<File> {
        let name = name.strip_prefix(r"\\?\").unwrap_or(name);
        let mut actual = Path::new(name).components();
        for wanted in Path::new(self.package_path).components() {
            if !actual
                .next()?
                .as_os_str()
                .eq_ignore_ascii_case(wanted.as_os_str())
            {
                return None;
            }
        }
        if actual.next().is_some() {
            return None;
        }
        let profile = profile();
        let file_name = self.package_path.rsplit(['/', '\\']).next()?;
        let candidate = self.profiles.join(profile).join(file_name);
        match OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&candidate)
        {
            Ok(file) => {
                tracing::info!(profile, path = %candidate.display(), "opened controller diagram");
                Some(file)
            }
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(profile, %error, "using package controller diagram");
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn each_open_uses_its_own_profile_and_preserves_existing_handles() {
        let dir = tempfile::tempdir().unwrap();
        let diagram = Diagram {
            package_path: r"G:\ui\controls.bin",
            profiles: dir.path().to_owned(),
        };
        for profile in ["playstation", "nintendo"] {
            std::fs::create_dir(dir.path().join(profile)).unwrap();
            std::fs::write(dir.path().join(profile).join("controls.bin"), profile).unwrap();
        }
        let mut first = diagram
            .open(r"g:\.\UI\Controls.bin", || "playstation")
            .unwrap();
        let mut second = diagram
            .open(r"\\?\G:\ui\controls.bin", || "nintendo")
            .unwrap();
        assert!(diagram.open("G:/ui/controls.bin", || "steam").is_none());
        for name in [
            "T:/ui/controls.bin",
            "G:/other/ui/controls.bin",
            "G:/data.rpf",
        ] {
            assert!(diagram.open(name, || "playstation").is_none());
        }
        let mut contents = String::new();
        second.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "nintendo");
        contents.clear();
        first.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "playstation");
    }
}
