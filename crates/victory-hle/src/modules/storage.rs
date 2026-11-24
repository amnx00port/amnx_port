//! Persistent local storage: a host folder of the title's own
//! (`runtime::Title::storage`, under the data directory, never the cache)
//! with the quota its manifest declares, seen by the title as a drive
//! (`files`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::path::Path;

use super::xasync::{self, AsyncBlock, E_ABORT};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, NOT_SUPPORTED, Object, S_OK, add_ref, missing, put,
    put_string,
};
use crate::runtime;

pub const PROVIDER: Guid = Guid::new(
    0xf4fa_f4d4,
    0x2d04,
    0x4fce,
    [0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84],
);
pub const IID: Guid = Guid::new(
    0xd294_11df,
    0x0794,
    0x4553,
    [0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d],
);

/// As Grand Theft Auto VI (2026) asks for it; it calls 0x20 (path) and 0x28
/// (space) with the same arguments.
pub const IID_2026: Guid = Guid::new(
    0x41a4_e10c,
    0x5a7e,
    0x41d9,
    [0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6],
);

pub static OBJECT: Object = Object::new("persistent local storage", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT), (PROVIDER, IID_2026, &OBJECT)];

const E_FAIL: HRESULT = 0x8000_4005_u32 as i32;

type This = *const Object;

/// The path titles get: the root of a virtual drive over the folder. On the
/// console storage is a drive of its own, and titles rely on it: Minecraft
/// keeps only what precedes the path's last `:\` and mounts that as a
/// drive, so a host folder such as `C:\Users\..` would make it write to the
/// root of `C:`. None (the path calls fail) before the title is set.
fn folder() -> Option<String> {
    let storage = runtime::title().storage;
    if storage.as_os_str().is_empty() {
        return None;
    }
    #[cfg(windows)]
    return super::files::mount(&storage).map(|letter| format!("{letter}:\\"));
    #[cfg(not(windows))]
    None
}

extern "system" fn path_size(_: This, out: *mut usize) -> HRESULT {
    match folder() {
        Some(path) => put(out, path.len() + 1),
        None => NOT_SUPPORTED,
    }
}

/// A buffer too small for the path fails with `E_FAIL`, the documented
/// Xbox result (a PC gets `ERROR_INSUFFICIENT_BUFFER`).
extern "system" fn path(_: This, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    match folder() {
        Some(path) if size <= path.len() => E_FAIL,
        Some(path) => put_string(&path, size, buffer, used),
        None => NOT_SUPPORTED,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceInfo {
    /// Writable now.
    pub available_free: u64,
    /// Left in the quota.
    pub total_free: u64,
    pub used: u64,
    /// The quota.
    pub total: u64,
}

fn used_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => used_bytes(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// What the host disk under `dir` has free (unlimited if unknown).
pub(crate) fn disk_free(dir: &Path) -> u64 {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let mut free = 0;
        let null = std::ptr::null_mut();
        // SAFETY: a NUL-terminated path and a valid output.
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free,
                null,
                null,
            )
        };
        if ok != 0 {
            return free;
        }
    }
    let _ = dir;
    u64::MAX
}

fn space() -> SpaceInfo {
    let title = runtime::title();
    let used = used_bytes(&title.storage);
    let free = title.storage_bytes.saturating_sub(used);
    SpaceInfo {
        // Writable now: the quota's rest, as far as the host disk allows.
        available_free: free.min(disk_free(&title.storage)),
        total_free: free,
        used,
        total: title.storage_bytes,
    }
}

extern "system" fn space_info(_: This, out: *mut SpaceInfo) -> HRESULT {
    put(out, space())
}

/// Tells our prompts apart from other calls on the same block.
static PROMPT: u8 = 0;

/// There is nobody to ask: granted when the quota has room, else declined.
extern "system" fn prompt_for_space(_: This, bytes: u64, block: *mut AsyncBlock) -> HRESULT {
    let hr = if bytes <= space().total_free {
        S_OK
    } else {
        E_ABORT
    };
    xasync::finished(block, (&raw const PROMPT).cast(), hr)
}

extern "system" fn prompt_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    xasync::XAsyncGetResult(block, (&raw const PROMPT).cast(), 0, none, none.cast())
}

static VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    path_size as *const (),        // 0x18
    path as *const (),             // 0x20
    space_info as *const (),       // 0x28
    prompt_for_space as *const (), // 0x30
    prompt_result as *const (),    // 0x38
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn serves_a_drive_over_the_title_folder_and_its_quota() {
        let _guard = crate::runtime::TEST_TITLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        type Size = extern "system" fn(This, *mut usize) -> HRESULT;
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        type Space = extern "system" fn(This, *mut SpaceInfo) -> HRESULT;
        type Prompt = extern "system" fn(This, u64, *mut AsyncBlock) -> HRESULT;
        type Result = extern "system" fn(This, *mut AsyncBlock) -> HRESULT;
        let this: This = &OBJECT;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("save.dat"), [0u8; 100]).unwrap();
        runtime::set_title(runtime::Title {
            id: 1,
            storage: dir.path().to_owned(),
            storage_bytes: 1000,
            ..Default::default()
        });

        // The path is a drive root, never the host folder.
        let mut size = 0;
        assert_eq!(method::<Size>(&OBJECT, 0x18)(this, &mut size), S_OK);
        assert_eq!(size, 4, "X:\\ and its NUL");
        let (mut buffer, mut used) = ([0u8; 4], 0);
        let get = method::<Get>(&OBJECT, 0x20);
        assert_eq!(get(this, 4, buffer.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(get(this, 3, buffer.as_mut_ptr(), &mut used), E_FAIL);
        let drive = std::str::from_utf8(&buffer[..3]).unwrap();
        assert!(drive.ends_with(":\\"), "{drive}");
        // Title file calls on it reach the folder.
        let path = format!("{drive}save.dat\0");
        let mut attributes = [0u8; 36];
        let get_attributes = super::super::files::GetFileAttributesExW;
        let wide: Vec<u16> = path.encode_utf16().collect();
        assert_ne!(
            get_attributes(wide.as_ptr(), 0, attributes.as_mut_ptr().cast()),
            0
        );

        let mut info = SpaceInfo {
            available_free: 0,
            total_free: 0,
            used: 0,
            total: 0,
        };
        assert_eq!(method::<Space>(&OBJECT, 0x28)(this, &mut info), S_OK);
        assert_eq!((info.used, info.total_free, info.total), (100, 900, 1000));

        let prompt = method::<Prompt>(&OBJECT, 0x30);
        let result = method::<Result>(&OBJECT, 0x38);
        for (bytes, expect) in [(900, S_OK), (901, E_ABORT)] {
            let mut block = AsyncBlock {
                queue: std::ptr::null(),
                context: std::ptr::null_mut(),
                callback: None,
                internal: [0; 4],
            };
            assert_eq!(prompt(this, bytes, &mut block), S_OK);
            assert_eq!(xasync::XAsyncGetStatus(&mut block, true), expect);
            assert_eq!(result(this, &mut block), expect, "{bytes} bytes");
        }
        runtime::set_title(runtime::Title::default());
    }

    #[test]
    fn path_traversal_cannot_escape_storage_root() {
        let _guard = crate::runtime::TEST_TITLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        let this: This = &OBJECT;
        let parent = tempfile::tempdir().unwrap();
        let storage = parent.path().join("storage_root");
        std::fs::create_dir(&storage).unwrap();
        let outside = parent.path().join("secret_outside.dat");
        std::fs::write(&outside, b"sensitive host data").unwrap();
        let inside = storage.join("inside.dat");
        std::fs::write(&inside, b"game save data").unwrap();
        let sub = storage.join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested.dat"), b"nested").unwrap();

        runtime::set_title(runtime::Title {
            id: 200,
            storage: storage.clone(),
            storage_bytes: 10_000,
            ..Default::default()
        });

        let (mut buffer, mut used) = ([0u8; 4], 0);
        let get = method::<Get>(&OBJECT, 0x20);
        assert_eq!(get(this, 4, buffer.as_mut_ptr(), &mut used), S_OK);
        let drive = std::str::from_utf8(&buffer[..3]).unwrap();

        let get_attributes = super::super::files::GetFileAttributesExW;
        let mut attributes = [0u8; 36];

        // 1. Attempts to traverse outside the virtual drive root must NOT reach outside file:
        let traversal_attempts = [
            format!("{drive}..\\secret_outside.dat\0"),
            format!("{drive}..\\..\\secret_outside.dat\0"),
            format!("{drive}sub\\..\\..\\secret_outside.dat\0"),
            format!("{drive}/../secret_outside.dat\0"),
            format!("{drive}sub/../../secret_outside.dat\0"),
        ];

        for attempt in &traversal_attempts {
            let wide: Vec<u16> = attempt.encode_utf16().collect();
            // GetFileAttributesExW returns 0 (fails) because the file does not exist inside storage_root
            let res = get_attributes(wide.as_ptr(), 0, attributes.as_mut_ptr().cast());
            assert_eq!(res, 0, "path traversal attempt '{attempt}' escaped storage root");
        }

        // 2. Intra-folder traversal within storage root must succeed:
        let valid_traversals = [
            format!("{drive}sub\\..\\inside.dat\0"),
            format!("{drive}sub/../inside.dat\0"),
            format!("{drive}.\\inside.dat\0"),
            format!("{drive}sub/./nested.dat\0"),
        ];
        for valid in &valid_traversals {
            let wide: Vec<u16> = valid.encode_utf16().collect();
            let res = get_attributes(wide.as_ptr(), 0, attributes.as_mut_ptr().cast());
            assert_ne!(res, 0, "intra-folder relative path '{valid}' failed to resolve");
        }
        runtime::set_title(runtime::Title::default());
    }

    #[test]
    fn mixed_path_separators_resolve_correctly() {
        let _guard = crate::runtime::TEST_TITLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        let this: This = &OBJECT;
        let dir = tempfile::tempdir().unwrap();
        let nested_dir = dir.path().join("folder").join("nested");
        std::fs::create_dir_all(&nested_dir).unwrap();
        std::fs::write(nested_dir.join("file.bin"), b"binary data").unwrap();

        runtime::set_title(runtime::Title {
            id: 201,
            storage: dir.path().to_owned(),
            storage_bytes: 10_000,
            ..Default::default()
        });

        let (mut buffer, mut used) = ([0u8; 4], 0);
        let get = method::<Get>(&OBJECT, 0x20);
        assert_eq!(get(this, 4, buffer.as_mut_ptr(), &mut used), S_OK);
        let drive = std::str::from_utf8(&buffer[..3]).unwrap();

        let get_attributes = super::super::files::GetFileAttributesExW;
        let mut attributes = [0u8; 36];

        let separator_variants = [
            format!("{drive}folder/nested/file.bin\0"),
            format!("{drive}folder\\nested\\file.bin\0"),
            format!("{drive}folder/nested\\file.bin\0"),
            format!("{drive}folder\\nested/file.bin\0"),
            format!("{drive}/folder/nested/file.bin\0"),
            format!("{drive}\\folder\\nested\\file.bin\0"),
            format!("{drive}/folder\\nested/file.bin\0"),
        ];

        for variant in &separator_variants {
            let wide: Vec<u16> = variant.encode_utf16().collect();
            let res = get_attributes(wide.as_ptr(), 0, attributes.as_mut_ptr().cast());
            assert_ne!(res, 0, "failed to open path with separator variant '{variant}'");
        }
        runtime::set_title(runtime::Title::default());
    }

    #[test]
    fn storage_quota_enforcement_and_overflow() {
        let _guard = crate::runtime::TEST_TITLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        type Space = extern "system" fn(This, *mut SpaceInfo) -> HRESULT;
        type Prompt = extern "system" fn(This, u64, *mut AsyncBlock) -> HRESULT;
        let this: This = &OBJECT;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("existing.dat"), [0u8; 300]).unwrap();

        runtime::set_title(runtime::Title {
            id: 202,
            storage: dir.path().to_owned(),
            storage_bytes: 1_000,
            ..Default::default()
        });

        let mut info = SpaceInfo {
            available_free: 0,
            total_free: 0,
            used: 0,
            total: 0,
        };
        assert_eq!(method::<Space>(&OBJECT, 0x28)(this, &mut info), S_OK);
        assert_eq!((info.used, info.total_free, info.total), (300, 700, 1_000));

        let prompt = method::<Prompt>(&OBJECT, 0x30);
        let mut block = AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        };

        // Allowed up to remaining free quota:
        assert_eq!(prompt(this, 700, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), S_OK);

        // Exceeding remaining quota returns E_ABORT:
        assert_eq!(prompt(this, 701, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), E_ABORT);

        // Extreme overflow (u64::MAX) safely rejected:
        assert_eq!(prompt(this, u64::MAX, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), E_ABORT);

        // Simulate file writes that exceed the configured quota (e.g. 1500 bytes used with quota 1000):
        std::fs::write(dir.path().join("large.dat"), [0u8; 1_200]).unwrap();
        assert_eq!(method::<Space>(&OBJECT, 0x28)(this, &mut info), S_OK);
        assert_eq!(info.used, 1_500);
        assert_eq!(info.total_free, 0, "saturating_sub must prevent underflow");
        assert_eq!(info.available_free, 0);

        // Any non-zero prompt fails when quota is exceeded:
        assert_eq!(prompt(this, 1, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), E_ABORT);

        // 0-byte prompt succeeds:
        assert_eq!(prompt(this, 0, &mut block), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), S_OK);
        runtime::set_title(runtime::Title::default());
    }

    #[test]
    fn unmounted_and_nonexistent_drive_access() {
        let _guard = crate::runtime::TEST_TITLE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        type Size = extern "system" fn(This, *mut usize) -> HRESULT;
        type Get = extern "system" fn(This, usize, *mut u8, *mut usize) -> HRESULT;
        type Space = extern "system" fn(This, *mut SpaceInfo) -> HRESULT;
        let this: This = &OBJECT;

        // Non-existent drive access via files::translate (drive B: is reserved and never mounted):
        let nonexistent = "B:\\nonexistent\\file.dat\0".encode_utf16().collect::<Vec<u16>>();
        let translated = crate::modules::files::translate(&nonexistent[..nonexistent.len() - 1]);
        assert!(translated.is_none(), "unmounted drive B: must not translate");

        let invalid = "1:\\invalid\\file.dat\0".encode_utf16().collect::<Vec<u16>>();
        assert!(crate::modules::files::translate(&invalid[..invalid.len() - 1]).is_none());

        let get_attributes = super::super::files::GetFileAttributesExW;
        let mut attributes = [0u8; 36];
        assert_eq!(
            get_attributes(nonexistent.as_ptr(), 0, attributes.as_mut_ptr().cast()),
            0,
            "querying nonexistent drive must fail"
        );

        // Empty storage configuration:
        runtime::set_title(runtime::Title {
            id: 203,
            storage: std::path::PathBuf::new(),
            storage_bytes: 0,
            ..Default::default()
        });

        let mut size = 0;
        assert_eq!(method::<Size>(&OBJECT, 0x18)(this, &mut size), NOT_SUPPORTED);

        let (mut buffer, mut used) = ([0u8; 4], 0);
        assert_eq!(method::<Get>(&OBJECT, 0x20)(this, 4, buffer.as_mut_ptr(), &mut used), NOT_SUPPORTED);

        // Null pointer handling when storage is unconfigured:
        assert_eq!(method::<Size>(&OBJECT, 0x18)(this, std::ptr::null_mut()), NOT_SUPPORTED);
        assert_eq!(method::<Space>(&OBJECT, 0x28)(this, std::ptr::null_mut()), xasync::E_INVALIDARG);
        runtime::set_title(runtime::Title::default());
    }
}
