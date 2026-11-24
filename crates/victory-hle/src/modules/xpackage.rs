//! XPackage: the title's own installation, complete and launchable, with
//! every chunk ready; no other package (content, add-ons) is installed.
//! The package is the folder of the title's executable (`G:` to the title).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::path::Path;
use std::sync::OnceLock;

use super::xasync::{self, AsyncBlock};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, put_string, text, token,
};
use crate::runtime;

pub const PROVIDER: Guid = Guid::new(
    0xaf40_6016,
    0xe850,
    0x4aa8,
    [0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e],
);

/// As Grand Theft Auto VI (2026) asks for it. Its slots up to 0xd0 take the
/// arguments the public API lists in the same order; 0x110 uninstalls a
/// package, 0x128/0x130 mount one with UI, 0x138 enumerates packages.
pub const IID_2026: Guid = Guid::new(
    0x0b9d_556e,
    0xab70,
    0x43f4,
    [0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40],
);

pub static OBJECT: Object = Object::new("package", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID_2026, &OBJECT)];

/// The documented "could not resolve an identifier for the package".
pub const NO_PACKAGE: HRESULT = 0x8924_5208_u32 as i32;

/// Our installation's identifier: the title id in hex (titles keep it in 33
/// bytes and hand it back to us).
pub fn identifier() -> String {
    format!("{:08X}", runtime::title().id)
}

fn is_ours(id: *const c_char) -> bool {
    text(id).is_some_and(|id| id == identifier())
}

/// Bytes in a directory tree, skipping reparse/symlink entries so directory
/// cycles cannot recurse infinitely.
fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let Ok(ft) = e.file_type() else {
                return 0;
            };
            if ft.is_symlink() {
                return 0;
            }
            if ft.is_dir() {
                dir_bytes(&e.path())
            } else {
                e.metadata().map_or(0, |m| m.len())
            }
        })
        .fold(0u64, u64::saturating_add)
}

/// Bytes in the package folder, counted once.
fn package_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let exe = runtime::main_image().and_then(runtime::image_path);
        exe.as_deref().and_then(Path::parent).map_or(0, dir_bytes)
    })
}

type This = *const Object;

extern "system" fn current_identifier(_: This, size: usize, buffer: *mut u8) -> HRESULT {
    put_string(&identifier(), size, buffer, std::ptr::null_mut())
}

extern "system" fn is_packaged(_: This) -> bool {
    true
}

/// Installation monitors all watch the same finished installation, so the
/// handle is a fixed address.
static MONITOR: u8 = 0;
fn monitor() -> *const c_void {
    (&raw const MONITOR).cast()
}

extern "system" fn create_monitor(
    _: This,
    id: *const c_char,
    _selector_count: u32,
    _selectors: *const c_void,
    _interval_ms: u32,
    _queue: *const c_void,
    out: *mut *const c_void,
) -> HRESULT {
    if !is_ours(id) {
        return NO_PACKAGE;
    }
    put(out, monitor())
}
extern "system" fn close_monitor(_: This, _: *const c_void) {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub total_bytes: u64,
    pub installed_bytes: u64,
    pub launch_bytes: u64,
    pub launchable: bool,
    pub completed: bool,
}
const _: () = assert!(size_of::<Progress>() == 32);

extern "system" fn progress(_: This, _: *const c_void, out: *mut Progress) {
    let bytes = package_bytes();
    put(
        out,
        Progress {
            total_bytes: bytes,
            installed_bytes: bytes,
            launch_bytes: bytes,
            launchable: true,
            completed: true,
        },
    );
}

/// Installation is already complete, so updating a monitor never observes a change.
extern "system" fn update_monitor(_: This, _monitor: *const c_void) -> bool {
    false
}

/// The installation is complete, so progress never changes.
extern "system" fn register_progress(
    _: This,
    _monitor: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    put(out, token())
}

extern "system" fn unregister_progress(
    _: This,
    _monitor: *const c_void,
    token: u64,
    _wait: bool,
) -> bool {
    token != 0
}

/// 0: ready.
// shortcut: every selector counts as installed, including ones the
// package's layout does not have (the console answers those with an
// invalid-selector error); Grand Theft Auto VI asks only about its own
// chunks, which a full install has.
extern "system" fn find_availability(
    _: This,
    id: *const c_char,
    _selector_count: u32,
    _selectors: *const c_void,
    out: *mut u32,
) -> HRESULT {
    if !is_ours(id) {
        return NO_PACKAGE;
    }
    put(out, 0)
}

extern "system" fn change_order(
    _: This,
    id: *const c_char,
    _count: u32,
    _selectors: *const c_void,
) -> HRESULT {
    if is_ours(id) { S_OK } else { NO_PACKAGE }
}

static INSTALL: u8 = 0;

/// Everything is installed: done at once, the result is a monitor.
extern "system" fn install_chunks_async(
    _: This,
    id: *const c_char,
    _count: u32,
    _selectors: *const c_void,
    _interval_ms: u32,
    _suppress_confirmation: bool,
    block: *mut AsyncBlock,
) -> HRESULT {
    let result = if is_ours(id) {
        Ok(xasync::bytes(monitor()))
    } else {
        Err(NO_PACKAGE)
    };
    xasync::finished_with(block, (&raw const INSTALL).cast(), result)
}
extern "system" fn install_chunks_result(
    _: This,
    block: *mut AsyncBlock,
    out: *mut *const c_void,
) -> HRESULT {
    xasync::result(block, (&raw const INSTALL).cast(), out)
}

/// Nothing to download.
extern "system" fn estimate_download(
    _: This,
    id: *const c_char,
    _count: u32,
    _selectors: *const c_void,
    bytes: *mut u64,
    confirm: *mut bool,
) -> HRESULT {
    if !is_ours(id) {
        return NO_PACKAGE;
    }
    if bytes.is_null() {
        return xasync::E_INVALIDARG;
    }
    if !confirm.is_null() {
        // SAFETY: checked non-null.
        unsafe { confirm.write(false) };
    }
    put(bytes, 0)
}

// shortcut: accepted but nothing is removed (the package folder is the
// user's), so the chunks stay ready; Grand Theft Auto VI's one call site
// (title+0x293b4b0, its chunk manager) would then see them still installed.
extern "system" fn uninstall_chunks(
    _: This,
    id: *const c_char,
    _count: u32,
    _selectors: *const c_void,
) -> HRESULT {
    if is_ours(id) { S_OK } else { NO_PACKAGE }
}

/// Mount handles only come from mounting another package, and there is
/// none.
extern "system" fn mount_path_size(_: This, _: *const c_void, _: *mut usize) -> HRESULT {
    xasync::E_INVALIDARG
}
extern "system" fn mount_path(_: This, _: *const c_void, _: usize, _: *mut u8) -> HRESULT {
    xasync::E_INVALIDARG
}
extern "system" fn close_mount(_: This, _: *const c_void) {}

/// Only other packages are uninstalled this way, and none is installed.
extern "system" fn uninstall_package(_: This, id: *const c_char) -> bool {
    tracing::info!(
        id = text(id).unwrap_or_default(),
        "package uninstall refused"
    );
    false
}

static MOUNT: u8 = 0;

extern "system" fn mount_async(_: This, id: *const c_char, block: *mut AsyncBlock) -> HRESULT {
    tracing::info!(
        id = text(id).unwrap_or_default(),
        "package mount: not installed"
    );
    xasync::finished(block, (&raw const MOUNT).cast(), NO_PACKAGE)
}
extern "system" fn mount_result(
    _: This,
    block: *mut AsyncBlock,
    out: *mut *const c_void,
) -> HRESULT {
    xasync::result(block, (&raw const MOUNT).cast(), out)
}

/// Titles enumerate content packages (kind 1); there are none.
// shortcut: the game kind (0) also lists nothing, not this installation;
// Grand Theft Auto VI enumerates only content (kind 1, title+0x2937bf0).
extern "system" fn enumerate(
    _: This,
    kind: u32,
    _scope: u32,
    _context: *mut c_void,
    _callback: *const c_void,
) -> HRESULT {
    tracing::debug!(kind, "packages enumerated: none");
    S_OK
}

static VTABLE: Methods<40> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    current_identifier as *const (),    // 0x18
    is_packaged as *const (),           // 0x20
    create_monitor as *const (),        // 0x28
    close_monitor as *const (),         // 0x30
    progress as *const (),              // 0x38
    update_monitor as *const (),        // 0x40
    register_progress as *const (),     // 0x48
    unregister_progress as *const (),   // 0x50
    missing::<11> as *const (),         // 0x58 user locale
    find_availability as *const (),     // 0x60
    missing::<13> as *const (),         // 0x68 enumerate chunk availability
    change_order as *const (),          // 0x70
    missing::<15> as *const (),         // 0x78 install chunks
    install_chunks_async as *const (),  // 0x80
    install_chunks_result as *const (), // 0x88
    estimate_download as *const (),     // 0x90
    uninstall_chunks as *const (),      // 0x98
    missing::<20> as *const (),         // 0xa0
    missing::<21> as *const (),         // 0xa8
    missing::<22> as *const (),         // 0xb0
    missing::<23> as *const (),         // 0xb8 mount (deprecated)
    mount_path_size as *const (),       // 0xc0
    mount_path as *const (),            // 0xc8
    close_mount as *const (),           // 0xd0
    missing::<27> as *const (),         // 0xd8
    missing::<28> as *const (),         // 0xe0 enumerate packages (older)
    missing::<29> as *const (),         // 0xe8
    missing::<30> as *const (),         // 0xf0
    missing::<31> as *const (),         // 0xf8
    missing::<32> as *const (),         // 0x100
    missing::<33> as *const (),         // 0x108
    uninstall_package as *const (),     // 0x110
    missing::<35> as *const (),         // 0x118
    missing::<36> as *const (),         // 0x120
    mount_async as *const (),           // 0x128
    mount_result as *const (),          // 0x130
    enumerate as *const (),             // 0x138
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    fn empty_block() -> AsyncBlock {
        AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        }
    }

    #[test]
    fn the_installation_is_complete_and_alone() {
        let this: This = &OBJECT;
        type Id = extern "system" fn(This, usize, *mut u8) -> HRESULT;
        let mut id = [0u8; 33];
        assert_eq!(method::<Id>(&OBJECT, 0x18)(this, 33, id.as_mut_ptr()), S_OK);
        let len = id.iter().position(|&b| b == 0).unwrap();
        assert_eq!(std::str::from_utf8(&id[..len]).unwrap(), identifier());
        assert!(method::<extern "system" fn(This) -> bool>(&OBJECT, 0x20)(
            this
        ));

        type Find =
            extern "system" fn(This, *const c_char, u32, *const c_void, *mut u32) -> HRESULT;
        let mut availability = 9;
        let ours = format!("{}\0", identifier());
        let find = method::<Find>(&OBJECT, 0x60);
        let none = std::ptr::null();
        assert_eq!(
            find(this, ours.as_ptr().cast(), 0, none, &mut availability),
            S_OK
        );
        assert_eq!(availability, 0, "ready");
        assert_eq!(
            find(this, c"dlc".as_ptr(), 0, none, &mut availability),
            NO_PACKAGE
        );

        type Progress_ = extern "system" fn(This, *const c_void, *mut Progress);
        let mut p = Progress {
            total_bytes: 1,
            installed_bytes: 0,
            launch_bytes: 0,
            launchable: false,
            completed: false,
        };
        method::<Progress_>(&OBJECT, 0x38)(this, monitor(), &mut p);
        assert!(p.completed && p.launchable && p.installed_bytes == p.total_bytes);

        type Mount = extern "system" fn(This, *const c_char, *mut AsyncBlock) -> HRESULT;
        let mut block = empty_block();
        assert_eq!(
            method::<Mount>(&OBJECT, 0x128)(this, c"dlc".as_ptr(), &mut block),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut block, true), NO_PACKAGE);
    }

    #[test]
    fn current_identifier_checks_buffer_size_and_null() {
        let this: This = &OBJECT;
        type Id = extern "system" fn(This, usize, *mut u8) -> HRESULT;
        let get_id = method::<Id>(&OBJECT, 0x18);
        let expected = identifier();
        let exact = expected.len() + 1;

        assert_eq!(
            get_id(this, exact, std::ptr::null_mut()),
            xasync::E_NOT_SUFFICIENT_BUFFER
        );
        let mut small = vec![0xAAu8; exact - 1];
        assert_eq!(
            get_id(this, small.len(), small.as_mut_ptr()),
            xasync::E_NOT_SUFFICIENT_BUFFER
        );
        let mut buf = vec![0xAAu8; exact];
        assert_eq!(get_id(this, buf.len(), buf.as_mut_ptr()), S_OK);
        assert_eq!(&buf[..expected.len()], expected.as_bytes());
        assert_eq!(buf[expected.len()], 0);
    }

    #[test]
    fn monitor_update_and_progress_registration_slots() {
        let this: This = &OBJECT;
        type CreateMonitor = extern "system" fn(
            This,
            *const c_char,
            u32,
            *const c_void,
            u32,
            *const c_void,
            *mut *const c_void,
        ) -> HRESULT;
        type UpdateMonitor = extern "system" fn(This, *const c_void) -> bool;
        type Register = extern "system" fn(
            This,
            *const c_void,
            *mut c_void,
            *const c_void,
            *mut u64,
        ) -> HRESULT;
        type Unregister = extern "system" fn(This, *const c_void, u64, bool) -> bool;
        type CloseMonitor = extern "system" fn(This, *const c_void);

        let ours = format!("{}\0", identifier());
        let create = method::<CreateMonitor>(&OBJECT, 0x28);
        let mut handle = std::ptr::null();

        assert_eq!(
            create(
                this,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                &mut handle
            ),
            NO_PACKAGE
        );
        assert_eq!(
            create(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            xasync::E_INVALIDARG
        );
        assert_eq!(
            create(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                &mut handle
            ),
            S_OK
        );
        assert_eq!(handle, monitor());

        // Installation is complete, so update reports no changes.
        assert!(!method::<UpdateMonitor>(&OBJECT, 0x40)(this, handle));

        // Register/unregister progress tokens.
        let reg = method::<Register>(&OBJECT, 0x48);
        let unreg = method::<Unregister>(&OBJECT, 0x50);
        assert_eq!(
            reg(
                this,
                handle,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            xasync::E_INVALIDARG
        );
        let (mut tok1, mut tok2) = (0u64, 0u64);
        assert_eq!(
            reg(this, handle, std::ptr::null_mut(), std::ptr::null(), &mut tok1),
            S_OK
        );
        assert_eq!(
            reg(this, handle, std::ptr::null_mut(), std::ptr::null(), &mut tok2),
            S_OK
        );
        assert_ne!(tok1, 0);
        assert_ne!(tok1, tok2);
        assert!(unreg(this, handle, tok1, true));
        assert!(!unreg(this, handle, 0, false));

        // Null progress output is tolerated without crashing.
        type ProgressFn = extern "system" fn(This, *const c_void, *mut Progress);
        method::<ProgressFn>(&OBJECT, 0x38)(this, handle, std::ptr::null_mut());
        method::<CloseMonitor>(&OBJECT, 0x30)(this, handle);
    }

    #[test]
    fn install_order_and_uninstall_chunks_handle_known_and_unknown_ids() {
        let this: This = &OBJECT;
        let ours = format!("{}\0", identifier());

        type OrderOrUninstall =
            extern "system" fn(This, *const c_char, u32, *const c_void) -> HRESULT;
        for slot in [0x70, 0x98] {
            let f = method::<OrderOrUninstall>(&OBJECT, slot);
            assert_eq!(f(this, ours.as_ptr().cast(), 0, std::ptr::null()), S_OK);
            assert_eq!(f(this, c"other".as_ptr(), 0, std::ptr::null()), NO_PACKAGE);
            assert_eq!(f(this, std::ptr::null(), 0, std::ptr::null()), NO_PACKAGE);
        }

        type InstallAsync = extern "system" fn(
            This,
            *const c_char,
            u32,
            *const c_void,
            u32,
            bool,
            *mut AsyncBlock,
        ) -> HRESULT;
        type InstallResult = extern "system" fn(This, *mut AsyncBlock, *mut *const c_void) -> HRESULT;

        let mut ok_block = empty_block();
        assert_eq!(
            method::<InstallAsync>(&OBJECT, 0x80)(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                0,
                true,
                &mut ok_block
            ),
            S_OK
        );
        let mut handle = std::ptr::null();
        assert_eq!(
            method::<InstallResult>(&OBJECT, 0x88)(this, &mut ok_block, &mut handle),
            S_OK
        );
        assert_eq!(handle, monitor());

        let mut err_block = empty_block();
        assert_eq!(
            method::<InstallAsync>(&OBJECT, 0x80)(
                this,
                c"other".as_ptr(),
                0,
                std::ptr::null(),
                0,
                false,
                &mut err_block
            ),
            S_OK
        );
        assert_eq!(
            method::<InstallResult>(&OBJECT, 0x88)(this, &mut err_block, &mut handle),
            NO_PACKAGE
        );
    }

    #[test]
    fn estimate_download_does_not_mutate_confirm_when_bytes_is_null() {
        let this: This = &OBJECT;
        let ours = format!("{}\0", identifier());
        type Estimate = extern "system" fn(
            This,
            *const c_char,
            u32,
            *const c_void,
            *mut u64,
            *mut bool,
        ) -> HRESULT;
        let est = method::<Estimate>(&OBJECT, 0x90);

        let mut confirm = true;
        assert_eq!(
            est(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut confirm
            ),
            xasync::E_INVALIDARG
        );
        assert!(confirm, "confirm must not be overwritten on E_INVALIDARG");

        let mut bytes = 1234u64;
        assert_eq!(
            est(this, c"other".as_ptr(), 0, std::ptr::null(), &mut bytes, &mut confirm),
            NO_PACKAGE
        );
        assert_eq!(bytes, 1234);
        assert!(confirm);

        assert_eq!(
            est(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                &mut bytes,
                &mut confirm
            ),
            S_OK
        );
        assert_eq!(bytes, 0);
        assert!(!confirm);

        // Optional `confirm` pointer may be null.
        bytes = 99;
        assert_eq!(
            est(
                this,
                ours.as_ptr().cast(),
                0,
                std::ptr::null(),
                &mut bytes,
                std::ptr::null_mut()
            ),
            S_OK
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn mount_uninstall_enumerate_and_dir_bytes() {
        let this: This = &OBJECT;
        type MountPathSize = extern "system" fn(This, *const c_void, *mut usize) -> HRESULT;
        type MountPath = extern "system" fn(This, *const c_void, usize, *mut u8) -> HRESULT;
        type UninstallPkg = extern "system" fn(This, *const c_char) -> bool;
        type Enumerate =
            extern "system" fn(This, u32, u32, *mut c_void, *const c_void) -> HRESULT;

        let mut size = 0usize;
        assert_eq!(
            method::<MountPathSize>(&OBJECT, 0xc0)(this, std::ptr::null(), &mut size),
            xasync::E_INVALIDARG
        );
        assert_eq!(
            method::<MountPath>(&OBJECT, 0xc8)(this, std::ptr::null(), 0, std::ptr::null_mut()),
            xasync::E_INVALIDARG
        );
        assert!(!method::<UninstallPkg>(&OBJECT, 0x110)(this, c"dlc".as_ptr()));
        assert!(!method::<UninstallPkg>(&OBJECT, 0x110)(this, std::ptr::null()));
        assert_eq!(
            method::<Enumerate>(&OBJECT, 0x138)(
                this,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            S_OK
        );

        // `dir_bytes` sums nested regular files and returns 0 for missing directories.
        let root = std::env::temp_dir().join(format!(
            "victory_xpackage_test_{}",
            std::process::id()
        ));
        let sub = root.join("nested");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(root.join("a.bin"), [1u8; 17]).unwrap();
        std::fs::write(sub.join("b.bin"), [2u8; 25]).unwrap();
        assert_eq!(dir_bytes(&root), 42);
        let _ = std::fs::remove_file(root.join("a.bin"));
        let _ = std::fs::remove_file(sub.join("b.bin"));
        let _ = std::fs::remove_dir(&sub);
        let _ = std::fs::remove_dir(&root);
        assert_eq!(dir_bytes(&root), 0);
    }
}
