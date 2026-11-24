//! The display and what the system shell does for a title: HDR mode, screen
//! timeout deferrals, launching URIs, screenshots.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};

#[cfg(windows)]
use super::xgameruntime::put;
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, NOT_SUPPORTED, Object, add_ref, missing, text,
};

/// HDR mode; asked for with this interface id since 2020.
pub const HDR_PROVIDER: Guid = Guid::new(
    0x03f0_fe74,
    0xfdd9,
    0x4e5c,
    [0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5],
);
pub const HDR_IID: Guid = Guid::new(
    0x35f0_7670,
    0x706e,
    0x4bfb,
    [0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3],
);
/// Launching URIs and screen timeout deferrals; the interface id is the
/// provider's.
pub const SHELL: Guid = Guid::new(
    0x1b33_9674,
    0x328d,
    0x4283,
    [0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39],
);
/// Screenshots and clips.
pub const CAPTURE_PROVIDER: Guid = Guid::new(
    0xa4f1_aee2,
    0x4bf1,
    0x4485,
    [0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27],
);
/// As Grand Theft Auto VI (2026) asks for it; it calls 0x28 (take a
/// screenshot) with the public API's arguments.
pub const CAPTURE_IID_2026: Guid = Guid::new(
    0x22e6_72d7,
    0xb4e3,
    0x406c,
    [0xbd, 0x50, 0x8f, 0x0d, 0x25, 0x23, 0x6f, 0x9e],
);

pub static HDR_OBJECT: Object = Object::new("hdr", &HDR_VTABLE);
pub static SHELL_OBJECT: Object = Object::new("shell", &SHELL_VTABLE);
pub static CAPTURE_OBJECT: Object = Object::new("capture", &CAPTURE_VTABLE);

pub const INTERFACES: &[Interface] = &[
    (HDR_PROVIDER, HDR_IID, &HDR_OBJECT),
    (SHELL, SHELL, &SHELL_OBJECT),
    (CAPTURE_PROVIDER, CAPTURE_IID_2026, &CAPTURE_OBJECT),
];

type This = *const Object;

/// 2: disabled. We present in SDR, so the title must render SDR.
// shortcut: HDR output is not done (see the presentation items in
// `TODO.md`); report enabled, with the host display's luminance, once the
// swap chain can present HDR.
extern "system" fn try_enable_hdr(_: This, preference: u32, _info: *mut c_void) -> u32 {
    tracing::info!(preference, "HDR mode: disabled (SDR presentation)");
    2
}

/// Nothing launches URIs for the title: no store, browser or console app
/// exists here, and the console's are online services.
extern "system" fn launch_uri(_: This, _user: *const c_void, uri: *const c_char) -> HRESULT {
    tracing::info!(uri = text(uri).unwrap_or_default(), "URI not launched");
    NOT_SUPPORTED
}

/// A deferral keeps the host display on, as on the console, through a
/// power request; the handle is that request.
#[cfg(windows)]
extern "system" fn acquire_deferral(_: This, out: *mut *mut c_void) -> HRESULT {
    use windows_sys::Win32::System::Power::{
        PowerCreateRequest, PowerRequestDisplayRequired, PowerSetRequest,
    };
    use windows_sys::Win32::System::Threading::{
        POWER_REQUEST_CONTEXT_SIMPLE_STRING, REASON_CONTEXT,
    };
    if out.is_null() {
        return super::xasync::E_INVALIDARG;
    }
    let mut reason: Vec<u16> = "Title display deferral\0".encode_utf16().collect();
    let mut context = REASON_CONTEXT {
        Flags: POWER_REQUEST_CONTEXT_SIMPLE_STRING,
        ..Default::default()
    };
    context.Reason.SimpleReasonString = reason.as_mut_ptr();
    // SAFETY: a valid context; the string is copied by the call.
    let request = unsafe { PowerCreateRequest(&context) };
    if request as isize == -1 {
        return last_error();
    }
    // SAFETY: the request we just created.
    if unsafe { PowerSetRequest(request, PowerRequestDisplayRequired) } == 0 {
        let hr = last_error();
        // SAFETY: as above; not handed out.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(request) };
        return hr;
    }
    put(out, request)
}

#[cfg(windows)]
extern "system" fn close_deferral(_: This, handle: *mut c_void) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Power::{PowerClearRequest, PowerRequestDisplayRequired};
    if !handle.is_null() {
        // SAFETY: a request from `acquire_deferral`.
        unsafe {
            PowerClearRequest(handle, PowerRequestDisplayRequired);
            CloseHandle(handle);
        }
    }
}

#[cfg(windows)]
fn last_error() -> HRESULT {
    // SAFETY: no preconditions.
    let code = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    (0x8007_0000 | (code & 0xffff)) as i32
}

// shortcut: screenshots fail as if capture were unavailable. Taking one needs
// a hook in the GPU side: the last presented frame read back as BGRA8 (size,
// row pitch, bytes) on request, e.g. `victory_gpu::last_frame() ->
// Option<Frame>`, copied from the back buffer after `PresentX`.
extern "system" fn take_screenshot(_: This, _user: *const c_void, _result: *mut c_void) -> HRESULT {
    tracing::info!("screenshot not taken");
    NOT_SUPPORTED
}

static HDR_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),
    try_enable_hdr as *const (), // 0x20
]);

#[cfg(windows)]
static SHELL_VTABLE: Methods<6> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    launch_uri as *const (),       // 0x18
    acquire_deferral as *const (), // 0x20
    close_deferral as *const (),   // 0x28
]);
#[cfg(not(windows))]
static SHELL_VTABLE: Methods<4> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    launch_uri as *const (),
]);

static CAPTURE_VTABLE: Methods<6> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),    // 0x18 diagnostic screenshot
    missing::<4> as *const (),    // 0x20 diagnostic clip
    take_screenshot as *const (), // 0x28
]);

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::{S_OK, method};

    #[test]
    fn sdr_and_display_deferrals() {
        type Hdr = extern "system" fn(This, u32, *mut c_void) -> u32;
        assert_eq!(
            method::<Hdr>(&HDR_OBJECT, 0x20)(&HDR_OBJECT, 0, std::ptr::null_mut()),
            2
        );
        type Acquire = extern "system" fn(This, *mut *mut c_void) -> HRESULT;
        type Close = extern "system" fn(This, *mut c_void);
        let mut handle = std::ptr::null_mut();
        assert_eq!(
            method::<Acquire>(&SHELL_OBJECT, 0x20)(&SHELL_OBJECT, &mut handle),
            S_OK
        );
        assert!(!handle.is_null());
        method::<Close>(&SHELL_OBJECT, 0x28)(&SHELL_OBJECT, handle);
    }
}

/// The Windows test above takes a host power request; these run anywhere.
#[cfg(all(test, not(windows)))]
mod host_free_tests {
    use super::*;
    use crate::modules::xasync::E_NOTIMPL;
    use crate::modules::xgameruntime::{QueryApiImpl, S_OK, method};

    type Hdr = extern "system" fn(This, u32, *mut c_void) -> u32;
    type Launch = extern "system" fn(This, *const c_void, *const c_char) -> HRESULT;
    type Shot = extern "system" fn(This, *const c_void, *mut c_void) -> HRESULT;
    type Plain = extern "system" fn(This) -> HRESULT;
    type Count = extern "system" fn(This) -> u32;

    fn query(provider: Guid, iid: Guid) -> (HRESULT, *mut c_void) {
        let mut out = std::ptr::null_mut();
        (QueryApiImpl(&provider, &iid, &mut out), out)
    }

    #[test]
    fn display_areas_resolve_and_refuse_what_the_host_cannot_do() {
        // Each pair resolves to its object; a provider asked with another
        // area's interface id does not.
        for (provider, iid, object) in INTERFACES {
            let (hr, out) = query(*provider, *iid);
            assert_eq!(hr, S_OK, "{provider}");
            assert_eq!(out.cast_const(), (*object as *const Object).cast());
        }
        assert_eq!(query(HDR_PROVIDER, SHELL).0, NOT_SUPPORTED);
        assert_eq!(query(CAPTURE_PROVIDER, HDR_IID).0, NOT_SUPPORTED);
        assert_eq!(query(SHELL, CAPTURE_IID_2026).0, NOT_SUPPORTED);

        // SDR whatever the title prefers, with or without room for info.
        let mut info = [0u8; 64];
        let hdr = method::<Hdr>(&HDR_OBJECT, 0x20);
        for preference in [0, 1, 2, u32::MAX] {
            assert_eq!(hdr(&HDR_OBJECT, preference, std::ptr::null_mut()), 2);
            assert_eq!(hdr(&HDR_OBJECT, preference, info.as_mut_ptr().cast()), 2);
        }
        assert_eq!(info, [0; 64], "the info is not written");

        // No URI is launched, named or not.
        let launch = method::<Launch>(&SHELL_OBJECT, 0x18);
        let uri = c"ms-windows-store://pdp/?productid=0";
        assert_eq!(
            launch(&SHELL_OBJECT, std::ptr::null(), uri.as_ptr()),
            NOT_SUPPORTED
        );
        assert_eq!(
            launch(&SHELL_OBJECT, std::ptr::null(), std::ptr::null()),
            NOT_SUPPORTED
        );
        // Off Windows there is no power request to defer the screen with:
        // the shell table ends at the launch slot.
        assert_eq!(SHELL_VTABLE.0.len(), 4);

        // Screenshots fail as unavailable; the diagnostic captures are
        // unimplemented slots, not crashes.
        let mut result = [0u8; 32];
        assert_eq!(
            method::<Shot>(&CAPTURE_OBJECT, 0x28)(
                &CAPTURE_OBJECT,
                std::ptr::null(),
                result.as_mut_ptr().cast()
            ),
            NOT_SUPPORTED
        );
        assert_eq!(result, [0; 32]);
        for slot in [0x18, 0x20] {
            assert_eq!(
                method::<Plain>(&CAPTURE_OBJECT, slot)(&CAPTURE_OBJECT),
                E_NOTIMPL
            );
        }
        assert_eq!(method::<Plain>(&HDR_OBJECT, 0x18)(&HDR_OBJECT), E_NOTIMPL);

        // AddRef and Release on every object are no-ops that report one.
        for object in [&HDR_OBJECT, &SHELL_OBJECT, &CAPTURE_OBJECT] {
            for slot in [0x08, 0x10] {
                assert_eq!(method::<Count>(object, slot)(object), 1);
            }
        }
    }
}
