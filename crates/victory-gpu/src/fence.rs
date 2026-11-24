//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(all(test, windows))] // every test here calls into host D3D12 / Win32
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }
}

/// The Windows test above waits on host events; off Windows setting one is
/// a no-op, so these check the fence itself: its value, flags and address,
/// which waits are queued and which a signal settles, and its ids.
#[cfg(all(test, not(windows)))]
mod host_free_tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, S_FALSE, slot};

    type Value = extern "system" fn(This) -> u64;
    type Wait = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
    type Signal = extern "system" fn(This, u64) -> HRESULT;
    type Flags = extern "system" fn(This) -> u32;
    type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
    type Release = extern "system" fn(This) -> u32;

    fn new(initial: u64, flags: u32) -> This {
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), initial, flags, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        f
    }

    /// The values still waited on, in the order they were asked for.
    fn waiting(f: This) -> Vec<u64> {
        let waiters = fence(f).waiters.lock().unwrap();
        waiters.iter().map(|&(value, _)| value).collect()
    }

    fn event(n: usize) -> *mut c_void {
        std::ptr::without_provenance_mut(n)
    }

    #[test]
    fn a_fence_keeps_its_value_flags_and_address() {
        for (initial, flags) in [(0, 0), (1, 0), (u64::MAX, 0x2), (7, 0x1)] {
            let f = new(initial, flags);
            assert_eq!(slot::<Value>(f, 0x40)(f), initial);
            assert_eq!(slot::<Flags>(f, 0x58)(f), flags);
            // Titles read the value at the fence's GPU address.
            let at = slot::<Value>(f, 0x60)(f);
            // SAFETY: the fence's value, alive until released.
            assert_eq!(unsafe { *(at as *const u64) }, initial);
            assert_eq!(slot::<Signal>(f, 0x50)(f, initial ^ 1), S_OK);
            // SAFETY: as above.
            assert_eq!(unsafe { *(at as *const u64) }, initial ^ 1);
            assert_eq!(slot::<Value>(f, 0x60)(f), at, "the address stays put");
            assert_eq!(slot::<Release>(f, 0x10)(f), 0);
        }
    }

    #[test]
    fn waits_are_queued_until_a_signal_reaches_them() {
        let f = new(3, 0);
        let (wait, signal) = (slot::<Wait>(f, 0x48), slot::<Signal>(f, 0x50));
        // At or below the value: done at once, nothing queued, with or
        // without an event.
        for value in [0, 3] {
            assert_eq!(wait(f, value, event(1)), S_OK);
            assert_eq!(wait(f, value, std::ptr::null_mut()), S_OK);
        }
        assert!(waiting(f).is_empty());
        // Above it with no event would block forever: refused, not queued.
        assert_eq!(wait(f, 4, std::ptr::null_mut()), E_NOTIMPL);
        assert!(waiting(f).is_empty());

        for value in [9, 5, 7, 5] {
            assert_eq!(wait(f, value, event(value as usize)), S_OK);
        }
        assert_eq!(waiting(f), [9, 5, 7, 5]);
        // Short of every wait: none settled.
        assert_eq!(signal(f, 4), S_OK);
        assert_eq!(waiting(f), [9, 5, 7, 5]);
        // A signal settles every wait it reaches, and only those.
        assert_eq!(signal(f, 6), S_OK);
        assert_eq!(waiting(f), [9, 7]);
        // Signals may move the value back; queued waits stay.
        assert_eq!(signal(f, 1), S_OK);
        assert_eq!(slot::<Value>(f, 0x40)(f), 1);
        assert_eq!(waiting(f), [9, 7]);
        assert_eq!(signal(f, u64::MAX), S_OK);
        assert!(waiting(f).is_empty());
        assert_eq!(slot::<Value>(f, 0x40)(f), u64::MAX);
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }

    #[test]
    fn fences_answer_their_ids_and_no_others() {
        let other = guid(1, 2, 3, [4; 8]);
        let mut out = std::ptr::dangling();
        // Made as something it is not: refused, out cleared.
        assert_eq!(
            create(std::ptr::null(), 0, 0, &other, &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());
        // No out: only asked whether it would succeed.
        assert_eq!(
            create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );

        let f = new(0, 0);
        let query = slot::<Query>(f, 0);
        for known in FENCE_IIDS {
            let mut again = std::ptr::null();
            assert_eq!(query(f, &known, &mut again), S_OK);
            assert_eq!(again, f);
            assert_eq!(slot::<Release>(f, 0x10)(f), 1);
        }
        let mut none = std::ptr::dangling();
        assert_eq!(query(f, &other, &mut none), E_NOINTERFACE);
        assert!(none.is_null());
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }
}
