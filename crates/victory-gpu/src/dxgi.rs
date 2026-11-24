//! The DXGI side of the device: one adapter with one output, reached from
//! the device by interface query. Tables follow the PC interfaces' order
//! (IUnknown, the object methods, then each interface's methods).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{
    self, ADAPTER, DXGI_DEVICE, E_INVALIDARG, E_POINTER, FACTORY, HRESULT, Iid, MORE_DATA,
    NOT_FOUND, OUTPUT, Object, S_OK, Table, This, guid,
};

/// The DXGI device interfaces, base first (as on PC).
pub(crate) const DEVICE_IIDS: [Iid; 3] = [
    guid(
        0x54ec77fa,
        0x1377,
        0x44e6,
        [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c],
    ),
    guid(
        0x77db970f,
        0x6276,
        0x48ba,
        [0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c],
    ),
    guid(
        0x05008617,
        0xfbfd,
        0x4051,
        [0xa7, 0x90, 0x14, 0x48, 0x84, 0xb4, 0xf6, 0xa9],
    ),
];
const ADAPTER_IIDS: [Iid; 3] = [
    guid(
        0x2411e7e1,
        0x12ac,
        0x4ccf,
        [0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0],
    ),
    guid(
        0x29038f61,
        0x3839,
        0x4626,
        [0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05],
    ),
    guid(
        0x0aa1ae0a,
        0xfa0e,
        0x4b84,
        [0x86, 0x44, 0xe0, 0x5f, 0xf8, 0xe5, 0xac, 0xb5],
    ),
];
const FACTORY_IIDS: [Iid; 3] = [
    guid(
        0x7b7166ec,
        0x21c7,
        0x44ae,
        [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69],
    ),
    guid(
        0x770aae78,
        0xf26f,
        0x4dba,
        [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    ),
    guid(
        0x50c83a1c,
        0xe072,
        0x4c48,
        [0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0],
    ),
];
const OUTPUT_IIDS: [Iid; 1] = [guid(
    0xae02eedb,
    0xc735,
    0x4690,
    [0x8d, 0x52, 0x5a, 0x8d, 0xc2, 0x02, 0x13, 0xaa],
)];

static DEVICE: Object = Object::new(&DEVICE_FUNCTIONS);
static ADAPTER_OBJECT: Object = Object::new(&ADAPTER_FUNCTIONS);
static FACTORY_OBJECT: Object = Object::new(&FACTORY_FUNCTIONS);
static OUTPUT_OBJECT: Object = Object::new(&OUTPUT_FUNCTIONS);

pub(crate) fn device() -> This {
    &DEVICE
}

/// `found` if `iid` is one of `known`.
fn query(kind: usize, iid: *const Iid, known: &[Iid], found: This, out: *mut This) -> HRESULT {
    object::answer(kind, iid, object::is(iid, known).then_some(found), out)
}

/// The DXGI device also answers for the D3D12 device.
extern "system" fn device_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if crate::device::is_device(iid) {
        return object::answer(DXGI_DEVICE, iid, Some(crate::device::device()), out);
    }
    query(DXGI_DEVICE, iid, &DEVICE_IIDS, this, out)
}
extern "system" fn device_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(DXGI_DEVICE, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}
extern "system" fn device_adapter(_: This, out: *mut This) -> HRESULT {
    object::answer(DXGI_DEVICE, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn adapter_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &ADAPTER_IIDS, this, out)
}
extern "system" fn adapter_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &FACTORY_IIDS, &FACTORY_OBJECT, out)
}
extern "system" fn adapter_outputs(_: This, index: u32, out: *mut This) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if index != 0 {
        // SAFETY: checked non-null; DXGI clears `*out` on `DXGI_ERROR_NOT_FOUND`.
        unsafe { out.write(std::ptr::null()) };
        return NOT_FOUND;
    }
    object::answer(ADAPTER, &OUTPUT_IIDS[0], Some(&OUTPUT_OBJECT), out)
}

#[repr(C)]
struct AdapterDescription {
    name: [u16; 128],
    vendor: u32,
    device: u32,
    subsystem: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    luid: [u32; 2],
}
const _: () = assert!(size_of::<AdapterDescription>() == 0x130);

#[repr(C)]
struct AdapterDescription1 {
    base: AdapterDescription,
    flags: u32,
}
const _: () = assert!(size_of::<AdapterDescription1>() == 0x138);

const AMD: u32 = 0x1002;

fn build_adapter_description() -> AdapterDescription {
    let mut name = [0; 128];
    for (slot, unit) in name.iter_mut().zip("VIctory console GPU".encode_utf16()) {
        *slot = unit;
    }
    // shortcut: memory sizes are the title's split (GPU-optimal, standard),
    // not measured; revisit once memory is emulated.
    let (gpu_optimal, standard) = crate::console::current().title_memory();
    AdapterDescription {
        name,
        vendor: AMD,
        device: 0,
        subsystem: 0,
        revision: 0,
        dedicated_video_memory: gpu_optimal,
        dedicated_system_memory: 0,
        shared_system_memory: standard,
        luid: [1, 0],
    }
}

extern "system" fn adapter_description(_: This, out: *mut AdapterDescription) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for the description.
    unsafe { out.write(build_adapter_description()) };
    S_OK
}

extern "system" fn adapter_description1(_: This, out: *mut AdapterDescription1) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let description = AdapterDescription1 {
        base: build_adapter_description(),
        flags: 0,
    };
    // SAFETY: checked non-null; the title passes room for `DXGI_ADAPTER_DESC1`.
    unsafe { out.write(description) };
    S_OK
}

extern "system" fn factory_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(FACTORY, iid, &FACTORY_IIDS, this, out)
}
extern "system" fn factory_adapters(_: This, index: u32, out: *mut This) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    if index != 0 {
        // SAFETY: checked non-null; DXGI clears `*out` on `DXGI_ERROR_NOT_FOUND`.
        unsafe { out.write(std::ptr::null()) };
        return NOT_FOUND;
    }
    object::answer(FACTORY, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn output_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &OUTPUT_IIDS, this, out)
}
extern "system" fn output_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}

/// IUnknown 3, object 4, device 5 + 2 + 3.
static DEVICE_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<DXGI_DEVICE, 32>();
    f[0] = device_query as *const ();
    f[6] = device_parent as *const (); // 0x30
    f[7] = device_adapter as *const (); // 0x38
    Table(f)
};

/// IUnknown 3, object 4, adapter 3 + 1 + 1.
static ADAPTER_FUNCTIONS: Table<24> = {
    let mut f = object::missing_table::<ADAPTER, 24>();
    f[0] = adapter_query as *const ();
    f[6] = adapter_parent as *const (); // 0x30
    f[7] = adapter_outputs as *const (); // 0x38
    f[8] = adapter_description as *const (); // 0x40
    f[10] = adapter_description1 as *const (); // 0x50
    Table(f)
};

/// IUnknown 3, object 4, factory 5 + 2 + 11.
static FACTORY_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<FACTORY, 32>();
    f[0] = factory_query as *const ();
    f[7] = factory_adapters as *const (); // 0x38
    f[12] = factory_adapters as *const (); // 0x60, the same for version 1
    Table(f)
};

/// A display mode as the console lists it: size, refresh rate (numerator,
/// denominator), format, scanline order, scaling, flags.
#[repr(C)]
struct Mode {
    width: u32,
    height: u32,
    refresh: [u32; 2],
    format: u32,
    scanlines: u32,
    scaling: u32,
    flags: u32,
}
const _: () = assert!(size_of::<Mode>() == 32);

/// The console's mode list: one 4K mode at 60 Hz (Grand Theft Auto VI looks
/// for a 120 Hz one to allow 120 Hz frame intervals).
/// shortcut: fixed mode, not the host display's; report the host's rates if
/// a title paces by them.
extern "system" fn modes(
    _: This,
    format: u32,
    _flags: u32,
    count: *mut u32,
    out: *mut Mode,
) -> HRESULT {
    // SAFETY: the title passes its count (null is tolerated).
    let Some(count) = (unsafe { count.as_mut() }) else {
        return E_INVALIDARG;
    };
    if out.is_null() {
        *count = 1;
        return S_OK;
    }
    if *count < 1 {
        return MORE_DATA;
    }
    *count = 1;
    let mode = Mode {
        width: 3840,
        height: 2160,
        refresh: [60, 1],
        format,
        scanlines: 1, // progressive
        scaling: 0,
        flags: 0,
    };
    // SAFETY: checked non-null; room for `count` >= 1 modes.
    unsafe { out.write(mode) };
    S_OK
}

/// IUnknown 3, object 4, output 15 (12 as on PC, then 3 console ones).
static OUTPUT_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<OUTPUT, 32>();
    f[0] = output_query as *const ();
    f[6] = output_parent as *const (); // 0x30
    f[21] = modes as *const (); // 0xa8
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, slot};

    type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
    type Get = extern "system" fn(This, *mut This) -> HRESULT;
    type Indexed = extern "system" fn(This, u32, *mut This) -> HRESULT;
    type Describe = extern "system" fn(This, *mut AdapterDescription) -> HRESULT;
    type Describe1 = extern "system" fn(This, *mut AdapterDescription1) -> HRESULT;
    type ModeList = extern "system" fn(This, u32, u32, *mut u32, *mut Mode) -> HRESULT;

    /// Minecraft's startup: device → DXGI device (newest first) → adapter →
    /// description, factory, first output.
    #[test]
    fn startup_walks_from_device_to_output() {
        let d3d = crate::device::device();
        let newest_pc = guid(
            0x6007896c,
            0x3244,
            0x4afd,
            [0xbf, 0x18, 0xa6, 0xd3, 0xbe, 0xda, 0x50, 0x23],
        );
        let mut dxgi = std::ptr::null();
        let qi = slot::<Query>(d3d, 0);
        assert_eq!(qi(d3d, &newest_pc, &mut dxgi), E_NOINTERFACE);
        assert!(dxgi.is_null());
        assert_eq!(qi(d3d, &DEVICE_IIDS[2], &mut dxgi), S_OK);
        assert_eq!(dxgi, device());

        let mut adapter = std::ptr::null();
        assert_eq!(slot::<Get>(dxgi, 0x38)(dxgi, &mut adapter), S_OK);
        let mut description = std::mem::MaybeUninit::<AdapterDescription>::uninit();
        assert_eq!(
            slot::<Describe>(adapter, 0x40)(adapter, description.as_mut_ptr()),
            S_OK
        );
        // SAFETY: written above.
        assert_eq!(unsafe { description.assume_init() }.vendor, AMD);

        let mut factory = std::ptr::null();
        let parent = slot::<Query>(adapter, 0x30);
        assert_eq!(parent(adapter, &FACTORY_IIDS[2], &mut factory), S_OK);
        let mut again = std::ptr::null();
        assert_eq!(slot::<Indexed>(factory, 0x38)(factory, 0, &mut again), S_OK);
        assert_eq!(again, adapter);

        let mut output = std::ptr::null();
        let outputs = slot::<Indexed>(adapter, 0x38);
        assert_eq!(outputs(adapter, 0, &mut output), S_OK);
        assert!(!output.is_null());
        assert_eq!(outputs(adapter, 1, &mut output), NOT_FOUND);
        assert!(output.is_null());
    }

    #[test]
    fn device_query_handles_dxgi_d3d_and_null_pointers() {
        let dxgi = device();
        let qi = slot::<Query>(dxgi, 0);

        for iid in &DEVICE_IIDS {
            let mut out = std::ptr::null();
            assert_eq!(qi(dxgi, iid, &mut out), S_OK);
            assert_eq!(out, dxgi);
        }

        // Null output pointer returns E_POINTER even for valid IIDs.
        assert_eq!(qi(dxgi, &DEVICE_IIDS[0], std::ptr::null_mut()), E_POINTER);

        // Null IID returns E_NOINTERFACE and clears output pointer.
        let mut out = dxgi;
        assert_eq!(qi(dxgi, std::ptr::null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());

        // DXGI device parent returns the adapter.
        let mut parent = std::ptr::null();
        assert_eq!(
            slot::<Query>(dxgi, 0x30)(dxgi, &ADAPTER_IIDS[1], &mut parent),
            S_OK
        );
        assert_eq!(parent, &ADAPTER_OBJECT as This);
    }

    #[test]
    fn enumerators_clear_out_on_not_found_and_reject_null() {
        let adapter: This = &ADAPTER_OBJECT;
        let factory: This = &FACTORY_OBJECT;

        let outputs = slot::<Indexed>(adapter, 0x38);
        assert_eq!(outputs(adapter, 0, std::ptr::null_mut()), E_POINTER);
        assert_eq!(outputs(adapter, 1, std::ptr::null_mut()), E_POINTER);

        let mut out = std::ptr::null();
        assert_eq!(outputs(adapter, 0, &mut out), S_OK);
        assert_eq!(out, &OUTPUT_OBJECT as This);
        assert_eq!(outputs(adapter, 1, &mut out), NOT_FOUND);
        assert!(out.is_null());

        // Both EnumAdapters (0x38) and EnumAdapters1 (0x60) behave identically.
        for offset in [0x38, 0x60] {
            let adapters = slot::<Indexed>(factory, offset);
            assert_eq!(adapters(factory, 0, std::ptr::null_mut()), E_POINTER);
            assert_eq!(adapters(factory, 2, std::ptr::null_mut()), E_POINTER);
            let mut found = std::ptr::null();
            assert_eq!(adapters(factory, 0, &mut found), S_OK);
            assert_eq!(found, adapter);
            assert_eq!(adapters(factory, 1, &mut found), NOT_FOUND);
            assert!(found.is_null());
        }
    }

    #[test]
    fn adapter_description_and_desc1_report_console_fields_and_guard_null() {
        let adapter: This = &ADAPTER_OBJECT;
        let get_desc = slot::<Describe>(adapter, 0x40);
        let get_desc1 = slot::<Describe1>(adapter, 0x50);

        assert_eq!(get_desc(adapter, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(get_desc1(adapter, std::ptr::null_mut()), E_INVALIDARG);

        let mut desc1 = std::mem::MaybeUninit::<AdapterDescription1>::uninit();
        assert_eq!(get_desc1(adapter, desc1.as_mut_ptr()), S_OK);
        // SAFETY: initialized by `get_desc1`.
        let desc1 = unsafe { desc1.assume_init() };

        let nul = desc1.base.name.iter().position(|&c| c == 0).unwrap();
        let name = String::from_utf16(&desc1.base.name[..nul]).unwrap();
        assert_eq!(name, "VIctory console GPU");
        assert_eq!(desc1.base.vendor, AMD);
        assert_eq!(desc1.base.luid, [1, 0]);
        assert_eq!(desc1.base.dedicated_system_memory, 0);
        let (gpu_optimal, standard) = crate::console::current().title_memory();
        assert_eq!(desc1.base.dedicated_video_memory, gpu_optimal);
        assert_eq!(desc1.base.shared_system_memory, standard);
        assert_eq!(desc1.flags, 0);

        #[cfg(windows)]
        {
            use windows::Win32::Graphics::Dxgi::{DXGI_ADAPTER_DESC, DXGI_ADAPTER_DESC1};
            assert_eq!(
                size_of::<AdapterDescription>(),
                size_of::<DXGI_ADAPTER_DESC>()
            );
            assert_eq!(
                size_of::<AdapterDescription1>(),
                size_of::<DXGI_ADAPTER_DESC1>()
            );
        }
    }

    #[test]
    fn output_parent_and_modes_preserve_requested_format() {
        let output: This = &OUTPUT_OBJECT;
        let mut parent = std::ptr::null();
        assert_eq!(
            slot::<Query>(output, 0x30)(output, &ADAPTER_IIDS[0], &mut parent),
            S_OK
        );
        assert_eq!(parent, &ADAPTER_OBJECT as This);

        let get_modes = slot::<ModeList>(output, 0xa8);
        const R8G8B8A8_UNORM: u32 = 28;

        // Null count pointer fails with E_INVALIDARG.
        assert_eq!(
            get_modes(output, R8G8B8A8_UNORM, 0, std::ptr::null_mut(), std::ptr::null_mut()),
            E_INVALIDARG
        );

        // Null output buffer queries mode count.
        let mut count = 0;
        assert_eq!(
            get_modes(output, R8G8B8A8_UNORM, 0, &mut count, std::ptr::null_mut()),
            S_OK
        );
        assert_eq!(count, 1);

        // Zero-capacity output buffer returns MORE_DATA.
        let mut mode = std::mem::MaybeUninit::<Mode>::uninit();
        let mut zero = 0;
        assert_eq!(
            get_modes(output, R8G8B8A8_UNORM, 0, &mut zero, mode.as_mut_ptr()),
            MORE_DATA
        );

        // Sufficient capacity populates 4K@60 progressive with the requested format.
        let mut cap = 4;
        assert_eq!(
            get_modes(output, R8G8B8A8_UNORM, 0, &mut cap, mode.as_mut_ptr()),
            S_OK
        );
        assert_eq!(cap, 1);
        // SAFETY: written above.
        let mode = unsafe { mode.assume_init() };
        assert_eq!(mode.width, 3840);
        assert_eq!(mode.height, 2160);
        assert_eq!(mode.refresh, [60, 1]);
        assert_eq!(mode.format, R8G8B8A8_UNORM);
        assert_eq!(mode.scanlines, 1);
    }
}
