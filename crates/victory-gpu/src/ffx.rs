//! The title's FidelityFX upscaler calls answered by AMD's PC FidelityFX
//! loader for D3D12, `amd_fidelityfx_loader_dx12.dll` from the public
//! FidelityFX SDK (2.x, `Kits/FidelityFX/signedbin`), in the folder
//! `VICTORY_FFX_DIR` names.
//!
//! The console's FidelityFX API has the PC API's structure layouts. Its
//! upscaler DLL only carries the fixed 1920x1080 Quality tables: a Series X
//! configuration made to pick it (3840x2160 output) gets "maxUpscaleSize
//! must be 1920x1080" and the title stops. Here the title's descriptions go
//! to the PC loader with the console objects swapped for the host's: the
//! console backend description for the DX12 one on the host device, the
//! command list for the host list behind the title's, and every texture for
//! its host copy, moved to the state the upscaler reads (inputs) or writes
//! (output) on entry. The PC upscaler leaves them in those states.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{Interface, PCSTR, PCWSTR};

use crate::object::This;

pub(crate) mod framegen;
pub(crate) mod rgb9e5;

type ReturnCode = u32;
const OK: ReturnCode = 0;
const ERROR: ReturnCode = 1;
const ERROR_UNKNOWN_DESC: ReturnCode = 2;
const ERROR_PARAMETER: ReturnCode = 6;

/// `ffxApiHeader`: a description's type and the next in its chain.
#[repr(C)]
pub struct Header {
    kind: u64,
    next: *mut Header,
}

const CREATE_UPSCALE: u64 = 0x0001_0000;
const UPSCALE_VERSION: u64 = 0x0001_000b;
const SDK_UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 2;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const QUERY_GET_VERSIONS: u64 = 4;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
/// The console's backend description; the device is at +0x10, as in the
/// PC's DX12 one.
const CONSOLE_BACKEND: u64 = 0x0100_0002;
const BACKEND_DX12: u64 = 2;

/// `ffxCreateBackendDX12Desc`.
#[repr(C)]
struct BackendDx12 {
    header: Header,
    device: *mut c_void,
}

/// `ffxCreateContextDescUpscale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    kind: u64,
    next: *mut Header,
    flags: u32,
    max_render: [u32; 2],
    max_upscale: [u32; 2],
    message: *const c_void,
}
const _: () = assert!(size_of::<CreateUpscale>() == 48);

/// `FfxApiResource`: the resource, its description, its state.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}
const _: () = assert!(size_of::<Resource>() == 48);

/// `ffxDispatchDescUpscale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    kind: u64,
    next: *mut Header,
    command_list: *mut c_void,
    /// Color, depth, motion vectors, exposure, reactive, transparency and
    /// composition, output.
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_vector_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera: [f32; 4],
    flags: u32,
}
const _: () = assert!(size_of::<DispatchUpscale>() == 432);
const OUTPUT: usize = 6;

/// `ffxQueryDescGetVersions`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GetVersions {
    kind: u64,
    next: *mut Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const i8,
}

/// FfxApiResourceState bits and the host states they stand for.
const FFX_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_COMPUTE_READ: u32 = 1 << 2;
const HOST_UNORDERED_ACCESS: u32 = 0x8;
const HOST_NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;

type Farproc = unsafe extern "system" fn() -> isize;
type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> ReturnCode;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> ReturnCode;
type Configure = unsafe extern "C" fn(*mut *mut c_void, *const Header) -> ReturnCode;
type Query = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> ReturnCode;
type Dispatch = unsafe extern "C" fn(*mut *mut c_void, *const Header) -> ReturnCode;

struct Api {
    create: Create,
    destroy: Destroy,
    configure: Configure,
    query: Query,
    dispatch: Dispatch,
}

/// Whether the title's FidelityFX calls come here: `VICTORY_FFX_DIR` set.
pub fn enabled() -> bool {
    std::env::var_os("VICTORY_FFX_DIR").is_some()
}

/// AMD's loader, loaded on first use. The upscaler it picks is loaded by
/// full path first, so the loader finds it without a change to the DLL
/// search path.
fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("VICTORY_FFX_DIR")?);
        let load = |name: &str| {
            let path: Vec<u16> = dir.join(name).as_os_str().encode_wide_nul();
            // SAFETY: a nul-terminated path.
            unsafe { LoadLibraryW(PCWSTR(path.as_ptr())) }.ok()
        };
        let _ = load("amd_fidelityfx_upscaler_dx12.dll");
        if framegen::enabled() {
            let _ = load("amd_fidelityfx_framegeneration_dx12.dll");
        }
        let Some(loader) = load("amd_fidelityfx_loader_dx12.dll") else {
            tracing::error!(?dir, "ffx: amd_fidelityfx_loader_dx12.dll not found; set VICTORY_FFX_DIR to the SDK's signedbin");
            return None;
        };
        let get = |name: &[u8]| {
            // SAFETY: a loaded module and a nul-terminated name.
            unsafe { GetProcAddress(loader, PCSTR(name.as_ptr())) }
        };
        // SAFETY: the SDK's exports, of these types (`ffx_api.h`).
        unsafe {
            Some(Api {
                create: std::mem::transmute::<Farproc, Create>(get(b"ffxCreateContext\0")?),
                destroy: std::mem::transmute::<Farproc, Destroy>(get(b"ffxDestroyContext\0")?),
                configure: std::mem::transmute::<Farproc, Configure>(get(b"ffxConfigure\0")?),
                query: std::mem::transmute::<Farproc, Query>(get(b"ffxQuery\0")?),
                dispatch: std::mem::transmute::<Farproc, Dispatch>(get(b"ffxDispatch\0")?),
            })
        }
    })
    .as_ref()
}

trait WideNul {
    fn encode_wide_nul(&self) -> Vec<u16>;
}

impl WideNul for std::ffi::OsStr {
    fn encode_wide_nul(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(Some(0)).collect()
    }
}

fn host_device() -> Option<*mut c_void> {
    crate::host::host().map(|h| h.device.as_raw())
}

/// Logs a description type the first time it is seen.
fn note(call: &str, kind: u64) {
    static SEEN: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|(c, k)| c == call && *k == kind) {
        seen.push((call.to_owned(), kind));
        tracing::info!("ffx: {call} with description type {kind:#x}");
    }
}

pub extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    _memory: *const c_void,
) -> ReturnCode {
    let (Some(api), Some(device)) = (api(), host_device()) else {
        return ERROR;
    };
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    let mut upscale = None;
    let mut backend = false;
    let mut h = desc;
    while !h.is_null() {
        // SAFETY: the title's chain of descriptions.
        let kind = unsafe { (*h).kind };
        match kind {
            // SAFETY: a description of this type.
            CREATE_UPSCALE => upscale = Some(unsafe { *h.cast::<CreateUpscale>() }),
            CONSOLE_BACKEND => backend = true,
            _ => note("ffxCreateContext", kind),
        }
        // SAFETY: as above.
        h = unsafe { (*h).next };
    }
    let (Some(mut upscale), true) = (upscale, backend) else {
        return ERROR_PARAMETER;
    };
    // The console's version and provider descriptions name console builds;
    // the PC loader picks the newest upscaler the GPU runs.
    let mut dx12 = BackendDx12 {
        header: Header {
            kind: BACKEND_DX12,
            next: std::ptr::null_mut(),
        },
        device,
    };
    #[repr(C)]
    struct Version {
        header: Header,
        version: u32,
    }
    let mut version = Version {
        header: Header { kind: UPSCALE_VERSION, next: &mut dx12.header },
        version: SDK_UPSCALER_VERSION,
    };
    if framegen::enabled() {
        static PROVIDERS: std::sync::Once = std::sync::Once::new();
        PROVIDERS.call_once(|| {
            log_versions(api, device, CREATE_UPSCALE, "upscaling");
            log_versions(api, device, 0x20001, "frame generation");
        });
    }
    upscale.next = &mut version.header;
    // SAFETY: the SDK's call with a chain that outlives it.
    let mut r = unsafe { (api.create)(context, (&raw mut upscale).cast(), std::ptr::null()) };
    // SAFETY: the caller's context slot is non-null and the SDK has returned from create.
    let retry_legacy = r == ERROR_UNKNOWN_DESC && unsafe { (*context).is_null() };
    if retry_legacy {
        upscale.next = &mut dx12.header;
        // SAFETY: this SDK rejected the version descriptor without constructing a context.
        r = unsafe { (api.create)(context, (&raw mut upscale).cast(), std::ptr::null()) };
    }
    if r == OK && framegen::enabled() {
        // SAFETY: a successful create wrote the handle.
        framegen::register(unsafe { *context }, upscale.flags);
    }
    let provider = provider_name(context);
    tracing::info!(
        render = ?upscale.max_render, upscale = ?upscale.max_upscale, flags = upscale.flags, result = r, provider,
        "ffx: upscale context on the host GPU"
    );
    r
}

fn log_versions(api: &Api, device: *mut c_void, kind: u64, effect: &str) {
    let mut count = 0;
    let mut query = GetVersions {
        kind: QUERY_GET_VERSIONS,
        next: std::ptr::null_mut(),
        create_kind: kind,
        device,
        count: &mut count,
        ids: std::ptr::null_mut(),
        names: std::ptr::null_mut(),
    };
    // SAFETY: the documented contextless version query writes the required array length.
    let status = unsafe { (api.query)(std::ptr::null_mut(), (&raw mut query).cast()) };
    if status != OK || count == 0 || count > 64 {
        tracing::info!(effect, count, status, "ffx: provider enumeration unavailable");
        return;
    }
    let mut names = vec![std::ptr::null::<i8>(); count as usize];
    query.names = names.as_mut_ptr();
    // SAFETY: the SDK fills up to the count it returned, and names are its own strings.
    let status = unsafe { (api.query)(std::ptr::null_mut(), (&raw mut query).cast()) };
    if status == OK {
        let providers: Vec<_> = names
            .into_iter()
            .take(count as usize)
            .filter(|p| !p.is_null())
            .map(|p| {
                // SAFETY: the successful query returned a nul-terminated provider name.
                unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
            })
            .collect();
        tracing::info!(effect, ?providers, "ffx: available PC providers");
    }
}

/// The upscaler version the loader chose for a context, for the log.
fn provider_name(context: *mut *mut c_void) -> String {
    #[repr(C)]
    struct ProviderVersion {
        kind: u64,
        next: *mut Header,
        id: u64,
        name: *const i8,
    }
    let Some(api) = api() else {
        return String::new();
    };
    let mut q = ProviderVersion {
        kind: 6,
        next: std::ptr::null_mut(),
        id: 0,
        name: std::ptr::null(),
    };
    // SAFETY: the context pointer the title passed, checked for null.
    if context.is_null() || unsafe { (*context).is_null() } {
        return String::new();
    }
    // SAFETY: a context the loader created.
    if unsafe { (api.query)(context, (&raw mut q).cast()) } != OK || q.name.is_null() {
        return String::new();
    }
    // SAFETY: the loader's nul-terminated name.
    unsafe { std::ffi::CStr::from_ptr(q.name) }
        .to_string_lossy()
        .into_owned()
}

pub extern "C" fn ffxDestroyContext(
    context: *mut *mut c_void,
    _memory: *const c_void,
) -> ReturnCode {
    let Some(api) = api() else { return ERROR };
    if framegen::enabled() && !context.is_null() {
        // SAFETY: the caller owns a live context slot.
        framegen::forget(unsafe { *context });
    }
    // SAFETY: a context this module created.
    unsafe { (api.destroy)(context, std::ptr::null()) }
}

pub extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *const Header) -> ReturnCode {
    let Some(api) = api() else { return ERROR };
    if !desc.is_null() {
        // SAFETY: the title's description.
        note("ffxConfigure", unsafe { (*desc).kind });
    }
    // SAFETY: passed on as the title made it.
    unsafe { (api.configure)(context, desc) }
}

pub extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> ReturnCode {
    let (Some(api), Some(device)) = (api(), host_device()) else {
        return ERROR;
    };
    if desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: the title's description.
    let kind = unsafe { (*desc).kind };
    note("ffxQuery", kind);
    if kind == QUERY_GET_VERSIONS {
        // It names the device to enumerate for: the host's.
        // SAFETY: a description of this type.
        let mut q = unsafe { *desc.cast::<GetVersions>() };
        q.device = device;
        // SAFETY: the SDK's call; the title's output arrays stay its own.
        return unsafe { (api.query)(context, (&raw mut q).cast()) };
    }
    // SAFETY: a null context pointer or one this module filled.
    let contextless = context.is_null() || unsafe { (*context).is_null() };
    if !contextless {
        // SAFETY: passed on with the PC context.
        let r = unsafe { (api.query)(context, desc) };
        log_query(kind, desc, r);
        return r;
    }
    // Without a context the loader learns the backend from the chain: the
    // DX12 one is added at its end for the call, then taken off.
    let mut dx12 = BackendDx12 {
        header: Header {
            kind: BACKEND_DX12,
            next: std::ptr::null_mut(),
        },
        device,
    };
    let mut last = desc;
    // SAFETY: the title's chain.
    while !unsafe { (*last).next }.is_null() {
        // SAFETY: as above.
        last = unsafe { (*last).next };
    }
    // SAFETY: the chain's end, restored before returning.
    unsafe { (*last).next = (&raw mut dx12).cast() };
    // SAFETY: the title's description with the backend chained for the call.
    let r = unsafe { (api.query)(context, desc) };
    // SAFETY: the chain's end again, as the title made it.
    unsafe { (*last).next = std::ptr::null_mut() };
    log_query(kind, desc, r);
    r
}

/// A query's result, once per type, with the jitter phase count it gave.
fn log_query(kind: u64, desc: *mut Header, r: ReturnCode) {
    static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&kind) {
        return;
    }
    seen.push(kind);
    #[repr(C)]
    struct JitterPhaseCount {
        kind: u64,
        next: *mut Header,
        render_width: u32,
        display_width: u32,
        out: *mut i32,
    }
    if kind == QUERY_JITTER_PHASE_COUNT {
        // SAFETY: a description of this type.
        let q = unsafe { &*desc.cast::<JitterPhaseCount>() };
        // SAFETY: the title's output, written by the call when it succeeded.
        let count = (r == OK && !q.out.is_null()).then(|| unsafe { *q.out });
        tracing::info!(
            r,
            q.render_width,
            q.display_width,
            ?count,
            "ffx: jitter phase count"
        );
    } else {
        tracing::info!(r, "ffx: query {kind:#x}");
    }
}

pub extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *const Header) -> ReturnCode {
    let Some(api) = api() else { return ERROR };
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: the title's description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        note("ffxDispatch", kind);
        return ERROR_PARAMETER;
    }
    // SAFETY: a description of this type.
    let mut d = unsafe { *desc.cast::<DispatchUpscale>() };
    let title_list = d.command_list as This;
    let Some(list) = crate::list::host_list(title_list) else {
        return ERROR_PARAMETER;
    };
    let geometry = framegen::enabled().then(|| {
        (
            crate::resource::host_copy(d.resources[1].resource as This),
            crate::resource::host_copy(d.resources[2].resource as This),
        )
    });
    let output = d.resources[OUTPUT].resource as This;
    let input = d.resources[0].resource as This;
    let Some(destination) = crate::resource::host_resource(output) else {
        missing_texture(OUTPUT, output);
        return ERROR_PARAMETER;
    };
    let packed_output = if crate::host::legal_state(destination, HOST_UNORDERED_ACCESS)
        != HOST_UNORDERED_ACCESS
    {
        let Some(packed) = crate::resource::rgb9e5_fsr_output(output)
            .filter(|packed| packed.matches(d.upscale_size))
        else {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                tracing::warn!(?output, size = ?d.upscale_size, "ffx: output is not writable and has no matching RGB9E5 fallback");
            });
            return ERROR_PARAMETER;
        };
        Some(packed)
    } else {
        None
    };
    for (i, r) in d.resources.iter_mut().enumerate() {
        if i == OUTPUT && let Some(packed) = packed_output {
            *r = packed.sdk_resource(list);
            continue;
        }
        if r.resource.is_null() {
            continue;
        }
        let title = r.resource as This;
        let Some(texture) = crate::resource::host_resource(title) else {
            missing_texture(i, title);
            r.resource = std::ptr::null_mut();
            continue;
        };
        let (wanted, ffx) = if i == OUTPUT {
            (HOST_UNORDERED_ACCESS, FFX_UNORDERED_ACCESS)
        } else {
            (HOST_NON_PIXEL_SHADER_RESOURCE, FFX_COMPUTE_READ)
        };
        crate::list::ensure_state(list, title, wanted);
        if crate::host::legal_state(texture, wanted) != wanted {
            tracing::warn!(
                slot = i,
                wanted,
                "ffx: the host texture cannot take the upscaler's state"
            );
        }
        r.resource = texture.as_raw();
        r.state = ffx;
    }
    if d.resources[0].resource.is_null() || d.resources[OUTPUT].resource.is_null() {
        return ERROR_PARAMETER;
    }
    d.command_list = list.list.as_raw();
    // SAFETY: the SDK's call on an open host list with live textures.
    let r = unsafe { (api.dispatch)(context, (&raw const d).cast()) };
    if r == OK {
        if let Some(packed) = packed_output {
            packed.copy_to(list, output, destination);
            static FIRST_PACK: std::sync::Once = std::sync::Once::new();
            FIRST_PACK.call_once(|| {
                tracing::info!(?output, size = ?d.upscale_size, "ffx: recorded RGB9E5 output conversion");
            });
        }
        crate::texture::host_written(output);
        #[cfg(feature = "diagnostics")]
        {
            crate::capture::note_target(input);
            crate::capture::note_target(output);
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
            let output_state = if packed_output.is_some() {
                D3D12_RESOURCE_STATE_COPY_DEST.0 as u32
            } else {
                HOST_UNORDERED_ACCESS
            };
            crate::capture::record_fsr(list, input, output, output_state);
        }
        if let Some((Some(depth), Some(motion))) = geometry {
            // SAFETY: the SDK just accepted this context.
            framegen::record(list, unsafe { *context }, &d, depth, motion);
        }
    }
    // The upscaler bound its own heaps, root signature and pipeline.
    crate::list::forget_host_bindings(title_list);
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        // SAFETY: the destination is a live host texture.
        let format = unsafe { destination.GetDesc() }.Format.0;
        tracing::info!(render = ?d.render_size, upscale = ?d.upscale_size, result = r, ?input, ?output, format, packed = packed_output.is_some(), "ffx: first upscale recorded on the host GPU");
    });
    r
}

fn missing_texture(slot: usize, title: This) {
    use std::sync::atomic::{AtomicU8, Ordering};
    static REPORTED: AtomicU8 = AtomicU8::new(0);
    let bit = 1 << slot;
    if REPORTED.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        let (title_address, desc) = crate::resource::texture_memory(title)
            .map_or((0, None), |(address, desc)| (address as usize, Some(desc)));
        tracing::warn!(slot, ?title, title_address, ?desc, "ffx: an upscale resource without a host texture");
    }
}
