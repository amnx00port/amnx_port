//! Experimental FidelityFX frame generation for presents after a title upscale.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
use windows::Win32::Graphics::Dxgi::{
    DXGI_SWAP_CHAIN_DESC1, IDXGIFactory2, IDXGISwapChain3, IDXGISwapChain4,
};
use windows::core::Interface;

use super::{
    Api, BACKEND_DX12, BackendDx12, DispatchUpscale, HOST_NON_PIXEL_SHADER_RESOURCE, Header, OK,
    Resource,
};
use crate::host::HostList;
use crate::resource::HostTexture;

const CREATE: u64 = 0x20001;
const CONFIGURE: u64 = 0x20002;
const PREPARE: u64 = 0x2000c;
const VERSION: u64 = 0x2000e;
const SWAPCHAIN_CREATE: u64 = 0x30006;
const SWAPCHAIN_WAIT: u64 = 0x30007;
const SWAPCHAIN_VERSION: u64 = 0x3000b;
const RGBA8_UNORM: u32 = 10;
// Both effect APIs assign the same bits to these four inputs.
const MOTION_DEPTH_FLAGS: u32 = 0b11110;

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        cfg!(feature = "diagnostics")
            && std::env::var_os("VICTORY_FFX_FRAMEGEN").is_some_and(|s| s == "1")
    })
}

fn contexts() -> &'static Mutex<HashMap<usize, u32>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, u32>>> = OnceLock::new();
    CONTEXTS.get_or_init(Mutex::default)
}

pub(super) fn register(handle: *mut c_void, flags: u32) {
    contexts()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle as usize, flags);
}

pub(super) fn forget(handle: *mut c_void) {
    contexts()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&(handle as usize));
}

pub(crate) struct Inputs {
    depth: Arc<HostTexture>,
    depth_desc: [u32; 8],
    motion: Arc<HostTexture>,
    motion_desc: [u32; 8],
    render: [u32; 2],
    jitter: [f32; 2],
    scale: [f32; 2],
    time_ms: f32,
    camera: [f32; 4],
    reset: bool,
    flags: u32,
}

pub(super) fn record(
    list: &HostList,
    ctx: *mut c_void,
    upscale: &DispatchUpscale,
    depth: Arc<HostTexture>,
    motion: Arc<HostTexture>,
) {
    let Some(flags) = contexts()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(ctx as usize))
        .copied()
    else {
        return;
    };
    if upscale.resources[1].resource.is_null() || upscale.resources[2].resource.is_null() {
        return;
    }
    *list
        .framegen_inputs
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(Inputs {
        depth,
        depth_desc: upscale.resources[1].description,
        motion,
        motion_desc: upscale.resources[2].description,
        render: upscale.render_size,
        jitter: upscale.jitter,
        scale: upscale.motion_vector_scale,
        time_ms: upscale.frame_time_delta,
        camera: upscale.camera,
        reset: upscale.reset,
        flags: flags & MOTION_DEPTH_FLAGS,
    });
}

static SUBMITTED: Mutex<Option<Inputs>> = Mutex::new(None);

pub(crate) fn submitted(runs: &[&HostList]) {
    let mut latest = None;
    for list in runs {
        if let Some(inputs) = list
            .framegen_inputs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            latest = Some(inputs);
        }
    }
    if latest.is_some() {
        *SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()) = latest;
    }
}

pub(crate) fn take_inputs() -> Option<Inputs> {
    SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[repr(C)]
struct Version {
    header: Header,
    version: u32,
}

#[repr(C)]
struct SwapchainDesc {
    header: Header,
    output: *mut *mut c_void,
    hwnd: HWND,
    desc: *const DXGI_SWAP_CHAIN_DESC1,
    fullscreen_desc: *const c_void,
    factory: *mut c_void,
    queue: *mut c_void,
}

#[repr(C)]
struct Create {
    header: Header,
    flags: u32,
    display: [u32; 2],
    max_render: [u32; 2],
    format: u32,
}

type Callback = unsafe extern "C" fn(*mut Generate, *mut c_void) -> u32;

#[repr(C)]
struct Configure {
    header: Header,
    swapchain: *mut c_void,
    present_callback: *const c_void,
    present_user: *mut c_void,
    generate_callback: Option<Callback>,
    generate_user: *mut c_void,
    enabled: bool,
    allow_async: bool,
    hudless: Resource,
    flags: u32,
    only_generated: bool,
    rect: [i32; 4],
    id: u64,
}

#[repr(C)]
struct Prepare {
    header: Header,
    id: u64,
    flags: u32,
    list: *mut c_void,
    render: [u32; 2],
    jitter: [f32; 2],
    scale: [f32; 2],
    time_ms: f32,
    reset: bool,
    camera: [f32; 4],
    depth: Resource,
    motion: Resource,
    camera_position: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_forward: [f32; 3],
}

#[repr(C)]
struct Generate {
    header: Header,
    list: *mut c_void,
    color: Resource,
    output: [Resource; 4],
    count: u32,
}

const _: () = {
    assert!(std::mem::size_of::<SwapchainDesc>() == 64);
    assert!(std::mem::size_of::<Create>() == 40);
    assert!(std::mem::size_of::<Configure>() == 144);
    assert!(std::mem::size_of::<Prepare>() == 232);
    assert!(std::mem::offset_of!(Generate, count) == 264);
};

fn header(kind: u64) -> Header {
    Header {
        kind,
        next: null_mut(),
    }
}

struct Context {
    api: &'static Api,
    handle: *mut c_void,
}

// SAFETY: the presenter mutex serializes use of all contexts, including callbacks made during Present.
unsafe impl Send for Context {}

impl Context {
    fn create(api: &'static Api, desc: &mut Header) -> Result<Self, u32> {
        let mut ctx = Self {
            api,
            handle: null_mut(),
        };
        // SAFETY: the typed descriptor chain and handle slot live through this call.
        let code = unsafe { (api.create)(&mut ctx.handle, desc, std::ptr::null()) };
        if code == OK && !ctx.handle.is_null() {
            Ok(ctx)
        } else {
            Err(code)
        }
    }

    fn configure(&self, desc: &Configure) -> u32 {
        // SAFETY: our context and the matching descriptor.
        unsafe { (self.api.configure)(&mut { self.handle }, &desc.header) }
    }

    fn dispatch(&self, desc: &Header) -> u32 {
        // SAFETY: our context and the matching descriptor.
        unsafe { (self.api.dispatch)(&mut { self.handle }, desc) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: exclusively owned and no presents are in flight when a generator is released.
            unsafe { (self.api.destroy)(&mut self.handle, std::ptr::null()) };
        }
    }
}

struct Generator {
    context: Context,
    generated: AtomicU64,
    error: AtomicU32,
}

unsafe extern "C" fn generate(desc: *mut Generate, user: *mut c_void) -> u32 {
    // SAFETY: the configured callback receives our boxed generator and an SDK-owned dispatch descriptor.
    let g = unsafe { &*user.cast::<Generator>() };
    // SAFETY: the SDK may update the generated-frame count through this live descriptor.
    let code = unsafe { (g.context.api.dispatch)(&mut { g.context.handle }, desc.cast()) };
    if code == OK {
        // SAFETY: the descriptor is still live for this callback.
        let count = unsafe { std::ptr::addr_of!((*desc).count).read() };
        g.generated.fetch_add(u64::from(count), Ordering::Relaxed);
    } else {
        g.error.store(code, Ordering::Relaxed);
    }
    code
}

pub(crate) struct Presentation {
    api: &'static Api,
    proxy: Context,
    generator: Option<Box<Generator>>,
    key: Option<([u32; 2], [u32; 2], u32)>,
    armed: bool,
    next_id: u64,
    continuous: u64,
    last_frame: u64,
    base_presents: u32,
    logged: bool,
}

impl Presentation {
    pub(crate) fn create(
        factory: &IDXGIFactory2,
        queue: &ID3D12CommandQueue,
        hwnd: HWND,
        desc: &DXGI_SWAP_CHAIN_DESC1,
    ) -> Option<(IDXGISwapChain3, Self)> {
        if !enabled() || desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM {
            return None;
        }
        let api = super::api()?;
        let mut version = Version {
            header: header(SWAPCHAIN_VERSION),
            version: 4 << 22,
        };
        let mut output = null_mut();
        let mut args = SwapchainDesc {
            header: Header {
                kind: SWAPCHAIN_CREATE,
                next: &mut version.header,
            },
            output: &mut output,
            hwnd,
            desc,
            fullscreen_desc: std::ptr::null(),
            factory: factory.as_raw(),
            queue: queue.as_raw(),
        };
        let proxy = Context::create(api, &mut args.header)
            .inspect_err(|code| {
                tracing::warn!(
                    code,
                    "ffx framegen: swapchain unavailable; using normal present"
                );
            })
            .ok()?;
        if output.is_null() {
            return None;
        }
        // SAFETY: the SDK returned an owned IDXGISwapChain4 reference.
        let swapchain = unsafe { IDXGISwapChain4::from_raw(output) }.cast().ok()?;
        tracing::info!("ffx framegen: proxy swapchain created");
        Some((
            swapchain,
            Self {
                api,
                proxy,
                generator: None,
                key: None,
                armed: false,
                next_id: 0,
                continuous: 0,
                last_frame: 0,
                base_presents: 0,
                logged: false,
            },
        ))
    }

    pub(crate) fn disable(&mut self, chain: &IDXGISwapChain3) {
        if !self.armed {
            return;
        }
        let Some(generator) = &self.generator else {
            return;
        };
        let code = generator
            .context
            .configure(&self.config(chain, false, [0; 2]));
        if code != OK {
            tracing::warn!(code, "ffx framegen: disable failed");
        }
        let _ = self.proxy.dispatch(&header(SWAPCHAIN_WAIT));
        self.armed = false;
    }

    fn config(&self, chain: &IDXGISwapChain3, armed: bool, display: [u32; 2]) -> Configure {
        let callback = self.generator.as_deref().filter(|_| armed);
        Configure {
            header: header(CONFIGURE),
            swapchain: chain.as_raw(),
            present_callback: std::ptr::null(),
            present_user: null_mut(),
            generate_callback: callback.map(|_| generate as Callback),
            generate_user: callback
                .map_or(null_mut(), |g| (g as *const Generator).cast_mut().cast()),
            enabled: armed,
            allow_async: false,
            hudless: Resource {
                resource: null_mut(),
                description: [0; 8],
                state: 0,
            },
            flags: 0,
            only_generated: false,
            rect: [0, 0, display[0] as i32, display[1] as i32],
            id: self.next_id,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        chain: &IDXGISwapChain3,
        device: &ID3D12Device,
        list: &HostList,
        display_size: [u32; 2],
        frame: u64,
        inputs: Option<Inputs>,
    ) -> bool {
        let Some(input) = inputs.filter(|i| {
            i.render
                .into_iter()
                .zip(display_size)
                .all(|(r, d)| r > 0 && r <= d)
        }) else {
            self.disable(chain);
            return false;
        };
        let key = (display_size, input.render, input.flags);
        if self.key != Some(key) {
            self.disable(chain);
            self.generator = None;
            self.key = Some(key);
            self.next_id = 0;
            self.continuous = 0;
            let _ = self.proxy.dispatch(&header(SWAPCHAIN_WAIT));
            // SAFETY: the proxy swapchain is live and serialized by the presenter.
            self.base_presents = unsafe { chain.GetLastPresentCount() }.unwrap_or(0);
            let mut backend = BackendDx12 {
                header: header(BACKEND_DX12),
                device: device.as_raw(),
            };
            let mut version = Version {
                header: Header {
                    kind: VERSION,
                    next: &mut backend.header,
                },
                version: (4 << 22) | 2,
            };
            let mut args = Create {
                header: Header {
                    kind: CREATE,
                    next: &mut version.header,
                },
                flags: input.flags,
                display: display_size,
                max_render: input.render,
                format: RGBA8_UNORM,
            };
            match Context::create(self.api, &mut args.header) {
                Ok(context) => {
                    let provider = super::provider_name(&mut { context.handle });
                    // The title upscale description has no view matrix. The FSR 3 provider uses
                    // depth, motion vectors and scalar camera values, and does not consume it.
                    if provider.starts_with("3.") {
                        tracing::info!(provider, ?display_size, render = ?input.render, "ffx framegen: game context created");
                        self.generator = Some(Box::new(Generator {
                            context,
                            generated: AtomicU64::new(0),
                            error: AtomicU32::new(0),
                        }));
                    } else {
                        tracing::warn!(
                            provider,
                            "ffx framegen: provider requires game view matrices"
                        );
                    }
                }
                Err(code) => {
                    tracing::warn!(code, ?display_size, render = ?input.render, "ffx framegen: game context failed")
                }
            }
        }
        let Some(g) = &self.generator else {
            return false;
        };
        let config = self.config(chain, true, display_size);
        let code = g.context.configure(&config);
        if code != OK {
            tracing::warn!(code, "ffx framegen: configure failed");
            self.disable(chain);
            return false;
        }
        let prepare = Prepare {
            header: header(PREPARE),
            id: self.next_id,
            flags: 0,
            list: list.list.as_raw(),
            render: input.render,
            jitter: input.jitter,
            scale: input.scale,
            time_ms: input.time_ms,
            reset: input.reset || !self.armed || self.last_frame + 1 != frame,
            camera: input.camera,
            depth: read(list, &input.depth, input.depth_desc),
            motion: read(list, &input.motion, input.motion_desc),
            camera_position: [0.0; 3],
            camera_up: [0., 1., 0.],
            camera_right: [1., 0., 0.],
            camera_forward: [0., 0., 1.],
        };
        self.armed = true;
        let code = g.context.dispatch(&prepare.header);
        if code != OK {
            tracing::warn!(code, "ffx framegen: prepare failed");
            self.disable(chain);
            return false;
        }
        self.last_frame = frame;
        self.next_id += 1;
        self.continuous += u64::from(!prepare.reset);
        if self.next_id == 1 {
            tracing::info!(frame, render = ?input.render, ?display_size, "ffx framegen: first game depth and motion submitted for presentation");
        }
        true
    }

    pub(crate) fn presented(&mut self, chain: &IDXGISwapChain3, frame: u64) {
        let Some(g) = &self.generator else { return };
        let error = g.error.load(Ordering::Relaxed);
        if error != OK {
            tracing::warn!(error, "ffx framegen: interpolation callback failed");
            self.disable(chain);
        } else if !self.logged && self.continuous >= 2 && g.generated.load(Ordering::Relaxed) >= 3 {
            let result = self.proxy.dispatch(&header(SWAPCHAIN_WAIT));
            // SAFETY: a live proxy swapchain with completed presents.
            let presents = unsafe { chain.GetLastPresentCount() }
                .unwrap_or(0)
                .saturating_sub(self.base_presents);
            let generated = g.generated.load(Ordering::Relaxed);
            tracing::info!(
                frame,
                rendered = self.next_id,
                continuous = self.continuous,
                generated,
                presents,
                result,
                "ffx framegen: generated game frames"
            );
            self.logged = true;
        }
    }
}

fn read(list: &HostList, copy: &Arc<HostTexture>, description: [u32; 8]) -> Resource {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES;
    if let Some(before) = list.transition_copy(copy, HOST_NON_PIXEL_SHADER_RESOURCE) {
        let barrier = crate::host::transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(HOST_NON_PIXEL_SHADER_RESOURCE as i32),
        );
        // SAFETY: an open list and a texture retained by its tracker.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    Resource {
        resource: copy.texture.as_raw(),
        description,
        state: super::FFX_COMPUTE_READ,
    }
}
