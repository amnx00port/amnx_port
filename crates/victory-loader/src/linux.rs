//! Linux native PE loader and runtime environment.

use std::collections::HashSet;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use victory_pe::{ImageInfo, ImportSymbol};

use crate::thunks::{ImportSite, ThunkBlock, UnimplementedPolicy};
use crate::{
    DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, ImportResolver, PreparedImage, Provenance, Resolution,
    prepare,
};

/// Linux syscall number for arch_prctl on x86_64.
const ARCH_SET_GS: libc::c_int = 0x1001;

/// Simulated Windows Thread Environment Block (TEB) for x86_64.
#[repr(C, align(64))]
pub struct SimulatedTeb {
    pub exception_list: usize,                  // 0x00
    pub stack_base: usize,                      // 0x08
    pub stack_limit: usize,                     // 0x10
    pub sub_system_tib: usize,                  // 0x18
    pub fiber_data: usize,                      // 0x20
    pub arbitrary_user_pointer: usize,          // 0x28
    pub self_ptr: usize,                        // 0x30
    pub environment_pointer: usize,             // 0x38
    pub process_id: usize,                      // 0x40 (ClientId.UniqueProcess)
    pub thread_id: usize,                       // 0x48 (ClientId.UniqueThread)
    pub active_rpc_handle: usize,               // 0x50
    pub thread_local_storage_pointer: *mut usize, // 0x58
    pub peb: usize,                             // 0x60
}

/// Global list of mapped images.
static NEXT_DLL_TLS_INDEX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Returns the emulated TEB for the current thread if one has been installed.
pub fn current_teb() -> Option<*mut SimulatedTeb> {
    let ptr: usize;
    // SAFETY: reading gs:[0x30] on x86_64 retrieves the emulated TEB self pointer if set.
    unsafe {
        std::arch::asm!("mov {0}, gs:[0x30]", out(reg) ptr);
    }
    if ptr != 0 {
        Some(ptr as *mut SimulatedTeb)
    } else {
        None
    }
}

static MAPPED_IMAGES: Mutex<Vec<Arc<LoadedImage>>> = Mutex::new(Vec::new());

pub(crate) fn register_mapped(image: Arc<LoadedImage>) {
    let mut images = MAPPED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    images.retain(|m| m.base != image.base);
    images.push(image);
}

pub(crate) fn unregister_mapped(base: usize) {
    let mut images = MAPPED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    images.retain(|m| m.base != base);
}

pub(crate) fn locate_address(address: usize) -> Option<(String, usize)> {
    let images = MAPPED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    for img in images.iter() {
        if (img.base..img.base + img.size).contains(&address) {
            let name = img.path.file_name()?.to_string_lossy().into_owned();
            return Some((name, address - img.base));
        }
    }
    None
}

/// Sets the x86-64 GS base register for the current thread to `addr`.
///
/// # Safety
/// `addr` must be a valid pointer to a `SimulatedTeb` or 0.
pub unsafe fn set_gs_base(addr: usize) {
    // SAFETY: caller guarantees `addr` is a valid SimulatedTeb or 0.
    unsafe {
        libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, addr);
    }
}

/// Allocates and initializes a TEB and TLS array for the current thread.
pub fn init_thread_teb(tls_slots: &[usize]) -> *mut SimulatedTeb {
    let max_slots = 64.max(tls_slots.len() + 1);
    let mut array = vec![0usize; max_slots];
    for (i, &slot) in tls_slots.iter().enumerate() {
        array[i] = slot;
    }
    let array_ptr = Box::into_raw(array.into_boxed_slice()) as *mut usize;

    let teb_box = Box::new(SimulatedTeb {
        exception_list: 0,
        stack_base: 0,
        stack_limit: 0,
        sub_system_tib: 0,
        fiber_data: 0,
        arbitrary_user_pointer: 0,
        self_ptr: 0,
        environment_pointer: 0,
        process_id: std::process::id() as usize,
        // SAFETY: SYS_gettid syscall has no preconditions.
        thread_id: unsafe { libc::syscall(libc::SYS_gettid) as usize },
        active_rpc_handle: 0,
        thread_local_storage_pointer: array_ptr,
        peb: 0,
    });
    let teb_ptr = Box::into_raw(teb_box);
    // SAFETY: `teb_ptr` is a freshly allocated valid pointer.
    unsafe {
        (*teb_ptr).self_ptr = teb_ptr as usize;
        set_gs_base(teb_ptr as usize);
    }
    teb_ptr
}

/// A title image mapped into this process on Linux.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    pub(crate) base: usize,
    pub(crate) size: usize,
    pub(crate) path: PathBuf,
    pub(crate) info: ImageInfo,
    pub(crate) tls_block: Option<usize>,
    pub(crate) tls_index: Option<u32>,
}

// SAFETY: the image mapping is process-wide; accessing it across threads is safe.
unsafe impl Send for LoadedImage {}
// SAFETY: as above; `LoadedImage` has no interior mutability.
unsafe impl Sync for LoadedImage {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingImport {
    pub dll: String,
    pub symbol: String,
}

#[derive(Debug, Default, Clone)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub missing: Vec<MissingImport>,
}

/// Writes the prepared copy into `dir` and maps it via mmap.
pub fn load(prepared: &PreparedImage, dir: &Path) -> Result<LoadedImage> {
    victory_core::ensure_outside_repo(dir)?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(prepared.file_name());
    write_if_changed(&path, &prepared.bytes)?;

    let size = (prepared.info.size_of_image as usize).next_multiple_of(4096);
    let preferred = prepared.info.image_base as usize;

    // Try mapping at preferred address first, fallback to dynamic address.
    // SAFETY: anonymous mapping reservation for the image.
    let mut mapped_ptr = unsafe {
        libc::mmap(
            preferred as *mut libc::c_void,
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if mapped_ptr == libc::MAP_FAILED {
        // SAFETY: fallback anonymous mapping if preferred address is unavailable.
        mapped_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
    }

    if mapped_ptr == libc::MAP_FAILED {
        bail!(
            "mmap failed to allocate {} bytes for {}: {}",
            size,
            path.display(),
            std::io::Error::last_os_error()
        );
    }

    let base = mapped_ptr as usize;

    // 1. Copy headers
    let header_size = (prepared.info.sections.first().map_or(4096, |s| s.file_offset as usize))
        .min(prepared.bytes.len());
    // SAFETY: `base` is newly mapped writable memory large enough for the headers.
    unsafe {
        std::ptr::copy_nonoverlapping(prepared.bytes.as_ptr(), base as *mut u8, header_size);
    }

    // 2. Map sections
    for section in &prepared.info.sections {
        let dest = base + section.rva as usize;
        let file_start = section.file_offset as usize;
        let file_end = (file_start + section.file_size as usize).min(prepared.bytes.len());
        if file_end > file_start {
            let src = &prepared.bytes[file_start..file_end];
            // SAFETY: `dest..dest+src.len()` is within the mapped image bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dest as *mut u8, src.len());
            }
        }
    }

    // 3. Apply base relocations if mapped at a different base
    if base != preferred {
        apply_relocations(base, prepared.info.image_base, &prepared.bytes, &prepared.info)?;
    }

    // 4. Allocate static TLS block if present
    let (tls_block, tls_index) = if let Some(tls) = &prepared.info.tls {
        let start = base + tls.directory_rva as usize;
        // In IMAGE_TLS_DIRECTORY64:
        // +0: StartAddressOfRawData (u64)
        // +8: EndAddressOfRawData (u64)
        // +16: AddressOfIndex (u64)
        // +24: AddressOfCallBacks (u64)
        // +32: SizeOfZeroFill (u32)
        // +36: Characteristics (u32)
        // SAFETY: `start` points to the mapped TLS directory within image bounds.
        let (tls_mem_ptr, assigned_index) = unsafe {
            let start_va = (start as *const u64).read();
            let end_va = ((start + 8) as *const u64).read();
            let index_ptr = ((start + 16) as *const u64).read() as *mut u32;
            let zero_fill = ((start + 32) as *const u32).read() as usize;

            let raw_len = if end_va >= start_va {
                (end_va - start_va) as usize
            } else {
                0
            };
            let total_len = raw_len + zero_fill;
            let layout = std::alloc::Layout::from_size_align(total_len.max(16), 16).unwrap();
            let tls_mem = std::alloc::alloc_zeroed(layout);
            if raw_len > 0 {
                // Copy template data from raw data VA (apply_relocations already adjusted start_va)
                let src_ptr = start_va as *const u8;
                std::ptr::copy_nonoverlapping(src_ptr, tls_mem, raw_len);
            }
            let idx = if prepared.info.is_dll() {
                NEXT_DLL_TLS_INDEX.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            } else {
                0
            };
            if !index_ptr.is_null() {
                index_ptr.write(idx);
            }
            (tls_mem as usize, idx)
        };
        (Some(tls_mem_ptr), Some(assigned_index))
    } else {
        (None, None)
    };

    if let (Some(block), Some(idx), Some(teb)) = (tls_block, tls_index, current_teb()) {
        // SAFETY: teb and thread_local_storage_pointer are valid if current_teb returned Some.
        unsafe {
            let array = (*teb).thread_local_storage_pointer;
            if !array.is_null() && (idx as usize) < 64 {
                *array.add(idx as usize) = block;
            }
        }
    }

    let loaded = LoadedImage {
        base,
        size,
        path,
        info: prepared.info.clone(),
        tls_block,
        tls_index,
    };

    tracing::info!(
        base = format_args!("{:#x}", loaded.base),
        preferred = format_args!("{:#x}", loaded.info.image_base),
        path = %loaded.path.display(),
        "image mapped on Linux"
    );

    register_mapped(Arc::new(loaded.clone()));
    Ok(loaded)
}

fn apply_relocations(base: usize, preferred: u64, bytes: &[u8], info: &ImageInfo) -> Result<()> {
    let delta = (base as i64).wrapping_sub(preferred as i64);
    let Some(section) = info.sections.iter().find(|s| s.name == ".reloc") else {
        return Ok(());
    };
    let start = section.file_offset as usize;
    let end = (start + section.file_size as usize).min(bytes.len());
    let mut cur = start;

    while cur + 8 <= end {
        let page_rva = u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        let block_size = u32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().unwrap()) as usize;
        if block_size < 8 || cur + block_size > end {
            break;
        }
        let entries = (block_size - 8) / 2;
        for i in 0..entries {
            let pos = cur + 8 + i * 2;
            let entry = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            let typ = entry >> 12;
            let offset = (entry & 0x0FFF) as usize;
            let target_addr = base + page_rva as usize + offset;
            if typ == 10 {
                // IMAGE_REL_BASED_DIR64
                // SAFETY: target_addr is within the mapped image bounds.
                unsafe {
                    let ptr = target_addr as *mut u64;
                    let val = ptr.read_unaligned();
                    ptr.write_unaligned(val.wrapping_add(delta as u64));
                }
            } else if typ == 3 {
                // IMAGE_REL_BASED_HIGHLOW
                // SAFETY: target_addr is within the mapped image bounds.
                unsafe {
                    let ptr = target_addr as *mut u32;
                    let val = ptr.read_unaligned();
                    ptr.write_unaligned(val.wrapping_add(delta as u32));
                }
            }
        }
        cur += block_size;
    }
    Ok(())
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<()> {
    if std::fs::read(path).is_ok_and(|existing| existing == bytes) {
        return Ok(());
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    Ok(())
}

pub fn load_title_dlls(
    main: &ImageInfo,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_imports(main, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

pub fn load_title_dll(
    name: &str,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_dll(name, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

fn map_imports(
    info: &ImageInfo,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    for module in info.imports.iter().chain(&info.delay_imports) {
        map_dll(&module.dll, dirs, wanted, seen, loaded)?;
    }
    Ok(())
}

fn map_dll(
    name: &str,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    let name = name.to_ascii_lowercase();
    let path = dirs.0.join(&name);
    if !seen.insert(name.clone()) || !wanted(&name) || !path.is_file() {
        return Ok(());
    }
    let original = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let prepared = prepare(&original).with_context(|| format!("preparing {}", path.display()))?;
    let image = load(&prepared, dirs.1).with_context(|| format!("loading {}", path.display()))?;
    map_imports(&image.info, dirs, wanted, seen, loaded)?;
    loaded.push((name, image));
    Ok(())
}

pub struct WithTitleDlls<'a> {
    pub inner: &'a dyn ImportResolver,
    pub dlls: &'a [(String, LoadedImage)],
}

impl ImportResolver for WithTitleDlls<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let found = self.inner.resolve(dll, symbol);
        if found != Resolution::Missing {
            return found;
        }
        self.dlls
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(dll))
            .and_then(|(_, image)| image.export_symbol(symbol))
            .map_or(Resolution::Missing, |address| Resolution::Found {
                address,
                provenance: Provenance::Title,
            })
    }
}

impl LoadedImage {
    pub fn base(&self) -> usize {
        self.base
    }

    pub fn tls_index(&self) -> Option<u32> {
        self.tls_index
    }

    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn export(&self, name: &str) -> Option<usize> {
        self.info
            .exports
            .iter()
            .find(|e| e.name.as_deref() == Some(name))
            .and_then(|e| e.rva)
            .map(|rva| self.base + rva as usize)
    }

    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match symbol {
            ImportSymbol::Name { name, .. } => self.export(name),
            ImportSymbol::Ordinal(o) => self
                .info
                .exports
                .iter()
                .find(|e| e.ordinal == *o)
                .and_then(|e| e.rva)
                .map(|rva| self.base + rva as usize),
        }
    }

    pub fn bind_imports(
        &self,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let mut report = BindReport::default();
        let mut slots: Vec<(u32, usize)> = Vec::with_capacity(self.info.import_count());
        let mut unresolved: Vec<(u32, &str, &ImportSymbol)> = Vec::new();

        for module in self.info.imports.iter().chain(&self.info.delay_imports) {
            for entry in &module.entries {
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found { address, provenance } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            Provenance::Host => report.host += 1,
                            Provenance::Title => report.title += 1,
                        }
                        slots.push((entry.iat_slot_rva, address));
                    }
                    Resolution::Missing => {
                        unresolved.push((entry.iat_slot_rva, &module.dll, &entry.symbol));
                    }
                }
            }
        }

        if !unresolved.is_empty() {
            let mut block = ThunkBlock::new(unresolved.len())?;
            for (slot, dll, symbol) in unresolved {
                let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
                    dll: dll.to_owned(),
                    symbol: symbol.to_string(),
                    policy,
                    calls: AtomicU64::new(0),
                }));
                slots.push((slot, block.emit(site)?));
                report.missing.push(MissingImport {
                    dll: site.dll.clone(),
                    symbol: site.symbol.clone(),
                });
            }
            block.seal()?;
        }

        self.write_slots(&slots)?;
        self.apply_section_protections()?;

        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "imports bound on Linux"
        );
        Ok(report)
    }

    fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        for &(rva, address) in slots {
            let slot = (self.base + rva as usize) as *mut u64;
            // Align to page boundary for mprotect
            let page = (slot as usize) & !4095;
            // SAFETY: changing memory protection to make IAT slot writable and writing address.
            unsafe {
                libc::mprotect(
                    page as *mut libc::c_void,
                    4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                );
                slot.write_volatile(address as u64);
            }
        }
        Ok(())
    }

    fn apply_section_protections(&self) -> Result<()> {
        const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
        const SCN_MEM_READ: u32 = 0x4000_0000;
        const SCN_MEM_WRITE: u32 = 0x8000_0000;

        for section in &self.info.sections {
            let start = (self.base + section.rva as usize) & !4095;
            let len = (section.virtual_size as usize).next_multiple_of(4096);
            let mut prot = 0;
            if section.characteristics & SCN_MEM_READ != 0 {
                prot |= libc::PROT_READ;
            }
            if section.characteristics & SCN_MEM_WRITE != 0 {
                prot |= libc::PROT_WRITE;
            }
            if section.characteristics & SCN_MEM_EXECUTE != 0 {
                prot |= libc::PROT_EXEC;
            }
            // SAFETY: start and len are within the mapped image memory.
            unsafe {
                libc::mprotect(start as *mut libc::c_void, len, prot);
            }
        }
        Ok(())
    }

    /// Runs static TLS callbacks for this image.
    ///
    /// # Safety
    /// Thread state must be initialized and image mapped.
    pub unsafe fn run_tls_callbacks(&self, reason: u32) {
        // SAFETY: caller guarantees thread state and image mapping are valid.
        unsafe { self.tls_callbacks(reason, std::ptr::null_mut()) }
    }

    // SAFETY: the caller ensures the image is mapped and thread state is ready.
    unsafe fn tls_callbacks(&self, reason: u32, reserved: *mut c_void) {
        let Some(tls) = &self.info.tls else { return };
        for &rva in &tls.callback_rvas {
            type TlsCallback = unsafe extern "win64" fn(*mut c_void, u32, *mut c_void);
            // SAFETY: rva is within the mapped image bounds.
            let callback: TlsCallback = unsafe { std::mem::transmute(self.base + rva as usize) };
            // SAFETY: invoking TLS callback with Win64 ABI.
            unsafe { callback(self.base as *mut c_void, reason, reserved) };
        }
    }

    /// Attaches the DLL (`DLL_PROCESS_ATTACH`).
    ///
    /// # Safety
    /// The DLL must be mapped, imports bound, and dependencies attached.
    pub unsafe fn attach(&self) -> Result<()> {
        // SAFETY: caller guarantees mapping, imports, and dependencies.
        if unsafe { self.notify(DLL_PROCESS_ATTACH, true, std::ptr::null_mut()) } == Some(0) {
            bail!("DllMain returned FALSE");
        }
        Ok(())
    }

    /// Detaches the DLL (`DLL_PROCESS_DETACH`).
    ///
    /// # Safety
    /// The DLL must still be mapped.
    pub unsafe fn detach(&self) {
        // SAFETY: caller guarantees the DLL is still mapped.
        unsafe { self.notify(DLL_PROCESS_DETACH, true, std::ptr::null_mut()) };
    }

    pub(crate) unsafe fn notify(
        &self,
        reason: u32,
        dll_main: bool,
        reserved: *mut c_void,
    ) -> Option<i32> {
        // SAFETY: notifying TLS callbacks.
        unsafe { self.tls_callbacks(reason, reserved) };
        if !dll_main || !self.info.is_dll() || self.info.entry_rva == 0 {
            return None;
        }
        type DllMain = unsafe extern "win64" fn(*mut c_void, u32, *mut c_void) -> i32;
        // SAFETY: entry_rva is within the mapped DLL bounds.
        let main: DllMain = unsafe { std::mem::transmute(self.base + self.info.entry_rva as usize) };
        // SAFETY: invoking DllMain with Win64 ABI.
        Some(unsafe { main(self.base as *mut c_void, reason, reserved) })
    }

    /// Spawns the main thread for the loaded image.
    ///
    /// # Safety
    /// The image and its dependencies must have all imports bound and be ready for execution.
    pub unsafe fn spawn_main(
        self: Arc<Self>,
        dlls: Vec<(String, LoadedImage)>,
    ) -> Result<std::thread::JoinHandle<Result<u32>>> {
        if self.info.entry_rva == 0 {
            bail!("image has no entry point");
        }
        let stack = usize::try_from(self.info.stack_reserve)
            .unwrap_or(usize::MAX)
            .max(1 << 20);

        let mut max_slot = self.tls_index.map_or(0, |idx| idx as usize);
        for (_, dll) in &dlls {
            if let Some(idx) = dll.tls_index {
                max_slot = max_slot.max(idx as usize);
            }
        }
        let mut tls_slots = vec![0usize; (max_slot + 1).max(64)];
        if let (Some(idx), Some(block)) = (self.tls_index, self.tls_block) {
            tls_slots[idx as usize] = block;
        }
        for (_, dll) in &dlls {
            if let (Some(idx), Some(block)) = (dll.tls_index, dll.tls_block) {
                tls_slots[idx as usize] = block;
            }
        }

        std::thread::Builder::new()
            .name("title-main".into())
            .stack_size(stack)
            .spawn(move || {
                // Initialize simulated TEB and set GS base for this thread
                let teb = init_thread_teb(&tls_slots);

                for (name, dll) in &dlls {
                    // SAFETY: imports are bound; attaching the DLL.
                    unsafe { dll.attach() }.with_context(|| format!("attaching {name}"))?;
                }
                // SAFETY: running TLS callbacks after attach.
                unsafe { self.run_tls_callbacks(DLL_PROCESS_ATTACH) };

                type Entry = unsafe extern "win64" fn(*mut c_void) -> u32;
                // SAFETY: entry_rva is non-zero and verified.
                let entry: Entry =
                    unsafe { std::mem::transmute(self.base + self.info.entry_rva as usize) };
                // SAFETY: invoking the title's main entry point with Win64 ABI.
                let result = unsafe { entry(std::ptr::null_mut()) };

                // Clean up TEB allocations
                // SAFETY: freeing the allocated TEB and TLS array.
                unsafe {
                    let _ = Box::from_raw((*teb).thread_local_storage_pointer);
                    let _ = Box::from_raw(teb);
                }

                Ok(result)
            })
            .context("spawning title main thread")
    }

    /// Unmaps the image from memory.
    ///
    /// # Safety
    /// No code from this image or pointers into it may be accessed after unmapping.
    pub unsafe fn unload(self) {
        unregister_mapped(self.base);
        // SAFETY: unmapping the previously mapped image memory.
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.size);
        }
    }
}

/// The file crash reports are appended to, besides being printed.
static LOG: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Also appends every unhandled report and panic to `file`.
pub fn log_crashes_to(file: std::path::PathBuf) {
    if LOG.set(file).is_ok() {
        let print = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            print(info);
            append_to_log(&format!(
                "VIctory: panic on thread {:?}: {info}\n{}",
                std::thread::current().name().unwrap_or("?"),
                title_output()
            ));
        }));
    }
}

pub fn log_crash(text: &str) {
    let text = format!("{text}\n{}", title_output());
    eprint!("{text}");
    append_to_log(&text);
}

fn append_to_log(text: &str) {
    use std::io::Write;
    let Some(file) = LOG.get() else { return };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let opened = std::fs::OpenOptions::new().create(true).append(true).open(file);
    if let Ok(mut f) = opened {
        let _ = write!(f, "unix time {now}, process {}\n{text}\n", std::process::id());
    }
}

static SAID: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
const SAID_LINES: usize = 8;

pub fn note_title_output(line: &str) {
    if line.is_empty() {
        return;
    }
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.len() == SAID_LINES {
        said.pop_front();
    }
    said.push_back(line.to_owned());
}

fn title_output() -> String {
    let Ok(said) = SAID.try_lock() else {
        return String::new();
    };
    said.iter().map(|l| format!("  title said: {l}\n")).collect()
}

pub fn locate(address: usize) -> String {
    locate_address(address).map_or_else(
        || format!("{address:#x}"),
        |(name, rva)| format!("{name}+{rva:#x}"),
    )
}

pub fn disable_thread_calls(_base: usize) {}

#[cfg(feature = "diagnostics")]
pub fn save_on_crash(_ranges: Vec<(usize, usize, std::path::PathBuf)>) {}

pub fn report_crashes() {
    unsafe extern "C" fn sig_handler(
        sig: libc::c_int,
        _info: *mut libc::siginfo_t,
        ucontext: *mut libc::c_void,
    ) {
        let ucontext = ucontext as *const libc::ucontext_t;
        let rip = if !ucontext.is_null() {
            // SAFETY: ucontext is provided by the kernel via SA_SIGINFO.
            unsafe { (*ucontext).uc_mcontext.gregs[libc::REG_RIP as usize] as usize }
        } else {
            0
        };
        let location = locate(rip);
        let text = format!("VIctory: crash signal {sig} at {location} (RIP={rip:#x})");
        log_crash(&text);
        // SAFETY: exiting immediately after logging fatal signal.
        unsafe {
            libc::_exit(1);
        }
    }

    // SAFETY: registering POSIX signal handlers for crash logging.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sig_handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGILL, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGFPE, &sa, std::ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linux_teb_and_gs_base() {
        let dummy_tls_slot = 0x12345678_usize;
        let teb = init_thread_teb(&[dummy_tls_slot]);
        assert!(!teb.is_null());

        let read_self: usize;
        let read_tls_array: usize;
        // SAFETY: testing inline assembly reading from emulated TEB.
        unsafe {
            std::arch::asm!(
                "mov {0}, gs:[0x30]",
                "mov {1}, gs:[0x58]",
                out(reg) read_self,
                out(reg) read_tls_array,
            );
        }
        assert_eq!(read_self, teb as usize);
        // SAFETY: freeing the allocated test TEB and TLS array.
        unsafe {
            let tls_ptr = *(read_tls_array as *const usize);
            assert_eq!(tls_ptr, dummy_tls_slot);
            let _ = Box::from_raw((*teb).thread_local_storage_pointer);
            let _ = Box::from_raw(teb);
        }
    }

    #[test]
    fn test_distinct_tls_indices_for_dlls() {
        let idx0 = 0_u32;
        let dll1_idx = NEXT_DLL_TLS_INDEX.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dll2_idx = NEXT_DLL_TLS_INDEX.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        assert_ne!(dll1_idx, dll2_idx);
        assert!(dll1_idx > 0);
        assert!(dll2_idx > dll1_idx);

        let dummy_slot_main = 0xAAAA_usize;
        let dummy_slot_dll1 = 0xBBBB_usize;
        let dummy_slot_dll2 = 0xCCCC_usize;

        let max_idx = dll2_idx as usize;
        let mut tls_slots = vec![0usize; max_idx + 1];
        tls_slots[idx0 as usize] = dummy_slot_main;
        tls_slots[dll1_idx as usize] = dummy_slot_dll1;
        tls_slots[dll2_idx as usize] = dummy_slot_dll2;

        let teb = init_thread_teb(&tls_slots);
        assert!(!teb.is_null());

        let read_tls_array: usize;
        // SAFETY: reading gs:[0x58] for TLS array pointer.
        unsafe {
            std::arch::asm!("mov {0}, gs:[0x58]", out(reg) read_tls_array);
        }
        let array = read_tls_array as *const usize;
        // SAFETY: reading slots by assigned module indices.
        unsafe {
            assert_eq!(*array.add(idx0 as usize), dummy_slot_main);
            assert_eq!(*array.add(dll1_idx as usize), dummy_slot_dll1);
            assert_eq!(*array.add(dll2_idx as usize), dummy_slot_dll2);
            let _ = Box::from_raw((*teb).thread_local_storage_pointer);
            let _ = Box::from_raw(teb);
        }
    }
}
