//! Pass-through to the host's own implementation of shared APIs.

use victory_pe::ImportSymbol;

/// Which DLLs may be satisfied by the host's own implementation.
///
/// Console titles link mostly against Windows API sets that behave the same
/// on PC. Passing those through lets bring-up focus on what really differs.
/// Anything whose semantics differ on console (memory, files, threads) is
/// overridden in `modules`, which always wins over the host.
#[derive(Debug, Clone)]
pub struct HostPolicy {
    /// Lower-case patterns; a trailing `*` matches a prefix.
    pub allow: Vec<String>,
}

impl Default for HostPolicy {
    fn default() -> Self {
        let allow = [
            "api-ms-win-core-*",
            "api-ms-win-eventing-*",
            "api-ms-win-security-*",
            "api-ms-win-crt-*",
            "kernel32.dll",
            "kernelbase.dll",
            "ntdll.dll",
            "ucrtbase.dll",
            "vcruntime140*",
            "msvcp140*",
            // Off-machine traffic is refused in `modules::net`.
            "ws2_32.dll",
            "bcrypt.dll",
            "crypt32.dll",
            "ncrypt.dll",
            "winhttp.dll",
            "sspicli.dll",
            "cabinet.dll",
            "avrt.dll",
            // Window, message and shell APIs: the host's desktop window
            // stands in for the console's single core window.
            "ext-ms-win-rtcore-ntuser-*",
            "api-ms-win-shcore-*",
            "msvcp_win.dll",
            "ole32.dll",
            "advapi32.dll",
            "iphlpapi.dll",
            // The title's crash handler writes minidumps with it (loaded by name).
            "dbghelp.dll",
            // Same APIs as the console's (XAudio2 ordinal 1 is XAudio2Create
            // on both); Media Foundation reads local media files.
            "xaudio2_9.dll",
            "mfplat.dll",
            "mfreadwrite.dll",
        ];
        HostPolicy {
            allow: allow.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl HostPolicy {
    /// No pass-through at all: shows the full surface a title needs.
    pub fn none() -> Self {
        HostPolicy { allow: Vec::new() }
    }

    pub fn allows(&self, dll: &str) -> bool {
        let dll = dll.to_ascii_lowercase();
        self.allow
            .iter()
            .any(|pattern| match pattern.strip_suffix('*') {
                Some(prefix) => dll.starts_with(prefix),
                None => dll == *pattern,
            })
    }

    pub(crate) fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Option<usize> {
        if !self.allows(dll) {
            return None;
        }
        imp::resolve(dll, symbol)
    }
}

/// The host's own `name` from the System32 DLL `dll`: what a title import
/// of it is bound to when we do not override it.
pub(crate) fn function(dll: &str, name: &str) -> Option<usize> {
    let name = ImportSymbol::Name {
        hint: 0,
        name: name.into(),
    };
    imp::resolve(dll, &name)
}

#[cfg(windows)]
mod imp {
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::sync::{LazyLock, Mutex};

    use victory_pe::ImportSymbol;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA,
    };

    /// Module handles by lower-case DLL name (stored as usize: Send + Sync).
    static MODULES: LazyLock<Mutex<HashMap<String, usize>>> = LazyLock::new(Default::default);

    pub(super) fn resolve(dll: &str, symbol: &ImportSymbol) -> Option<usize> {
        let module = module(dll)?;
        let proc = match symbol {
            ImportSymbol::Name { name, .. } => {
                let name = CString::new(name.as_str()).ok()?;
                // SAFETY: valid module handle, NUL-terminated name.
                unsafe { GetProcAddress(module as _, name.as_ptr().cast()) }
            }
            // SAFETY: MAKEINTRESOURCE-style ordinal lookup on a valid module.
            ImportSymbol::Ordinal(o) => unsafe { GetProcAddress(module as _, *o as usize as _) },
        };
        proc.map(|f| f as *const () as usize)
    }

    fn module(dll: &str) -> Option<usize> {
        let key = dll.to_ascii_lowercase();
        let mut cache = MODULES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&m) = cache.get(&key) {
            return Some(m);
        }
        let name = CString::new(key.as_str()).ok()?;
        // System32 only: never pick up a same-named DLL from the current folder or PATH.
        // SAFETY: NUL-terminated DLL name; only allow-listed system DLLs get here.
        let handle = unsafe {
            LoadLibraryExA(
                name.as_ptr().cast(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        } as usize;
        (handle != 0).then(|| *cache.entry(key).or_insert(handle))
    }
}

#[cfg(not(windows))]
mod imp {
    use std::ffi::c_void;
    use victory_pe::ImportSymbol;

    pub(super) fn resolve(dll: &str, symbol: &ImportSymbol) -> Option<usize> {
        let name = match symbol {
            ImportSymbol::Name { name, .. } => name.as_str(),
            ImportSymbol::Ordinal(ord) => {
                if dll.eq_ignore_ascii_case("ws2_32.dll") {
                    match *ord {
                        2 => "bind",
                        4 => "connect",
                        20 => "sendto",
                        51 => "gethostbyaddr",
                        52 => "gethostbyname",
                        102 => "WSAAsyncGetHostByAddr",
                        103 => "WSAAsyncGetHostByName",
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
        };
        fallback_resolve(dll, name)
    }

    fn fallback_resolve(dll: &str, name: &str) -> Option<usize> {
        let key = dll.to_ascii_lowercase();
        if key.starts_with("api-ms-win-")
            || key == "kernel32.dll"
            || key == "kernelbase.dll"
            || key == "ntdll.dll"
            || key == "ucrtbase.dll"
            || key == "msvcp140.dll"
            || key == "ws2_32.dll"
        {
            let func: Option<usize> = match name {
                "WSASetLastError" => Some(fake_set_last_error_i32 as *const () as usize),
                "WSAGetLastError" => Some(fake_get_last_error_i32 as *const () as usize),
                "socket" => Some(fake_socket as *const () as usize),
                "closesocket" => Some(fake_close as *const () as usize),
                "connect" => Some(fake_connect as *const () as usize),
                "bind" => Some(fake_bind as *const () as usize),
                "getsockopt" => Some(fake_getsockopt as *const () as usize),
                "sendto" => Some(fake_sendto as *const () as usize),
                "getnameinfo" => Some(fake_getnameinfo as *const () as usize),
                "gethostbyname" => Some(fake_gethostbyname as *const () as usize),
                "gethostbyaddr" => Some(fake_gethostbyaddr as *const () as usize),
                "WSAAsyncGetHostByName" => Some(fake_null as *const () as usize),
                "WSAAsyncGetHostByAddr" => Some(fake_null as *const () as usize),
                "WSAIoctl" => Some(fake_wsaoctl as *const () as usize),
                "GetTickCount" => Some(fake_get_tick_count as *const () as usize),
                "GetTickCount64" => Some(fake_get_tick_count64 as *const () as usize),
                "QueryPerformanceCounter" => Some(fake_qpc as *const () as usize),
                "QueryPerformanceFrequency" => Some(fake_qpf as *const () as usize),
                "GetCurrentProcessId" => Some(fake_get_pid as *const () as usize),
                "GetCurrentThreadId" => Some(fake_get_tid as *const () as usize),
                "Sleep" => Some(fake_sleep as *const () as usize),
                "SleepEx" => Some(fake_sleep_ex as *const () as usize),
                "CloseHandle" => Some(fake_close_handle as *const () as usize),
                "GetLastError" => Some(fake_get_last_error as *const () as usize),
                "SetLastError" => Some(fake_set_last_error as *const () as usize),
                "VirtualAlloc" => Some(fake_virtual_alloc as *const () as usize),
                "VirtualFree" => Some(fake_virtual_free as *const () as usize),
                "VirtualProtect" => Some(fake_virtual_protect as *const () as usize),
                "CreateFileW" => Some(fake_create_file_w as *const () as usize),
                "CreateFileA" => Some(fake_create_file_a as *const () as usize),
                "CreateFile2" => Some(fake_create_file_2 as *const () as usize),
                "ReadFile" => Some(fake_read_file as *const () as usize),
                "WriteFile" => Some(fake_write_file as *const () as usize),
                "CreateDirectoryW" => Some(fake_create_directory_w as *const () as usize),
                "RemoveDirectoryW" => Some(fake_remove_directory_w as *const () as usize),
                "DeleteFileW" => Some(fake_delete_file_w as *const () as usize),
                "DeleteFileA" => Some(fake_delete_file_a as *const () as usize),
                "SetFileAttributesW" => Some(fake_set_file_attributes_w as *const () as usize),
                "GetFileAttributesW" => Some(fake_get_file_attributes_w as *const () as usize),
                "GetFileAttributesA" => Some(fake_get_file_attributes_a as *const () as usize),
                "GetFileAttributesExW" => Some(fake_get_file_attributes_ex_w as *const () as usize),
                "GetDiskFreeSpaceW" => Some(fake_get_disk_free_space_w as *const () as usize),
                "GetDiskFreeSpaceExW" => Some(fake_get_disk_free_space_ex_w as *const () as usize),
                "FindFirstFileW" => Some(fake_find_first_file_w as *const () as usize),
                "FindFirstFileExW" => Some(fake_find_first_file_ex_w as *const () as usize),
                "MoveFileExW" => Some(fake_move_file_ex_w as *const () as usize),
                "SetCurrentDirectoryW" => Some(fake_set_current_directory_w as *const () as usize),
                "SetCurrentDirectoryA" => Some(fake_set_current_directory_a as *const () as usize),
                "_wfopen" => Some(fake_wfopen as *const () as usize),
                "fopen" => Some(fake_fopen as *const () as usize),
                "_wfopen_s" => Some(fake_wfopen_s as *const () as usize),
                "fopen_s" => Some(fake_fopen_s as *const () as usize),
                "_wfsopen" => Some(fake_wfsopen as *const () as usize),
                "fread" => Some(fake_fread as *const () as usize),
                "fclose" => Some(fake_fclose as *const () as usize),
                "_open" => Some(fake_open as *const () as usize),
                "_wopen" => Some(fake_wopen as *const () as usize),
                "_close" => Some(fake_close as *const () as usize),
                "_wstat64" => Some(fake_wstat64 as *const () as usize),
                "_wstat64i32" => Some(fake_wstat64 as *const () as usize),
                "_stat64i32" => Some(fake_wstat64 as *const () as usize),
                "_wremove" => Some(fake_wremove as *const () as usize),
                "remove" => Some(fake_remove as *const () as usize),
                "_wchmod" => Some(fake_wchmod as *const () as usize),
                "_wutime64" => Some(fake_wutime64 as *const () as usize),
                "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z" => Some(fake_fiopen_wide as *const () as usize),
                "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z" => Some(fake_fiopen_narrow as *const () as usize),
                _ => None,
            };
            if func.is_some() {
                return func;
            }
        }
        None
    }

    extern "win64" fn fake_get_tick_count() -> u32 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start = *START.get_or_init(std::time::Instant::now);
        start.elapsed().as_millis() as u32
    }

    extern "win64" fn fake_get_tick_count64() -> u64 {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start = *START.get_or_init(std::time::Instant::now);
        start.elapsed().as_millis() as u64
    }

    extern "win64" fn fake_qpc(count: *mut i64) -> i32 {
        if count.is_null() {
            return 0;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            let mut ts: libc::timespec = std::mem::zeroed();
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
            let nanos = (ts.tv_sec as i64) * 1_000_000_000 + (ts.tv_nsec as i64);
            count.write(nanos);
        }
        1
    }

    extern "win64" fn fake_qpf(freq: *mut i64) -> i32 {
        if freq.is_null() {
            return 0;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            freq.write(1_000_000_000);
        }
        1
    }

    extern "win64" fn fake_get_pid() -> u32 {
        std::process::id()
    }

    extern "win64" fn fake_get_tid() -> u32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::syscall(libc::SYS_gettid) as u32 }
    }

    extern "win64" fn fake_sleep(ms: u32) {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }

    extern "win64" fn fake_sleep_ex(ms: u32, _: i32) -> u32 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
        0
    }

    thread_local! {
        static TLS_ERROR: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    extern "win64" fn fake_get_last_error() -> u32 {
        TLS_ERROR.with(|e| e.get())
    }

    extern "win64" fn fake_get_last_error_i32() -> i32 {
        TLS_ERROR.with(|e| e.get()) as i32
    }

    extern "win64" fn fake_set_last_error(code: u32) {
        TLS_ERROR.with(|e| e.set(code));
    }

    extern "win64" fn fake_set_last_error_i32(code: i32) {
        TLS_ERROR.with(|e| e.set(code as u32));
    }

    extern "win64" fn fake_socket(af: i32, type_: i32, protocol: i32) -> usize {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let s = unsafe { libc::socket(af, type_, protocol) };
        if s < 0 {
            usize::MAX
        } else {
            s as usize
        }
    }

    extern "win64" fn fake_connect(s: usize, name: *const u8, len: i32) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::connect(s as i32, name as *const libc::sockaddr, len as libc::socklen_t) }
    }

    extern "win64" fn fake_sendto(
        s: usize,
        buf: *const u8,
        len: i32,
        flags: i32,
        to: *const u8,
        tolen: i32,
    ) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            let ret = libc::sendto(
                s as i32,
                buf as *const libc::c_void,
                len as usize,
                flags,
                to as *const libc::sockaddr,
                tolen as libc::socklen_t,
            );
            if ret < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EISCONN) {
                libc::send(s as i32, buf as *const libc::c_void, len as usize, flags) as i32
            } else {
                ret as i32
            }
        }
    }

    extern "win64" fn fake_bind(s: usize, name: *const u8, len: i32) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::bind(s as i32, name as *const libc::sockaddr, len as libc::socklen_t) }
    }

    extern "win64" fn fake_getsockopt(
        s: usize,
        level: i32,
        optname: i32,
        optval: *mut u8,
        optlen: *mut i32,
    ) -> i32 {
        let (posix_level, posix_optname) = match (level, optname) {
            (0xffff, 0x1008) => (libc::SOL_SOCKET, libc::SO_TYPE),
            (0xffff, 0x1007) => (libc::SOL_SOCKET, libc::SO_ERROR),
            (0xffff, _) => (libc::SOL_SOCKET, optname),
            _ => (level, optname),
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            libc::getsockopt(
                s as i32,
                posix_level,
                posix_optname,
                optval as *mut libc::c_void,
                optlen as *mut libc::socklen_t,
            )
        }
    }

    extern "win64" fn fake_getnameinfo(
        sa: *const u8,
        sa_len: i32,
        host: *mut libc::c_char,
        host_len: u32,
        serv: *mut libc::c_char,
        serv_len: u32,
        flags: i32,
    ) -> i32 {
        let mut posix_flags = 0;
        if flags & 0x01 != 0 {
            posix_flags |= libc::NI_NOFQDN;
        }
        if flags & 0x02 != 0 {
            posix_flags |= libc::NI_NUMERICHOST;
        }
        if flags & 0x04 != 0 {
            posix_flags |= libc::NI_NAMEREQD;
        }
        if flags & 0x08 != 0 {
            posix_flags |= libc::NI_NUMERICSERV;
        }
        if flags & 0x10 != 0 {
            posix_flags |= libc::NI_DGRAM;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            libc::getnameinfo(
                sa as *const libc::sockaddr,
                sa_len as libc::socklen_t,
                host,
                host_len as libc::socklen_t,
                serv,
                serv_len as libc::socklen_t,
                posix_flags,
            )
        }
    }

    extern "win64" fn fake_gethostbyname(_name: *const libc::c_char) -> *mut c_void {
        std::ptr::null_mut()
    }

    extern "win64" fn fake_gethostbyaddr(
        _addr: *const libc::c_char,
        _len: i32,
        _type_: i32,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    extern "win64" fn fake_null() -> *mut c_void {
        std::ptr::null_mut()
    }

    extern "win64" fn fake_wsaoctl(
        _s: usize,
        _code: u32,
        _in_buf: *const c_void,
        _in_len: u32,
        out_buf: *mut c_void,
        _out_len: u32,
        returned: *mut u32,
        _ov: *mut c_void,
        _comp: *const c_void,
    ) -> i32 {
        if !returned.is_null() {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                returned.write(0);
            }
        }
        if !out_buf.is_null() {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                (out_buf as *mut usize).write(0);
            }
        }
        0
    }

    extern "win64" fn fake_close_handle(handle: *mut c_void) -> i32 {
        let val = handle as usize;
        if val > 0 && val < 65536 {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                libc::close(val as i32);
            }
        }
        1
    }

    extern "win64" fn fake_virtual_alloc(
        addr: *mut c_void,
        size: usize,
        _alloc_type: u32,
        protect: u32,
    ) -> *mut c_void {
        let mut prot = 0;
        if protect & 0x02 != 0 {
            prot |= libc::PROT_READ;
        }
        if protect & 0x04 != 0 {
            prot |= libc::PROT_READ | libc::PROT_WRITE;
        }
        if protect & 0x20 != 0 {
            prot |= libc::PROT_READ | libc::PROT_EXEC;
        }
        if protect & 0x40 != 0 {
            prot |= libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        }
        if prot == 0 {
            prot = libc::PROT_READ | libc::PROT_WRITE;
        }

        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            let res = libc::mmap(
                addr,
                size,
                prot,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if res == libc::MAP_FAILED {
                std::ptr::null_mut()
            } else {
                res
            }
        }
    }

    extern "win64" fn fake_virtual_free(addr: *mut c_void, size: usize, free_type: u32) -> i32 {
        if free_type == 0x8000 && !addr.is_null() {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                libc::munmap(addr, size.max(4096));
            }
        }
        1
    }

    extern "win64" fn fake_virtual_protect(
        addr: *mut c_void,
        size: usize,
        protect: u32,
        old_protect: *mut u32,
    ) -> i32 {
        if !old_protect.is_null() {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                old_protect.write(0x04);
            }
        }
        let mut prot = 0;
        if protect & 0x02 != 0 {
            prot |= libc::PROT_READ;
        }
        if protect & 0x04 != 0 {
            prot |= libc::PROT_READ | libc::PROT_WRITE;
        }
        if protect & 0x20 != 0 {
            prot |= libc::PROT_READ | libc::PROT_EXEC;
        }
        if protect & 0x40 != 0 {
            prot |= libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        }
        if prot == 0 {
            prot = libc::PROT_READ | libc::PROT_WRITE;
        }

        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let ret = unsafe { libc::mprotect(addr, size, prot) };
        if ret == 0 { 1 } else { 0 }
    }

    fn to_posix_path(wide_or_narrow: &str) -> std::path::PathBuf {
        let mut s = wide_or_narrow.replace('\\', "/");
        if let Some(stripped) = s.strip_prefix("//?/") {
            s = stripped.to_string();
        }
        if let Some(stripped) = s.strip_prefix("/?/") {
            s = stripped.to_string();
        }
        if s.len() >= 2 && s.as_bytes()[0].is_ascii_alphabetic() && s.as_bytes()[1] == b':' {
            s = s[2..].to_string();
        }
        std::path::PathBuf::from(s)
    }

    fn wide_to_path(wide: *const u16) -> Option<std::path::PathBuf> {
        if wide.is_null() {
            return None;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let len = unsafe { (0..).take_while(|&i| *wide.add(i) != 0).count() };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let slice = unsafe { std::slice::from_raw_parts(wide, len) };
        let s = String::from_utf16_lossy(slice);
        Some(to_posix_path(&s))
    }

    fn narrow_to_path(narrow: *const libc::c_char) -> Option<std::path::PathBuf> {
        if narrow.is_null() {
            return None;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let s = unsafe { std::ffi::CStr::from_ptr(narrow) }.to_string_lossy();
        Some(to_posix_path(&s))
    }

    extern "win64" fn fake_create_file_w(
        path: *const u16,
        access: u32,
        _share: u32,
        _sec: *const c_void,
        disposition: u32,
        _flags: u32,
        _template: *mut c_void,
    ) -> *mut c_void {
        let Some(path) = wide_to_path(path) else {
            return (-1isize) as *mut c_void;
        };
        let mut flags = libc::O_CLOEXEC;
        if (access & 0x8000_0000 != 0) && (access & 0x4000_0000 != 0) {
            flags |= libc::O_RDWR;
        } else if access & 0x4000_0000 != 0 {
            flags |= libc::O_WRONLY;
        } else {
            flags |= libc::O_RDONLY;
        }
        if disposition == 1 {
            flags |= libc::O_CREAT | libc::O_EXCL;
        } else if disposition == 2 {
            flags |= libc::O_CREAT | libc::O_TRUNC;
        } else if disposition == 4 {
            flags |= libc::O_CREAT;
        }
        let c_str = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return (-1isize) as *mut c_void,
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let fd = unsafe { libc::open(c_str.as_ptr(), flags, 0o666) };
        if fd < 0 {
            (-1isize) as *mut c_void
        } else {
            fd as usize as *mut c_void
        }
    }

    extern "win64" fn fake_create_file_a(
        path: *const libc::c_char,
        access: u32,
        share: u32,
        sec: *const c_void,
        disposition: u32,
        flags: u32,
        template: *mut c_void,
    ) -> *mut c_void {
        if path.is_null() {
            return (-1isize) as *mut c_void;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let s = unsafe { std::ffi::CStr::from_ptr(path) };
        let wide: Vec<u16> = s.to_string_lossy().encode_utf16().chain(Some(0)).collect();
        fake_create_file_w(wide.as_ptr(), access, share, sec, disposition, flags, template)
    }

    extern "win64" fn fake_create_file_2(
        path: *const u16,
        access: u32,
        share: u32,
        disposition: u32,
        _info: *const c_void,
    ) -> *mut c_void {
        fake_create_file_w(path, access, share, std::ptr::null(), disposition, 0, std::ptr::null_mut())
    }

    extern "win64" fn fake_read_file(
        handle: *mut c_void,
        buffer: *mut u8,
        bytes_to_read: u32,
        bytes_read: *mut u32,
        _overlapped: *mut c_void,
    ) -> i32 {
        let fd = handle as usize as i32;
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let n = unsafe { libc::read(fd, buffer as *mut libc::c_void, bytes_to_read as usize) };
        if n >= 0 {
            if !bytes_read.is_null() {
                // SAFETY: calling host libc FFI or dereferencing caller pointer.
                unsafe {
                    bytes_read.write(n as u32);
                }
            }
            1
        } else {
            if !bytes_read.is_null() {
                // SAFETY: calling host libc FFI or dereferencing caller pointer.
                unsafe {
                    bytes_read.write(0);
                }
            }
            0
        }
    }

    extern "win64" fn fake_write_file(
        handle: *mut c_void,
        buffer: *const u8,
        bytes_to_write: u32,
        bytes_written: *mut u32,
        _overlapped: *mut c_void,
    ) -> i32 {
        let fd = handle as usize as i32;
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        let n = unsafe { libc::write(fd, buffer as *const libc::c_void, bytes_to_write as usize) };
        if n >= 0 {
            if !bytes_written.is_null() {
                // SAFETY: calling host libc FFI or dereferencing caller pointer.
                unsafe {
                    bytes_written.write(n as u32);
                }
            }
            1
        } else {
            if !bytes_written.is_null() {
                // SAFETY: calling host libc FFI or dereferencing caller pointer.
                unsafe {
                    bytes_written.write(0);
                }
            }
            0
        }
    }

    extern "win64" fn fake_create_directory_w(path: *const u16, _sec: *const c_void) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        if std::fs::create_dir_all(&path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_remove_directory_w(path: *const u16) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        if std::fs::remove_dir(&path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_delete_file_w(path: *const u16) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        if std::fs::remove_file(&path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_delete_file_a(path: *const libc::c_char) -> i32 {
        let Some(path) = narrow_to_path(path) else {
            return 0;
        };
        if std::fs::remove_file(&path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_set_file_attributes_w(path: *const u16, _attr: u32) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        if path.exists() { 1 } else { 0 }
    }

    const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

    extern "win64" fn fake_get_file_attributes_w(path: *const u16) -> u32 {
        let Some(path) = wide_to_path(path) else {
            return INVALID_FILE_ATTRIBUTES;
        };
        match std::fs::metadata(&path) {
            Ok(meta) => {
                if meta.is_dir() {
                    FILE_ATTRIBUTE_DIRECTORY
                } else {
                    FILE_ATTRIBUTE_NORMAL
                }
            }
            Err(_) => INVALID_FILE_ATTRIBUTES,
        }
    }

    extern "win64" fn fake_get_file_attributes_a(path: *const libc::c_char) -> u32 {
        let Some(path) = narrow_to_path(path) else {
            return INVALID_FILE_ATTRIBUTES;
        };
        match std::fs::metadata(&path) {
            Ok(meta) => {
                if meta.is_dir() {
                    FILE_ATTRIBUTE_DIRECTORY
                } else {
                    FILE_ATTRIBUTE_NORMAL
                }
            }
            Err(_) => INVALID_FILE_ATTRIBUTES,
        }
    }

    #[repr(C)]
    struct WIN32_FILE_ATTRIBUTE_DATA {
        dw_file_attributes: u32,
        ft_creation_time: [u32; 2],
        ft_last_access_time: [u32; 2],
        ft_last_write_time: [u32; 2],
        n_file_size_high: u32,
        n_file_size_low: u32,
    }

    extern "win64" fn fake_get_file_attributes_ex_w(
        path: *const u16,
        _level: i32,
        out: *mut c_void,
    ) -> i32 {
        if out.is_null() {
            return 0;
        }
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        let Ok(meta) = std::fs::metadata(&path) else {
            return 0;
        };
        let len = meta.len();
        let data = WIN32_FILE_ATTRIBUTE_DATA {
            dw_file_attributes: if meta.is_dir() {
                FILE_ATTRIBUTE_DIRECTORY
            } else {
                FILE_ATTRIBUTE_NORMAL
            },
            ft_creation_time: [0, 0],
            ft_last_access_time: [0, 0],
            ft_last_write_time: [0, 0],
            n_file_size_high: (len >> 32) as u32,
            n_file_size_low: len as u32,
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            (out as *mut WIN32_FILE_ATTRIBUTE_DATA).write(data);
        }
        1
    }

    extern "win64" fn fake_get_disk_free_space_w(
        _dir: *const u16,
        sectors_per_cluster: *mut u32,
        bytes_per_sector: *mut u32,
        free_clusters: *mut u32,
        total_clusters: *mut u32,
    ) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            if !sectors_per_cluster.is_null() {
                sectors_per_cluster.write(8);
            }
            if !bytes_per_sector.is_null() {
                bytes_per_sector.write(512);
            }
            if !free_clusters.is_null() {
                free_clusters.write(10_000_000);
            }
            if !total_clusters.is_null() {
                total_clusters.write(20_000_000);
            }
        }
        1
    }

    extern "win64" fn fake_get_disk_free_space_ex_w(
        _dir: *const u16,
        free_bytes_available: *mut u64,
        total_bytes: *mut u64,
        total_free_bytes: *mut u64,
    ) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            if !free_bytes_available.is_null() {
                free_bytes_available.write(100 << 30);
            }
            if !total_bytes.is_null() {
                total_bytes.write(500 << 30);
            }
            if !total_free_bytes.is_null() {
                total_free_bytes.write(100 << 30);
            }
        }
        1
    }

    extern "win64" fn fake_find_first_file_w(path: *const u16, _data: *mut c_void) -> *mut c_void {
        let Some(path) = wide_to_path(path) else {
            return (-1isize) as *mut c_void;
        };
        if path.exists() {
            std::ptr::NonNull::dangling().as_ptr()
        } else {
            (-1isize) as *mut c_void
        }
    }

    extern "win64" fn fake_find_first_file_ex_w(
        path: *const u16,
        _info_level: i32,
        data: *mut c_void,
        _search_op: i32,
        _search_filter: *mut c_void,
        _flags: u32,
    ) -> *mut c_void {
        fake_find_first_file_w(path, data)
    }

    extern "win64" fn fake_move_file_ex_w(from: *const u16, to: *const u16, _flags: u32) -> i32 {
        let (Some(from), Some(to)) = (wide_to_path(from), wide_to_path(to)) else {
            return 0;
        };
        if std::fs::rename(from, to).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_set_current_directory_w(path: *const u16) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return 0;
        };
        if std::env::set_current_dir(path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_set_current_directory_a(path: *const libc::c_char) -> i32 {
        let Some(path) = narrow_to_path(path) else {
            return 0;
        };
        if std::env::set_current_dir(path).is_ok() { 1 } else { 0 }
    }

    extern "win64" fn fake_wfopen(path: *const u16, mode: *const u16) -> *mut libc::FILE {
        let Some(path) = wide_to_path(path) else {
            return std::ptr::null_mut();
        };
        let mode_str = if mode.is_null() {
            "rb".to_string()
        } else {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            let len = unsafe { (0..).take_while(|&i| *mode.add(i) != 0).count() };
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            let slice = unsafe { std::slice::from_raw_parts(mode, len) };
            String::from_utf16_lossy(slice).replace("ccs=UTF-8", "").replace(',', "")
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return std::ptr::null_mut(),
        };
        let c_mode = match std::ffi::CString::new(mode_str.trim().as_bytes()) {
            Ok(c) => c,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fopen(c_path.as_ptr(), c_mode.as_ptr()) }
    }

    extern "win64" fn fake_fopen(path: *const libc::c_char, mode: *const libc::c_char) -> *mut libc::FILE {
        let Some(path) = narrow_to_path(path) else {
            return std::ptr::null_mut();
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fopen(c_path.as_ptr(), mode) }
    }

    extern "win64" fn fake_wfopen_s(file: *mut *mut libc::FILE, path: *const u16, mode: *const u16) -> i32 {
        if file.is_null() {
            return 22;
        }
        let f = fake_wfopen(path, mode);
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            file.write(f);
        }
        if f.is_null() { 2 } else { 0 }
    }

    extern "win64" fn fake_fopen_s(file: *mut *mut libc::FILE, path: *const libc::c_char, mode: *const libc::c_char) -> i32 {
        if file.is_null() {
            return 22;
        }
        let f = fake_fopen(path, mode);
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe {
            file.write(f);
        }
        if f.is_null() { 2 } else { 0 }
    }

    extern "win64" fn fake_wfsopen(path: *const u16, mode: *const u16, _shflag: i32) -> *mut libc::FILE {
        fake_wfopen(path, mode)
    }

    extern "win64" fn fake_fread(ptr: *mut u8, size: usize, count: usize, stream: *mut libc::FILE) -> usize {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fread(ptr as *mut libc::c_void, size, count, stream) }
    }

    extern "win64" fn fake_fclose(stream: *mut libc::FILE) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fclose(stream) }
    }

    extern "win64" fn fake_close(fd: i32) -> i32 {
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::close(fd) }
    }

    extern "win64" fn fake_open(path: *const libc::c_char, flags: i32, mode: i32) -> i32 {
        let Some(path) = narrow_to_path(path) else {
            return -1;
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return -1,
        };
        let mut posix_flags = libc::O_CLOEXEC;
        if flags & 1 != 0 {
            posix_flags |= libc::O_WRONLY;
        } else if flags & 2 != 0 {
            posix_flags |= libc::O_RDWR;
        } else {
            posix_flags |= libc::O_RDONLY;
        }
        if flags & 0x100 != 0 {
            posix_flags |= libc::O_CREAT;
        }
        if flags & 0x200 != 0 {
            posix_flags |= libc::O_TRUNC;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::open(c_path.as_ptr(), posix_flags, (mode as libc::mode_t) | 0o666) }
    }

    extern "win64" fn fake_wopen(path: *const u16, flags: i32, mode: i32) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return -1;
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return -1,
        };
        let mut posix_flags = libc::O_CLOEXEC;
        if flags & 1 != 0 {
            posix_flags |= libc::O_WRONLY;
        } else if flags & 2 != 0 {
            posix_flags |= libc::O_RDWR;
        } else {
            posix_flags |= libc::O_RDONLY;
        }
        if flags & 0x100 != 0 {
            posix_flags |= libc::O_CREAT;
        }
        if flags & 0x200 != 0 {
            posix_flags |= libc::O_TRUNC;
        }
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::open(c_path.as_ptr(), posix_flags, (mode as libc::mode_t) | 0o666) }
    }

    extern "win64" fn fake_wstat64(path: *const u16, stat: *mut c_void) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return -1;
        };
        let Ok(meta) = std::fs::metadata(&path) else {
            return -1;
        };
        if !stat.is_null() {
            // SAFETY: calling host libc FFI or dereferencing caller pointer.
            unsafe {
                let ptr = stat as *mut u64;
                ptr.add(3).write(meta.len());
            }
        }
        0
    }

    extern "win64" fn fake_wremove(path: *const u16) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return -1;
        };
        let _ = std::fs::remove_file(&path).or_else(|_| std::fs::remove_dir(&path));
        0
    }

    extern "win64" fn fake_remove(path: *const libc::c_char) -> i32 {
        let Some(path) = narrow_to_path(path) else {
            return -1;
        };
        let _ = std::fs::remove_file(&path).or_else(|_| std::fs::remove_dir(&path));
        0
    }

    extern "win64" fn fake_wchmod(path: *const u16, _mode: i32) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return -1;
        };
        if path.exists() { 0 } else { -1 }
    }

    extern "win64" fn fake_wutime64(path: *const u16, _times: *mut c_void) -> i32 {
        let Some(path) = wide_to_path(path) else {
            return -1;
        };
        if path.exists() { 0 } else { -1 }
    }

    extern "win64" fn fake_fiopen_wide(path: *const u16, mode: i32, _prot: i32) -> *mut libc::FILE {
        let mode_str = match mode & 0x3f {
            1 => c"rb",
            2 => c"wb",
            3 => c"w+b",
            8 | 10 => c"ab",
            _ => c"r+b",
        };
        let Some(path) = wide_to_path(path) else {
            return std::ptr::null_mut();
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fopen(c_path.as_ptr(), mode_str.as_ptr()) }
    }

    extern "win64" fn fake_fiopen_narrow(path: *const libc::c_char, mode: i32, _prot: i32) -> *mut libc::FILE {
        let mode_str = match mode & 0x3f {
            1 => c"rb",
            2 => c"wb",
            3 => c"w+b",
            8 | 10 => c"ab",
            _ => c"r+b",
        };
        let Some(path) = narrow_to_path(path) else {
            return std::ptr::null_mut();
        };
        let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: calling host libc FFI or dereferencing caller pointer.
        unsafe { libc::fopen(c_path.as_ptr(), mode_str.as_ptr()) }
    }
}
