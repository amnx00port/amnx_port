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
    pub(super) fn resolve(_dll: &str, _symbol: &victory_pe::ImportSymbol) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(text: &str) -> ImportSymbol {
        ImportSymbol::Name {
            hint: 0,
            name: text.into(),
        }
    }

    /// What the console shares with the PC passes through; console-only
    /// DLLs and look-alike names never do.
    #[test]
    fn the_default_policy_is_an_allow_list() {
        let policy = HostPolicy::default();
        for dll in [
            "KERNEL32.DLL",
            "kernelbase.dll",
            "ntdll.dll",
            "ucrtbase.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "msvcp140.dll",
            "msvcp_win.dll",
            "api-ms-win-core-synch-l1-2-0.dll",
            "api-ms-win-eventing-provider-l1-1-0.dll",
            "api-ms-win-security-base-l1-2-0.dll",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "ext-ms-win-rtcore-ntuser-window-l1-1-0.dll",
            "api-ms-win-shcore-scaling-l1-1-1.dll",
            "ws2_32.dll",
            "bcrypt.dll",
            "crypt32.dll",
            "ncrypt.dll",
            "winhttp.dll",
            "sspicli.dll",
            "cabinet.dll",
            "avrt.dll",
            "dbghelp.dll",
            "xaudio2_9.dll",
            "mfplat.dll",
            "mfreadwrite.dll",
            "ole32.dll",
            "advapi32.dll",
            "iphlpapi.dll",
        ] {
            assert!(policy.allows(dll), "{dll}");
        }
        for dll in [
            "",
            "xmem.dll",
            "xgameruntime.dll",
            "d3d12_xs.dll",
            "d3d11.dll",
            "dxgi.dll",
            "wininet.dll",
            "urlmon.dll",
            "user32.dll",
            "xinput1_4.dll",
            "kernel32.dll.evil",
            "kernel32",
            "notkernel32.dll",
            "api-ms-win-corex.dll",
            "msvcp.dll",
        ] {
            assert!(!policy.allows(dll), "{dll}");
        }
        assert!(!HostPolicy::none().allows("kernel32.dll"));
        assert!(!HostPolicy::none().allows(""));
    }

    /// A pattern is a whole name or, with a trailing `*`, a prefix; a DLL
    /// name's case never matters.
    #[test]
    fn dll_names_match_case_insensitively_by_name_or_prefix() {
        let policy = HostPolicy {
            allow: vec!["kernel32.dll".into(), "api-ms-win-core-*".into()],
        };
        assert!(policy.allows("Kernel32.DLL"));
        assert!(policy.allows("api-ms-win-core-file-l1-1-0.dll"));
        assert!(policy.allows("API-MS-WIN-CORE-FILE-L1-1-0.DLL"));
        assert!(!policy.allows("kernel32.dll.old"));
        assert!(!policy.allows("kernel32"));
        assert!(!policy.allows("my-kernel32.dll"));
        assert!(!policy.allows("api-ms-win-core"));
        assert!(!policy.allows("api-ms-win-corex.dll"));
        assert!(!policy.allows("api-ms-win-eventing-provider-l1-1-0.dll"));

        assert!(!HostPolicy { allow: Vec::new() }.allows("kernel32.dll"));
    }

    /// Allowed DLLs resolve to the host's functions; console-only names
    /// and nameless lookups resolve to nothing.
    #[cfg(windows)]
    #[test]
    fn allowed_dlls_resolve_to_the_hosts_own_functions() {
        let policy = HostPolicy::default();
        let tick = policy.resolve("kernel32.dll", &name("GetTickCount"));
        assert!(tick.is_some());
        assert_eq!(
            policy.resolve("KERNEL32.DLL", &name("GetTickCount")),
            tick,
            "one module per DLL, whatever the case"
        );
        assert_eq!(function("kernel32.dll", "GetTickCount"), tick);
        assert_eq!(policy.resolve("kernel32.dll", &name("NotAnExport")), None);
        assert_eq!(function("kernel32.dll", "NotAnExport"), None);

        // The policy decides, not the presence of the name on this host.
        assert_eq!(
            policy.resolve("xmem.dll", &name("XMemSetAllocationHooks")),
            None
        );
        assert_eq!(
            HostPolicy::none().resolve("kernel32.dll", &name("GetTickCount")),
            None
        );
        // An allow-listed name the host does not have is a miss, not a panic.
        assert_eq!(
            policy.resolve("api-ms-win-core-nothing-l1-1-0.dll", &name("Nothing")),
            None
        );
        // Winsock keeps its documented ordinals.
        assert!(
            policy
                .resolve("ws2_32.dll", &ImportSymbol::Ordinal(1))
                .is_some()
        );
        // GetProcAddress cannot take an empty or split name.
        assert_eq!(policy.resolve("kernel32.dll", &name("")), None);
        assert_eq!(
            policy.resolve("kernel32.dll", &name("GetTick\0Count")),
            None
        );
    }
}
