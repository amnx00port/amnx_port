//! Our implementation of the Xbox system API surface.
//!
//! A title imports from three kinds of DLLs, and each gets its own policy:
//!
//! | kind | examples | handled by |
//! |------|----------|------------|
//! | Windows API sets the console shares with PC | `api-ms-win-core-synch-*`, `ws2_32.dll` | host pass-through, overridden here where console behaviour differs (paths, memory) |
//! | console-only system APIs | `xgameruntime.dll`, `xmem.dll`, `d3d12_xs.dll` | implemented here, in [`modules`] |
//! | anything else | | an unimplemented-import stub from the loader |
//!
//! Resolution order for every import: an implementation in [`modules`],
//! then the host if [`HostPolicy`] allows that DLL, then `Missing`.

mod host;
pub mod modules;
mod registry;
pub mod runtime;

pub use host::HostPolicy;
pub use registry::{HleExport, HleModule, registry};
pub use victory_gpu::console;

use victory_loader::{ImportResolver, Provenance, Resolution};
use victory_pe::ImportSymbol;

/// Resolves a title's imports against our runtime and the host.
#[derive(Debug, Default)]
pub struct HleResolver {
    pub host: HostPolicy,
}

impl HleResolver {
    pub fn new(host: HostPolicy) -> Self {
        HleResolver { host }
    }

    /// Where an import would come from, without binding anything.
    pub fn classify(&self, dll: &str, symbol: &ImportSymbol) -> Option<Provenance> {
        match self.resolve(dll, symbol) {
            Resolution::Found { provenance, .. } => Some(provenance),
            Resolution::Missing => None,
        }
    }
}

impl ImportResolver for HleResolver {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        // Ordinal imports look up `#n`.
        if let Some(address) = registry::lookup(dll, &symbol.to_string()) {
            return Resolution::Found {
                address,
                provenance: Provenance::Hle,
            };
        }
        match self.host.resolve(dll, symbol) {
            Some(address) => Resolution::Found {
                address,
                provenance: Provenance::Host,
            },
            None => Resolution::Missing,
        }
    }
}

// The registry is code-only, so the "Missing" outcome can be checked on any
// host; host pass-through itself needs Windows.
#[cfg(test)]
mod tests {
    use super::*;

    fn name(n: &str) -> ImportSymbol {
        ImportSymbol::Name {
            hint: 0,
            name: n.into(),
        }
    }

    #[test]
    fn console_api_resolves_to_hle() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("XGameRuntime.dll", &name("XGameRuntimeInitialize")),
            Some(Provenance::Hle)
        );
    }

    #[test]
    fn strict_policy_reports_missing() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-1-0.dll", &name("Sleep")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn shared_api_sets_pass_through_to_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-2-0.dll", &name("Sleep")),
            Some(Provenance::Host)
        );
        // Console-only DLLs never pass through.
        assert_eq!(
            r.classify("xmem.dll", &name("XMemSetAllocationHooks")),
            None
        );
    }

    #[cfg(windows)]
    #[test]
    fn overrides_win_over_host() {
        let r = HleResolver::default();
        assert_eq!(
            r.classify(
                "api-ms-win-core-libraryloader-l1-2-0.dll",
                &name("GetModuleHandleW")
            ),
            Some(Provenance::Hle)
        );
    }

    // Off Windows the two tests above are gated (pass-through needs the
    // host's DLLs). Precedence and the policy itself are code-only.

    #[cfg(not(windows))]
    #[test]
    fn overrides_win_and_allowed_host_dlls_go_missing_on_any_host() {
        let r = HleResolver::default();
        // Our override answers before the policy is consulted, at any
        // API-set version and in any case.
        for dll in [
            "api-ms-win-core-libraryloader-l1-2-0.dll",
            "API-MS-WIN-CORE-LIBRARYLOADER-L1-1-1.DLL",
        ] {
            assert_eq!(
                r.classify(dll, &name("GetModuleHandleW")),
                Some(Provenance::Hle),
                "{dll}"
            );
            assert_eq!(
                HleResolver::new(HostPolicy::none()).classify(dll, &name("GetModuleHandleW")),
                Some(Provenance::Hle),
                "{dll}"
            );
        }
        // Allowed for pass-through, but there is no host to pass to.
        assert!(r.host.allows("api-ms-win-core-synch-l1-2-0.dll"));
        assert_eq!(
            r.classify("api-ms-win-core-synch-l1-2-0.dll", &name("Sleep")),
            None
        );
        assert_eq!(r.classify("ws2_32.dll", &ImportSymbol::Ordinal(115)), None);
        // Console-only DLLs are never allowed through.
        assert!(!r.host.allows("xmem.dll"));
        assert_eq!(
            r.classify("xmem.dll", &name("XMemSetAllocationHooks")),
            None
        );
    }

    #[test]
    fn host_policy_patterns_are_exact_or_prefix_and_ignore_case() {
        let policy = HostPolicy::default();
        for allowed in [
            "KERNEL32.DLL",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "vcruntime140_1.dll",
            "msvcp140_atomic_wait.dll",
            "Ext-MS-Win-RTCore-NTUser-Window-L1-1-0.dll",
        ] {
            assert!(policy.allows(allowed), "{allowed}");
        }
        for refused in [
            // Exact entries are not prefixes, and carry their extension.
            "kernel32.dll.bak",
            "kernel32",
            "xkernel32.dll",
            // A prefix entry matches from the start only.
            "my-api-ms-win-core-synch.dll",
            "xgameruntime.dll",
            "d3d12_xs.dll",
            "wininet.dll",
            "",
        ] {
            assert!(!policy.allows(refused), "{refused}");
        }
        assert!(!HostPolicy::none().allows("kernel32.dll"));
        let star = HostPolicy {
            allow: vec!["*".into()],
        };
        assert!(star.allows("anything.dll") && star.allows(""));
    }
}
