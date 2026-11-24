# Architecture

user's files (outside repo)             VIctory
───────────────────────────             ─────────────────────────────────────────
game assets ──────────────────────────► victory-pipeline + victory-game
(victory-formats, victory-shader)
│
▼
cache_dir: converted data
│
game executable ─► victory-loader ────► title code (native x86-64)
prepare, map, bind      │ imports
▼
victory-hle ── host Windows (shared APIs)
│ D3D12.X
▼
victory-gpu ─► host D3D12
│ shaders at pipeline creation
▼
victory-shader ◄─► cache_dir

Shaders are translated when the title creates a pipeline, using the byte data it passes from wherever it was loaded (executable, archive, or memory). Converting them ahead of time from files acts as a preliminary warm-up step.

## Crates

| Crate | Role |
|-------|------|
| `victory-core` | Configuration, asset-policy guard, and system logging. |
| `victory-pe` | Read-only PE32+ and import-library analysis. |
| `victory-loader` | Prepares, maps, binds, and starts the title image. |
| `victory-hle` | Console API implementations and host pass-through policy. |
| `victory-gpu` | Coordinates D3D12.X objects, command lists, and GPU memory mapping to the host D3D12 interface. |
| `victory-shader` | Translates Xbox shader containers into host-compatible shaders. |
| `victory-formats` | Handles GPU texture layouts (sizes, tiling, detiling) and audio codecs; BCPack support is planned. |
| `victory-pipeline` | Manages the converter trait and content-addressed cache data. |
| `victory-game` | Handles Grand Theft Auto VI-specific formats, build identities, and custom hooks. |
| `apps/victory` | Serves as the primary command-line front end. |
| `xtask` | Runs asset-policy checks and manages Git hooks. |
| `tests/fixtures/hle-probe` | Acts as a stand-in "title" using internal code specifically for loader validation tests. |

Everything except `victory-game` operates at the platform level and can be tested against any Xbox Series title.

## Import Resolution

For every `dll!function` the title attempts to import, resolution occurs in the following priority order:

1. Our custom emulation implementation inside `victory-hle::modules` (always takes precedence);
2. The host operating system's native function, provided the active `HostPolicy` explicitly allows that specific DLL;
3. An exported function from a local DLL shipped directly alongside the game executable, mapped identically to the executable (restricted to DLLs not handled by options 1 or 2);
4. A dynamically generated fallback stub that identifies the import name when called. Running with the `--unimplemented trap` flag triggers a debugger break and aborts execution, while the `zero` flag logs the occurrence once and returns 0.

Invoking `LoadLibrary*` on a console-specific DLL implemented by this layer (such as `xgameruntime.dll`) yields a pseudo-handle, allowing subsequent `GetProcAddress` requests to locate our native overrides. 

Calling `LoadLibrary*` on a specific title DLL returns its mapped memory image, initializing the mapping process immediately if no other components have imported it yet. Both by explicit name and standard import pathways, a game title can only access host system DLLs that `HostPolicy` safely permits, and calling `GetProcAddress` on those modules redirects back to our override hooks where binding matching occurs.

The `QueryApiImpl` subsystem distributes our runtime objects based on the requested provider and interface ID (the exact coverage layout is detailed under `victory-hle::modules::xgameruntime`). Connected or online service requests are answered as an offline console instance. Game saves are stored as flat files directed inside `VICTORY_DATA_DIR`.

Title storage is mounted as a virtual drive matching the console interface, allowing titles to construct internal file paths seamlessly. The `victory-hle::modules::files` subsystem binds to every path-dependent import routine (including `kernel32` configurations and the standard C/C++ runtime engines), automatically rewriting paths targeting the virtual drive over to the designated host destination directory under `VICTORY_DATA_DIR`. Unmapped paths pass directly through to the host environment without adjustments.

An executable's core code accesses its static Thread Local Storage (TLS) allocation as slot 0. When mapped explicitly as a DLL, it is assigned an alternate slot identifier, causing the loader environment to smoothly redirect those memory loads via `victory-loader::tls`. Running the execution command as `victory run --execute` injects an active crash reporting engine that classifies system faults using `module+rva` markers while safely unwinding the faulting thread's calling stack through `victory-loader::crash`.

Executing `victory census` details this entire structure and module classification breakdown statically without executing any target files (it completely avoids mapping the title DLL dependencies, whereas executing `victory run` without the `--execute` flag will map them).

## Shaders

The `victory_shader::ir::build` routine converts a native program into Static Single Assignment (SSA) form, mapping exactly one intermediate instruction to every machine instruction. The `victory_shader::structure` layer then strips out `EXEC` parameters: lane masks transition into distinct per-lane booleans, wave-wide mask tests evolve into structured per-lane conditional branches, and standard vector instructions maintain operations only when a specific execution lane can be reached while its active `EXEC` bit is clear. 

The `victory_shader::resource` component traces every active descriptor reference and scalar memory address back to its fundamental root configuration parameter (whether originating from an internal table, target heap, root descriptor, or fixed constant). A resource descriptor read by a waterfall loop from the primary lane (`v_readfirstlane_b32`) is tracked directly back to the specific per-lane memory load or conditional evaluation that assigns each execution lane its unique values (such as a condition mask's `v_cndmask_b32`, or a conditional branch whose individual paths set it via structurisation), enabling access using a non-uniform heap evaluation index in a single pass. 

Running `victory translate` scans every discovered shader asset container (including packaged containers and structured `SGD5` files) through these verification layers, generating a comprehensive report itemizing structural elements the translation layer cannot successfully process yet.
