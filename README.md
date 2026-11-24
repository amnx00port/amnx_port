# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

## Status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `VICTORY_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |
| `VICTORY_PROFILE=1` or `=<dir>` | any build: frame times, memory, caches and loads; see [Profiling](#profiling) |
| `VICTORY_PROFILE_OVERLAY=1` | any build: the same with an on-screen panel, F3 to hide and show it; see [Profiling](#profiling) |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder.

## Profiling

Any build, release included, profiles a run with `VICTORY_PROFILE` set;
unset, it costs one flag test per hook.

| Variable | Effect |
|----------|--------|
| `VICTORY_PROFILE=1` | a summary in the log (target `victory_profile`) every 5 s |
| `VICTORY_PROFILE=<dir>` | the same, plus `<dir>\frames.csv` and `<dir>\trace.json` (outside the repository) |
| `VICTORY_PROFILE_OVERLAY=1` or `VICTORY_PROFILE=overlay` | the summaries, plus a panel in the window's top left corner (with `VICTORY_PROFILE=<dir>`, the files too) |
| `VICTORY_PROFILE_INTERVAL=<seconds>` | time between summaries |
| `VICTORY_PROFILE_HITCH_MS=<ms>` | a frame interval above this is a hitch (default 50) |
| `VICTORY_PROFILE_TRACE_US=<us>` | timed scopes shorter than this are counted, not traced (default 100) |
| `VICTORY_PROFILE_DUMP=30,90` | a full report at those seconds (with every thread's stack in a diagnostics build); one is also logged when the host GPU fails |

A summary has the frame rate and 1% low, frame interval avg, p50, p95,
p99 and max, hitches and stutters (over twice the median); per frame, the
time preparing submissions, waiting for the GPU and presenting, and the
submissions, draws and dispatches; process memory, video memory and its
budget, and resident bytes (GPU memory, host textures, shader containers,
the default allocator); cache hits and misses (shader containers, the
shader disk cache, host pipelines, detiled textures); the timed scopes that took longest
(shader translation, host pipelines, `ReadFile`, `XMemAlloc`, audio
decoding...); and loads (file reads, DirectStorage requests) with the
biggest and slowest of the period.

`frames.csv` has one row per presented frame. Its `frame` column is the
present count `VICTORY_CAPTURE` names its files by, so a captured frame
finds its row. `trace.json` opens in <https://ui.perfetto.dev> or
`chrome://tracing`: a row of frames, each thread's timed scopes and loads,
and counters for frame times and memory. It stays readable when a run
ends abruptly. Timing a new place is one line:

```rust
victory_profile::scope!("gpu", "my pass", bytes); // times the rest of the block
```

The overlay shows, redrawn four times a second: the frame rate, average
frame time, 1% low and longest frame of the last 120 frames, with a
graph of them (newest at the right, red above `VICTORY_PROFILE_HITCH_MS`,
a line at it); the GPU wait, draws and submissions per frame; the
process's working set and commit; video memory used, its budget and
shared memory; each cache's hit rate (shader containers, the shader disk
cache, host pipelines, detiled textures); and file and DirectStorage
read rates in MB/s over the last second. F3, while the window has focus,
hides and shows it; the title sees the key as well. It is copied into the
back buffer after the frame (no shaders), so `VICTORY_CAPTURE` images do
not have it, and it scales with the window: twice as big from 1440 lines.

`victory_profile`'s documentation lists the other hooks (`count!`,
`cache!`, `resident!`, `load`).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
