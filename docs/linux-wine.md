# Linux with Wine/Proton (experimental)

VIctory is a Windows application: the title loader uses Windows APIs and the
GPU backend uses D3D12. Linux reports have successfully run it through Wine
with vkd3d-proton translating D3D12 to Vulkan. Cross-build the Windows
executable; native Linux builds are not currently supported.

## Reported working setup

A community report (#879) used Arch/CachyOS, an AMD RADV GPU, Wine 11.22, and
vkd3d-proton 3.2.0 from Proton Experimental. DXVK supplied `dxgi.dll` (and
`d3d11.dll`/`d3d10core.dll`); the report used:

```sh
WINEDLLOVERRIDES="d3d12=n,b;d3d12core=n,b;dxgi=n,b"
```

The `n,b` overrides are important: they select the native Proton DLLs before
Wine's built-ins. Use a vkd3d-proton version compatible with your GPU and
driver. The report found 3.1 crashed during world loading, while 3.2.0 reached
menus and loading. This is community-reported, not a tested or supported
configuration; other reports describe crashes and severe performance issues.

## Build and launch

Enter the project shell (Nix with `<nixpkgs>` available), then cross-compile
the Windows MSVC target. The shell provides Rustup, `cargo-xwin`, Vulkan
tools, Clang, LLVM, and LLD:

```sh
nix-shell
cargo xwin build --release -p victory --target x86_64-pc-windows-msvc
```

Rustup reads `rust-toolchain.toml` and installs the pinned toolchain/target on
first use. Outside Nix, install Rustup and `cargo-xwin` yourself.

The project shell prefers an installed GE-Proton compatibility tool (on this
machine it selects `GE-Proton-nix`). The helper builds if needed, creates a
dedicated prefix outside the repository, and launches through GE-Proton's own
runner. This is the path verified with the supplied game installation:

```sh
nix-shell
./scripts/run-ge-proton.sh
```

The helper defaults to the game executable path supplied for this machine.
For another installation, pass the executable or set `VICTORY_TITLE_EXE`.
Override `GE_PROTON` to select another installed GE-Proton release. A GE-Proton
installation can be discovered under Steam's `compatibilitytools.d`. The helper
uses GE-Proton directly because that path was verified with the supplied game
files; UMU is another option but requires explicit pressure-vessel access for
game files outside the home directory.

## Controller

VIctory reads up to four XInput gamepads. Proton should expose an Xbox or
Xbox-compatible controller as XInput; connect it before starting the title.
The gamepad path includes buttons, D-pad, sticks, and analog triggers. Basic
motor rumble is sent through XInput, but the title's trigger/adaptive rumble
is not implemented.

The launch helper runs GE-Proton outside the Steam client, so Steam Input's
virtual controller mapping is not automatically applied. If your controller
is not seen, or it is not XInput-compatible, add the built `victory.exe` to
Steam as a non-Steam game, force the installed GE-Proton compatibility tool,
enable Steam Input, and use these launch options (adjust the game path if
needed):

```text
WINEDLLOVERRIDES="d3d12=n,b;d3d12core=n,b;dxgi=n,b" %command% run "Z:\mnt\data\Games\GTA6\GTA6.Xbox.v1.0.2.1\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe" --execute
```

As a fallback, when no gamepad is connected, the keyboard acts as a pad while
the title window is focused: WASD/IJKL are the sticks, arrows the D-pad,
Enter/Backspace/X/Y the face buttons, Q/E the shoulders, Z/C the triggers,
F/H the stick clicks, and Esc/Tab Menu/View.

For manual Wine/Proton launches, use a win64 prefix with the Proton versions of
`d3d12.dll`, `d3d12core.dll`, and DXVK's `dxgi.dll` available to that prefix.
The exact installation method depends on whether you use system Wine or
Proton; do not mix arbitrary Wine built-ins with the Proton D3D12 stack.
The shell sets the reported DLL overrides by default; Wine falls back to its
built-ins if native DLLs are not installed, which is not a working D3D12 setup.

Run from an active graphical session. For a KDE Wayland session launched
remotely, one report needed the session's `XDG_RUNTIME_DIR`, `DISPLAY` (for
XWayland), and `XAUTHORITY`; obtain these from that session rather than
hard-coding another user's runtime directory.

For example, with a prefix already configured and the game directory visible
to Wine as `Z:\path\to\game`:

```sh
export WINEPREFIX="$HOME/.wine-victory"
export WINEDLLOVERRIDES="d3d12=n,b;d3d12core=n,b;dxgi=n,b"
wine target/x86_64-pc-windows-msvc/release/victory.exe config
wine target/x86_64-pc-windows-msvc/release/victory.exe run \
  'Z:\path\to\game\game.exe' --execute
```

Configure paths in `victory.toml` (or `VICTORY_*_DIR` variables) so they are
valid inside the Wine process. Keep the repository, game data, and derived
cache/data directories separate as required by the asset policy.

## Current limitations

Linux/Wine reports have reached responsive menus, but performance can be
extremely low and the world is not reliably playable. Missing shaders and
indirect draws are still dropped; the title/runtime also has reported GPU
crashes and hangs. Enabling indirect execution may require Microsoft's
`d3dcompiler_47.dll`, since some Wine versions cannot compile the required
`cs_5_1` shader. No Linux setup is currently part of the project's tested
platform matrix.
