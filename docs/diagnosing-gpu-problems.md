# Diagnosing GPU problems

How to narrow down a hang, a removed device, a missing image or a slow
frame in the title, with the switches the [README](../README.md#debugging)
lists. [gpu.md](gpu.md) explains the parts named here.

Build with diagnostics first: most of what follows needs it.

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

## The lines to read first

Every 5 seconds the log has one line for the frames since the last one
(wrapped here):

```
<fps> fps, <ms> ms/frame (<cpu> preparing: copies <ms> ms <MB> MB, textures <ms> ms <MB> MB;
<gpu> on the GPU; <ms> copying back <MB> MB), <n> draws/frame,
dropped <n> without a host pipeline (<n> being made) and <n> indirect
```

* **preparing** is CPU time before submissions: copies of title memory
  into host copies, and texture detiling. **on the GPU** is time waiting
  for submissions. **copying back** is what shaders wrote going back into
  title memory. Their sum against the frame time shows where a slow frame
  goes. With diagnostics, a breakdown of the copies follows (`copystat`).
* **dropped without a host pipeline** counts draws and dispatches that did
  not run. While pipelines are **being made**, drops are expected and fall
  as the pipeline library fills (a second run loads what the first made).
  With diagnostics, the reasons follow, one line each:
  `dropped <n>/frame without a host pipeline: <reason>`.
* **indirect** counts indirect executions dropped.

At every power-of-two frame, `frame presented` gives draws and dispatches
recorded against those run on the host. Past frame 256 it also gives the
shader cache's hits and misses.

`crashes.log` in the data folder keeps every crash and host GPU failure,
from any build.

## The device was removed, or the screen went black

Windows resets the display driver when a submission runs for about two
seconds. Every monitor goes black for a moment and the run ends with
`host GPU device removed after a submission of <n> lists; device removed:
<reason>`.

1. **Which command.** `VICTORY_D3D12_DRED=1` adds the host's device-removed
   data: a faulting address with the allocation or title memory copy it
   lies in, and for each list that did not finish, the command it stopped
   at and the pipelines its draws and dispatches use. On an adapter
   without that data (the software one), the last six commands each list
   noted are given instead.
2. **Which list, how slow.** `VICTORY_GPU_TIMING=1` runs each list of a
   submission on its own and times it. Lists over 50 ms are logged with
   their slowest commands. Submissions over 250 ms are logged as `slow host
   submission` with their pipelines, even without it.
3. **A loop that never ends?** A hang that goes away, or becomes a slow
   frame, with a low `VICTORY_SHADER_LOOP_LIMIT` (say 1024, or 64) is a
   translated loop whose exit condition is never met. A barrier that can
   never be passed does not end because loops do, unless it sits inside
   one. A loop that waits for memory another workgroup or an earlier
   dispatch writes points at what makes that write visible.
4. **One program?** `VICTORY_SKIP_PIPELINES=<hash>,...` drops the draws and
   dispatches whose pipeline contains a listed shader. If the hang moves
   elsewhere, the hash was not the only one. Some titles make many
   variants of one program.
5. **Safely.** `VICTORY_D3D12_WARP=1` renders on the software adapter,
   which cannot reset the driver. It is slow, and its timing says nothing
   about real GPUs.

## A pipeline is refused

`host pipeline refused` gives the host's error and the shaders. With
`VICTORY_D3D12_DEBUG=1` the debug layer's reason is logged right after it.
Known causes:

| Debug layer says | Meaning |
|------------------|---------|
| `Shader wants wave lane count of 64 but maximum wave lane count for device is 32` | The program is emitted with `WaveSize(64)` (see [gpu.md](gpu.md#shaders)) and this GPU runs 32-lane waves only, as NVIDIA's do. Such programs run on GPUs with 64-lane waves. |
| a validation error | The emitted DXIL is invalid. `victory translate --validate` checks every program the title has with `dxil.dll`'s validator, without running the title. |

`no host pipeline: <reason>` (logged once per reason) means a shader did
not translate. `victory translate` tallies the reasons over all of the
title's shaders without running it, and `--list` prints one line per
program.

## The image is black or missing

`VICTORY_CAPTURE=<dir>` saves presented frames, and with
`VICTORY_DUMP_TARGETS=1` every render target of those frames. A target
that holds the scene while the presented frame does not shows that a
later pass fails. That pass is often one whose draws or dispatches are
dropped or refused (see above).

## The title crashed

A crash report names the faulting code as `module+rva` and unwinds the
faulting thread. A fault in the title's own code is often a consequence:
of data our GPU work wrote wrongly or did not write, or of an import we
answer differently. A break exception (`0x80000003`) is either the stub of
an import we have not written (`--unimplemented trap`, the default), which
names the import, or the title's own check failing, which its stack
shows. `VICTORY_STACKS`,
`VICTORY_WATCH` and `VICTORY_CRASH_SAVE` help from there.

## Reporting what you found

Give the commit or branch, the GPU and driver, the console mode, the
switches set, and what the log says: error text, counts, shader hashes,
`module+rva`. Say what you did not test.

Nothing from the game goes into an issue or the repository: no shader
code or disassembly, no dumps, no captured frames
([asset policy](asset-policy.md)).
