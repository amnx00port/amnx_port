# GPU: from title memory to the host GPU

How `victory-gpu` runs the title's D3D12.X work on host D3D12, in the order
a frame goes through it. The module docs hold the details; this page ties
them together.

## Title memory

Console memory is unified: a title GPU address is a CPU address, and the
title's resources are plain memory it reads and writes from its threads
while the GPU runs. The host GPU cannot read title memory, so:

* a **buffer** a draw or dispatch uses gets a persistent host copy of its
  size on first use (`shadow`). A title address inside the buffer is the
  same offset in its copy. Memory a title uses as GPU memory outside any
  buffer object gets copies of the aligned windows it touches;
* a **texture** gets a host texture. Contents the title writes itself
  (streamed pages, movie frames, glyph atlases) are detiled from title
  memory into a staging buffer and copied in when they changed
  (`texture`);
* **descriptor heaps** are title memory too. Each shader-visible heap has a
  host heap in the same order, and slot `i` is rebuilt from what the title
  wrote at slot `i` (`descriptor::host`).

## Knowing what the title wrote

Copying every buffer before every submission is far too slow in the world,
so only the 64 KiB pages the title wrote since their last copy are copied
(`watch`):

* **Write-watched memory.** `victory-hle`'s xmem reserves the title's GPU
  memory with `MEM_WRITE_WATCH` where it can (not placeholders for
  physical pages), so the kernel keeps a dirty bit per 4 KiB page.
  Asking for a range's written pages also clears them. Host copies of the
  same memory overlap (nested buffers, textures in buffers), so every
  answer goes into a journal: each page's last write, numbered by the
  asking. Each copy remembers, per page, the asking after which it was
  last copied.
* **Physical-page views** (xmem's sections) cannot be write-watched. After
  a copy takes a 64 KiB block, the block is made read-only. A vectored
  exception handler takes the title's first write to it, notes the block
  written and gives write access back. The runtime reports when it maps
  or unmaps views (`title_memory_remapped`), which resets their state.
* **System calls into a guarded block** fail instead of faulting into the
  handler (`ReadFile` returns `ERROR_NOACCESS`). So the runtime opens the
  blocks first with `victory_gpu::kernel_will_write`: `ReadFile` and
  `fread` in `files.rs`, and DirectStorage's raw reads in `dstorage.rs`.
* Memory that is neither is copied whole.

## A submission, step by step

1. **Recording.** The title's inlined code writes packets only while a
   list's put pointer is below its limit. Ours keep the limits at zero, so
   every command arrives as a call. Each call records onto a host command
   list. A draw or dispatch first brings the host list up to date: the
   host pipeline and our root signature, the user data the translated
   shaders read, the host copies and descriptor slots they read, and the
   states of the textures involved (`draw`).
2. **Ranges to refresh and ranges the GPU writes.** Each draw declares the
   buffer ranges it reads (vertex and index buffers, constant buffers,
   root views, buffer views in tables) and the ranges its shaders may
   write (unordered access).
3. **At submission:** each declared read range is refreshed from title
   memory (the written pages only), changed textures are detiled, and the
   bytes of every range the GPU may write are kept as a snapshot.
4. **On the host GPU.** The lists run and the submission is waited for:
   CPU and GPU never overlap (a `shortcut:` in `host.rs`). After the wait
   the device is checked. A removed device ends the run with a report
   (see [diagnosing](diagnosing-gpu-problems.md)).
5. **Copy-back.** For each range shaders wrote, the copy is compared with
   its snapshot, and the bytes the GPU changed go back into title memory.
   Pages that are not mapped are left alone.
6. **Completion.** Only then do the lists' bottom-of-pipe writes land:
   the fence and timestamp values that tell the title the GPU is done.

### Ordering inside a list

D3D12 starts the commands of one host list in order, but draws and
dispatches may overlap: a later one may read what an earlier one has not
finished writing, unless a barrier separates them. What the title's
ordering commands become on the host:

| Title command | On the host |
|---------------|-------------|
| Resource barrier on a texture | its transition |
| Resource barrier on a buffer | a UAV barrier on every resource: buffers have host copies, which a barrier cannot name |
| Enhanced barriers (`Barrier`) | a UAV barrier on every resource |
| Pipeline syncs (list slots 129 and 130) | nothing |
| GPU wait for a memory value | nothing: lists run to completion, so another queue's or the CPU's write cannot be waited for |

Texture uses by the host (clears, copies) move the host texture to the
state they need first (`list::ensure_state`).

## Pipelines

Titles create far more pipelines than they draw with, so a host pipeline
is made on a pipeline's first draw or dispatch, not at creation
(`pipeline`). Workers make them off the title's threads (`pipeline::make`).
A first use waits for its pipeline within a budget per frame. Past it, a
draw or dispatch whose pipeline is not ready is dropped, as an
untranslated one is. The 5-second frame line counts those draws and the
pipelines still being made.

A host pipeline goes into the host's pipeline library, named by a hash of
its host description: the translated code, our root signature and the
graphics state (`pipeline::library`). The library is saved in the cache
folder, so a second run loads instead of compiling. The host refuses a
library from another adapter or driver.

A pipeline the host refuses to create is logged as `host pipeline refused`
and its draws are dropped.

## Shaders

The translator (`victory-shader`, see [architecture](architecture.md#shaders))
works on the console's GFX10 code and emits DXIL for shader model 6.6,
which indexes the descriptor heaps directly. Translations are kept on
disk, keyed by the container's content and by the build of VIctory that
made them (`shader_cache`). Points that matter on the host:

* **Wave size.** Console programs run in 32- or 64-lane waves. A compute
  program whose meaning depends on its lanes running together (reading
  other lanes, lane masks read as numbers, or one wave sharing LDS) is
  emitted with `WaveSize` set to the console's width (`emit::lower`). A
  host whose waves are 32 lanes wide refuses a pipeline that requires 64.
* **Loops.** Every translated loop stops after a number of iterations per
  lane, 2^20 by default, so a loop whose exit condition is never met ends
  instead of running forever. Nested loops multiply, so such a loop can
  still run long enough for the host to reset the driver.
  `VICTORY_SHADER_LOOP_LIMIT` changes the number, and its translations get
  their own cache entries.
* **Refusals.** An instruction or pattern the translator cannot handle
  yet stops that program's translation with a reason. Pipelines that use
  it are not made, and their draws are dropped.

## Indirect execution

The title's own compute passes write most indirect arguments, so a pass
that is dropped or translated wrongly leaves whatever the memory held. A
compute shader of ours clamps every record on the GPU just before it
runs, and each list has a work budget (`indirect`). Without that, a draw
of billions of vertices would run until the host resets the driver.
