# Compute lane masks restored from VGPRs

In the local Series S run that reached the first mission's title
card, the log reported 251 unique refused compute shaders, 13 mesh,
13 pixel, and four vertex shaders. These counts describe distinct
programs. Fifty-seven compute programs reached the refusal
`a lane mask holding lanes elsewhere read as data`.

This branch follows [PR #298](https://github.com/Marvisak/VIctory/pull/298),
which tracks a mask saved by `v_writelane_b32` and restored by
`v_readlane_b32` when the saved mask can be identified. The translator
can then use the original predicate for the lanes that consume it.

The regression constructs a single synthetic compute shader. It
inverts a comparison mask, saves it to a VGPR lane, restores it, and
uses it in `v_cndmask_b32`. Equal inputs must write 17 to the output
UAV, and unequal inputs must write 29. On upstream base `ffe310e`
the test fails during lowering with `a select on an untracked mask`.
With `54f97db` it validates and writes both expected results on the
hardware D3D12 adapter and WARP.

```powershell
cargo test --locked -p victory-shader compute_uses_a_reloaded_lane_mask_for_select -- --nocapture
```

## Offline comparison

The same local shader archive was processed with
`victory translate --validate --list --jobs 6` on both revisions,
using the Windows SDK 10.0.26100.0 DXIL validator. The archive scan
reported 26,503 containers and 26,084 native programs on each side.

| Stage | Validated on ffe310e | Validated with #298 | Added | Lost |
| --- | ---: | ---: | ---: | ---: |
| Compute | 1,417 | 1,424 | 7 | 0 |
| Mesh | 8,152 | 8,152 | 0 | 0 |
| Vertex | 2,104 | 2,104 | 0 | 0 |
| Pixel | 10,051 | 10,078 | 27 | 0 |

The lane-mask-as-data refusal fell from 272 to 12, all in compute.
Compute programs that emitted DXIL increased from 1,441 to 1,458.
The counts measure translation and DXIL validation. Dispatch drops,
frame time, and visual impact in the first mission remain unmeasured.

The workspace suite passed 343 tests with five ignored. Clippy for
all shader targets passed with warnings denied. The shader archive,
logs, and validation output stayed outside the repository.
