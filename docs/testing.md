# Testing

## Running the tests

```powershell
cargo test --workspace
```

It needs no game data. What tests may use is set by the
[asset policy](asset-policy.md):

* tests that need real files take their paths from environment variables
  and skip when they are unset, and they assert on properties, never on
  snapshots;
* GPU tests run on the host's D3D12 device, and return early without one;
* shader tests that need the DXIL validator (`dxil.dll`) or a console
  shader compiler return early without it. A green run on a machine
  without them has checked less.

Most GPU tests share process-wide state: one host device, the write-watch
journal, and the guards on view memory (`watch`). When a test fails only
alongside others, run it alone with `-- --test-threads=1` to rule that
out. A test that frees write-watched memory leaves the journal's pages for
its addresses behind for the next allocation there.

## What CI runs

On every pull request, on `windows-latest`
(`.github/workflows/ci.yml`):

```powershell
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
```

The second line builds with `diagnostics`, so code behind
`#[cfg(feature = "diagnostics")]` is checked too. Run all three before
opening one.

## Tests for a fix

* Show that the test fails without the fix: run it on `main` and give the
  failure in the pull request, next to the passing run.
* Make races deterministic. Run the other thread's step at the exact point
  it matters, for example a title write between a function's read and its
  write, instead of hoping a loop hits it.
* Build test data in the test, from our own code. A fix found in the title
  gets a test on synthetic data that reproduces its cause.

## Working from Linux

The workspace builds for Windows only. From Linux:

* **Lints.** Clippy checks the Windows target without linking:

  ```bash
  rustup target add x86_64-pc-windows-msvc
  CARGO_FEATURE_PURE=1 cargo clippy --workspace --all-targets --target x86_64-pc-windows-msvc -- -D warnings
  ```

  `CARGO_FEATURE_PURE=1` makes `blake3` build without a C compiler for
  the target.
* **Tests on Windows.** Push the branch to your fork with a workflow of
  your own that runs the tests on `windows-latest`. Keep that workflow out
  of the pull request.
* **Tests under Wine or Proton.** Most tests run under Wine with the MSVC
  target. A few fail there for reasons of Wine's, not of the change being
  tested. Run the same tests on `main` for a baseline, and compare.
