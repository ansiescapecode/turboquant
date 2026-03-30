# Coverage gap inventory

This inventory is generated from:

- `cargo llvm-cov --no-default-features --features "std stdlib cpu burn-ext experimental-huffman" --workspace --summary-only -- --include-ignored`
- `cargo llvm-cov --no-default-features --features "std stdlib cpu burn-ext experimental-huffman" --workspace --text --show-missing-lines --output-path /tmp/cov.txt -- --include-ignored`

## Baseline summary

- Total line coverage: `59.39%`
- Primary uncovered modules:
  - `src/api/burn_ext.rs`
  - `src/api/kernel.rs`
  - `src/burn_ext/mod.rs`
  - `src/kernels/mod.rs`
  - (secondary) `src/bench/mod.rs`

## Gap-to-test mapping

### `src/api/kernel.rs`

- Gap type: fluent wrapper methods not invoked.
- Test owner: add `src/api/kernel.rs` unit tests (`#[cfg(test)]`) that call every fluent method, including:
  - launch helpers (`launch`, `launch_device`, `launch_device_pipeline`, handle-based variants),
  - encode/decode wrappers (`encode_device`, `entropy_device`, `decode_device`),
  - experimental Huffman wrappers behind feature gates.

### `src/api/burn_ext.rs`

- Gap type: Burn fluent wrappers not invoked.
- Test owner: add `src/api/burn_ext.rs` unit tests (`#[cfg(all(test, feature = "burn-ext", feature = "cpu"))]`) that:
  - construct CPU `CubeBackend` tensors,
  - execute `TurboQuantFluent` and `TurboQuantCubeFluent` methods,
  - cover launch/encode/decode wrapper paths.

### `src/burn_ext/mod.rs`

- Gap type: public Burn extension functions and trait impl paths not directly exercised.
- Test owner: add `src/burn_ext/mod.rs` unit tests (`#[cfg(all(test, feature = "burn-ext", feature = "cpu"))]`) that call:
  - `turboquant_mse`, `turboquant_prod`,
  - `turboquant_mse_encode_bitpacked`, `turboquant_mse_encode_entropy`, `turboquant_mse_decode_indices`,
  - experimental Huffman helpers behind `experimental-huffman`.

### `src/kernels/mod.rs`

- Gap type A: host reference helper branches and math helpers.
- Gap type B: fail-closed panic branches and uncommon validation paths.
- Gap type C: pipeline option branches not hit in current matrix.
- Test owner:
  - expand `src/kernels/tests.rs` with targeted branch tests:
    - invalid packet invariants per check branch,
    - missing checksum branch,
    - non-Huffman decode mismatch variants,
    - pipeline option combinations and launch override paths,
    - experimental Huffman branch-only paths still uncovered.

### `src/bench/mod.rs`

- Gap type: helper-only lines and low-probability codepaths in report utilities.
- Test owner:
  - add focused bench unit tests (`#[cfg(test)]`) for pure helper functions and print/report branches,
  - keep `--include-ignored` coverage execution to include ignored profiling tests.

## Execution order

1. API wrapper coverage (`src/api/kernel.rs`, `src/api/burn_ext.rs`).
2. Burn extension module coverage (`src/burn_ext/mod.rs`).
3. Kernels fail-closed and branch closures (`src/kernels/tests.rs`).
4. Bench helper edge-path tests (`src/bench/mod.rs`).
5. Re-run coverage command and iterate until all uncovered lists are empty.
