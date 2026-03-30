# Contributing

## Development setup

- Install stable Rust.
- Clone this repository.

## Required checks before opening a PR

- `cargo fmt --check`
- `cargo clippy --all-targets --no-default-features --features "std stdlib cpu" -- -D warnings`
- `cargo test`
- `cargo test --no-default-features --features "std stdlib cpu"`
- `cargo test --features burn-ext`

## Strict-equivalence contract

Changes touching `src/kernels/mod.rs`, `src/api/kernel.rs`, or `src/burn_ext/mod.rs` must preserve strict equivalence between:

- host reference paths (`quantize_prod`, `dequantize_mse`, `dequantize_prod`)
- fused/device validation paths (`launch_turboquant_fused_device`, `validate_fused_outputs_on_device`)
- fluent API wrappers in `src/api/`

If behavior changes are intentional, update:

- `docs/references/paper-code-traceability.md`
- `README.md`
- tests covering both direct and fluent APIs

## Documentation

- Keep public API docs and examples aligned with implementation.
- Use `docs/references/turboquant.txt` as the canonical paper reference source.
