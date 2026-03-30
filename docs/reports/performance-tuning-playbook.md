# Performance tuning playbook (Burn + CubeCL)

This playbook summarizes practical tuning levers for this repository and how to apply them safely.

## 1) Launch geometry autotuning (implemented)

- Use the built-in cube-dim sweep benchmarks in `src/bench/mod.rs`:
  - `autotune_cube_dim_cpu(...)`
  - `autotune_cube_dim_wgpu_msl(...)`
- Current measured bests (see `profiling-2026-03-30.md`):
  - CPU best cube-dim: `8`
  - WGPU Metal best cube-dim (single sweep): `4`
  - WGPU Metal best cube-dim (median e2e): `32`
  - WGPU Metal best cube-dim (median dispatch): `4`

Recommended next step:

- Thread these best defaults into a runtime-selected launch policy for perf-mode deployments while keeping strict-reference mode unchanged.

## 2) Separate correctness mode from perf mode

Correctness mode should keep:

- strict equivalence checks
- eager readback for output parity validation

Perf mode should prefer:

- fewer readbacks
- batched launches
- warmup before timing
- device-native packet encoding paths (`launch_turboquant_pipeline_device_with_options`)

## 3) Warm/cold profiling discipline

Always report both:

- cold run (includes compilation/autotune)
- warm run (cache-reuse steady state)

This repo now supports stable all-in-one ignored profile runs after consolidating WGPU runtime initialization to one shared helper.

## 4) CubeCL runtime/configuration levers

Use CubeCL config and env controls when profiling runtime behavior:

- `cubecl.toml` / `CubeCL.toml` discovery in working directory or parent
- environment overrides for debugging/tuning behavior
- one-time `GlobalConfig::set` style initialization before CubeCL usage

Useful env toggles for diagnostics:

- `CUBECL_DEBUG_LOG`
- `CUBECL_DEBUG_OPTION`
- `CUBECL_AUTOTUNE_LEVEL`

## 5) Burn-side performance strategy

- Keep model composition in Burn and move only narrow hotspots to CubeCL kernels.
- Preserve reference paths for parity tests.
- Profile release builds only.
- Measure CPU and WGPU separately; do not assume backend parity.

## 6) Immediate optimization backlog

1. Add p95 reporting for each device timing metric (dispatch/sync/export).
2. Add batch-size sweep for KV model-shape profiles in device-native mode.
3. Add startup warmup helper for production inference entrypoints.
4. Persist and reuse autotune decisions in deployment environments.
5. Add CI artifact upload for profiling output snapshots.
