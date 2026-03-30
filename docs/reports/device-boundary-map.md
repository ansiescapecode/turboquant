# Device boundary map

This map tracks where host/device crossings occurred before the device-resident pipeline work and how each call site is treated now.

## Remove from normal path

- `src/kernels/mod.rs`
  - `read_fused_outputs`: remove from default execution path.
  - `read_f32_buffer`: keep only as explicit debug/export helper.
  - `launch_turboquant_fused`: keep as legacy host-readback wrapper only.
  - Host codec APIs removed.
- `src/api/kernel.rs`
  - `TurboQuantKernelFluent::launch`: host readback wrapper only.
  - Host codec helpers removed.
- `src/api/burn_ext.rs`
  - Host codec helpers removed.

## Keep as debug/export only

- `src/kernels/mod.rs`
  - `read_fused_outputs`
  - `read_f32_buffer`
  - `read_u32_buffer`

## Replace with device stages

- `src/kernels/mod.rs`
  - Add device-resident state and encoded packet structs.
  - Add bitpack/entropy/decode kernels operating on device handles.
  - Add single-kernel fused packet launch (`launch_turboquant_pipeline_device_from_handle_with_options`).
  - Add options-aware asset and pipeline entrypoints (`*_with_options`) to expose CubeK controls through APIs.
- `src/api/kernel.rs`
  - Add default device-only fluent methods returning device handles.
- `src/api/burn_ext.rs`
  - Add CubeBackend fluent device API (`turboquant_cube`) with encode/decode and options parity.
- `src/bench/mod.rs`
  - Use device-handle pipeline timings with optional export timing, not mandatory per-iteration readback.
  - Use one shared WGPU runtime init helper to avoid profiling collisions.
