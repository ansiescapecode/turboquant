# turboquant

TurboQuant reference implementation with device-native quantization pipelines and CubeCL fused launch helpers.


## WARNING: I'M SOMEWHAT OF A SCIENTIST MYSELF...

This project is an experiment. The kernel path **may** or **may not** production-safe... choose your own adventure.

If you choose to run this kernel, you do so knowing that you are using something written by a chimpanzee that gets lucky with his fingers sometimes, armed with a terminal degree in studio art, and a statistical ghost he has trapped in a metal box in his bedroom.

## Project files

- `LICENSE`: project license.
- `CONTRIBUTING.md`: contributor workflow and required checks.
- `SECURITY.md`: security reporting policy.
- `CHANGELOG.md`: release history.

## Paper citations for code

Primary paper reference:

- `docs/references/turboquant.pdf`

## Profiling reports

- `docs/reports/profiling-2026-03-30.md`
- `docs/reports/performance-tuning-playbook.md`
- `docs/reports/device-boundary-map.md`

## Measured codec snapshot

All values below are measured from repository tests (not paper estimates). Full raw output, formulas, and workload definitions are in `docs/reports/profiling-2026-03-30.md`.

### Speed (QPS)

| Path | Regular roundtrip | Bitpacked roundtrip | Delta-xor roundtrip | Huffman/entropy roundtrip |
| --- | ---: | ---: | ---: | ---: |
| Kernel CPU | 5,242,463.958 | 1,612.556 | 1,384.859 | 248.698 |
| Kernel WGPU Metal | 132,699.967 | 2,375.051 | 1,925.562 | 266.226 |
| Burn ext WGPU Metal | 22.154 | 433.544 | n/a | 383.128 |

### Compression (memory savings)

Representative measured savings versus regular fp16 KV cache and regular u32 index buffers:

| Model shape | Bitpacked save vs KV | Huffman/entropy save vs KV | Bitpacked save vs u32 indices | Huffman/entropy save vs u32 indices | Huffman/entropy wire ratio vs bitpacked |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dim=2048` (Llama/Mistral cases) | 81.250% | ~89.8-89.9% | 90.625% | ~94.9% | ~0.54x |
| `dim=1024` (Qwen2.5-7B case) | 81.250% | ~89.5-89.8% | 90.625% | ~94.8-94.9% | ~0.54-0.56x |

Notes:
- Huffman figures above are from the shared-codebook path and are now experimental.
- Report includes both packet-only and packet+shared-codebook resident footprints.

## Strict equivalence contract

The fused launch path in `src/kernels/mod.rs` is implemented to match host `quantize_prod` stage semantics for the same `(input, bit_width, seed)`:

- MSE stage uses `mse_bit_width = max(bit_width - 1, 1)`.
- Input is transformed with the same deterministic signed permutation used by host MSE quantization.
- QJL signs use the same seeded Gaussian projection (`seed ^ PROD_PROJECTION_SALT`) with a rotated-basis projection transform that preserves host dot-product values.
- Device validation checks strict parity against host-stage semantics for the same inputs.

The fused `mse` output buffer is in rotated coordinates. Recover original-coordinate MSE values by applying:

- `invert_signed_permutation(mse_rot, dim, seed ^ MSE_ROTATION_SALT)`


## Compression

**Seriously** probably best to not use compression but if you do use xor but if you're feeling squirrely Huffman is here too under the experimental flag

Default compressed path is bitpacked + delta-xor entropy (XOR) for production stability.
Huffman paths are gated behind the Cargo feature `experimental-huffman`.
This is intentional: I am still hardening long-run codebook drift handling and paged-attention/vLLM integration behavior to someone's production standards (I'm trying to figure it out still tbh but this sounds more github like).

Why Huffman is still experimental:
- Huffman adds a stateful codebook lifecycle, while XOR is stateless and simpler to operate under concurrency.
- Long-running serving workloads can rebuild codebooks over time; paging systems must prevent stale generation reuse.
- Paged-attention runtimes (for example vLLM) introduce async page movement/eviction paths that can decode out-of-order unless metadata/version checks are strict.
- I already fail closed on integrity mismatches, but production rollout also needs sustained operational validation at serving scale.

What must be true before promoting Huffman to default:
- Stable long-duration serving runs with no integrity rejects under expected load patterns.
- Proven compatibility with paged-attention page lifecycle events (allocate/evict/reuse/rewrite) without stale decode incidents.
- Explicit vLLM integration tests covering out-of-order page decode, page-version mismatches, and policy/codebook rollover behavior.
- Clear operational playbook for rebuild policy tuning, alerting, and safe fallback to XOR.

XOR decode hardening (fail-closed):
- mandatory payload checksum on decode entry (`payload_crc32c`)
- strict invariant checks (`dim > 0`, `bit_width >= 1`, `word_count == ceil(valid_bits / 32)`, `valid_bits` capacity bound)
- strict non-Huffman bit-width contract (`valid_bits == dim * bit_width`)
- decode rejects corrupted payloads and malformed metadata instead of returning partial/undefined indices

## Fluent APIs

### Kernel fluent API

```rust
use turboquant::api::kernel::turboquant_kernel;

let fluent = turboquant_kernel::<cubecl::cpu::CpuRuntime>(&Default::default(), &input)
    .bit_width(4)
    .seed(77)
    .emit_qjl(true);

let outputs = fluent.launch();
let (mse_rot, qjl) = turboquant::kernels::read_fused_outputs(&outputs);
let ok = fluent.validate_on_device(&outputs, 1e-6);

let device_packet = fluent.launch_device_encoded(true);
let _decoded_indices = fluent.decode_device(&device_packet);

let encoded_bitpacked = fluent.encode_device(&outputs);
let encoded_entropy = fluent.entropy_device(&encoded_bitpacked);
let decoded_indices = fluent.decode_device(&encoded_entropy);
```

### Burn extension fluent API

```rust
use turboquant::api::burn_ext::TurboQuantTensorFluentExt;

let y = x.turboquant().bit_width(4).seed(33).prod();
```

```rust
use turboquant::api::burn_ext::TurboQuantCubeTensorFluentExt;

let cube = x.turboquant_cube().bit_width(4).seed(33).emit_entropy(true);
let packet = cube.launch_device_entropy();
let indices_handle = cube.decode_indices(&packet);
```

### Huffman fluent API (gated)

`experimental-huffman` must be enabled to use these methods.

```rust
use turboquant::api::kernel::turboquant_kernel;
use turboquant::api::burn_ext::TurboQuantCubeTensorFluentExt;

// Kernel fluent API (Huffman auto-policy).
let fluent = turboquant_kernel::<cubecl::cpu::CpuRuntime>(&Default::default(), &input)
    .bit_width(4)
    .seed(77)
    .emit_qjl(true);
let outputs = fluent.launch();
let mut auto_policy = fluent.huffman_policy_auto();
let huffman_packet = fluent.huffman_device_auto(&outputs, &mut auto_policy);
let _decoded_auto = fluent.decode_device_auto(&huffman_packet, &mut auto_policy);

// Burn cube fluent API (Huffman auto-policy).
let cube = x.turboquant_cube().bit_width(4).seed(33).emit_qjl(true);
let mut cube_policy = cube.huffman_policy_auto();
let packet_a = cube.huffman_device_auto(&cube.launch_device(), &mut cube_policy);
let _indices_a = cube.decode_device_auto(&packet_a, &mut cube_policy);
```

## Auto-policy correctness coverage

Automatic Huffman reuse behavior is validated with explicit edge-case tests in `src/kernels/tests.rs`:

- Runtime state/introspection behavior (`test_auto_policy_exposes_runtime_state_cpu`)
- Single-symbol distribution (`test_auto_policy_roundtrip_single_symbol_cpu`)
- Tiny dimensions (`test_auto_policy_roundtrip_tiny_dims_cpu`)
- Uniform-like random distribution (`test_auto_policy_roundtrip_uniform_like_cpu`)
- Shape and bit-width changes with one reused policy (`test_auto_policy_handles_bit_width_and_dim_changes_cpu`)
- Long-running drift stream (`test_auto_policy_long_run_drift_roundtrip_cpu`)
- Rebuild-boundary counter/reset semantics (`test_auto_policy_rebuild_boundary_cpu`)
- Decode-before-encode fallback (`test_auto_policy_decode_before_encode_fallback_cpu`)
- Invalidate/rebuild lifecycle (`test_auto_policy_invalidate_then_rebuild_cpu`)
- Missing Huffman written-bits metadata fallback (`test_huffman_decode_without_written_bits_handle_uses_valid_bits_cpu`)
- Large-dimension stress (`test_auto_policy_large_dim_stress_cpu`)
- Cadence bound sweep (`test_auto_policy_cadence_bounds_cpu`)
- Wrong-policy decode rejection (`test_auto_policy_decode_rejects_wrong_policy_cpu`)
- Backend parity on macOS WGPU/Metal (`test_auto_policy_roundtrip_wgpu_msl`)

Fail-closed guard:
- policy-managed Huffman packets now carry a codebook generation tag, and decode through policy rejects mismatched/stale policies instead of silently decoding with the wrong codebook.
- policy-managed Huffman packets now also carry:
  - a BLAKE3 codebook fingerprint over `parent/left/right/root + bit_width + node_cap + generation + policy_id`
  - a CRC32C payload checksum over `payload_words + written_bits` (or `valid_bits` fallback)
- decode verifies these checks before decoding; mismatches are rejected and policy decode invalidates cached codebook state before failing.

Codebook retention model:
- one active codebook per policy instance (not all historical codebooks)
- periodic or shape-driven rebuild replaces the active codebook
- stale/wrong policy usage is rejected by policy identity + generation + fingerprint checks

## Paged attention and vLLM integration model

TurboQuant with compression (xor, huffman) is **theoretically** compatible with paged attention when packet metadata is treated as part of page identity and decode is fail-closed.

### Recommended packet metadata for paging

For each compressed KV page/chunk, include:
- `sequence_id`
- `layer_id`
- `head_group_id` (or KV-head group key)
- `page_id`
- `page_version` (monotonic per page rewrite)
- `codec` (`Bitpacked` or `DeltaXorEntropy`; Huffman only with `experimental-huffman`)
- existing integrity fields (`payload_crc32c`, and Huffman policy fields when enabled)

### vLLM-style control flow

1. Prefill or decode produces KV tensors for a page.
2. TurboQuant encodes page payload on-device (`bitpacked` + optional XOR entropy).
3. Page table stores payload handle + metadata fields above.
4. Attention read path resolves page table entry and validates metadata match.
5. TurboQuant decode validates checksum/invariants and reconstructs indices on-device.
6. Any mismatch/corruption rejects page decode (no partial decode, no silent fallback).

### Operational rules for correctness

- Keep decode stateless: decode depends only on packet bytes + explicit metadata.
- Keep order explicit: use `(sequence_id, page_id, page_version)` to prevent stale page reuse.
- Never apply decoded output if page-version check fails.
- Prefer XOR in production paging paths; use Huffman only behind `experimental-huffman`.

### Codebook lifecycle note (experimental Huffman only)

- Maintain one active codebook per stream key (for example `(layer, head_group, page_size_class)`).
- Rebuild replaces the active codebook; do not retain unbounded historical codebooks.
- If you need out-of-order decode tolerance, keep a small bounded generation window with strict generation checks.

## Running tests

- CPU strict profile (portable):
  - `cargo test --no-default-features --features "std stdlib cpu"`
- Default profile (includes current default feature set):
  - `cargo test`
- Profiling report (prints QPS lines):
  - `cargo test bench::tests::print_profile_report_cpu -- --ignored --nocapture`
  - `cargo test bench::tests::print_profile_report_wgpu_msl -- --ignored --nocapture`
  - `cargo test bench::tests::print_profile_report_wgpu_msl_kv_models -- --ignored --nocapture`
  - `cargo test bench::tests::print_profile_report_cpu_autotune_cube_dim -- --ignored --nocapture`
  - `cargo test bench::tests::print_profile_report_wgpu_msl_autotune_cube_dim -- --ignored --nocapture`
  - `cargo test bench::tests:: -- --ignored --nocapture`
- Auto-policy edge-case validation:
  - `cargo test --no-default-features --features "std stdlib cpu" auto_policy`
  - `cargo test --features "wgpu wgpu-msl" test_auto_policy_roundtrip_wgpu_msl` (macOS only)
- Experimental Huffman validation:
  - `cargo test --no-default-features --features "std stdlib cpu experimental-huffman" auto_policy`
- Coverage gate (100% line coverage target for `src/**` in CI profile):
  - `cargo llvm-cov --no-default-features --features "std stdlib cpu burn-ext experimental-huffman" --workspace -- --include-ignored`
  - CI enforces the same profile and runs with a strict line-threshold gate.

## CI

GitHub Actions workflow is defined in `.github/workflows/ci.yml`:

- `linux-quality`: formatting + clippy with warnings denied.
- `linux-cpu-strict`: runs CPU strict-equivalence profile.
- `linux-burn-ext`: runs burn extension profile.
- `macos-default`: runs default profile, including macOS-gated runtime tests.
