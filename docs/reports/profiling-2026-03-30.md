# Profiling report (2026-03-30)

This report captures the latest device-only throughput metrics for TurboQuant fused kernel and packet pipeline paths on CPU and WGPU Metal.

Default extra-compression path is now delta-xor entropy (XOR). Huffman is gated behind the `experimental-huffman` feature.

## Environment

- Timestamp (UTC): `2026-03-30T15:11:27Z`
- OS/kernel: `Darwin 25.4.0 (arm64)`
- `rustc`: `1.94.0 (4a4ef493e 2026-03-02)`
- `cargo`: `1.94.0 (85eff7c80 2026-01-15)`

## Commands used

- CPU fused profile:
  - `cargo test bench::tests::print_profile_report_cpu -- --ignored --nocapture`
- WGPU Metal fused profile (macOS):
  - `cargo test bench::tests::print_profile_report_wgpu_msl -- --ignored --nocapture`
- WGPU Metal fused profile for known open-weight KV-cache sizes:
  - `cargo test bench::tests::print_profile_report_wgpu_msl_kv_models -- --ignored --nocapture`
- CPU fused launch autotune (cube-dim sweep):
  - `cargo test bench::tests::print_profile_report_cpu_autotune_cube_dim -- --ignored --nocapture`
- WGPU Metal fused launch autotune (cube-dim sweep):
  - `cargo test bench::tests::print_profile_report_wgpu_msl_autotune_cube_dim -- --ignored --nocapture`
- Full ignored profiling suite:
  - `cargo test bench::tests:: -- --ignored --nocapture`
- Kernel codec variants (CPU):
  - `cargo test bench::tests::print_profile_report_cpu_codecs_kernel --features "burn-ext" -- --ignored --nocapture`
- Kernel codec variants (WGPU Metal):
  - `cargo test bench::tests::print_profile_report_wgpu_msl_codecs_kernel --features "burn-ext" -- --ignored --nocapture`
- Burn extension codec variants (WGPU Metal):
  - `cargo test bench::tests::print_profile_report_wgpu_msl_codecs_burn_ext --features "burn-ext" -- --ignored --nocapture`
- Codec memory/compression calculations from measured packets (CPU):
  - `cargo test bench::tests::print_profile_report_cpu_codec_memory_kv_models --features "burn-ext" -- --ignored --nocapture`
- Codec memory/compression calculations from measured packets (Burn extension, WGPU Metal):
  - `cargo test bench::tests::print_profile_report_wgpu_msl_codec_memory_kv_models_burn_ext --features "burn-ext" -- --ignored --nocapture`
- Auto-policy correctness matrix (CPU):
  - `cargo test --no-default-features --features "std stdlib cpu" auto_policy`
- Auto-policy backend parity (macOS WGPU Metal):
  - `cargo test --features "wgpu wgpu-msl" test_auto_policy_roundtrip_wgpu_msl`
- Remaining AutoHuffman hardening cases (CPU):
  - `cargo test --no-default-features --features "std stdlib cpu" auto_policy`
  - `cargo test --no-default-features --features "std stdlib cpu" test_huffman_decode_without_written_bits_handle_uses_valid_bits_cpu`

## Results

- `profile.cpu.fused_kernel_qps=1698.541`
- `profile.cpu.device.dispatch_qps=675.049`
- `profile.cpu.device.sync_qps=138916.473`
- `profile.cpu.device.export_qps=1068893.528`
- `profile.wgpu_msl.fused_kernel_qps=1021.886`
- `profile.wgpu_msl.device.dispatch_qps=952.822`
- `profile.wgpu_msl.device.sync_qps=30150.749`
- `profile.wgpu_msl.device.export_qps=91.456`

### Old vs new comparison

Compared against the earlier pre-device-only baseline collected in this repository, the latest run shows the following deltas:

- CPU fused kernel: `22.570 -> 1698.541` (~`75.3x`)
- CPU dispatch-only pipeline: `22.025 -> 675.049` (~`30.6x`)
- WGPU Metal fused kernel: `21.662 -> 1021.886` (~`47.2x`)
- WGPU Metal autotune best fused: `19.321 -> 704.333` (~`36.5x`)
- WGPU KV (Llama-3.1-8B): `0.365 -> 83.830` (~`229.7x`)
- WGPU KV (Llama-3.1-70B): `16.882 -> 208.437` (~`12.3x`)
- WGPU KV (Mistral-7B-v0.1): `16.823 -> 229.796` (~`13.7x`)
- WGPU KV (Qwen2.5-7B): `1.453 -> 398.786` (~`274.5x`)

### CPU cube-dim autotune results

- `profile.cpu.autotune.cube_dim=1 fused_qps=751.412`
- `profile.cpu.autotune.cube_dim=2 fused_qps=1645.476`
- `profile.cpu.autotune.cube_dim=4 fused_qps=893.000`
- `profile.cpu.autotune.cube_dim=8 fused_qps=1654.294`
- `profile.cpu.autotune.cube_dim=16 fused_qps=1541.149`
- `profile.cpu.autotune.cube_dim=32 fused_qps=1312.017`
- `profile.cpu.autotune.cube_dim=64 fused_qps=1052.989`
- `profile.cpu.autotune.best_cube_dim=8 fused_qps=1654.294`

### WGPU Metal cube-dim autotune results

- `profile.wgpu_msl.autotune.cube_dim=1 fused_qps=238.781`
- `profile.wgpu_msl.autotune.cube_dim=2 fused_qps=67.043`
- `profile.wgpu_msl.autotune.cube_dim=4 fused_qps=704.333`
- `profile.wgpu_msl.autotune.cube_dim=8 fused_qps=204.554`
- `profile.wgpu_msl.autotune.cube_dim=16 fused_qps=562.851`
- `profile.wgpu_msl.autotune.cube_dim=32 fused_qps=214.817`
- `profile.wgpu_msl.autotune.cube_dim=64 fused_qps=560.679`
- `profile.wgpu_msl.autotune.cube_dim=128 fused_qps=393.154`
- `profile.wgpu_msl.autotune.best_cube_dim=4 fused_qps=704.333`

### WGPU Metal KV-cache model-size results

- `profile.wgpu_msl.kv.model=Llama-3.1-8B dim=2048 layers=32 kv_bytes_fp16_per_token_total=131072 fused_qps=83.830`
- `profile.wgpu_msl.kv.model=Llama-3.1-70B dim=2048 layers=80 kv_bytes_fp16_per_token_total=327680 fused_qps=208.437`
- `profile.wgpu_msl.kv.model=Mistral-7B-v0.1 dim=2048 layers=32 kv_bytes_fp16_per_token_total=131072 fused_qps=229.796`
- `profile.wgpu_msl.kv.model=Qwen2.5-7B dim=1024 layers=28 kv_bytes_fp16_per_token_total=57344 fused_qps=398.786`

### WGPU Metal median autotune results

- `profile.wgpu_msl.autotune_median.e2e.cube_dim=2 fused_qps=333.573`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=4 fused_qps=380.345`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=8 fused_qps=493.270`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=16 fused_qps=543.727`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=32 fused_qps=546.486`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=64 fused_qps=540.967`
- `profile.wgpu_msl.autotune_median.e2e.cube_dim=128 fused_qps=516.421`
- `profile.wgpu_msl.autotune_median.e2e.best_cube_dim=32 fused_qps=546.486`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=2 fused_qps=650.466`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=4 fused_qps=1506.314`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=8 fused_qps=1384.374`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=16 fused_qps=1421.931`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=32 fused_qps=1047.207`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=64 fused_qps=1198.088`
- `profile.wgpu_msl.autotune_median.dispatch.cube_dim=128 fused_qps=1129.262`
- `profile.wgpu_msl.autotune_median.dispatch.best_cube_dim=4 fused_qps=1506.314`

### Codec speed by variant (kernel, CPU)

- `profile.cpu.kernel_codec.regular.encode_qps=164753.976`
- `profile.cpu.kernel_codec.regular.decode_qps=2243999.930`
- `profile.cpu.kernel_codec.regular.roundtrip_qps=5242463.958`
- `profile.cpu.kernel_codec.bitpacked.encode_qps=2411.939`
- `profile.cpu.kernel_codec.bitpacked.decode_qps=2453.700`
- `profile.cpu.kernel_codec.bitpacked.roundtrip_qps=1612.556`
- `profile.cpu.kernel_codec.delta_xor.encode_qps=2725.107`
- `profile.cpu.kernel_codec.delta_xor.decode_qps=2529.782`
- `profile.cpu.kernel_codec.delta_xor.roundtrip_qps=1384.859`
- `profile.cpu.kernel_codec.huffman.encode_qps=266.939`
- `profile.cpu.kernel_codec.huffman.decode_qps=1796.171`
- `profile.cpu.kernel_codec.huffman.roundtrip_qps=248.698`

### Codec speed by variant (kernel, WGPU Metal)

- `profile.wgpu_msl.kernel_codec.regular.encode_qps=25079.598`
- `profile.wgpu_msl.kernel_codec.regular.decode_qps=106711.130`
- `profile.wgpu_msl.kernel_codec.regular.roundtrip_qps=132699.967`
- `profile.wgpu_msl.kernel_codec.bitpacked.encode_qps=5738.581`
- `profile.wgpu_msl.kernel_codec.bitpacked.decode_qps=5785.137`
- `profile.wgpu_msl.kernel_codec.bitpacked.roundtrip_qps=2375.051`
- `profile.wgpu_msl.kernel_codec.delta_xor.encode_qps=4162.218`
- `profile.wgpu_msl.kernel_codec.delta_xor.decode_qps=4048.679`
- `profile.wgpu_msl.kernel_codec.delta_xor.roundtrip_qps=1925.562`
- `profile.wgpu_msl.kernel_codec.huffman.encode_qps=285.887`
- `profile.wgpu_msl.kernel_codec.huffman.decode_qps=2456.877`
- `profile.wgpu_msl.kernel_codec.huffman.roundtrip_qps=266.226`

### Codec speed by variant (Burn extension, WGPU Metal)

- `profile.wgpu_msl.burn_ext_codec.regular.qps=22.154`
- `profile.wgpu_msl.burn_ext_codec.bitpacked.encode_qps=585.352`
- `profile.wgpu_msl.burn_ext_codec.bitpacked.decode_qps=6359.599`
- `profile.wgpu_msl.burn_ext_codec.bitpacked.roundtrip_qps=433.544`
- `profile.wgpu_msl.burn_ext_codec.entropy.encode_qps=22.265`
- `profile.wgpu_msl.burn_ext_codec.entropy.decode_qps=4409.451`
- `profile.wgpu_msl.burn_ext_codec.entropy.roundtrip_qps=383.128`

### Codec memory and compression (measured from tests, not paper)

Formulas used in tests:

- `regular_kv_fp16_bytes_per_layer = dim * sizeof(u16)`
- `regular_indices_u32_bytes_per_layer = dim * sizeof(u32)`
- `wire_bytes_per_layer = ceil(valid_bits / 8)` for fixed-width codecs; Huffman uses measured `written_bits` from device output.
- `resident_packet_only_bytes_per_layer = payload_words_bytes`
- `resident_with_shared_codebook_bytes_per_layer = payload_words_bytes + (shared_codebook_bytes / sharing_window)`
- `save_vs_kv_pct = 100 * (1 - codec_total_bytes / regular_kv_fp16_total_bytes)`
- `save_vs_regular_indices_pct = 100 * (1 - codec_wire_bytes_per_layer / regular_indices_u32_bytes_per_layer)`

CPU kernel packet-derived results:

- Llama-3.1-8B (`dim=2048`, `layers=32`):
  - bitpacked: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - delta_xor: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - huffman (shared-codebook path): `wire=416.094`, `resident_packet_only=2048`, `resident_with_shared_codebook=2054.125`, `save_vs_kv=89.841%`, `save_vs_regular_indices=94.921%`
  - ratio: `huffman_vs_bitpacked_wire=0.542`, `delta_vs_bitpacked_wire=1.000`
- Llama-3.1-70B (`dim=2048`, `layers=80`):
  - bitpacked: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - delta_xor: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - huffman (shared-codebook path): `wire=414.469`, `resident_packet_only=2048`, `resident_with_shared_codebook=2054.125`, `save_vs_kv=89.881%`, `save_vs_regular_indices=94.941%`
  - ratio: `huffman_vs_bitpacked_wire=0.540`, `delta_vs_bitpacked_wire=1.000`
- Mistral-7B-v0.1 (`dim=2048`, `layers=32`):
  - bitpacked: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - delta_xor: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - huffman (shared-codebook path): `wire=415.656`, `resident_packet_only=2048`, `resident_with_shared_codebook=2054.125`, `save_vs_kv=89.852%`, `save_vs_regular_indices=94.926%`
  - ratio: `huffman_vs_bitpacked_wire=0.541`, `delta_vs_bitpacked_wire=1.000`
- Qwen2.5-7B (`dim=1024`, `layers=28`):
  - bitpacked: `wire=384`, `resident=384`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - delta_xor: `wire=384`, `resident=384`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - huffman (shared-codebook path): `wire=214.906`, `resident_packet_only=1024`, `resident_with_shared_codebook=1030.125`, `save_vs_kv=89.507%`, `save_vs_regular_indices=94.753%`
  - ratio: `huffman_vs_bitpacked_wire=0.560`, `delta_vs_bitpacked_wire=1.000`

Burn extension WGPU Metal packet-derived results:

- Llama-3.1-8B (`dim=2048`, `layers=32`):
  - bitpacked: `wire=768`, `resident=768`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - entropy (shared-codebook path): `wire=415`, `resident_packet_only=2048`, `resident_with_shared_codebook=2244`, `save_vs_kv=89.868%`, `save_vs_regular_indices=94.934%`
  - ratio: `entropy_vs_bitpacked_wire=0.540`
- Qwen2.5-7B (`dim=1024`, `layers=28`):
  - bitpacked: `wire=384`, `resident=384`, `save_vs_kv=81.250%`, `save_vs_regular_indices=90.625%`
  - entropy (shared-codebook path): `wire=209`, `resident_packet_only=1024`, `resident_with_shared_codebook=1220`, `save_vs_kv=89.795%`, `save_vs_regular_indices=94.897%`
  - ratio: `entropy_vs_bitpacked_wire=0.544`

## Workload definitions

Metrics are produced by `src/bench/mod.rs`:

- Fused kernel:
  - `fused_kernel_throughput_cpu(dim=256, samples=256, bit_width=4, seed=77)`
- Fused kernel (WGPU Metal):
  - `fused_kernel_throughput_wgpu_msl(dim=256, samples=128, bit_width=4, seed=177)`
- Fused kernel autotune sweeps:
  - CPU: `autotune_cube_dim_cpu(dim=256, samples=128, bit_width=4, seed=277, candidates=[1,2,4,8,16,32,64])`
  - WGPU Metal: `autotune_cube_dim_wgpu_msl(dim=2048, samples=64, bit_width=4, seed=377, candidates=[1,2,4,8,16,32,64,128])`
- Fused kernel (WGPU Metal, open-weight KV model shapes):
  - `Llama-3.1-8B: dim=2*8*128=2048, layers=32, kv_bytes_fp16_per_token_total=131072`
  - `Llama-3.1-70B: dim=2*8*128=2048, layers=80, kv_bytes_fp16_per_token_total=327680`
  - `Mistral-7B-v0.1: dim=2*8*128=2048, layers=32, kv_bytes_fp16_per_token_total=131072`
  - `Qwen2.5-7B: dim=2*4*128=1024, layers=28, kv_bytes_fp16_per_token_total=57344`
- Device-native pipeline timing:
  - CPU: `pipeline_device_timing_cpu(dim=256, samples=128, bit_width=4, seed=0xAA77, entropy=true)`
  - WGPU Metal: `pipeline_device_timing_wgpu_msl(dim=256, samples=64, bit_width=4, seed=0xBB77, entropy=true)`

## Notes

- These are local baselines for this specific machine/runtime and should not be compared across machines without normalization.
- Fused-kernel metric includes output readback in each sample iteration.
- Device-native timing split isolates:
  - dispatch-only pipeline QPS,
  - explicit flush/sync proxy cost,
  - explicit final payload export cost.
- WGPU profiling previously collided on repeated runtime registration; `src/bench/mod.rs` now uses one shared WGPU init helper for all profiling functions.
- For publication, report command lines and workload definitions along with raw values.
- Full ignored profiling suite wall-clock: ~37.40s in this run.
- Auto-policy correctness edge cases are now covered by tests for single-symbol, tiny-dim, uniform-like, drift-over-time, rebuild-boundary, and shape/bit-width change scenarios.
- Additional hardening now covers decode-before-encode fallback, explicit invalidate/rebuild lifecycle, Huffman missing `written_bits` metadata fallback, large-dimension stress, and cadence-bound sweep checks.
- Policy-managed Huffman decode is now fail-closed for stale/wrong policy use via codebook-generation tag validation.
- Integrity checks now enforce:
  - BLAKE3 codebook fingerprint validation for policy-managed Huffman decode.
  - CRC32C payload checksum validation at decode entry.
  - policy cache invalidation + hard failure on policy-managed mismatch to prevent silent KV corruption.
