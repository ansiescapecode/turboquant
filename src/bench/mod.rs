//! Benchmark helpers for TurboQuant APIs.

use std::time::Instant;

use cubecl::CubeElement;
use rand::{rngs::StdRng, RngExt, SeedableRng};

#[cfg(feature = "experimental-huffman")]
use crate::kernels::encode_device_huffman;
use crate::kernels::{
    decode_device_indices, encode_device_bitpacked, encode_device_entropy,
    launch_turboquant_fused_device_from_handle, launch_turboquant_fused_device_with_assets,
    launch_turboquant_pipeline_device_from_handle, prepare_turboquant_launch_assets,
    read_fused_outputs, TurboQuantLaunchOverride,
};

fn median(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty(), "median requires non-empty values");
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

#[cfg(test)]
fn bits_to_bytes(bits: usize) -> usize {
    bits.div_ceil(8)
}

#[cfg(test)]
fn packet_wire_bytes<R: cubecl::Runtime>(packet: &crate::kernels::DeviceEncodedPacket<R>) -> usize {
    match packet.encoding {
        crate::kernels::DeviceEncodingKind::Huffman => {
            let bits = packet
                .huffman_written_bits_handle
                .as_ref()
                .map(|handle| {
                    let bytes = packet.client.read_one(handle.clone());
                    let values = u32::from_bytes(&bytes);
                    values.first().copied().unwrap_or(packet.valid_bits) as usize
                })
                .unwrap_or(packet.valid_bits as usize);
            bits_to_bytes(bits)
        }
        _ => bits_to_bytes(packet.valid_bits as usize),
    }
}

#[cfg(test)]
fn packet_resident_bytes<R: cubecl::Runtime>(
    packet: &crate::kernels::DeviceEncodedPacket<R>,
) -> usize {
    let payload = packet.word_count * core::mem::size_of::<u32>();
    let huffman_state = if matches!(packet.encoding, crate::kernels::DeviceEncodingKind::Huffman)
        && packet.huffman_parent_handle.is_some()
    {
        let node_cap = (1usize << packet.bit_width) * 2usize;
        (node_cap * 3usize + 2usize) * core::mem::size_of::<u32>()
    } else {
        0usize
    };
    payload + huffman_state
}

#[cfg(test)]
fn huffman_codebook_resident_bytes<R: cubecl::Runtime>(
    codebook: &crate::kernels::DeviceHuffmanCodebook<R>,
) -> usize {
    (codebook.node_cap * 3usize + 1usize) * core::mem::size_of::<u32>()
}

/// Timing breakdown for device-resident benchmark mode.
#[derive(Debug, Clone, Copy)]
pub struct DeviceTimingBreakdown {
    /// Average pipeline dispatch throughput (no immediate export).
    pub dispatch_qps: f64,
    /// Average post-dispatch flush throughput proxy.
    pub sync_qps: f64,
    /// Final explicit export throughput proxy.
    pub export_qps: f64,
}

/// Throughput for a specific encode/decode codec path.
#[derive(Debug, Clone, Copy)]
pub struct CodecPathTiming {
    /// Encode-only throughput in ops/second.
    pub encode_qps: f64,
    /// Decode-only throughput in ops/second.
    pub decode_qps: f64,
    /// Combined encode+decode throughput in ops/second.
    pub roundtrip_qps: f64,
}

/// Aggregate kernel codec profile for all supported paths.
#[derive(Debug, Clone, Copy)]
pub struct KernelCodecProfile {
    pub regular: CodecPathTiming,
    pub bitpacked: CodecPathTiming,
    pub delta_xor: CodecPathTiming,
    pub huffman: CodecPathTiming,
}

/// Profile kernel codec encode/decode paths on CPU runtime.
pub fn kernel_codec_profile_cpu(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
) -> KernelCodecProfile {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260342);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = Default::default();
    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let input_handles = dataset
        .iter()
        .map(|data| assets.client.create_from_slice(f32::as_bytes(data)))
        .collect::<Vec<_>>();

    let mut fused_states = Vec::with_capacity(samples);
    for handle in &input_handles {
        fused_states.push(launch_turboquant_fused_device_from_handle::<
            cubecl::cpu::CpuRuntime,
        >(&assets, handle, true));
    }

    fn path_timing(
        states: &[crate::kernels::DeviceFusedOutputs<cubecl::cpu::CpuRuntime>],
        encode: impl Fn(
            &crate::kernels::DeviceFusedOutputs<cubecl::cpu::CpuRuntime>,
        ) -> crate::kernels::DeviceEncodedPacket<cubecl::cpu::CpuRuntime>,
    ) -> CodecPathTiming {
        let encode_start = Instant::now();
        let packets = states.iter().map(&encode).collect::<Vec<_>>();
        if let Some(last) = packets.last() {
            let _ = std::hint::black_box(last.client.read_one(last.payload_words_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = states.len() as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = packets
            .iter()
            .map(decode_device_indices::<cubecl::cpu::CpuRuntime>)
            .collect::<Vec<_>>();
        if let (Some(last_packet), Some(last_handle)) = (packets.last(), decoded.last()) {
            let _ = std::hint::black_box(last_packet.client.read_one(last_handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = states.len() as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let mut last = None;
        for state in states {
            let packet = encode(state);
            let handle = decode_device_indices::<cubecl::cpu::CpuRuntime>(&packet);
            last = Some((packet, handle));
        }
        if let Some((packet, handle)) = last {
            let _ = std::hint::black_box(packet.client.read_one(handle));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = states.len() as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    }

    let regular = {
        let encode_start = Instant::now();
        let handles = fused_states
            .iter()
            .map(|state| state.mse_index_handle.clone())
            .collect::<Vec<_>>();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), handles.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = samples as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = handles.clone();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), decoded.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = samples as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let decoded = fused_states
            .iter()
            .map(|state| state.mse_index_handle.clone())
            .collect::<Vec<_>>();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), decoded.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = samples as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    };
    let bitpacked = path_timing(
        &fused_states,
        encode_device_bitpacked::<cubecl::cpu::CpuRuntime>,
    );
    let delta_xor = path_timing(&fused_states, |state| {
        let packed = encode_device_bitpacked::<cubecl::cpu::CpuRuntime>(state);
        encode_device_entropy::<cubecl::cpu::CpuRuntime>(&packed)
    });
    #[cfg(feature = "experimental-huffman")]
    let huffman = path_timing(
        &fused_states,
        encode_device_huffman::<cubecl::cpu::CpuRuntime>,
    );
    #[cfg(not(feature = "experimental-huffman"))]
    let huffman = CodecPathTiming {
        encode_qps: 0.0,
        decode_qps: 0.0,
        roundtrip_qps: 0.0,
    };

    KernelCodecProfile {
        regular,
        bitpacked,
        delta_xor,
        huffman,
    }
}

/// Profile kernel codec encode/decode paths on WGPU Metal runtime.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn kernel_codec_profile_wgpu_msl(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
) -> KernelCodecProfile {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260344);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = cubecl::wgpu::WgpuDevice::default();
    init_wgpu_msl_once(&device);
    let assets = prepare_turboquant_launch_assets::<cubecl::wgpu::WgpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let input_handles = dataset
        .iter()
        .map(|data| assets.client.create_from_slice(f32::as_bytes(data)))
        .collect::<Vec<_>>();

    let mut fused_states = Vec::with_capacity(samples);
    for handle in &input_handles {
        fused_states.push(launch_turboquant_fused_device_from_handle::<
            cubecl::wgpu::WgpuRuntime,
        >(&assets, handle, true));
    }

    fn path_timing(
        states: &[crate::kernels::DeviceFusedOutputs<cubecl::wgpu::WgpuRuntime>],
        encode: impl Fn(
            &crate::kernels::DeviceFusedOutputs<cubecl::wgpu::WgpuRuntime>,
        ) -> crate::kernels::DeviceEncodedPacket<cubecl::wgpu::WgpuRuntime>,
    ) -> CodecPathTiming {
        let encode_start = Instant::now();
        let packets = states.iter().map(&encode).collect::<Vec<_>>();
        if let Some(last) = packets.last() {
            let _ = std::hint::black_box(last.client.read_one(last.payload_words_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = states.len() as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = packets
            .iter()
            .map(decode_device_indices::<cubecl::wgpu::WgpuRuntime>)
            .collect::<Vec<_>>();
        if let (Some(last_packet), Some(last_handle)) = (packets.last(), decoded.last()) {
            let _ = std::hint::black_box(last_packet.client.read_one(last_handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = states.len() as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let mut last = None;
        for state in states {
            let packet = encode(state);
            let handle = decode_device_indices::<cubecl::wgpu::WgpuRuntime>(&packet);
            last = Some((packet, handle));
        }
        if let Some((packet, handle)) = last {
            let _ = std::hint::black_box(packet.client.read_one(handle));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = states.len() as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    }

    let regular = {
        let encode_start = Instant::now();
        let handles = fused_states
            .iter()
            .map(|state| state.mse_index_handle.clone())
            .collect::<Vec<_>>();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), handles.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = samples as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = handles.clone();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), decoded.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = samples as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let decoded = fused_states
            .iter()
            .map(|state| state.mse_index_handle.clone())
            .collect::<Vec<_>>();
        if let (Some(last_state), Some(last_handle)) = (fused_states.last(), decoded.last()) {
            let _ = std::hint::black_box(last_state.client.read_one(last_handle.clone()));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = samples as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    };
    let bitpacked = path_timing(
        &fused_states,
        encode_device_bitpacked::<cubecl::wgpu::WgpuRuntime>,
    );
    let delta_xor = path_timing(&fused_states, |state| {
        let packed = encode_device_bitpacked::<cubecl::wgpu::WgpuRuntime>(state);
        encode_device_entropy::<cubecl::wgpu::WgpuRuntime>(&packed)
    });
    #[cfg(feature = "experimental-huffman")]
    let huffman = path_timing(
        &fused_states,
        encode_device_huffman::<cubecl::wgpu::WgpuRuntime>,
    );
    #[cfg(not(feature = "experimental-huffman"))]
    let huffman = CodecPathTiming {
        encode_qps: 0.0,
        decode_qps: 0.0,
        roundtrip_qps: 0.0,
    };

    KernelCodecProfile {
        regular,
        bitpacked,
        delta_xor,
        huffman,
    }
}

/// Aggregate Burn extension codec profile (CPU Cube backend).
#[cfg(all(
    feature = "burn-ext",
    feature = "wgpu",
    feature = "wgpu-msl",
    target_os = "macos"
))]
#[derive(Debug, Clone, Copy)]
pub struct BurnExtCodecProfile {
    pub regular_qps: f64,
    pub bitpacked: CodecPathTiming,
    pub entropy: CodecPathTiming,
}

/// Profile Burn extension encode/decode paths on WGPU Metal backend.
#[cfg(all(
    feature = "burn-ext",
    feature = "wgpu",
    feature = "wgpu-msl",
    target_os = "macos"
))]
pub fn burn_ext_codec_profile_wgpu_msl(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
) -> BurnExtCodecProfile {
    type B = burn_wgpu::Wgpu<f32, i32, u32>;
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260343);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_msl_once(&device);
    let tensors = dataset
        .iter()
        .map(|data| {
            burn::tensor::Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(data.clone(), [dim]),
                &device,
            )
        })
        .collect::<Vec<_>>();

    let regular_start = Instant::now();
    let mut last_regular = None;
    for (idx, tensor) in tensors.iter().enumerate() {
        let out = crate::burn_ext::turboquant_mse(tensor.clone(), bit_width, seed ^ (idx as u64));
        last_regular = Some(out);
    }
    if let Some(out) = last_regular {
        let _ = std::hint::black_box(
            out.into_data()
                .to_vec::<f32>()
                .expect("regular data readback"),
        );
    }
    let regular_elapsed = regular_start.elapsed().as_secs_f64().max(1e-9);
    let regular_qps = samples as f64 / regular_elapsed;

    let bitpacked = {
        let encode_start = Instant::now();
        let encoded =
            tensors
                .iter()
                .enumerate()
                .map(|(idx, tensor)| {
                    crate::burn_ext::turboquant_mse_encode_bitpacked::<
                        burn_wgpu::WgpuRuntime,
                        i32,
                        u32,
                    >(tensor.clone(), bit_width, seed ^ (idx as u64))
                })
                .collect::<Vec<_>>();
        if let Some(last) = encoded.last() {
            let _ = std::hint::black_box(last.client.read_one(last.payload_words_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = samples as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = encoded
            .iter()
            .map(crate::burn_ext::turboquant_mse_decode_indices)
            .collect::<Vec<_>>();
        if let (Some(packet), Some(handle)) = (encoded.last(), decoded.last()) {
            let _ = std::hint::black_box(packet.client.read_one(handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = samples as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let mut last = None;
        for (idx, tensor) in tensors.iter().enumerate() {
            let packet = crate::burn_ext::turboquant_mse_encode_bitpacked::<
                burn_wgpu::WgpuRuntime,
                i32,
                u32,
            >(tensor.clone(), bit_width, seed ^ (idx as u64));
            let decoded = crate::burn_ext::turboquant_mse_decode_indices(&packet);
            last = Some((packet, decoded));
        }
        if let Some((packet, decoded)) = last {
            let _ = std::hint::black_box(packet.client.read_one(decoded));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = samples as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    };

    let entropy = {
        let entropy_seed = seed ^ 0xA5A5_0000_u64;
        let codebook = crate::burn_ext::turboquant_mse_build_huffman_codebook::<
            burn_wgpu::WgpuRuntime,
            i32,
            u32,
        >(tensors[0].clone(), bit_width, entropy_seed);
        let encode_start = Instant::now();
        let encoded = tensors
            .iter()
            .enumerate()
            .map(|(idx, tensor)| {
                crate::burn_ext::turboquant_mse_encode_entropy_with_codebook::<
                    burn_wgpu::WgpuRuntime,
                    i32,
                    u32,
                >(
                    tensor.clone(),
                    bit_width,
                    entropy_seed ^ (idx as u64),
                    &codebook,
                )
            })
            .collect::<Vec<_>>();
        if let Some(last) = encoded.last() {
            let _ = std::hint::black_box(last.client.read_one(last.payload_words_handle.clone()));
        }
        let encode_elapsed = encode_start.elapsed().as_secs_f64().max(1e-9);
        let encode_qps = samples as f64 / encode_elapsed;

        let decode_start = Instant::now();
        let decoded = encoded
            .iter()
            .map(|packet| {
                crate::burn_ext::turboquant_mse_decode_indices_with_codebook(packet, &codebook)
            })
            .collect::<Vec<_>>();
        if let (Some(packet), Some(handle)) = (encoded.last(), decoded.last()) {
            let _ = std::hint::black_box(packet.client.read_one(handle.clone()));
        }
        let decode_elapsed = decode_start.elapsed().as_secs_f64().max(1e-9);
        let decode_qps = samples as f64 / decode_elapsed;

        let roundtrip_start = Instant::now();
        let mut last = None;
        for (idx, tensor) in tensors.iter().enumerate() {
            let packet = crate::burn_ext::turboquant_mse_encode_entropy_with_codebook::<
                burn_wgpu::WgpuRuntime,
                i32,
                u32,
            >(
                tensor.clone(),
                bit_width,
                entropy_seed ^ (idx as u64),
                &codebook,
            );
            let decoded =
                crate::burn_ext::turboquant_mse_decode_indices_with_codebook(&packet, &codebook);
            last = Some((packet, decoded));
        }
        if let Some((packet, decoded)) = last {
            let _ = std::hint::black_box(packet.client.read_one(decoded));
        }
        let roundtrip_elapsed = roundtrip_start.elapsed().as_secs_f64().max(1e-9);
        let roundtrip_qps = samples as f64 / roundtrip_elapsed;

        CodecPathTiming {
            encode_qps,
            decode_qps,
            roundtrip_qps,
        }
    };

    BurnExtCodecProfile {
        regular_qps,
        bitpacked,
        entropy,
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
fn init_wgpu_msl_once(device: &cubecl::wgpu::WgpuDevice) {
    static WGPU_MSL_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WGPU_MSL_INIT.get_or_init(|| {
        let _setup = cubecl::wgpu::init_setup::<cubecl::wgpu::Metal>(
            device,
            cubecl::wgpu::RuntimeOptions::default(),
        );
    });
}

/// Measure fused TurboQuant kernel throughput on CubeCL CPU runtime.
///
/// Returns vectors/second for fused launches (`launch_turboquant_fused_device` + readback).
pub fn fused_kernel_throughput_cpu(dim: usize, samples: usize, bit_width: u8, seed: u64) -> f64 {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260331);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = Default::default();
    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::cpu::CpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure fused TurboQuant kernel throughput on CubeCL CPU runtime with explicit cube-dim.
pub fn fused_kernel_throughput_cpu_with_cube_dim(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    cube_dim: u32,
) -> f64 {
    assert!(cube_dim > 0, "cube_dim must be > 0");
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260333);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = Default::default();
    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device,
        dim,
        bit_width,
        seed,
        Some(TurboQuantLaunchOverride { cube_dim }),
    );
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::cpu::CpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure fused TurboQuant CPU dispatch throughput with only one terminal readback.
///
/// This approximates a lower-overhead "compute path" by avoiding per-iteration host
/// synchronization. A final single readback is still performed to keep completion semantics.
pub fn fused_kernel_dispatch_throughput_cpu_with_cube_dim(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    cube_dim: u32,
) -> f64 {
    assert!(cube_dim > 0, "cube_dim must be > 0");
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260335);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = Default::default();
    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device,
        dim,
        bit_width,
        seed,
        Some(TurboQuantLaunchOverride { cube_dim }),
    );
    let mut last = None;
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::cpu::CpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        last = Some(outputs);
    }
    if let Some(outputs) = last {
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure device-native pipeline timings on CPU runtime.
pub fn pipeline_device_timing_cpu(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    entropy: bool,
) -> DeviceTimingBreakdown {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260337);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = Default::default();
    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let input_handles = dataset
        .iter()
        .map(|data| assets.client.create_from_slice(f32::as_bytes(data)))
        .collect::<Vec<_>>();
    let mut last_packet = None;
    let dispatch_start = Instant::now();
    for input_handle in &input_handles {
        let (_state, packet) = launch_turboquant_pipeline_device_from_handle::<
            cubecl::cpu::CpuRuntime,
        >(&assets, std::hint::black_box(input_handle), true, entropy);
        last_packet = Some(packet);
    }
    let dispatch_elapsed = dispatch_start.elapsed().as_secs_f64().max(1e-9);
    let dispatch_qps = samples as f64 / dispatch_elapsed;

    let mut sync_qps = 0.0;
    let mut export_qps = 0.0;
    if let Some(packet) = last_packet {
        let sync_start = Instant::now();
        packet.client.flush();
        let sync_elapsed = sync_start.elapsed().as_secs_f64().max(1e-9);
        sync_qps = samples as f64 / sync_elapsed;

        let export_start = Instant::now();
        let _ = std::hint::black_box(packet.client.read_one(packet.payload_words_handle));
        let export_elapsed = export_start.elapsed().as_secs_f64().max(1e-9);
        export_qps = samples as f64 / export_elapsed;
    }

    DeviceTimingBreakdown {
        dispatch_qps,
        sync_qps,
        export_qps,
    }
}

/// Sweep cube-dim candidates on CPU runtime and return best result.
pub fn autotune_cube_dim_cpu(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    candidates: &[u32],
) -> (u32, f64, Vec<(u32, f64)>) {
    assert!(!candidates.is_empty(), "candidates must not be empty");
    let mut results = Vec::with_capacity(candidates.len());
    let mut best = (candidates[0], f64::NEG_INFINITY);
    for &cube_dim in candidates {
        let qps =
            fused_kernel_throughput_cpu_with_cube_dim(dim, samples, bit_width, seed, cube_dim);
        if qps > best.1 {
            best = (cube_dim, qps);
        }
        results.push((cube_dim, qps));
    }
    (best.0, best.1, results)
}

/// Run repeated CPU autotune sweeps and return medians per candidate.
#[allow(clippy::too_many_arguments)]
pub fn autotune_cube_dim_cpu_profiled(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    candidates: &[u32],
    warmup_rounds: usize,
    trials: usize,
    readback_each_iteration: bool,
) -> (u32, f64, Vec<(u32, f64)>) {
    assert!(trials > 0, "trials must be > 0");
    for _ in 0..warmup_rounds {
        for &cube_dim in candidates {
            if readback_each_iteration {
                let _ = fused_kernel_throughput_cpu_with_cube_dim(
                    dim, samples, bit_width, seed, cube_dim,
                );
            } else {
                let _ = fused_kernel_dispatch_throughput_cpu_with_cube_dim(
                    dim, samples, bit_width, seed, cube_dim,
                );
            }
        }
    }

    let mut per_candidate = Vec::with_capacity(candidates.len());
    let mut best = (candidates[0], f64::NEG_INFINITY);
    for &cube_dim in candidates {
        let mut run_values = Vec::with_capacity(trials);
        for t in 0..trials {
            let trial_seed = seed ^ ((t as u64) << 20);
            let qps = if readback_each_iteration {
                fused_kernel_throughput_cpu_with_cube_dim(
                    dim, samples, bit_width, trial_seed, cube_dim,
                )
            } else {
                fused_kernel_dispatch_throughput_cpu_with_cube_dim(
                    dim, samples, bit_width, trial_seed, cube_dim,
                )
            };
            run_values.push(qps);
        }
        let med = median(&mut run_values);
        if med > best.1 {
            best = (cube_dim, med);
        }
        per_candidate.push((cube_dim, med));
    }
    (best.0, best.1, per_candidate)
}

/// Measure fused TurboQuant kernel throughput on CubeCL WGPU Metal runtime.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn fused_kernel_throughput_wgpu_msl(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
) -> f64 {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260332);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let assets = prepare_turboquant_launch_assets::<cubecl::wgpu::WgpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::wgpu::WgpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure fused TurboQuant throughput on WGPU Metal with explicit cube-dim.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn fused_kernel_throughput_wgpu_msl_with_cube_dim(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    cube_dim: u32,
) -> f64 {
    assert!(cube_dim > 0, "cube_dim must be > 0");
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260334);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let assets = prepare_turboquant_launch_assets::<cubecl::wgpu::WgpuRuntime>(
        &device,
        dim,
        bit_width,
        seed,
        Some(TurboQuantLaunchOverride { cube_dim }),
    );
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::wgpu::WgpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure fused TurboQuant WGPU Metal dispatch throughput with only one terminal readback.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn fused_kernel_dispatch_throughput_wgpu_msl_with_cube_dim(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    cube_dim: u32,
) -> f64 {
    assert!(cube_dim > 0, "cube_dim must be > 0");
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260336);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let assets = prepare_turboquant_launch_assets::<cubecl::wgpu::WgpuRuntime>(
        &device,
        dim,
        bit_width,
        seed,
        Some(TurboQuantLaunchOverride { cube_dim }),
    );
    let mut last = None;
    let start = Instant::now();
    for data in &dataset {
        let outputs = launch_turboquant_fused_device_with_assets::<cubecl::wgpu::WgpuRuntime>(
            &assets,
            std::hint::black_box(data),
            true,
        );
        last = Some(outputs);
    }
    if let Some(outputs) = last {
        let _ = std::hint::black_box(read_fused_outputs(&outputs));
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    samples as f64 / elapsed
}

/// Measure device-native pipeline timings on WGPU Metal runtime.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn pipeline_device_timing_wgpu_msl(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    entropy: bool,
) -> DeviceTimingBreakdown {
    assert!(dim > 0, "dim must be positive");
    assert!(samples > 0, "samples must be positive");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mut rng = StdRng::seed_from_u64(20260338);
    let mut dataset = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut data = vec![0.0_f32; dim];
        for value in &mut data {
            *value = rng.random_range(-1.0_f32..1.0_f32);
        }
        dataset.push(data);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let assets = prepare_turboquant_launch_assets::<cubecl::wgpu::WgpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let input_handles = dataset
        .iter()
        .map(|data| assets.client.create_from_slice(f32::as_bytes(data)))
        .collect::<Vec<_>>();
    let mut last_packet = None;
    let dispatch_start = Instant::now();
    for input_handle in &input_handles {
        let (_state, packet) = launch_turboquant_pipeline_device_from_handle::<
            cubecl::wgpu::WgpuRuntime,
        >(&assets, std::hint::black_box(input_handle), true, entropy);
        last_packet = Some(packet);
    }
    let dispatch_elapsed = dispatch_start.elapsed().as_secs_f64().max(1e-9);
    let dispatch_qps = samples as f64 / dispatch_elapsed;

    let mut sync_qps = 0.0;
    let mut export_qps = 0.0;
    if let Some(packet) = last_packet {
        let sync_start = Instant::now();
        packet.client.flush();
        let sync_elapsed = sync_start.elapsed().as_secs_f64().max(1e-9);
        sync_qps = samples as f64 / sync_elapsed;

        let export_start = Instant::now();
        let _ = std::hint::black_box(packet.client.read_one(packet.payload_words_handle));
        let export_elapsed = export_start.elapsed().as_secs_f64().max(1e-9);
        export_qps = samples as f64 / export_elapsed;
    }

    DeviceTimingBreakdown {
        dispatch_qps,
        sync_qps,
        export_qps,
    }
}

/// Sweep cube-dim candidates on WGPU Metal runtime and return best result.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn autotune_cube_dim_wgpu_msl(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    candidates: &[u32],
) -> (u32, f64, Vec<(u32, f64)>) {
    assert!(!candidates.is_empty(), "candidates must not be empty");
    let mut results = Vec::with_capacity(candidates.len());
    let mut best = (candidates[0], f64::NEG_INFINITY);
    for &cube_dim in candidates {
        let qps =
            fused_kernel_throughput_wgpu_msl_with_cube_dim(dim, samples, bit_width, seed, cube_dim);
        if qps > best.1 {
            best = (cube_dim, qps);
        }
        results.push((cube_dim, qps));
    }
    (best.0, best.1, results)
}

/// Run repeated WGPU Metal autotune sweeps and return medians per candidate.
#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
pub fn autotune_cube_dim_wgpu_msl_profiled(
    dim: usize,
    samples: usize,
    bit_width: u8,
    seed: u64,
    candidates: &[u32],
    warmup_rounds: usize,
    trials: usize,
    readback_each_iteration: bool,
) -> (u32, f64, Vec<(u32, f64)>) {
    assert!(trials > 0, "trials must be > 0");
    for _ in 0..warmup_rounds {
        for &cube_dim in candidates {
            if readback_each_iteration {
                let _ = fused_kernel_throughput_wgpu_msl_with_cube_dim(
                    dim, samples, bit_width, seed, cube_dim,
                );
            } else {
                let _ = fused_kernel_dispatch_throughput_wgpu_msl_with_cube_dim(
                    dim, samples, bit_width, seed, cube_dim,
                );
            }
        }
    }

    let mut per_candidate = Vec::with_capacity(candidates.len());
    let mut best = (candidates[0], f64::NEG_INFINITY);
    for &cube_dim in candidates {
        let mut run_values = Vec::with_capacity(trials);
        for t in 0..trials {
            let trial_seed = seed ^ ((t as u64) << 20);
            let qps = if readback_each_iteration {
                fused_kernel_throughput_wgpu_msl_with_cube_dim(
                    dim, samples, bit_width, trial_seed, cube_dim,
                )
            } else {
                fused_kernel_dispatch_throughput_wgpu_msl_with_cube_dim(
                    dim, samples, bit_width, trial_seed, cube_dim,
                )
            };
            run_values.push(qps);
        }
        let med = median(&mut run_values);
        if med > best.1 {
            best = (cube_dim, med);
        }
        per_candidate.push((cube_dim, med));
    }
    (best.0, best.1, per_candidate)
}

#[cfg(test)]
mod tests {
    use cubecl::prelude::Runtime;
    use cubecl::CubeElement;
    use rand::{RngExt, SeedableRng};

    use super::{
        autotune_cube_dim_cpu, fused_kernel_throughput_cpu, kernel_codec_profile_cpu,
        pipeline_device_timing_cpu,
    };

    #[test]
    fn fused_kernel_throughput_benchmark_cpu() {
        fn assert_runtime<R: Runtime>() {}
        assert_runtime::<cubecl::cpu::CpuRuntime>();
        let qps = fused_kernel_throughput_cpu(128, 128, 4, 55);
        assert!(qps > 0.0, "fused throughput must be positive");
    }

    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_cpu() {
        let fused_qps = fused_kernel_throughput_cpu(256, 256, 4, 77);
        let device = pipeline_device_timing_cpu(256, 128, 4, 0xAA77, true);
        println!("profile.cpu.fused_kernel_qps={fused_qps:.3}");
        println!("profile.cpu.device.dispatch_qps={:.3}", device.dispatch_qps);
        println!("profile.cpu.device.sync_qps={:.3}", device.sync_qps);
        println!("profile.cpu.device.export_qps={:.3}", device.export_qps);
    }

    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_cpu_codecs_kernel() {
        let profile = kernel_codec_profile_cpu(256, 128, 4, 0xCC77);
        println!(
            "profile.cpu.kernel_codec.regular.encode_qps={:.3}",
            profile.regular.encode_qps
        );
        println!(
            "profile.cpu.kernel_codec.regular.decode_qps={:.3}",
            profile.regular.decode_qps
        );
        println!(
            "profile.cpu.kernel_codec.regular.roundtrip_qps={:.3}",
            profile.regular.roundtrip_qps
        );
        println!(
            "profile.cpu.kernel_codec.bitpacked.encode_qps={:.3}",
            profile.bitpacked.encode_qps
        );
        println!(
            "profile.cpu.kernel_codec.bitpacked.decode_qps={:.3}",
            profile.bitpacked.decode_qps
        );
        println!(
            "profile.cpu.kernel_codec.bitpacked.roundtrip_qps={:.3}",
            profile.bitpacked.roundtrip_qps
        );
        println!(
            "profile.cpu.kernel_codec.delta_xor.encode_qps={:.3}",
            profile.delta_xor.encode_qps
        );
        println!(
            "profile.cpu.kernel_codec.delta_xor.decode_qps={:.3}",
            profile.delta_xor.decode_qps
        );
        println!(
            "profile.cpu.kernel_codec.delta_xor.roundtrip_qps={:.3}",
            profile.delta_xor.roundtrip_qps
        );
        println!(
            "profile.cpu.kernel_codec.huffman.encode_qps={:.3}",
            profile.huffman.encode_qps
        );
        println!(
            "profile.cpu.kernel_codec.huffman.decode_qps={:.3}",
            profile.huffman.decode_qps
        );
        println!(
            "profile.cpu.kernel_codec.huffman.roundtrip_qps={:.3}",
            profile.huffman.roundtrip_qps
        );
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_codecs_kernel() {
        let profile = super::kernel_codec_profile_wgpu_msl(256, 64, 4, 0xCC88);
        println!(
            "profile.wgpu_msl.kernel_codec.regular.encode_qps={:.3}",
            profile.regular.encode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.regular.decode_qps={:.3}",
            profile.regular.decode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.regular.roundtrip_qps={:.3}",
            profile.regular.roundtrip_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.bitpacked.encode_qps={:.3}",
            profile.bitpacked.encode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.bitpacked.decode_qps={:.3}",
            profile.bitpacked.decode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.bitpacked.roundtrip_qps={:.3}",
            profile.bitpacked.roundtrip_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.delta_xor.encode_qps={:.3}",
            profile.delta_xor.encode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.delta_xor.decode_qps={:.3}",
            profile.delta_xor.decode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.delta_xor.roundtrip_qps={:.3}",
            profile.delta_xor.roundtrip_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.huffman.encode_qps={:.3}",
            profile.huffman.encode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.huffman.decode_qps={:.3}",
            profile.huffman.decode_qps
        );
        println!(
            "profile.wgpu_msl.kernel_codec.huffman.roundtrip_qps={:.3}",
            profile.huffman.roundtrip_qps
        );
    }

    #[cfg(all(
        feature = "burn-ext",
        feature = "wgpu",
        feature = "wgpu-msl",
        target_os = "macos"
    ))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_codecs_burn_ext() {
        let profile = super::burn_ext_codec_profile_wgpu_msl(256, 96, 4, 0xDD77);
        println!(
            "profile.wgpu_msl.burn_ext_codec.regular.qps={:.3}",
            profile.regular_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.bitpacked.encode_qps={:.3}",
            profile.bitpacked.encode_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.bitpacked.decode_qps={:.3}",
            profile.bitpacked.decode_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.bitpacked.roundtrip_qps={:.3}",
            profile.bitpacked.roundtrip_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.entropy.encode_qps={:.3}",
            profile.entropy.encode_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.entropy.decode_qps={:.3}",
            profile.entropy.decode_qps
        );
        println!(
            "profile.wgpu_msl.burn_ext_codec.entropy.roundtrip_qps={:.3}",
            profile.entropy.roundtrip_qps
        );
    }

    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_cpu_codec_memory_kv_models() {
        struct KvCase {
            model: &'static str,
            kv_heads: usize,
            head_dim: usize,
            layers: usize,
        }

        let cases = [
            KvCase {
                model: "Llama-3.1-8B",
                kv_heads: 8,
                head_dim: 128,
                layers: 32,
            },
            KvCase {
                model: "Llama-3.1-70B",
                kv_heads: 8,
                head_dim: 128,
                layers: 80,
            },
            KvCase {
                model: "Mistral-7B-v0.1",
                kv_heads: 8,
                head_dim: 128,
                layers: 32,
            },
            KvCase {
                model: "Qwen2.5-7B",
                kv_heads: 4,
                head_dim: 128,
                layers: 28,
            },
        ];

        for (idx, case) in cases.iter().enumerate() {
            let dim = 2 * case.kv_heads * case.head_dim;
            let samples = 32usize;
            let bit_width = 4_u8;
            let seed = 0xEE00_u64 + idx as u64;

            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xABCDEF_u64);
            let mut dataset = Vec::with_capacity(samples);
            for _ in 0..samples {
                let mut data = vec![0.0_f32; dim];
                for value in &mut data {
                    *value = rng.random_range(-1.0_f32..1.0_f32);
                }
                dataset.push(data);
            }

            let device = Default::default();
            let assets = super::prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
                &device, dim, bit_width, seed, None,
            );
            let input_handles = dataset
                .iter()
                .map(|data| assets.client.create_from_slice(f32::as_bytes(data)))
                .collect::<Vec<_>>();
            let states = input_handles
                .iter()
                .map(|handle| {
                    super::launch_turboquant_fused_device_from_handle::<cubecl::cpu::CpuRuntime>(
                        &assets, handle, true,
                    )
                })
                .collect::<Vec<_>>();

            let mut bitpacked_wire = 0usize;
            let mut bitpacked_resident = 0usize;
            let mut delta_wire = 0usize;
            let mut delta_resident = 0usize;
            let mut huffman_wire = 0usize;
            let mut huffman_resident_packet_only = 0usize;
            let codebook = crate::kernels::build_device_huffman_codebook::<cubecl::cpu::CpuRuntime>(
                &states[0],
            );
            let codebook_resident = super::huffman_codebook_resident_bytes(&codebook);
            for state in &states {
                let bit = super::encode_device_bitpacked::<cubecl::cpu::CpuRuntime>(state);
                let delta = super::encode_device_entropy::<cubecl::cpu::CpuRuntime>(&bit);
                let huff = crate::kernels::encode_device_huffman_with_codebook::<
                    cubecl::cpu::CpuRuntime,
                >(state, &codebook);
                let _decoded =
                    crate::kernels::decode_device_indices_with_codebook(&huff, Some(&codebook));

                bitpacked_wire += super::packet_wire_bytes(&bit);
                bitpacked_resident += super::packet_resident_bytes(&bit);
                delta_wire += super::packet_wire_bytes(&delta);
                delta_resident += super::packet_resident_bytes(&delta);
                huffman_wire += super::packet_wire_bytes(&huff);
                huffman_resident_packet_only += super::packet_resident_bytes(&huff);
            }

            let avg_bitpacked_wire = bitpacked_wire as f64 / samples as f64;
            let avg_delta_wire = delta_wire as f64 / samples as f64;
            let avg_huffman_wire = huffman_wire as f64 / samples as f64;
            let avg_bitpacked_resident = bitpacked_resident as f64 / samples as f64;
            let avg_delta_resident = delta_resident as f64 / samples as f64;
            let avg_huffman_resident_packet_only =
                huffman_resident_packet_only as f64 / samples as f64;
            let avg_huffman_resident_with_shared_codebook =
                avg_huffman_resident_packet_only + (codebook_resident as f64 / samples as f64);

            let regular_indices_bytes_per_layer = (dim * core::mem::size_of::<u32>()) as f64;
            let regular_kv_fp16_bytes_per_layer = (dim * core::mem::size_of::<u16>()) as f64;
            let regular_kv_fp16_total = regular_kv_fp16_bytes_per_layer * case.layers as f64;

            let bitpacked_total = avg_bitpacked_wire * case.layers as f64;
            let delta_total = avg_delta_wire * case.layers as f64;
            let huffman_total = avg_huffman_wire * case.layers as f64;

            let save_bit_vs_kv = 100.0 * (1.0 - bitpacked_total / regular_kv_fp16_total);
            let save_delta_vs_kv = 100.0 * (1.0 - delta_total / regular_kv_fp16_total);
            let save_huff_vs_kv = 100.0 * (1.0 - huffman_total / regular_kv_fp16_total);
            let save_bit_vs_regular_indices =
                100.0 * (1.0 - avg_bitpacked_wire / regular_indices_bytes_per_layer);
            let save_delta_vs_regular_indices =
                100.0 * (1.0 - avg_delta_wire / regular_indices_bytes_per_layer);
            let save_huff_vs_regular_indices =
                100.0 * (1.0 - avg_huffman_wire / regular_indices_bytes_per_layer);

            println!(
                "profile.cpu.codec_memory.model={} dim={} layers={} regular_kv_fp16_bytes_total={:.0}",
                case.model, dim, case.layers, regular_kv_fp16_total
            );
            println!(
                "profile.cpu.codec_memory.model={} regular_indices_u32_bytes_per_layer={:.0}",
                case.model, regular_indices_bytes_per_layer
            );
            println!(
                "profile.cpu.codec_memory.model={} bitpacked_wire_bytes_per_layer_avg={:.3} bitpacked_resident_bytes_per_layer_avg={:.3} save_vs_kv_pct={:.3} save_vs_regular_indices_pct={:.3}",
                case.model, avg_bitpacked_wire, avg_bitpacked_resident, save_bit_vs_kv, save_bit_vs_regular_indices
            );
            println!(
                "profile.cpu.codec_memory.model={} delta_xor_wire_bytes_per_layer_avg={:.3} delta_xor_resident_bytes_per_layer_avg={:.3} save_vs_kv_pct={:.3} save_vs_regular_indices_pct={:.3}",
                case.model, avg_delta_wire, avg_delta_resident, save_delta_vs_kv, save_delta_vs_regular_indices
            );
            println!(
                "profile.cpu.codec_memory.model={} huffman_wire_bytes_per_layer_avg={:.3} huffman_resident_packet_only_bytes_per_layer_avg={:.3} huffman_resident_with_shared_codebook_bytes_per_layer_avg={:.3} save_vs_kv_pct={:.3} save_vs_regular_indices_pct={:.3}",
                case.model,
                avg_huffman_wire,
                avg_huffman_resident_packet_only,
                avg_huffman_resident_with_shared_codebook,
                save_huff_vs_kv,
                save_huff_vs_regular_indices
            );
            println!(
                "profile.cpu.codec_memory.model={} ratio.huffman_vs_bitpacked_wire={:.3} ratio.delta_vs_bitpacked_wire={:.3}",
                case.model,
                avg_huffman_wire / avg_bitpacked_wire,
                avg_delta_wire / avg_bitpacked_wire
            );
        }
    }

    #[cfg(all(
        feature = "burn-ext",
        feature = "wgpu",
        feature = "wgpu-msl",
        target_os = "macos"
    ))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_codec_memory_kv_models_burn_ext() {
        type B = burn_wgpu::Wgpu<f32, i32, u32>;
        struct KvCase {
            model: &'static str,
            kv_heads: usize,
            head_dim: usize,
            layers: usize,
        }

        let cases = [
            KvCase {
                model: "Llama-3.1-8B",
                kv_heads: 8,
                head_dim: 128,
                layers: 32,
            },
            KvCase {
                model: "Qwen2.5-7B",
                kv_heads: 4,
                head_dim: 128,
                layers: 28,
            },
        ];

        let device = burn_wgpu::WgpuDevice::default();
        super::init_wgpu_msl_once(&device);
        for (idx, case) in cases.iter().enumerate() {
            let dim = 2 * case.kv_heads * case.head_dim;
            let bit_width = 4_u8;
            let seed = 0xFA00_u64 + idx as u64;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x123456_u64);
            let mut data = vec![0.0_f32; dim];
            for value in &mut data {
                *value = rng.random_range(-1.0_f32..1.0_f32);
            }

            let tensor = burn::tensor::Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(data, [dim]),
                &device,
            );
            let bit = crate::burn_ext::turboquant_mse_encode_bitpacked::<
                burn_wgpu::WgpuRuntime,
                i32,
                u32,
            >(tensor.clone(), bit_width, seed);
            let codebook = crate::burn_ext::turboquant_mse_build_huffman_codebook::<
                burn_wgpu::WgpuRuntime,
                i32,
                u32,
            >(tensor.clone(), bit_width, seed ^ 0xABC0);
            let ent = crate::burn_ext::turboquant_mse_encode_entropy_with_codebook::<
                burn_wgpu::WgpuRuntime,
                i32,
                u32,
            >(tensor, bit_width, seed ^ 0xABC0, &codebook);

            let regular_kv_fp16_total = (case.layers * dim * core::mem::size_of::<u16>()) as f64;
            let regular_indices_per_layer = (dim * core::mem::size_of::<u32>()) as f64;
            let bit_total = super::packet_wire_bytes(&bit) as f64 * case.layers as f64;
            let ent_total = super::packet_wire_bytes(&ent) as f64 * case.layers as f64;
            let save_bit_vs_kv = 100.0 * (1.0 - bit_total / regular_kv_fp16_total);
            let save_ent_vs_kv = 100.0 * (1.0 - ent_total / regular_kv_fp16_total);
            let save_bit_vs_regular_indices =
                100.0 * (1.0 - super::packet_wire_bytes(&bit) as f64 / regular_indices_per_layer);
            let save_ent_vs_regular_indices =
                100.0 * (1.0 - super::packet_wire_bytes(&ent) as f64 / regular_indices_per_layer);
            let ent_resident_with_shared_codebook = super::packet_resident_bytes(&ent)
                + super::huffman_codebook_resident_bytes(&codebook);

            println!(
                "profile.wgpu_msl.burn_ext_codec_memory.model={} dim={} layers={} regular_kv_fp16_bytes_total={:.0}",
                case.model, dim, case.layers, regular_kv_fp16_total
            );
            println!(
                "profile.wgpu_msl.burn_ext_codec_memory.model={} bitpacked_wire_bytes_per_layer={} resident_bytes_per_layer={} save_vs_kv_pct={:.3} save_vs_regular_indices_pct={:.3}",
                case.model,
                super::packet_wire_bytes(&bit),
                super::packet_resident_bytes(&bit),
                save_bit_vs_kv,
                save_bit_vs_regular_indices
            );
            println!(
                "profile.wgpu_msl.burn_ext_codec_memory.model={} entropy_wire_bytes_per_layer={} resident_packet_only_bytes_per_layer={} resident_with_shared_codebook_bytes_per_layer={} save_vs_kv_pct={:.3} save_vs_regular_indices_pct={:.3} ratio.entropy_vs_bitpacked_wire={:.3}",
                case.model,
                super::packet_wire_bytes(&ent),
                super::packet_resident_bytes(&ent),
                ent_resident_with_shared_codebook,
                save_ent_vs_kv,
                save_ent_vs_regular_indices,
                super::packet_wire_bytes(&ent) as f64 / super::packet_wire_bytes(&bit) as f64
            );
        }
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl() {
        let qps = super::fused_kernel_throughput_wgpu_msl(256, 128, 4, 177);
        let device = super::pipeline_device_timing_wgpu_msl(256, 64, 4, 0xBB77, true);
        println!("profile.wgpu_msl.fused_kernel_qps={qps:.3}");
        println!(
            "profile.wgpu_msl.device.dispatch_qps={:.3}",
            device.dispatch_qps
        );
        println!("profile.wgpu_msl.device.sync_qps={:.3}", device.sync_qps);
        println!(
            "profile.wgpu_msl.device.export_qps={:.3}",
            device.export_qps
        );
    }

    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_cpu_autotune_cube_dim() {
        let candidates = [1_u32, 2, 4, 8, 16, 32, 64];
        let (best_dim, best_qps, results) = autotune_cube_dim_cpu(256, 128, 4, 277, &candidates);
        for (cube_dim, qps) in results {
            println!("profile.cpu.autotune.cube_dim={cube_dim} fused_qps={qps:.3}");
        }
        println!("profile.cpu.autotune.best_cube_dim={best_dim} fused_qps={best_qps:.3}");
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_kv_models() {
        struct KvCase {
            model: &'static str,
            kv_heads: usize,
            head_dim: usize,
            layers: usize,
        }

        // Common open-weight model KV layouts (per published config defaults).
        let cases = [
            KvCase {
                model: "Llama-3.1-8B",
                kv_heads: 8,
                head_dim: 128,
                layers: 32,
            },
            KvCase {
                model: "Llama-3.1-70B",
                kv_heads: 8,
                head_dim: 128,
                layers: 80,
            },
            KvCase {
                model: "Mistral-7B-v0.1",
                kv_heads: 8,
                head_dim: 128,
                layers: 32,
            },
            KvCase {
                model: "Qwen2.5-7B",
                kv_heads: 4,
                head_dim: 128,
                layers: 28,
            },
        ];

        for (idx, case) in cases.iter().enumerate() {
            let kv_dim_per_token_per_layer = 2 * case.kv_heads * case.head_dim;
            let kv_bytes_fp16_per_token_total = case.layers * kv_dim_per_token_per_layer * 2;
            let qps = super::fused_kernel_throughput_wgpu_msl(
                kv_dim_per_token_per_layer,
                96,
                4,
                0x6000 + idx as u64,
            );
            println!(
                "profile.wgpu_msl.kv.model={} dim={} layers={} kv_bytes_fp16_per_token_total={} fused_qps={:.3}",
                case.model,
                kv_dim_per_token_per_layer,
                case.layers,
                kv_bytes_fp16_per_token_total,
                qps
            );
        }
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_autotune_cube_dim() {
        let candidates = [1_u32, 2, 4, 8, 16, 32, 64, 128];
        let (best_dim, best_qps, results) =
            super::autotune_cube_dim_wgpu_msl(2048, 64, 4, 377, &candidates);
        for (cube_dim, qps) in results {
            println!("profile.wgpu_msl.autotune.cube_dim={cube_dim} fused_qps={qps:.3}");
        }
        println!("profile.wgpu_msl.autotune.best_cube_dim={best_dim} fused_qps={best_qps:.3}");
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
    #[test]
    #[ignore = "profiling report test; run manually with --ignored --nocapture"]
    fn print_profile_report_wgpu_msl_autotune_cube_dim_median_modes() {
        let candidates = [2_u32, 4, 8, 16, 32, 64, 128];
        let (best_e2e_dim, best_e2e_qps, e2e) =
            super::autotune_cube_dim_wgpu_msl_profiled(2048, 48, 4, 477, &candidates, 1, 3, true);
        for (cube_dim, qps) in e2e {
            println!("profile.wgpu_msl.autotune_median.e2e.cube_dim={cube_dim} fused_qps={qps:.3}");
        }
        println!(
            "profile.wgpu_msl.autotune_median.e2e.best_cube_dim={best_e2e_dim} fused_qps={best_e2e_qps:.3}"
        );

        let (best_dispatch_dim, best_dispatch_qps, dispatch) =
            super::autotune_cube_dim_wgpu_msl_profiled(2048, 48, 4, 577, &candidates, 1, 3, false);
        for (cube_dim, qps) in dispatch {
            println!(
                "profile.wgpu_msl.autotune_median.dispatch.cube_dim={cube_dim} fused_qps={qps:.3}"
            );
        }
        println!(
            "profile.wgpu_msl.autotune_median.dispatch.best_cube_dim={best_dispatch_dim} fused_qps={best_dispatch_qps:.3}"
        );
    }
}
