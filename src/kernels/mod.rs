//! TurboQuant kernel-facing APIs and backend launch helpers.
//!
//! This module includes:
//! - host reference implementations for TurboQuantmse and TurboQuantprod,
//! - device-native compact packet encoding for MSE indices,
//! - fused CubeCL kernel launch helpers for CPU and WGPU/MSL runtimes,
//! - device-only validation helpers used by integration tests.

use core::f32::consts::PI;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use cubecl::prelude::*;
use cubecl::server::Handle;
use rand::RngExt;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const MSE_ROTATION_SALT: u64 = 0xA5A5_17C3_913D_20FF;
const PROD_PROJECTION_SALT: u64 = 0xCEFA_EDFE_1987_0011;

type QuantCache<K> = OnceLock<Mutex<HashMap<K, Vec<f32>>>>;
type CentroidKey = (u8, usize);
type ProjectionKey = (usize, u64);

static CENTROID_CACHE: QuantCache<CentroidKey> = OnceLock::new();
static PROJECTION_CACHE: QuantCache<ProjectionKey> = OnceLock::new();
static HUFFMAN_POLICY_ID_GEN: AtomicU64 = AtomicU64::new(1);

#[inline]
fn assert_huffman_experimental() {
    #[cfg(not(feature = "experimental-huffman"))]
    panic!("huffman is experimental; enable feature \"experimental-huffman\"");
}

/// Quantized packet for TurboQuantmse (Algorithm 1).
#[derive(Debug, Clone)]
#[cfg(test)]
pub(crate) struct QuantMsePacket {
    /// Quantized centroid indices in rotated coordinate order.
    pub indices: Vec<u16>,
    /// Input dimensionality.
    pub dim: usize,
    /// Bits used per coordinate during MSE quantization.
    pub bit_width: u8,
    /// Seed used for deterministic rotation/permutation generation.
    pub seed: u64,
}

/// Quantized packet for TurboQuantprod (Algorithm 2).
#[derive(Debug, Clone)]
#[cfg(test)]
#[allow(dead_code)]
pub(crate) struct QuantProdPacket {
    /// First-stage MSE packet (bit-width `b - 1`).
    pub mse: QuantMsePacket,
    /// One-bit QJL sign sketch of the residual.
    pub qjl_signs: Vec<i8>,
    /// Residual norm used to scale dequantization.
    pub residual_norm: f32,
    /// Input dimensionality.
    pub dim: usize,
    /// Seed used for deterministic projection generation.
    pub seed: u64,
}

/// Kernel variant selected at compile-time by the blueprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantVariant {
    /// Quantize only the MSE stage.
    MseOnly,
    /// Quantize MSE plus QJL sign stage in one fused surface.
    ProdFused,
}

/// Minimal CubeK-style blueprint: only structural kernel choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantBlueprint {
    /// Structural variant selected for kernel specialization.
    pub variant: TurboQuantVariant,
    /// Number of quantization levels (`2^bit_width`).
    pub levels: u32,
    /// Whether out-of-bounds handling uses masking.
    pub masked_oob: bool,
    /// Whether QJL signs are emitted by the kernel.
    pub emit_qjl: bool,
    /// Whether entropy transform is fused into packet emission.
    pub emit_entropy: bool,
    /// Whether packet emission is fused into a single kernel launch.
    pub single_kernel_packet: bool,
}

/// Runtime launch settings owned by launch/routine layers.
#[derive(Debug, Clone)]
pub struct TurboQuantLaunchSettings {
    /// Number of cubes to dispatch.
    pub cube_count: CubeCount,
    /// Cube dimensions for the dispatch.
    pub cube_dim: CubeDim,
    /// Tensor line size used for launch arguments.
    pub line_size: u8,
}

/// Runtime problem description used by the routine.
#[derive(Debug, Clone, Copy)]
pub struct TurboQuantProblem {
    /// Input dimensionality.
    pub dim: usize,
    /// Target quantization bit-width.
    pub bit_width: u8,
}

/// CubeK-style kernel options toggled by the caller.
#[derive(Debug, Clone, Copy)]
pub struct TurboQuantKernelOptions {
    /// Emit QJL signs from the fused kernel.
    pub emit_qjl: bool,
    /// Fuse entropy transform into packet emission.
    pub emit_entropy: bool,
    /// Use a single-kernel packet pipeline.
    pub single_kernel_packet: bool,
}

/// Routine that adapts runtime constraints into blueprint + launch settings.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurboQuantRoutine;

impl TurboQuantRoutine {
    /// Prepare a minimal blueprint and launch settings for a runtime problem.
    ///
    /// # Parameters
    ///
    /// - `problem`: runtime problem shape and bit-width.
    ///
    /// # Returns
    ///
    /// `(blueprint, launch_settings)` following the CubeK split.
    pub fn prepare(
        problem: TurboQuantProblem,
        options: TurboQuantKernelOptions,
    ) -> (TurboQuantBlueprint, TurboQuantLaunchSettings) {
        let levels = 1_u32 << problem.bit_width.max(1);
        let blueprint = TurboQuantBlueprint {
            variant: if options.emit_qjl {
                TurboQuantVariant::ProdFused
            } else {
                TurboQuantVariant::MseOnly
            },
            levels,
            masked_oob: true,
            emit_qjl: options.emit_qjl,
            emit_entropy: options.emit_entropy,
            single_kernel_packet: options.single_kernel_packet,
        };
        let launch = TurboQuantLaunchSettings {
            cube_count: CubeCount::new_1d(problem.dim as u32),
            cube_dim: CubeDim::new_1d(1),
            line_size: 1,
        };
        (blueprint, launch)
    }
}

/// Quantize a vector with TurboQuantmse (Algorithm 1).
///
/// Paper trace:
/// - `docs/references/turboquant.txt`, Algorithm 1 (lines 419-438).
/// - Theorem 1 distortion guarantees (lines 468-546).
///
/// # Parameters
///
/// - `x`: input vector.
/// - `bit_width`: bits per coordinate in `[1, 8]`.
/// - `seed`: deterministic seed for rotation/permutation generation.
///
/// # Returns
///
/// A `QuantMsePacket` containing quantized centroid indices.
///
/// # Panics
///
/// Panics if `x` is empty or `bit_width` is outside `[1, 8]`.
#[cfg(test)]
pub(crate) fn quantize_mse(x: &[f32], bit_width: u8, seed: u64) -> QuantMsePacket {
    assert!(!x.is_empty(), "input vector must be non-empty");
    assert!((1..=8).contains(&bit_width), "bit_width must be in [1, 8]");

    let dim = x.len();
    let centroids = centroids_for(bit_width, dim);
    let rotated = apply_signed_permutation(x, dim, seed ^ MSE_ROTATION_SALT);

    let mut indices = Vec::with_capacity(dim);
    for value in rotated {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (idx, centroid) in centroids.iter().enumerate() {
            let d = value - *centroid;
            let dist = d * d;
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        indices.push(best_idx as u16);
    }

    QuantMsePacket {
        indices,
        dim,
        bit_width,
        seed,
    }
}

/// Dequantize a TurboQuantmse packet back to vector space.
///
/// # Parameters
///
/// - `packet`: quantized MSE packet.
///
/// # Returns
///
/// Reconstructed vector in the original basis.
#[cfg(test)]
pub(crate) fn dequantize_mse(packet: &QuantMsePacket) -> Vec<f32> {
    let centroids = centroids_for(packet.bit_width, packet.dim);
    let mut rotated = vec![0.0_f32; packet.dim];
    for (i, out) in rotated.iter_mut().enumerate() {
        let idx = packet.indices[i] as usize;
        *out = centroids[idx];
    }
    invert_signed_permutation(&rotated, packet.dim, packet.seed ^ MSE_ROTATION_SALT)
}

/// Quantize a vector with TurboQuantprod (Algorithm 2).
///
/// This performs MSE quantization at bit-width `max(b-1, 1)`, then computes a one-bit
/// QJL sketch over the residual.
///
/// Paper trace:
/// - `docs/references/turboquant.txt`, Algorithm 2 (lines 562-582).
/// - Residual-QJL step `sign(S * r)` (line 570).
/// - Theorem 2 properties (lines 616-735).
///
/// # Parameters
///
/// - `x`: input vector.
/// - `bit_width`: total target bit-width (`>= 1`).
/// - `seed`: deterministic seed.
///
/// # Panics
///
/// Panics if `x` is empty or `bit_width == 0`.
#[cfg(test)]
pub(crate) fn quantize_prod(x: &[f32], bit_width: u8, seed: u64) -> QuantProdPacket {
    assert!(!x.is_empty(), "input vector must be non-empty");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let mse_bit_width = bit_width.saturating_sub(1).max(1);
    let mse = quantize_mse(x, mse_bit_width, seed);
    let dim = x.len();
    let centroids = centroids_for(mse.bit_width, dim);
    let (perm, signs) = signed_permutation(dim, seed ^ MSE_ROTATION_SALT);
    let inverse_perm = inverse_permutation(&perm);

    // Fused residual handling: no x_mse or residual vector materialization.
    let mut residual_norm_sq = 0.0_f32;
    for (coord, x_coord) in x.iter().enumerate() {
        let mse_coord =
            dequantized_coordinate(&mse.indices, &centroids, &signs, &inverse_perm, coord);
        let r = *x_coord - mse_coord;
        residual_norm_sq += r * r;
    }
    let residual_norm = residual_norm_sq.sqrt();

    let projection_seed = seed ^ PROD_PROJECTION_SALT;
    let projection = projection_matrix(dim, projection_seed);
    let mut qjl_signs = Vec::with_capacity(dim);
    for row in 0..dim {
        let mut dot = 0.0_f32;
        let base = row * dim;
        for (col, x_col) in x.iter().enumerate() {
            let mse_col =
                dequantized_coordinate(&mse.indices, &centroids, &signs, &inverse_perm, col);
            let residual = *x_col - mse_col;
            dot += projection[base + col] * residual;
        }
        qjl_signs.push(if dot >= 0.0 { 1 } else { -1 });
    }

    QuantProdPacket {
        mse,
        qjl_signs,
        residual_norm,
        dim,
        seed,
    }
}

/// Dequantize a TurboQuantprod packet back to vector space.
///
/// Paper trace:
/// - `docs/references/turboquant.txt`, Algorithm 2 dequantization
///   (lines 572-582), including line 11 scaling term.
///
/// # Parameters
///
/// - `packet`: quantized product packet.
///
/// # Returns
///
/// Reconstructed vector from MSE + QJL stages.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn dequantize_prod(packet: &QuantProdPacket) -> Vec<f32> {
    let dim = packet.dim;
    let centroids = centroids_for(packet.mse.bit_width, dim);
    let (perm, signs) = signed_permutation(dim, packet.mse.seed ^ MSE_ROTATION_SALT);
    let inverse_perm = inverse_permutation(&perm);

    let mut out = vec![0.0_f32; dim];
    for (coord, value) in out.iter_mut().enumerate() {
        *value = dequantized_coordinate(
            &packet.mse.indices,
            &centroids,
            &signs,
            &inverse_perm,
            coord,
        );
    }

    let projection_seed = packet.seed ^ PROD_PROJECTION_SALT;
    let projection = projection_matrix(dim, projection_seed);
    let scale = (PI / 2.0).sqrt() * packet.residual_norm / (dim as f32);
    for row in 0..dim {
        let sign = packet.qjl_signs[row] as f32;
        let base = row * dim;
        for (col, value) in out.iter_mut().enumerate() {
            *value += projection[base + col] * sign * scale;
        }
    }
    out
}

/// CubeK-style fused kernel entrypoint for MSE + QJL stage outputs.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn turboquant_fused_kernel<F: Float>(
    input: &Tensor<F>,
    centroids: &Tensor<F>,
    projection: &Tensor<F>,
    mse_out: &mut Tensor<F>,
    mse_index_out: &mut Tensor<u32>,
    qjl_out: &mut Tensor<F>,
    #[comptime] levels: u32,
    #[comptime] emit_qjl: bool,
) {
    let idx = ABSOLUTE_POS;
    if idx < input.shape(0) {
        let value = input[idx];
        let mut best = centroids[0];
        let mut best_idx = 0u32;
        let mut best_dist = (value - centroids[0]) * (value - centroids[0]);
        for level in 1..levels {
            let c = centroids[level as usize];
            let d = value - c;
            let dist = d * d;
            if dist < best_dist {
                best_dist = dist;
                best = c;
                best_idx = level;
            }
        }
        mse_out[idx] = best;
        mse_index_out[idx] = best_idx;

        if emit_qjl {
            // Compute one QJL row-sign from a row-major projection matrix.
            let dim = input.shape(0);
            let mut dot = F::new(0.0);
            let base = idx * dim;
            for col in 0..dim {
                let x_col = input[col];
                let mut best_col = centroids[0];
                let mut best_col_dist = (x_col - centroids[0]) * (x_col - centroids[0]);
                for level in 1..levels {
                    let c = centroids[level as usize];
                    let d = x_col - c;
                    let dist = d * d;
                    if dist < best_col_dist {
                        best_col_dist = dist;
                        best_col = c;
                    }
                }
                let residual_col = x_col - best_col;
                dot += projection[base + col] * residual_col;
            }
            qjl_out[idx] = if dot >= F::new(0.0) {
                F::new(1.0)
            } else {
                F::new(-1.0)
            };
        }
    }
}

/// Single-launch fused pipeline kernel:
/// quantization + optional QJL + bitpacking + optional entropy.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn turboquant_pipeline_fused_kernel(
    input: &Tensor<f32>,
    centroids: &Tensor<f32>,
    projection: &Tensor<f32>,
    permutation: &Tensor<u32>,
    signs: &Tensor<f32>,
    mse_out: &mut Tensor<f32>,
    mse_index_out: &mut Tensor<u32>,
    qjl_out: &mut Tensor<f32>,
    payload_words: &mut Tensor<u32>,
    valid_bits: usize,
    #[comptime] levels: u32,
    #[comptime] emit_qjl: bool,
    #[comptime] emit_entropy: bool,
) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let dim = input.shape(0);
        let bit_width = valid_bits / dim;

        for word_idx in 0..payload_words.shape(0) {
            payload_words[word_idx] = 0u32;
        }

        for i in 0..dim {
            let src = permutation[i] as usize;
            let value = signs[i] * input[src];
            let mut best = centroids[0];
            let mut best_idx = 0u32;
            let mut best_dist = (value - centroids[0]) * (value - centroids[0]);
            for level in 1..levels {
                let c = centroids[level as usize];
                let d = value - c;
                let dist = d * d;
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                    best_idx = level;
                }
            }

            mse_out[i] = best;
            mse_index_out[i] = best_idx;

            let base_bit = i * bit_width;
            for bit in 0..bit_width {
                let global_bit = base_bit + bit;
                if global_bit < valid_bits {
                    let value_shift = bit as u32;
                    let bit_value = (best_idx >> value_shift) & 1u32;
                    if bit_value == 1u32 {
                        let word_idx = global_bit / 32usize;
                        let local_bit = global_bit % 32usize;
                        let local_shift = local_bit as u32;
                        payload_words[word_idx] |= 1u32 << local_shift;
                    }
                }
            }
        }

        if emit_qjl {
            for row in 0..dim {
                let mut dot = 0.0f32;
                let base = row * dim;
                for col in 0..dim {
                    let src_col = permutation[col] as usize;
                    let x_col = signs[col] * input[src_col];
                    let mut best_col = centroids[0];
                    let mut best_col_dist = (x_col - centroids[0]) * (x_col - centroids[0]);
                    for level in 1..levels {
                        let c = centroids[level as usize];
                        let d = x_col - c;
                        let dist = d * d;
                        if dist < best_col_dist {
                            best_col_dist = dist;
                            best_col = c;
                        }
                    }
                    let residual_col = x_col - best_col;
                    dot += projection[base + col] * residual_col;
                }
                qjl_out[row] = if dot >= f32::new(0.0) {
                    f32::new(1.0)
                } else {
                    f32::new(-1.0)
                };
            }
        }

        if emit_entropy {
            let mut prev = 0u32;
            for i in 0..payload_words.shape(0) {
                let current = payload_words[i];
                payload_words[i] = current ^ prev;
                prev = current;
            }
        }
    }
}

/// Apply deterministic signed permutation on-device.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn apply_signed_permutation_kernel(
    input: &Tensor<f32>,
    permutation: &Tensor<u32>,
    signs: &Tensor<f32>,
    rotated_out: &mut Tensor<f32>,
) {
    let idx = ABSOLUTE_POS;
    if idx < input.shape(0) {
        let src = permutation[idx] as usize;
        rotated_out[idx] = signs[idx] * input[src];
    }
}

/// Device-native bitpacker for MSE indices.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn bitpack_indices_kernel(
    indices: &Tensor<u32>,
    payload_words: &mut Tensor<u32>,
    bit_width: usize,
    valid_bits: usize,
) {
    let word_idx = ABSOLUTE_POS;
    if word_idx < payload_words.shape(0) {
        let mut packed = 0u32;
        let base_bit = word_idx * 32usize;
        for local_bit in 0usize..32usize {
            let global_bit = base_bit + local_bit;
            if global_bit < valid_bits {
                let symbol_idx = global_bit / bit_width;
                let symbol_bit = global_bit % bit_width;
                let symbol = indices[symbol_idx];
                let symbol_shift = symbol_bit as u32;
                let local_shift = local_bit as u32;
                let bit = (symbol >> symbol_shift) & 1u32;
                packed |= bit << local_shift;
            }
        }
        payload_words[word_idx] = packed;
    }
}

/// Device-native reversible entropy transform (delta-xor over packed words).
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn entropy_delta_xor_encode_kernel(input_words: &Tensor<u32>, output_words: &mut Tensor<u32>) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let mut prev = 0u32;
        for i in 0..input_words.shape(0) {
            let current = input_words[i];
            output_words[i] = current ^ prev;
            prev = current;
        }
    }
}

/// Decode device-native delta-xor entropy transform.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn entropy_delta_xor_decode_kernel(input_words: &Tensor<u32>, output_words: &mut Tensor<u32>) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let mut prev = 0u32;
        for i in 0..input_words.shape(0) {
            let current = input_words[i] ^ prev;
            output_words[i] = current;
            prev = current;
        }
    }
}

/// Device-native unpacker from packed words into index tensor.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn unpack_indices_kernel(
    payload_words: &Tensor<u32>,
    output_indices: &mut Tensor<u32>,
    bit_width: usize,
    valid_bits: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx < output_indices.shape(0) {
        let mut value = 0u32;
        let base_bit = idx * bit_width;
        for bit in 0usize..bit_width {
            let global_bit = base_bit + bit;
            if global_bit < valid_bits {
                let word_idx = global_bit / 32usize;
                let local_bit = global_bit % 32usize;
                let packed_word = payload_words[word_idx];
                let local_shift = local_bit as u32;
                let value_shift = bit as u32;
                let bit_value = (packed_word >> local_shift) & 1u32;
                value |= bit_value << value_shift;
            }
        }
        output_indices[idx] = value;
    }
}

/// Encode indices into Huffman bitstream payload on-device.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn histogram_indices_kernel(indices: &Tensor<u32>, frequencies: &mut Tensor<u32>, levels: usize) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        for i in 0..levels {
            frequencies[i] = 0u32;
        }
        for i in 0..indices.shape(0) {
            let symbol = indices[i] as usize;
            if symbol < levels {
                frequencies[symbol] += 1u32;
            }
        }
        // Ensure at least one active symbol to keep tree construction well-defined.
        if indices.shape(0) == 0usize {
            frequencies[0] = 1u32;
        }
    }
}

#[allow(dead_code)]
#[cube(launch_unchecked)]
fn huffman_init_tree_kernel(
    frequencies: &Tensor<u32>,
    node_weights: &mut Tensor<u32>,
    node_parent: &mut Tensor<u32>,
    node_left: &mut Tensor<u32>,
    node_right: &mut Tensor<u32>,
    levels: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let invalid = u32::MAX;
        let node_cap = node_weights.shape(0);
        for i in 0..node_cap {
            node_weights[i] = 0u32;
            node_parent[i] = invalid;
            node_left[i] = invalid;
            node_right[i] = invalid;
        }
        for i in 0..levels {
            // Add one-count smoothing so every symbol keeps a valid code path.
            node_weights[i] = frequencies[i] + 1u32;
        }
    }
}

#[allow(dead_code)]
#[cube(launch_unchecked)]
fn huffman_merge_step_kernel(
    node_weights: &mut Tensor<u32>,
    node_parent: &mut Tensor<u32>,
    node_left: &mut Tensor<u32>,
    node_right: &mut Tensor<u32>,
    scratch_min1: &mut Tensor<u32>,
    scratch_min2: &mut Tensor<u32>,
    next_node: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let invalid = u32::MAX;
        scratch_min1[0] = invalid;
        scratch_min2[0] = invalid;
        for n in 0..next_node {
            if node_parent[n] == invalid {
                let m1 = scratch_min1[0];
                let m2 = scratch_min2[0];
                if m1 == invalid || node_weights[n] < node_weights[m1 as usize] {
                    scratch_min2[0] = m1;
                    scratch_min1[0] = n as u32;
                } else if m2 == invalid || node_weights[n] < node_weights[m2 as usize] {
                    scratch_min2[0] = n as u32;
                }
            }
        }
        let min1 = scratch_min1[0];
        let min2 = scratch_min2[0];
        if min1 != invalid && min2 != invalid {
            node_left[next_node] = min1;
            node_right[next_node] = min2;
            node_parent[min1 as usize] = next_node as u32;
            node_parent[min2 as usize] = next_node as u32;
            node_weights[next_node] = node_weights[min1 as usize] + node_weights[min2 as usize];
        }
    }
}

/// Encode indices into Huffman bitstream payload on-device.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn huffman_encode_indices_kernel(
    indices: &Tensor<u32>,
    node_parent: &Tensor<u32>,
    _node_left: &Tensor<u32>,
    node_right: &Tensor<u32>,
    _root: &Tensor<u32>,
    path_bits: &mut Tensor<u32>,
    payload_words: &mut Tensor<u32>,
    written_bits_out: &mut Tensor<u32>,
    #[comptime] max_nodes: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let invalid = u32::MAX;
        for i in 0..payload_words.shape(0) {
            payload_words[i] = 0u32;
        }

        let mut write_cursor = 0usize;
        for i in 0..indices.shape(0) {
            let symbol = indices[i] as usize;
            let node = RuntimeCell::<u32>::new(symbol as u32);
            let path_len = RuntimeCell::<u32>::new(0u32);

            for _ in 0..max_nodes {
                let parent = node_parent[node.read() as usize];
                if parent != invalid {
                    let bit = if node_right[parent as usize] == node.read() {
                        u32::new(1)
                    } else {
                        u32::new(0)
                    };
                    path_bits[path_len.read() as usize] = bit;
                    path_len.store(path_len.read() + 1u32);
                    node.store(parent);
                }
            }
            if path_len.read() == 0u32 {
                path_bits[0] = 0u32;
                path_len.store(1u32);
            }

            for _ in 0..max_nodes {
                let len = path_len.read();
                if len > 0u32 {
                    let next = len - 1u32;
                    let bit = path_bits[next as usize];
                    let word_idx = write_cursor / 32usize;
                    let bit_idx = write_cursor % 32usize;
                    payload_words[word_idx] |= bit << (bit_idx as u32);
                    write_cursor += 1usize;
                    path_len.store(next);
                }
            }
        }
        written_bits_out[0] = write_cursor as u32;
    }
}

/// Decode Huffman bitstream payload back into indices on-device.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn huffman_decode_indices_kernel(
    payload_words: &Tensor<u32>,
    node_left: &Tensor<u32>,
    node_right: &Tensor<u32>,
    root: &Tensor<u32>,
    output_indices: &mut Tensor<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        let invalid = u32::MAX;
        let root_idx = root[0];
        let mut read_cursor = 0usize;
        for i in 0..output_indices.shape(0) {
            let mut node = root_idx;
            while node_left[node as usize] != invalid && node_right[node as usize] != invalid {
                let word_idx = read_cursor / 32usize;
                let bit_idx = read_cursor % 32usize;
                let bit = (payload_words[word_idx] >> (bit_idx as u32)) & 1u32;
                read_cursor += 1usize;
                node = if bit == 0u32 {
                    node_left[node as usize]
                } else {
                    node_right[node as usize]
                };
            }
            output_indices[i] = node;
        }
    }
}

/// On-device validation kernel for fused output semantics.
///
/// The kernel sets `status[0]` to `1.0` if it detects a mismatch.
#[allow(dead_code)]
#[cube(launch_unchecked)]
fn validate_fused_kernel(
    input: &Tensor<f32>,
    centroids: &Tensor<f32>,
    projection: &Tensor<f32>,
    mse_out: &Tensor<f32>,
    qjl_out: &Tensor<f32>,
    status: &mut Tensor<f32>,
    tolerance: f32,
    #[comptime] levels: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < input.shape(0) {
        let value = input[idx];
        let mut best = centroids[0];
        let mut best_dist = (value - centroids[0]) * (value - centroids[0]);
        for level in 1..levels {
            let c = centroids[level as usize];
            let d = value - c;
            let dist = d * d;
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }

        let tol2 = tolerance * tolerance;
        let mse_diff = mse_out[idx] - best;
        if mse_diff * mse_diff > tol2 {
            status[0] = f32::new(1.0);
        }

        let dim = input.shape(0);
        let mut dot = f32::new(0.0);
        let base = idx * dim;
        for col in 0..dim {
            let x_col = input[col];
            let mut best_col = centroids[0];
            let mut best_col_dist = (x_col - centroids[0]) * (x_col - centroids[0]);
            for level in 1..levels {
                let c = centroids[level as usize];
                let d = x_col - c;
                let dist = d * d;
                if dist < best_col_dist {
                    best_col_dist = dist;
                    best_col = c;
                }
            }
            let residual_col = x_col - best_col;
            dot += projection[base + col] * residual_col;
        }
        let expected_sign = if dot >= f32::new(0.0) {
            f32::new(1.0)
        } else {
            f32::new(-1.0)
        };
        let sign_diff = qjl_out[idx] - expected_sign;
        if sign_diff * sign_diff > tol2 {
            status[0] = f32::new(1.0);
        }
    }
}

/// Execute the fused TurboQuant kernel on a concrete CubeCL runtime.
///
/// This is a convenience wrapper that launches on-device and then reads both
/// output tensors back to host memory.
///
/// # Parameters
///
/// - `device`: backend device handle.
/// - `input`: input vector.
/// - `bit_width`: total TurboQuant bit-width (`>= 1`).
/// - `seed`: deterministic seed used for rotation/permutation and projection.
/// - `emit_qjl`: enables QJL sign output when `true`.
pub fn launch_turboquant_fused<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
) -> (Vec<f32>, Vec<f32>) {
    let device_outputs =
        launch_turboquant_fused_device::<R>(device, input, bit_width, seed, emit_qjl);
    read_fused_outputs(&device_outputs)
}

/// Device-resident fused outputs. Buffers remain on the backend device until explicitly read.
pub struct DeviceFusedOutputs<R: Runtime> {
    /// Client bound to the active runtime and device.
    pub client: ComputeClient<R>,
    /// Device buffer containing rotated-basis MSE-stage output.
    ///
    /// Use `invert_signed_permutation(..., seed ^ MSE_ROTATION_SALT)` to recover
    /// original-coordinate MSE dequantized values.
    pub mse_handle: Handle,
    /// Device buffer containing MSE centroid indices in rotated coordinate order.
    pub mse_index_handle: Handle,
    /// Device buffer containing QJL signs in original row order.
    pub qjl_handle: Handle,
    /// Number of elements stored in each output buffer.
    pub dim: usize,
    /// MSE-stage quantization bit-width used on device.
    pub bit_width: u8,
    /// Seed used for deterministic transforms in this launch.
    pub seed: u64,
}

/// Reusable device-resident launch assets for fused TurboQuant kernels.
///
/// Build this once and reuse across many launches with matching
/// `(dim, bit_width, seed, launch_override)` to avoid repeated host preparation/uploads.
pub struct DeviceLaunchAssets<R: Runtime> {
    /// Client bound to the active runtime and device.
    pub client: ComputeClient<R>,
    /// Input dimensionality.
    pub dim: usize,
    /// MSE-stage quantization bit-width.
    pub bit_width: u8,
    /// Seed used to derive transforms/projection.
    pub seed: u64,
    /// Quantization levels (`2^bit_width`).
    pub levels: u32,
    /// Launch cube count.
    pub cube_count: CubeCount,
    /// Launch cube dimensions.
    pub cube_dim: CubeDim,
    /// Tensor line size used by launch arguments.
    pub line_size: usize,
    /// Device buffer with centroids.
    pub centroids_handle: Handle,
    /// Device buffer with rotated-basis projection matrix.
    pub projection_handle: Handle,
    /// Device buffer with permutation indices (u32).
    pub permutation_handle: Handle,
    /// Device buffer with sign multipliers (+1/-1).
    pub signs_handle: Handle,
}

/// Device-native encoding kind used for fully on-device packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceEncodingKind {
    /// Fixed-width packed payload.
    Bitpacked,
    /// Delta-xor transformed packed payload.
    DeltaXorEntropy,
    /// Huffman-coded payload built and decoded on-device.
    Huffman,
}

/// Device-resident encoded packet representation.
pub struct DeviceEncodedPacket<R: Runtime> {
    /// Client bound to the active runtime and device.
    pub client: ComputeClient<R>,
    /// Device buffer containing packed payload words (`u32` each).
    pub payload_words_handle: Handle,
    /// Number of valid payload bits.
    pub valid_bits: u32,
    /// Number of payload words.
    pub word_count: usize,
    /// Number of quantized indices.
    pub dim: usize,
    /// Bits per index.
    pub bit_width: u8,
    /// Seed used by quantization pipeline.
    pub seed: u64,
    /// Device-native encoding kind.
    pub encoding: DeviceEncodingKind,
    /// Optional Huffman tree parent links for decode.
    pub huffman_parent_handle: Option<Handle>,
    /// Optional Huffman tree left-child links for decode.
    pub huffman_left_handle: Option<Handle>,
    /// Optional Huffman tree right-child links for decode.
    pub huffman_right_handle: Option<Handle>,
    /// Optional Huffman root node index buffer (`len=1`).
    pub huffman_root_handle: Option<Handle>,
    /// Optional Huffman payload bit count buffer (`len=1`).
    pub huffman_written_bits_handle: Option<Handle>,
    /// Optional policy-managed Huffman codebook generation tag for fail-closed decode.
    pub huffman_codebook_generation: Option<u64>,
    /// Optional policy identity tag for fail-closed decode.
    pub huffman_policy_id: Option<u64>,
    /// Optional policy-managed codebook fingerprint for fail-closed decode.
    pub huffman_codebook_fingerprint: Option<[u8; 32]>,
    /// Optional payload integrity checksum (CRC32C over payload + written bits).
    pub payload_crc32c: Option<u32>,
}

/// Reusable device-resident Huffman codebook.
pub struct DeviceHuffmanCodebook<R: Runtime> {
    /// Client bound to the active runtime and device.
    pub client: ComputeClient<R>,
    /// Parent links for each Huffman node.
    pub parent_handle: Handle,
    /// Left-child links for each Huffman node.
    pub left_handle: Handle,
    /// Right-child links for each Huffman node.
    pub right_handle: Handle,
    /// Root node index buffer (`len=1`).
    pub root_handle: Handle,
    /// Number of nodes in the codebook arrays.
    pub node_cap: usize,
    /// Bit width this codebook is built for.
    pub bit_width: u8,
}

/// Stateful Huffman codebook reuse policy.
///
/// Rebuilds the codebook every `rebuild_every_tokens` encodes, and reuses it in-between.
pub struct HuffmanCodebookReusePolicy<R: Runtime> {
    rebuild_every_tokens: usize,
    tokens_since_rebuild: usize,
    codebook: Option<DeviceHuffmanCodebook<R>>,
    codebook_generation: u64,
    policy_id: u64,
    codebook_fingerprint: Option<[u8; 32]>,
}

impl<R: Runtime> HuffmanCodebookReusePolicy<R> {
    /// Create a new policy with periodic rebuild cadence.
    pub fn new(rebuild_every_tokens: usize) -> Self {
        assert!(rebuild_every_tokens > 0, "rebuild_every_tokens must be > 0");
        Self {
            rebuild_every_tokens,
            tokens_since_rebuild: 0,
            codebook: None,
            codebook_generation: 0,
            policy_id: HUFFMAN_POLICY_ID_GEN.fetch_add(1, Ordering::Relaxed).max(1),
            codebook_fingerprint: None,
        }
    }

    /// Create a policy initialized with an existing codebook.
    pub fn from_codebook(rebuild_every_tokens: usize, codebook: DeviceHuffmanCodebook<R>) -> Self {
        assert!(rebuild_every_tokens > 0, "rebuild_every_tokens must be > 0");
        Self {
            rebuild_every_tokens,
            tokens_since_rebuild: 0,
            codebook: Some(codebook),
            codebook_generation: 1,
            policy_id: HUFFMAN_POLICY_ID_GEN.fetch_add(1, Ordering::Relaxed).max(1),
            codebook_fingerprint: None,
        }
    }

    /// Number of encodes since last rebuild.
    pub fn tokens_since_rebuild(&self) -> usize {
        self.tokens_since_rebuild
    }

    /// Inspect current codebook, if available.
    pub fn codebook(&self) -> Option<&DeviceHuffmanCodebook<R>> {
        self.codebook.as_ref()
    }

    /// Force the next encode to rebuild.
    pub fn invalidate(&mut self) {
        self.codebook = None;
        self.tokens_since_rebuild = 0;
        self.codebook_fingerprint = None;
    }

    /// Encode using the reuse policy, rebuilding automatically when needed.
    pub fn encode(&mut self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        let should_rebuild = match self.codebook.as_ref() {
            None => true,
            Some(codebook) => {
                codebook.bit_width != outputs.bit_width
                    || self.tokens_since_rebuild >= self.rebuild_every_tokens
            }
        };
        if should_rebuild {
            self.codebook = Some(build_device_huffman_codebook(outputs));
            self.tokens_since_rebuild = 0;
            self.codebook_generation = self.codebook_generation.wrapping_add(1).max(1);
            let codebook = self
                .codebook
                .as_ref()
                .expect("huffman policy must have a codebook after rebuild");
            self.codebook_fingerprint = Some(huffman_codebook_fingerprint(
                codebook,
                self.codebook_generation,
                self.policy_id,
            ));
        }
        let mut packet = encode_device_huffman_with_codebook(
            outputs,
            self.codebook
                .as_ref()
                .expect("huffman policy must have a codebook after rebuild"),
        );
        packet.huffman_codebook_generation = Some(self.codebook_generation);
        packet.huffman_policy_id = Some(self.policy_id);
        packet.huffman_codebook_fingerprint = self.codebook_fingerprint;
        packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
        self.tokens_since_rebuild += 1;
        packet
    }

    /// Decode using the currently cached codebook when available.
    pub fn decode(&mut self, packet: &DeviceEncodedPacket<R>) -> Handle {
        if matches!(packet.encoding, DeviceEncodingKind::Huffman) {
            if packet.huffman_policy_id != Some(self.policy_id) {
                self.invalidate();
                panic!("huffman decode rejected: policy identity mismatch (stale or wrong policy)");
            }
            if packet.huffman_codebook_generation != Some(self.codebook_generation) {
                self.invalidate();
                panic!(
                    "huffman decode rejected: codebook generation mismatch (stale or wrong policy)"
                );
            }
            let expected_fingerprint = self.codebook_fingerprint.unwrap_or_else(|| {
                let codebook = self
                    .codebook
                    .as_ref()
                    .expect("huffman decode requires available codebook");
                huffman_codebook_fingerprint(codebook, self.codebook_generation, self.policy_id)
            });
            if packet.huffman_codebook_fingerprint != Some(expected_fingerprint) {
                self.invalidate();
                panic!("huffman decode rejected: codebook fingerprint mismatch");
            }
            let expected_crc = packet
                .payload_crc32c
                .expect("huffman decode rejected: missing payload checksum");
            let actual_crc = packet_payload_crc32c(packet);
            if expected_crc != actual_crc {
                self.invalidate();
                panic!("huffman decode rejected: payload checksum mismatch");
            }
        }
        decode_device_indices_with_codebook(packet, self.codebook.as_ref())
    }
}

fn bits_to_bytes(bits: usize) -> usize {
    bits.div_ceil(8)
}

fn validate_packet_invariants<R: Runtime>(packet: &DeviceEncodedPacket<R>) {
    assert!(packet.dim > 0, "packet decode rejected: dim must be > 0");
    assert!(
        packet.bit_width >= 1,
        "packet decode rejected: bit_width must be >= 1"
    );
    assert!(
        packet.word_count > 0,
        "packet decode rejected: word_count must be > 0"
    );
    let max_bits = packet.word_count.saturating_mul(32);
    assert!(
        (packet.valid_bits as usize) <= max_bits,
        "packet decode rejected: valid_bits exceeds payload capacity"
    );
    assert_eq!(
        packet.word_count,
        (packet.valid_bits as usize).div_ceil(32),
        "packet decode rejected: word_count does not match valid_bits"
    );
    if !matches!(packet.encoding, DeviceEncodingKind::Huffman) {
        let expected_bits = (packet.dim as u64).saturating_mul(packet.bit_width as u64);
        assert_eq!(
            packet.valid_bits as u64, expected_bits,
            "packet decode rejected: non-huffman valid_bits mismatch"
        );
    }
}

fn packet_payload_crc32c<R: Runtime>(packet: &DeviceEncodedPacket<R>) -> u32 {
    let payload = packet.client.read_one(packet.payload_words_handle.clone());
    let payload_words = u32::from_bytes(&payload);
    let mut data = u32::as_bytes(payload_words).to_vec();
    if let Some(handle) = packet.huffman_written_bits_handle.as_ref() {
        let written = packet.client.read_one(handle.clone());
        let written_words = u32::from_bytes(&written);
        data.extend_from_slice(u32::as_bytes(written_words));
    } else {
        data.extend_from_slice(&packet.valid_bits.to_le_bytes());
    }
    crc32c::crc32c(data.as_slice())
}

fn huffman_codebook_fingerprint<R: Runtime>(
    codebook: &DeviceHuffmanCodebook<R>,
    generation: u64,
    policy_id: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&codebook.client.read_one(codebook.parent_handle.clone()));
    hasher.update(&codebook.client.read_one(codebook.left_handle.clone()));
    hasher.update(&codebook.client.read_one(codebook.right_handle.clone()));
    hasher.update(&codebook.client.read_one(codebook.root_handle.clone()));
    hasher.update(&[codebook.bit_width]);
    hasher.update(&codebook.node_cap.to_le_bytes());
    hasher.update(&generation.to_le_bytes());
    hasher.update(&policy_id.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn huffman_packet_wire_bytes<R: Runtime>(packet: &DeviceEncodedPacket<R>) -> usize {
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

fn huffman_codebook_bytes(bit_width: u8) -> usize {
    let node_cap = (1usize << bit_width) * 2usize;
    (node_cap * 3usize + 1usize) * core::mem::size_of::<u32>()
}

/// Fully automatic Huffman codebook policy.
///
/// The policy estimates an automatic rebuild cadence from observed compression
/// delta against bitpacked output, then reuses and rebuilds codebooks as needed.
pub struct AutoHuffmanCodebookPolicy<R: Runtime> {
    inner: Option<HuffmanCodebookReusePolicy<R>>,
    last_dim: Option<usize>,
    last_bit_width: Option<u8>,
}

impl<R: Runtime> Default for AutoHuffmanCodebookPolicy<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Runtime> AutoHuffmanCodebookPolicy<R> {
    /// Create a new automatic policy.
    pub fn new() -> Self {
        Self {
            inner: None,
            last_dim: None,
            last_bit_width: None,
        }
    }

    /// Current automatically selected rebuild cadence.
    pub fn rebuild_every_tokens(&self) -> Option<usize> {
        self.inner.as_ref().map(|p| p.rebuild_every_tokens)
    }

    /// Number of encodes since last rebuild, if initialized.
    pub fn tokens_since_rebuild(&self) -> Option<usize> {
        self.inner.as_ref().map(|p| p.tokens_since_rebuild())
    }

    /// Force the next encode to rebuild and clear shape tracking.
    pub fn invalidate(&mut self) {
        if let Some(inner) = self.inner.as_mut() {
            inner.invalidate();
        }
        self.last_dim = None;
        self.last_bit_width = None;
    }

    /// Encode with automatic codebook reuse and rebuild strategy.
    pub fn encode(&mut self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        let shape_changed =
            self.last_dim != Some(outputs.dim) || self.last_bit_width != Some(outputs.bit_width);
        if self.inner.is_none() || shape_changed {
            let codebook = build_device_huffman_codebook(outputs);
            let mut packet = encode_device_huffman_with_codebook(outputs, &codebook);
            let bitpacked = encode_device_bitpacked(outputs);
            let bitpacked_bytes = bits_to_bytes(bitpacked.valid_bits as usize);
            let huffman_bytes = huffman_packet_wire_bytes(&packet);
            let codebook_bytes = huffman_codebook_bytes(outputs.bit_width);

            let delta = bitpacked_bytes.saturating_sub(huffman_bytes);
            let rebuild_every = if delta == 0 {
                1usize
            } else {
                codebook_bytes.div_ceil(delta).clamp(1usize, 256usize)
            };

            let mut inner = HuffmanCodebookReusePolicy::from_codebook(rebuild_every, codebook);
            let next_generation = self
                .inner
                .as_ref()
                .map_or(1_u64, |p| p.codebook_generation.wrapping_add(1).max(1));
            inner.codebook_generation = next_generation;
            inner.codebook_fingerprint = Some(huffman_codebook_fingerprint(
                inner
                    .codebook
                    .as_ref()
                    .expect("auto policy must have codebook"),
                next_generation,
                inner.policy_id,
            ));
            // Initial encode already consumed one token.
            inner.tokens_since_rebuild = 1usize;
            packet.huffman_codebook_generation = Some(next_generation);
            packet.huffman_policy_id = Some(inner.policy_id);
            packet.huffman_codebook_fingerprint = inner.codebook_fingerprint;
            packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
            self.inner = Some(inner);
            self.last_dim = Some(outputs.dim);
            self.last_bit_width = Some(outputs.bit_width);
            return packet;
        }

        self.inner
            .as_mut()
            .expect("auto policy must be initialized")
            .encode(outputs)
    }

    /// Decode using automatically managed shared codebook when available.
    pub fn decode(&mut self, packet: &DeviceEncodedPacket<R>) -> Handle {
        if let Some(inner) = self.inner.as_mut() {
            inner.decode(packet)
        } else {
            decode_device_indices(packet)
        }
    }
}

/// Optional launch overrides for fused TurboQuant kernels.
#[derive(Debug, Clone, Copy)]
pub struct TurboQuantLaunchOverride {
    /// Override cube dimension used for launch.
    pub cube_dim: u32,
}

/// Launch fused TurboQuant and keep outputs device-resident.
///
/// # Parameters
///
/// - `device`: backend device handle.
/// - `input`: input vector.
/// - `bit_width`: total TurboQuant bit-width (`>= 1`).
/// - `seed`: deterministic seed used for rotation/permutation and projection.
/// - `emit_qjl`: enables QJL sign output when `true`.
///
/// # Returns
///
/// Device handles for both fused outputs.
///
/// # Panics
///
/// Panics if input/centroid buffers are empty or shape invariants are violated.
pub fn launch_turboquant_fused_device<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
) -> DeviceFusedOutputs<R> {
    launch_turboquant_fused_device_with_launch::<R>(device, input, bit_width, seed, emit_qjl, None)
}

/// Prepare reusable device launch assets for fused TurboQuant.
pub fn prepare_turboquant_launch_assets<R: Runtime>(
    device: &R::Device,
    dim: usize,
    bit_width: u8,
    seed: u64,
    launch_override: Option<TurboQuantLaunchOverride>,
) -> DeviceLaunchAssets<R> {
    prepare_turboquant_launch_assets_with_options(
        device,
        dim,
        bit_width,
        seed,
        launch_override,
        TurboQuantKernelOptions {
            emit_qjl: true,
            emit_entropy: false,
            single_kernel_packet: true,
        },
    )
}

/// Prepare reusable device launch assets for fused TurboQuant with explicit CubeK options.
pub fn prepare_turboquant_launch_assets_with_options<R: Runtime>(
    device: &R::Device,
    dim: usize,
    bit_width: u8,
    seed: u64,
    launch_override: Option<TurboQuantLaunchOverride>,
    kernel_options: TurboQuantKernelOptions,
) -> DeviceLaunchAssets<R> {
    assert!(dim > 0, "dim must be > 0");
    assert!(bit_width >= 1, "bit_width must be >= 1");

    let client = R::client(device);
    let mse_bit_width = bit_width.saturating_sub(1).max(1);
    let levels = 1_u32 << mse_bit_width;
    let centroids = centroids_for(mse_bit_width, dim);
    let (perm, signs) = signed_permutation(dim, seed ^ MSE_ROTATION_SALT);
    let perm_u32 = perm.iter().map(|v| *v as u32).collect::<Vec<u32>>();
    let projection_seed = seed ^ PROD_PROJECTION_SALT;
    let projection = projection_matrix(dim, projection_seed);
    let projection_rot = transform_projection_for_rotated_basis(&projection, &perm, &signs);
    assert!(levels.is_power_of_two(), "levels must be a power of two");
    let problem = TurboQuantProblem {
        dim,
        bit_width: mse_bit_width,
    };
    let (blueprint, launch) = TurboQuantRoutine::prepare(problem, kernel_options);
    assert_eq!(
        blueprint.levels, levels,
        "routine levels must match launch levels"
    );
    if kernel_options.emit_qjl {
        assert!(
            matches!(blueprint.variant, TurboQuantVariant::ProdFused),
            "emit_qjl=true expects ProdFused variant"
        );
    }
    let line_size = launch.line_size as usize;
    let (cube_count, cube_dim) = if let Some(override_cfg) = launch_override {
        assert!(override_cfg.cube_dim > 0, "cube_dim override must be > 0");
        let cube_dim = CubeDim::new_1d(override_cfg.cube_dim);
        let cube_count = CubeCount::new_1d((dim as u32).div_ceil(override_cfg.cube_dim));
        (cube_count, cube_dim)
    } else {
        (launch.cube_count, launch.cube_dim)
    };

    let centroids_handle = client.create_from_slice(f32::as_bytes(&centroids));
    let projection_handle = client.create_from_slice(f32::as_bytes(&projection_rot));
    let permutation_handle = client.create_from_slice(u32::as_bytes(&perm_u32));
    let signs_handle = client.create_from_slice(f32::as_bytes(&signs));

    DeviceLaunchAssets {
        client,
        dim,
        bit_width: mse_bit_width,
        seed,
        levels,
        cube_count,
        cube_dim,
        line_size,
        centroids_handle,
        projection_handle,
        permutation_handle,
        signs_handle,
    }
}

fn launch_turboquant_fused_with_input_handle<R: Runtime>(
    assets: &DeviceLaunchAssets<R>,
    input_handle: &Handle,
    emit_qjl: bool,
) -> DeviceFusedOutputs<R> {
    let shape = vec![assets.dim];
    let projection_shape = vec![assets.dim * assets.dim];
    let strides = vec![1usize];
    let centroid_shape = vec![1usize << assets.bit_width];
    let rotated_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<f32>());
    let mse_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<f32>());
    let mse_index_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<u32>());
    let qjl_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<f32>());

    // SAFETY: all tensor args use same client-owned contiguous 1D metadata.
    let input_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(input_handle, &strides, &shape, assets.line_size)
    };
    let perm_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(
            &assets.permutation_handle,
            &strides,
            &shape,
            assets.line_size,
        )
    };
    let sign_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&assets.signs_handle, &strides, &shape, assets.line_size)
    };
    let rotated_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&rotated_handle, &strides, &shape, assets.line_size)
    };
    unsafe {
        apply_signed_permutation_kernel::launch_unchecked::<R>(
            &assets.client,
            assets.cube_count.clone(),
            assets.cube_dim,
            input_arg,
            perm_arg,
            sign_arg,
            rotated_arg,
        )
        .expect("signed permutation kernel launch failed");
    };

    // SAFETY: all TensorArg views below are created from buffers allocated by the same client,
    // with 1D contiguous strides and shapes that exactly match the allocated element counts.
    let input_rot_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&rotated_handle, &strides, &shape, assets.line_size)
    };
    let centroids_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(
            &assets.centroids_handle,
            &strides,
            &centroid_shape,
            assets.line_size,
        )
    };
    let projection_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(
            &assets.projection_handle,
            &strides,
            &projection_shape,
            assets.line_size,
        )
    };
    let mse_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&mse_handle, &strides, &shape, assets.line_size)
    };
    let mse_index_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&mse_index_handle, &strides, &shape, assets.line_size)
    };
    let qjl_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&qjl_handle, &strides, &shape, assets.line_size)
    };

    // SAFETY: launch dimensions and argument shapes are validated above and agree with kernel
    // expectations (`dim` elements for input/output and `dim*dim` for projection).
    unsafe {
        turboquant_fused_kernel::launch_unchecked::<f32, R>(
            &assets.client,
            assets.cube_count.clone(),
            assets.cube_dim,
            input_rot_arg,
            centroids_arg,
            projection_arg,
            mse_arg,
            mse_index_arg,
            qjl_arg,
            assets.levels,
            emit_qjl,
        )
        .expect("turboquant fused kernel launch failed");
    };

    DeviceFusedOutputs {
        client: assets.client.clone(),
        mse_handle,
        mse_index_handle,
        qjl_handle,
        dim: assets.dim,
        bit_width: assets.bit_width,
        seed: assets.seed,
    }
}

/// Launch fused TurboQuant using prebuilt reusable launch assets.
///
/// This avoids rebuilding and re-uploading centroids/projection each launch.
pub fn launch_turboquant_fused_device_with_assets<R: Runtime>(
    assets: &DeviceLaunchAssets<R>,
    input: &[f32],
    emit_qjl: bool,
) -> DeviceFusedOutputs<R> {
    assert_eq!(
        input.len(),
        assets.dim,
        "input length must match prepared dim"
    );
    let input_handle = assets.client.create_from_slice(f32::as_bytes(input));
    launch_turboquant_fused_with_input_handle(assets, &input_handle, emit_qjl)
}

/// Launch fused TurboQuant directly from an existing device input handle.
///
/// This path avoids host-side input rematerialization between device stages.
pub fn launch_turboquant_fused_device_from_handle<R: Runtime>(
    assets: &DeviceLaunchAssets<R>,
    input_handle: &Handle,
    emit_qjl: bool,
) -> DeviceFusedOutputs<R> {
    launch_turboquant_fused_with_input_handle(assets, input_handle, emit_qjl)
}

/// Launch fused TurboQuant with optional launch-geometry override.
pub fn launch_turboquant_fused_device_with_launch<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
    launch_override: Option<TurboQuantLaunchOverride>,
) -> DeviceFusedOutputs<R> {
    assert!(!input.is_empty(), "input must be non-empty");
    assert!(bit_width >= 1, "bit_width must be >= 1");
    let assets = prepare_turboquant_launch_assets::<R>(
        device,
        input.len(),
        bit_width,
        seed,
        launch_override,
    );
    launch_turboquant_fused_device_with_assets(&assets, input, emit_qjl)
}

/// Explicit host readback helper (primarily for tests and diagnostics).
pub fn read_fused_outputs<R: Runtime>(outputs: &DeviceFusedOutputs<R>) -> (Vec<f32>, Vec<f32>) {
    let mse = read_f32_buffer(&outputs.client, outputs.mse_handle.clone());
    let qjl = read_f32_buffer(&outputs.client, outputs.qjl_handle.clone());
    (mse, qjl)
}

/// Read device MSE indices back to host (debug/export only).
pub fn read_fused_indices<R: Runtime>(outputs: &DeviceFusedOutputs<R>) -> Vec<u32> {
    read_u32_buffer(&outputs.client, outputs.mse_index_handle.clone())
}

/// Encode device-resident MSE indices using on-device bitpacking.
pub fn encode_device_bitpacked<R: Runtime>(
    outputs: &DeviceFusedOutputs<R>,
) -> DeviceEncodedPacket<R> {
    let valid_bits = (outputs.dim as u32) * (outputs.bit_width as u32);
    let word_count = (valid_bits as usize).div_ceil(32);
    let payload_words_handle = outputs
        .client
        .empty(word_count * core::mem::size_of::<u32>());
    let shape_payload = vec![word_count];
    let shape_indices = vec![outputs.dim];
    let strides = vec![1usize];
    let line_size = 1usize;

    // SAFETY: all tensor args use same client-owned buffers with contiguous 1D shape metadata.
    let indices_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(
            &outputs.mse_index_handle,
            &strides,
            &shape_indices,
            line_size,
        )
    };
    let payload_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&payload_words_handle, &strides, &shape_payload, line_size)
    };
    unsafe {
        bitpack_indices_kernel::launch_unchecked::<R>(
            &outputs.client,
            CubeCount::new_1d(word_count as u32),
            CubeDim::new_1d(1),
            indices_arg,
            payload_arg,
            ScalarArg::new(outputs.bit_width as usize),
            ScalarArg::new(valid_bits as usize),
        )
        .expect("device bitpack kernel launch failed");
    }

    let mut packet = DeviceEncodedPacket {
        client: outputs.client.clone(),
        payload_words_handle,
        valid_bits,
        word_count,
        dim: outputs.dim,
        bit_width: outputs.bit_width,
        seed: outputs.seed,
        encoding: DeviceEncodingKind::Bitpacked,
        huffman_parent_handle: None,
        huffman_left_handle: None,
        huffman_right_handle: None,
        huffman_root_handle: None,
        huffman_written_bits_handle: None,
        huffman_codebook_generation: None,
        huffman_policy_id: None,
        huffman_codebook_fingerprint: None,
        payload_crc32c: None,
    };
    packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
    packet
}

/// Apply on-device reversible entropy transform to a device bitpacked payload.
pub fn encode_device_entropy<R: Runtime>(
    packet: &DeviceEncodedPacket<R>,
) -> DeviceEncodedPacket<R> {
    let encoded_handle = packet
        .client
        .empty(packet.word_count * core::mem::size_of::<u32>());
    let shape = vec![packet.word_count];
    let strides = vec![1usize];
    let line_size = 1usize;

    // SAFETY: single-client contiguous tensor args with matching shape metadata.
    let input_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&packet.payload_words_handle, &strides, &shape, line_size)
    };
    let output_arg =
        unsafe { TensorArg::from_raw_parts::<u32>(&encoded_handle, &strides, &shape, line_size) };
    unsafe {
        entropy_delta_xor_encode_kernel::launch_unchecked::<R>(
            &packet.client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            input_arg,
            output_arg,
        )
        .expect("device entropy encode kernel launch failed");
    }

    let mut packet = DeviceEncodedPacket {
        client: packet.client.clone(),
        payload_words_handle: encoded_handle,
        valid_bits: packet.valid_bits,
        word_count: packet.word_count,
        dim: packet.dim,
        bit_width: packet.bit_width,
        seed: packet.seed,
        encoding: DeviceEncodingKind::DeltaXorEntropy,
        huffman_parent_handle: None,
        huffman_left_handle: None,
        huffman_right_handle: None,
        huffman_root_handle: None,
        huffman_written_bits_handle: None,
        huffman_codebook_generation: None,
        huffman_policy_id: None,
        huffman_codebook_fingerprint: None,
        payload_crc32c: None,
    };
    packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
    packet
}

/// Build Huffman tree and encode indices on-device.
pub fn encode_device_huffman<R: Runtime>(
    outputs: &DeviceFusedOutputs<R>,
) -> DeviceEncodedPacket<R> {
    assert_huffman_experimental();
    let codebook = build_device_huffman_codebook(outputs);
    let mut packet = encode_device_huffman_with_codebook(outputs, &codebook);
    // Backward-compatible path: embed codebook handles in packet.
    packet.huffman_parent_handle = Some(codebook.parent_handle);
    packet.huffman_left_handle = Some(codebook.left_handle);
    packet.huffman_right_handle = Some(codebook.right_handle);
    packet.huffman_root_handle = Some(codebook.root_handle);
    packet
}

/// Build a reusable Huffman codebook from device indices.
pub fn build_device_huffman_codebook<R: Runtime>(
    outputs: &DeviceFusedOutputs<R>,
) -> DeviceHuffmanCodebook<R> {
    assert_huffman_experimental();
    let levels = 1usize << outputs.bit_width;
    let node_cap = levels * 2usize;
    let frequencies_handle = outputs.client.empty(levels * core::mem::size_of::<u32>());
    let node_weights_handle = outputs.client.empty(node_cap * core::mem::size_of::<u32>());
    let node_parent_handle = outputs.client.empty(node_cap * core::mem::size_of::<u32>());
    let node_left_handle = outputs.client.empty(node_cap * core::mem::size_of::<u32>());
    let node_right_handle = outputs.client.empty(node_cap * core::mem::size_of::<u32>());
    let root_handle = outputs
        .client
        .create_from_slice(u32::as_bytes(&[((levels * 2usize) - 2usize) as u32]));
    let scratch_min1_handle = outputs.client.empty(core::mem::size_of::<u32>());
    let scratch_min2_handle = outputs.client.empty(core::mem::size_of::<u32>());

    let shape_levels = vec![levels];
    let shape_nodes = vec![node_cap];
    let shape_dim = vec![outputs.dim];
    let shape_one = vec![1usize];
    let strides = vec![1usize];
    let line_size = 1usize;

    let indices_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&outputs.mse_index_handle, &strides, &shape_dim, line_size)
    };
    let freq_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&frequencies_handle, &strides, &shape_levels, line_size)
    };
    unsafe {
        histogram_indices_kernel::launch_unchecked::<R>(
            &outputs.client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            indices_arg,
            freq_arg,
            ScalarArg::new(levels),
        )
        .expect("device histogram kernel launch failed");
    }

    let freq_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&frequencies_handle, &strides, &shape_levels, line_size)
    };
    let weights_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&node_weights_handle, &strides, &shape_nodes, line_size)
    };
    let parent_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&node_parent_handle, &strides, &shape_nodes, line_size)
    };
    let left_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&node_left_handle, &strides, &shape_nodes, line_size)
    };
    let right_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&node_right_handle, &strides, &shape_nodes, line_size)
    };
    unsafe {
        huffman_init_tree_kernel::launch_unchecked::<R>(
            &outputs.client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            freq_arg,
            weights_arg,
            parent_arg,
            left_arg,
            right_arg,
            ScalarArg::new(levels),
        )
        .expect("device huffman tree init failed");
    }

    for step in 0..levels.saturating_sub(1) {
        let next_node = levels + step;
        let weights_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(
                &node_weights_handle,
                &strides,
                &shape_nodes,
                line_size,
            )
        };
        let parent_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&node_parent_handle, &strides, &shape_nodes, line_size)
        };
        let left_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&node_left_handle, &strides, &shape_nodes, line_size)
        };
        let right_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&node_right_handle, &strides, &shape_nodes, line_size)
        };
        let scratch_min1_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&scratch_min1_handle, &strides, &shape_one, line_size)
        };
        let scratch_min2_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&scratch_min2_handle, &strides, &shape_one, line_size)
        };
        unsafe {
            huffman_merge_step_kernel::launch_unchecked::<R>(
                &outputs.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                weights_arg,
                parent_arg,
                left_arg,
                right_arg,
                scratch_min1_arg,
                scratch_min2_arg,
                ScalarArg::new(next_node),
            )
            .expect("device huffman merge step failed");
        }
    }
    DeviceHuffmanCodebook {
        client: outputs.client.clone(),
        parent_handle: node_parent_handle,
        left_handle: node_left_handle,
        right_handle: node_right_handle,
        root_handle,
        node_cap,
        bit_width: outputs.bit_width,
    }
}

/// Encode indices with a reusable Huffman codebook.
pub fn encode_device_huffman_with_codebook<R: Runtime>(
    outputs: &DeviceFusedOutputs<R>,
    codebook: &DeviceHuffmanCodebook<R>,
) -> DeviceEncodedPacket<R> {
    assert_huffman_experimental();
    assert_eq!(
        codebook.bit_width, outputs.bit_width,
        "huffman codebook bit-width mismatch"
    );
    let levels = 1usize << outputs.bit_width;
    let worst_valid_bits = outputs.dim.saturating_mul(levels);
    let word_count = worst_valid_bits.div_ceil(32);
    let payload_words_handle = outputs
        .client
        .empty(word_count * core::mem::size_of::<u32>());
    let written_bits_handle = outputs.client.empty(core::mem::size_of::<u32>());
    let path_bits_handle = outputs
        .client
        .empty(codebook.node_cap * core::mem::size_of::<u32>());

    let shape_nodes = vec![codebook.node_cap];
    let shape_dim = vec![outputs.dim];
    let shape_one = vec![1usize];
    let shape_payload = vec![word_count];
    let strides = vec![1usize];
    let line_size = 1usize;

    let payload_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&payload_words_handle, &strides, &shape_payload, line_size)
    };
    let indices_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&outputs.mse_index_handle, &strides, &shape_dim, line_size)
    };
    let parent_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&codebook.parent_handle, &strides, &shape_nodes, line_size)
    };
    let left_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&codebook.left_handle, &strides, &shape_nodes, line_size)
    };
    let right_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&codebook.right_handle, &strides, &shape_nodes, line_size)
    };
    let root_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&codebook.root_handle, &strides, &shape_one, line_size)
    };
    let path_bits_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&path_bits_handle, &strides, &shape_nodes, line_size)
    };
    let written_bits_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&written_bits_handle, &strides, &shape_one, line_size)
    };
    unsafe {
        huffman_encode_indices_kernel::launch_unchecked::<R>(
            &outputs.client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            indices_arg,
            parent_arg,
            left_arg,
            right_arg,
            root_arg,
            path_bits_arg,
            payload_arg,
            written_bits_arg,
            codebook.node_cap as u32,
        )
        .expect("device huffman encode failed");
    }

    let mut packet = DeviceEncodedPacket {
        client: outputs.client.clone(),
        payload_words_handle,
        valid_bits: worst_valid_bits as u32,
        word_count,
        dim: outputs.dim,
        bit_width: outputs.bit_width,
        seed: outputs.seed,
        encoding: DeviceEncodingKind::Huffman,
        huffman_parent_handle: None,
        huffman_left_handle: None,
        huffman_right_handle: None,
        huffman_root_handle: None,
        huffman_written_bits_handle: Some(written_bits_handle),
        huffman_codebook_generation: None,
        huffman_policy_id: None,
        huffman_codebook_fingerprint: None,
        payload_crc32c: None,
    };
    packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
    packet
}

/// Decode on-device payload back to device-resident MSE indices.
pub fn decode_device_indices<R: Runtime>(packet: &DeviceEncodedPacket<R>) -> Handle {
    decode_device_indices_with_codebook(packet, None)
}

/// Decode on-device payload back to indices with an optional shared Huffman codebook.
pub fn decode_device_indices_with_codebook<R: Runtime>(
    packet: &DeviceEncodedPacket<R>,
    codebook: Option<&DeviceHuffmanCodebook<R>>,
) -> Handle {
    validate_packet_invariants(packet);
    let expected_crc = packet
        .payload_crc32c
        .expect("packet decode rejected: missing payload checksum");
    let actual_crc = packet_payload_crc32c(packet);
    assert_eq!(
        actual_crc, expected_crc,
        "packet decode rejected: payload checksum mismatch"
    );

    if matches!(packet.encoding, DeviceEncodingKind::Huffman) {
        assert_huffman_experimental();
        let out_indices = packet
            .client
            .empty(packet.dim * core::mem::size_of::<u32>());
        let shape_words = vec![packet.word_count];
        let shape_nodes = vec![(1usize << packet.bit_width) * 2usize];
        let shape_indices = vec![packet.dim];
        let shape_one = vec![1usize];
        let strides = vec![1usize];
        let line_size = 1usize;

        let left_handle = if let Some(codebook) = codebook {
            &codebook.left_handle
        } else {
            packet
                .huffman_left_handle
                .as_ref()
                .expect("huffman decode requires codebook or embedded left-handle")
        };
        let right_handle = if let Some(codebook) = codebook {
            &codebook.right_handle
        } else {
            packet
                .huffman_right_handle
                .as_ref()
                .expect("huffman decode requires codebook or embedded right-handle")
        };
        let root_handle = if let Some(codebook) = codebook {
            &codebook.root_handle
        } else {
            packet
                .huffman_root_handle
                .as_ref()
                .expect("huffman decode requires codebook or embedded root-handle")
        };

        let payload_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(
                &packet.payload_words_handle,
                &strides,
                &shape_words,
                line_size,
            )
        };
        let left_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(left_handle, &strides, &shape_nodes, line_size)
        };
        let right_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(right_handle, &strides, &shape_nodes, line_size)
        };
        let root_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(root_handle, &strides, &shape_one, line_size)
        };
        let out_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&out_indices, &strides, &shape_indices, line_size)
        };

        unsafe {
            huffman_decode_indices_kernel::launch_unchecked::<R>(
                &packet.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                payload_arg,
                left_arg,
                right_arg,
                root_arg,
                out_arg,
            )
            .expect("device huffman decode kernel launch failed");
        }
        return out_indices;
    }

    let shape_words = vec![packet.word_count];
    let shape_indices = vec![packet.dim];
    let strides = vec![1usize];
    let line_size = 1usize;

    let unpack_source = if matches!(packet.encoding, DeviceEncodingKind::DeltaXorEntropy) {
        let decoded_words = packet
            .client
            .empty(packet.word_count * core::mem::size_of::<u32>());
        // SAFETY: contiguous 1D metadata with same client buffers.
        let in_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(
                &packet.payload_words_handle,
                &strides,
                &shape_words,
                line_size,
            )
        };
        let out_arg = unsafe {
            TensorArg::from_raw_parts::<u32>(&decoded_words, &strides, &shape_words, line_size)
        };
        unsafe {
            entropy_delta_xor_decode_kernel::launch_unchecked::<R>(
                &packet.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                in_arg,
                out_arg,
            )
            .expect("device entropy decode kernel launch failed");
        }
        decoded_words
    } else {
        packet.payload_words_handle.clone()
    };

    let out_indices = packet
        .client
        .empty(packet.dim * core::mem::size_of::<u32>());
    // SAFETY: contiguous 1D metadata with same client buffers.
    let payload_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&unpack_source, &strides, &shape_words, line_size)
    };
    let out_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&out_indices, &strides, &shape_indices, line_size)
    };
    unsafe {
        unpack_indices_kernel::launch_unchecked::<R>(
            &packet.client,
            CubeCount::new_1d(packet.dim as u32),
            CubeDim::new_1d(1),
            payload_arg,
            out_arg,
            ScalarArg::new(packet.bit_width as usize),
            ScalarArg::new(packet.valid_bits as usize),
        )
        .expect("device unpack kernel launch failed");
    }
    out_indices
}

/// Launch full device pipeline and return fully device-resident encoded packet.
pub fn launch_turboquant_pipeline_device_from_handle<R: Runtime>(
    assets: &DeviceLaunchAssets<R>,
    input_handle: &Handle,
    emit_qjl: bool,
    entropy: bool,
) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
    launch_turboquant_pipeline_device_from_handle_with_options(
        assets,
        input_handle,
        TurboQuantKernelOptions {
            emit_qjl,
            emit_entropy: entropy,
            single_kernel_packet: true,
        },
    )
}

/// Launch full device pipeline with explicit CubeK options.
pub fn launch_turboquant_pipeline_device_from_handle_with_options<R: Runtime>(
    assets: &DeviceLaunchAssets<R>,
    input_handle: &Handle,
    kernel_options: TurboQuantKernelOptions,
) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
    if kernel_options.emit_entropy {
        let state = launch_turboquant_fused_with_input_handle(
            assets,
            input_handle,
            kernel_options.emit_qjl,
        );
        let packed = encode_device_bitpacked(&state);
        let encoded = encode_device_entropy(&packed);
        return (state, encoded);
    }

    let valid_bits = (assets.dim as u32) * (assets.bit_width as u32);
    let word_count = (valid_bits as usize).div_ceil(32);

    let shape = vec![assets.dim];
    let projection_shape = vec![assets.dim * assets.dim];
    let centroid_shape = vec![1usize << assets.bit_width];
    let payload_shape = vec![word_count];
    let strides = vec![1usize];

    let mse_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<f32>());
    let mse_index_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<u32>());
    let qjl_handle = assets
        .client
        .empty(assets.dim * core::mem::size_of::<f32>());
    let payload_words_handle = assets
        .client
        .empty(word_count * core::mem::size_of::<u32>());

    let input_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(input_handle, &strides, &shape, assets.line_size)
    };
    let centroids_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(
            &assets.centroids_handle,
            &strides,
            &centroid_shape,
            assets.line_size,
        )
    };
    let projection_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(
            &assets.projection_handle,
            &strides,
            &projection_shape,
            assets.line_size,
        )
    };
    let perm_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(
            &assets.permutation_handle,
            &strides,
            &shape,
            assets.line_size,
        )
    };
    let sign_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&assets.signs_handle, &strides, &shape, assets.line_size)
    };
    let mse_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&mse_handle, &strides, &shape, assets.line_size)
    };
    let mse_index_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(&mse_index_handle, &strides, &shape, assets.line_size)
    };
    let qjl_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&qjl_handle, &strides, &shape, assets.line_size)
    };
    let payload_arg = unsafe {
        TensorArg::from_raw_parts::<u32>(
            &payload_words_handle,
            &strides,
            &payload_shape,
            assets.line_size,
        )
    };

    if kernel_options.single_kernel_packet {
        // SAFETY: all launch arguments are allocated by the same client with matching contiguous
        // metadata; this dispatch intentionally runs as one cube for single-kernel pipeline fusion.
        unsafe {
            turboquant_pipeline_fused_kernel::launch_unchecked::<R>(
                &assets.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(1),
                input_arg,
                centroids_arg,
                projection_arg,
                perm_arg,
                sign_arg,
                mse_arg,
                mse_index_arg,
                qjl_arg,
                payload_arg,
                ScalarArg::new(valid_bits as usize),
                assets.levels,
                kernel_options.emit_qjl,
                kernel_options.emit_entropy,
            )
            .expect("turboquant fused pipeline kernel launch failed");
        }
    } else {
        let state = launch_turboquant_fused_with_input_handle(
            assets,
            input_handle,
            kernel_options.emit_qjl,
        );
        let bitpacked = encode_device_bitpacked(&state);
        return (state, bitpacked);
    }

    let state = DeviceFusedOutputs {
        client: assets.client.clone(),
        mse_handle,
        mse_index_handle,
        qjl_handle,
        dim: assets.dim,
        bit_width: assets.bit_width,
        seed: assets.seed,
    };
    let mut packet = DeviceEncodedPacket {
        client: assets.client.clone(),
        payload_words_handle,
        valid_bits,
        word_count,
        dim: assets.dim,
        bit_width: assets.bit_width,
        seed: assets.seed,
        encoding: if kernel_options.emit_entropy {
            DeviceEncodingKind::DeltaXorEntropy
        } else {
            DeviceEncodingKind::Bitpacked
        },
        huffman_parent_handle: None,
        huffman_left_handle: None,
        huffman_right_handle: None,
        huffman_root_handle: None,
        huffman_written_bits_handle: None,
        huffman_codebook_generation: None,
        huffman_policy_id: None,
        huffman_codebook_fingerprint: None,
        payload_crc32c: None,
    };
    packet.payload_crc32c = Some(packet_payload_crc32c(&packet));
    (state, packet)
}

/// Launch full device pipeline and return fully device-resident encoded packet.
pub fn launch_turboquant_pipeline_device<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
    launch_override: Option<TurboQuantLaunchOverride>,
    entropy: bool,
) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
    launch_turboquant_pipeline_device_with_options(
        device,
        input,
        bit_width,
        seed,
        launch_override,
        TurboQuantKernelOptions {
            emit_qjl,
            emit_entropy: entropy,
            single_kernel_packet: true,
        },
    )
}

/// Launch full device pipeline with explicit CubeK options.
pub fn launch_turboquant_pipeline_device_with_options<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    bit_width: u8,
    seed: u64,
    launch_override: Option<TurboQuantLaunchOverride>,
    kernel_options: TurboQuantKernelOptions,
) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
    assert!(!input.is_empty(), "input must be non-empty");
    let assets = prepare_turboquant_launch_assets_with_options::<R>(
        device,
        input.len(),
        bit_width,
        seed,
        launch_override,
        kernel_options,
    );
    let input_handle = assets.client.create_from_slice(f32::as_bytes(input));
    launch_turboquant_pipeline_device_from_handle_with_options(
        &assets,
        &input_handle,
        kernel_options,
    )
}

/// Validate fused outputs entirely on the backend device.
/// Returns true when validation passes.
///
/// This routine performs correctness checks on-device and reads back only a
/// single status scalar. It validates strict equivalence with host `quantize_prod`
/// stage semantics for the same `bit_width` and `seed`.
pub fn validate_fused_outputs_on_device<R: Runtime>(
    input: &[f32],
    bit_width: u8,
    seed: u64,
    outputs: &DeviceFusedOutputs<R>,
    tolerance: f32,
) -> bool {
    let dim = outputs.dim;
    assert_eq!(input.len(), dim, "input size must match output size");
    assert!(bit_width >= 1, "bit_width must be >= 1");
    let mse_bit_width = bit_width.saturating_sub(1).max(1);
    let levels = 1_u32 << mse_bit_width;
    let centroids = centroids_for(mse_bit_width, dim);
    let rotated = apply_signed_permutation(input, dim, seed ^ MSE_ROTATION_SALT);
    let (perm, signs) = signed_permutation(dim, seed ^ MSE_ROTATION_SALT);
    let projection_seed = seed ^ PROD_PROJECTION_SALT;
    let projection = projection_matrix(dim, projection_seed);
    let projection_rot = transform_projection_for_rotated_basis(&projection, &perm, &signs);

    let shape = vec![dim];
    let projection_shape = vec![dim * dim];
    let strides = vec![1usize];
    let centroid_shape = vec![centroids.len()];
    let status_shape = vec![1usize];
    let line_size = 1usize;

    let input_handle = outputs.client.create_from_slice(f32::as_bytes(&rotated));
    let centroids_handle = outputs.client.create_from_slice(f32::as_bytes(&centroids));
    let projection_handle = outputs
        .client
        .create_from_slice(f32::as_bytes(&projection_rot));
    let status_handle = outputs.client.create_from_slice(f32::as_bytes(&[0.0_f32]));

    // SAFETY: validation-kernel TensorArg views use buffers allocated by the same client and
    // shape/stride metadata verified above.
    let input_arg =
        unsafe { TensorArg::from_raw_parts::<f32>(&input_handle, &strides, &shape, line_size) };
    let centroids_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&centroids_handle, &strides, &centroid_shape, line_size)
    };
    let projection_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&projection_handle, &strides, &projection_shape, line_size)
    };
    let mse_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&outputs.mse_handle, &strides, &shape, line_size)
    };
    let qjl_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&outputs.qjl_handle, &strides, &shape, line_size)
    };
    let status_arg = unsafe {
        TensorArg::from_raw_parts::<f32>(&status_handle, &strides, &status_shape, line_size)
    };

    // SAFETY: launch geometry and argument layout match `validate_fused_kernel` contract.
    unsafe {
        validate_fused_kernel::launch_unchecked::<R>(
            &outputs.client,
            CubeCount::new_1d(dim as u32),
            CubeDim::new_1d(1),
            input_arg,
            centroids_arg,
            projection_arg,
            mse_arg,
            qjl_arg,
            status_arg,
            ScalarArg::new(tolerance),
            levels,
        )
        .expect("device validation kernel launch failed");
    }

    let status = read_f32_buffer(&outputs.client, status_handle);
    status[0] == 0.0
}

/// Read a device buffer and decode it as `Vec<f32>`.
fn read_f32_buffer<R: Runtime>(client: &ComputeClient<R>, handle: Handle) -> Vec<f32> {
    let bytes = client.read_one(handle);
    f32::from_bytes(&bytes).to_vec()
}

/// Read a device buffer and decode it as `Vec<u32>`.
fn read_u32_buffer<R: Runtime>(client: &ComputeClient<R>, handle: Handle) -> Vec<u32> {
    let bytes = client.read_one(handle);
    u32::from_bytes(&bytes).to_vec()
}

/// Retrieve centroid table from cache or build it on demand.
pub(crate) fn centroids_for(bit_width: u8, dim: usize) -> Vec<f32> {
    let cache = CENTROID_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("centroid cache lock poisoned");
    if let Some(found) = cache.get(&(bit_width, dim)) {
        return found.clone();
    }
    let built = build_centroids(bit_width, dim);
    cache.insert((bit_width, dim), built.clone());
    built
}

/// Build scalar quantization centroids for a given bit-width and dimension.
fn build_centroids(bit_width: u8, dim: usize) -> Vec<f32> {
    if bit_width == 2 {
        // Paper's moderate-d normal approximation values from Algorithm 1 discussion.
        let inv_sqrt_d = 1.0 / (dim as f32).sqrt();
        return vec![-1.51, -0.453, 0.453, 1.51]
            .into_iter()
            .map(|v| v * inv_sqrt_d)
            .collect();
    }

    let levels = 1usize << bit_width;
    let sigma = 1.0 / (dim as f32).sqrt();
    let mut rng = StdRng::seed_from_u64(((dim as u64) << 16) ^ (bit_width as u64));
    let sample_count = 20_000usize;
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        samples.push(sample_standard_normal(&mut rng) * sigma);
    }

    let mut centroids = (0..levels)
        .map(|i| {
            let t = (i as f32 + 0.5) / levels as f32;
            (t * 6.0 - 3.0) * sigma
        })
        .collect::<Vec<_>>();

    for _ in 0..32 {
        let mut sums = vec![0.0_f32; levels];
        let mut counts = vec![0usize; levels];
        for sample in &samples {
            let mut best_idx = 0usize;
            let mut best_dist = f32::INFINITY;
            for (idx, centroid) in centroids.iter().enumerate() {
                let d = sample - centroid;
                let dist = d * d;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            sums[best_idx] += *sample;
            counts[best_idx] += 1;
        }

        for i in 0..levels {
            if counts[i] > 0 {
                centroids[i] = sums[i] / counts[i] as f32;
            }
        }
        centroids.sort_by(|a, b| a.total_cmp(b));
    }

    centroids
}

/// Build deterministic signed permutation used as a lightweight random rotation proxy.
fn signed_permutation(dim: usize, seed: u64) -> (Vec<usize>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm = (0..dim).collect::<Vec<_>>();
    perm.shuffle(&mut rng);
    let mut signs = vec![1.0_f32; dim];
    for sign in &mut signs {
        *sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    (perm, signs)
}

/// Apply deterministic signed permutation to an input vector.
fn apply_signed_permutation(x: &[f32], dim: usize, seed: u64) -> Vec<f32> {
    let (perm, signs) = signed_permutation(dim, seed);
    let mut out = vec![0.0_f32; dim];
    for i in 0..dim {
        out[i] = signs[i] * x[perm[i]];
    }
    out
}

/// Apply inverse signed permutation to recover original coordinate order.
#[cfg(test)]
fn invert_signed_permutation(x: &[f32], dim: usize, seed: u64) -> Vec<f32> {
    let (perm, signs) = signed_permutation(dim, seed);
    let mut out = vec![0.0_f32; dim];
    for i in 0..dim {
        out[perm[i]] = signs[i] * x[i];
    }
    out
}

/// Build inverse permutation indices.
#[cfg(test)]
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, p) in perm.iter().enumerate() {
        inv[*p] = i;
    }
    inv
}

/// Recover a dequantized coordinate directly from packed index state.
#[cfg(test)]
fn dequantized_coordinate(
    indices: &[u16],
    centroids: &[f32],
    signs: &[f32],
    inverse_perm: &[usize],
    coord: usize,
) -> f32 {
    let rotated_idx = inverse_perm[coord];
    let centroid_idx = indices[rotated_idx] as usize;
    signs[rotated_idx] * centroids[centroid_idx]
}

/// Sample one standard normal value with Box-Muller transform.
fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    // Box-Muller transform with open interval for ln.
    let u1 = rng.random_range(f32::EPSILON..1.0_f32);
    let u2 = rng.random_range(0.0_f32..1.0_f32);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Retrieve projection matrix from cache or build it on demand.
fn projection_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let cache = PROJECTION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("projection cache lock poisoned");
    if let Some(found) = cache.get(&(dim, seed)) {
        return found.clone();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut matrix = vec![0.0_f32; dim * dim];
    for value in &mut matrix {
        *value = sample_standard_normal(&mut rng);
    }
    cache.insert((dim, seed), matrix.clone());
    matrix
}

/// Transform an original-basis projection matrix into the rotated residual basis.
///
/// Given:
/// - `res_orig[perm[i]] = signs[i] * res_rot[i]`
///
/// this returns `projection_rot` such that:
/// - `projection_orig * res_orig == projection_rot * res_rot`.
fn transform_projection_for_rotated_basis(
    projection_orig: &[f32],
    perm: &[usize],
    signs: &[f32],
) -> Vec<f32> {
    let dim = perm.len();
    assert_eq!(
        projection_orig.len(),
        dim * dim,
        "projection must be row-major dim x dim"
    );
    assert_eq!(signs.len(), dim, "signs length mismatch");
    let mut projection_rot = vec![0.0_f32; dim * dim];
    for row in 0..dim {
        let base = row * dim;
        for ridx in 0..dim {
            let coord = perm[ridx];
            projection_rot[base + ridx] = projection_orig[base + coord] * signs[ridx];
        }
    }
    projection_rot
}

#[cfg(test)]
mod tests;
