use super::*;
use rand::{rngs::StdRng, Rng, SeedableRng};


#[cfg(feature = "cpu")]
#[test]
fn test_cpu_runtime_gate() {
    fn assert_runtime<R: Runtime>() {}
    assert_runtime::<cubecl::cpu::CpuRuntime>();
}

#[cfg(feature = "cpu")]
#[test]
fn test_fused_launch_cpu_runtime() {
    let dim = 128usize;
    let bit_width = 4_u8;
    let seed = 909_u64;
    let mut rng = StdRng::seed_from_u64(404);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-0.5_f32..0.5_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let (mse_rot, qjl) = read_fused_outputs(&outputs);
    let mse = invert_signed_permutation(&mse_rot, dim, seed ^ MSE_ROTATION_SALT);
    let packet = quantize_prod(&input, bit_width, seed);
    let mse_expected = dequantize_mse(&packet.mse);

    for i in 0..dim {
        let expected = mse_expected[i];
        assert!((mse[i] - expected).abs() <= 1e-6, "mse mismatch at {i}");
        let expected_sign = packet.qjl_signs[i] as f32;
        assert!((qjl[i] - expected_sign).abs() <= 1e-6, "qjl mismatch at {i}");
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_fused_launch_cpu_runtime_device_validation() {
    let dim = 128usize;
    let bit_width = 4_u8;
    let seed = 1909_u64;
    let mut rng = StdRng::seed_from_u64(1404);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-0.5_f32..0.5_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );

    let ok = validate_fused_outputs_on_device(&input, bit_width, seed, &outputs, 1e-6);
    assert!(ok, "on-device validation failed for CPU runtime");
}

#[cfg(feature = "cpu")]
#[test]
#[should_panic(expected = "bit_width must be >= 1")]
fn test_fused_launch_cpu_runtime_rejects_zero_bit_width() {
    let input = vec![0.1_f32, -0.2, 0.3, 0.4];
    let _ =
        launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(&Default::default(), &input, 0, 11, true);
}

#[cfg(feature = "cpu")]
#[test]
fn test_fused_launch_cpu_runtime_strict_equivalence_sweep() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let bit_widths = [2_u8, 3_u8, 4_u8, 5_u8];
    let dims = [16_usize, 32, 64];
    let tolerance = 1e-6_f32;

    for case_idx in 0..12 {
        let dim = dims[case_idx % dims.len()];
        let bit_width = bit_widths[case_idx % bit_widths.len()];
        let seed = 10_000_u64 + case_idx as u64;

        let mut input = vec![0.0_f32; dim];
        for value in &mut input {
            *value = rng.gen_range(-1.0_f32..1.0_f32);
        }

        let packet = quantize_prod(&input, bit_width, seed);
        let mse_expected = dequantize_mse(&packet.mse);
        let qjl_expected = packet
            .qjl_signs
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<f32>>();

        let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
            &Default::default(),
            &input,
            bit_width,
            seed,
            true,
        );
        let (mse_rot, qjl) = read_fused_outputs(&outputs);
        let mse = invert_signed_permutation(&mse_rot, dim, seed ^ MSE_ROTATION_SALT);

        for i in 0..dim {
            assert!(
                (mse[i] - mse_expected[i]).abs() <= tolerance,
                "case {case_idx} mse mismatch at {i}: got={} expected={}",
                mse[i],
                mse_expected[i]
            );
            assert!(
                (qjl[i] - qjl_expected[i]).abs() <= tolerance,
                "case {case_idx} qjl mismatch at {i}: got={} expected={}",
                qjl[i],
                qjl_expected[i]
            );
        }

        let ok = validate_fused_outputs_on_device(&input, bit_width, seed, &outputs, tolerance);
        assert!(ok, "case {case_idx} on-device validation failed");
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_host_reference_quantize_dequantize_roundtrips_cpu() {
    let mut rng = StdRng::seed_from_u64(0xABCDEF01);
    let mut input = vec![0.0_f32; 32];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let mse_packet = quantize_mse(&input, 4, 55);
    let mse = dequantize_mse(&mse_packet);
    assert_eq!(mse.len(), input.len());

    let prod_packet = quantize_prod(&input, 5, 77);
    let prod = dequantize_prod(&prod_packet);
    assert_eq!(prod.len(), input.len());
}

#[cfg(feature = "cpu")]
#[test]
fn test_host_fused_launch_wrapper_cpu() {
    let input = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let (mse, qjl) =
        launch_turboquant_fused::<cubecl::cpu::CpuRuntime>(&Default::default(), &input, 4, 123, true);
    assert_eq!(mse.len(), input.len());
    assert_eq!(qjl.len(), input.len());
}

#[cfg(feature = "cpu")]
#[test]
fn test_device_packet_pipeline_roundtrip_indices_cpu() {
    let dim = 64usize;
    let bit_width = 4_u8;
    let seed = 0xFACE_u64;
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let bitpacked = encode_device_bitpacked(&outputs);
    let entropy = encode_device_entropy(&bitpacked);
    let decoded_indices_handle = decode_device_indices(&entropy);
    let decoded = read_u32_buffer(&entropy.client, decoded_indices_handle);

    assert_eq!(decoded, expected, "device packet roundtrip mismatch");
}

#[cfg(feature = "cpu")]
#[test]
#[should_panic(expected = "payload checksum mismatch")]
fn test_xor_decode_rejects_payload_corruption_cpu() {
    let dim = 64usize;
    let bit_width = 4_u8;
    let seed = 0xFACE_u64;
    let mut rng = StdRng::seed_from_u64(0xABCD0001);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let bitpacked = encode_device_bitpacked(&outputs);
    let entropy = encode_device_entropy(&bitpacked);

    let payload_bytes = entropy.client.read_one(entropy.payload_words_handle.clone());
    let mut words = u32::from_bytes(&payload_bytes).to_vec();
    if let Some(first) = words.get_mut(0) {
        *first ^= 0x01;
    }
    let corrupted_handle = entropy.client.create_from_slice(u32::as_bytes(&words));
    let corrupted = DeviceEncodedPacket::<cubecl::cpu::CpuRuntime> {
        client: entropy.client.clone(),
        payload_words_handle: corrupted_handle,
        valid_bits: entropy.valid_bits,
        word_count: entropy.word_count,
        dim: entropy.dim,
        bit_width: entropy.bit_width,
        seed: entropy.seed,
        encoding: entropy.encoding,
        huffman_parent_handle: entropy.huffman_parent_handle.clone(),
        huffman_left_handle: entropy.huffman_left_handle.clone(),
        huffman_right_handle: entropy.huffman_right_handle.clone(),
        huffman_root_handle: entropy.huffman_root_handle.clone(),
        huffman_written_bits_handle: entropy.huffman_written_bits_handle.clone(),
        huffman_codebook_generation: entropy.huffman_codebook_generation,
        huffman_policy_id: entropy.huffman_policy_id,
        huffman_codebook_fingerprint: entropy.huffman_codebook_fingerprint,
        payload_crc32c: entropy.payload_crc32c,
    };

    let _ = decode_device_indices(&corrupted);
}

#[cfg(feature = "cpu")]
#[test]
#[should_panic(expected = "non-huffman valid_bits mismatch")]
fn test_xor_decode_rejects_valid_bits_mismatch_cpu() {
    let dim = 64usize;
    let bit_width = 4_u8;
    let seed = 0xFACE_u64;
    let mut rng = StdRng::seed_from_u64(0xABCD0002);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let bitpacked = encode_device_bitpacked(&outputs);
    let entropy = encode_device_entropy(&bitpacked);

    let mut malformed = DeviceEncodedPacket::<cubecl::cpu::CpuRuntime> {
        client: entropy.client.clone(),
        payload_words_handle: entropy.payload_words_handle.clone(),
        valid_bits: entropy.valid_bits.saturating_sub(1),
        word_count: entropy.word_count,
        dim: entropy.dim,
        bit_width: entropy.bit_width,
        seed: entropy.seed,
        encoding: entropy.encoding,
        huffman_parent_handle: entropy.huffman_parent_handle.clone(),
        huffman_left_handle: entropy.huffman_left_handle.clone(),
        huffman_right_handle: entropy.huffman_right_handle.clone(),
        huffman_root_handle: entropy.huffman_root_handle.clone(),
        huffman_written_bits_handle: entropy.huffman_written_bits_handle.clone(),
        huffman_codebook_generation: entropy.huffman_codebook_generation,
        huffman_policy_id: entropy.huffman_policy_id,
        huffman_codebook_fingerprint: entropy.huffman_codebook_fingerprint,
        payload_crc32c: None,
    };
    malformed.payload_crc32c = Some(packet_payload_crc32c(&malformed));

    let _ = decode_device_indices(&malformed);
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_device_huffman_roundtrip_indices_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0xDA7A_u64;
    let mut rng = StdRng::seed_from_u64(0x1234ABCD);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let huffman = encode_device_huffman(&outputs);
    let decoded = read_u32_buffer(&huffman.client, decode_device_indices(&huffman));

    assert_eq!(decoded, expected, "device huffman roundtrip mismatch");
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_device_huffman_roundtrip_with_shared_codebook_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0xDA7B_u64;
    let mut rng = StdRng::seed_from_u64(0x1234ABCE);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let codebook = build_device_huffman_codebook(&outputs);
    let huffman = encode_device_huffman_with_codebook(&outputs, &codebook);
    let decoded =
        read_u32_buffer(&huffman.client, decode_device_indices_with_codebook(&huffman, Some(&codebook)));

    assert_eq!(decoded, expected, "device huffman shared-codebook roundtrip mismatch");
    assert!(huffman.huffman_parent_handle.is_none(), "shared codebook packets should not embed tree");
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_device_huffman_policy_rebuild_cadence_cpu() {
    let dim = 64usize;
    let bit_width = 4_u8;
    let seed = 0xDA7C_u64;
    let mut rng = StdRng::seed_from_u64(0x1234ABCF);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = HuffmanCodebookReusePolicy::<cubecl::cpu::CpuRuntime>::new(2);
    let packet1 = policy.encode(&outputs);
    let decoded1 = read_u32_buffer(&packet1.client, policy.decode(&packet1));
    assert_eq!(decoded1, expected);

    let packet2 = policy.encode(&outputs);
    let decoded2 = read_u32_buffer(&packet2.client, policy.decode(&packet2));
    assert_eq!(decoded2, expected);

    // Third encode triggers rebuild due to cadence=2.
    let packet3 = policy.encode(&outputs);
    let decoded3 = read_u32_buffer(&packet3.client, policy.decode(&packet3));
    assert_eq!(decoded3, expected);
    assert_eq!(policy.tokens_since_rebuild(), 1);
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_device_huffman_policy_auto_roundtrip_cpu() {
    let dim = 72usize;
    let bit_width = 4_u8;
    let seed = 0xDA7D_u64;
    let mut rng = StdRng::seed_from_u64(0x1234ABD0);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet1 = policy.encode(&outputs);
    let packet2 = policy.encode(&outputs);
    let decoded1 = read_u32_buffer(&packet1.client, policy.decode(&packet1));
    let decoded2 = read_u32_buffer(&packet2.client, policy.decode(&packet2));

    assert_eq!(decoded1, expected);
    assert_eq!(decoded2, expected);
    assert!(
        policy.rebuild_every_tokens().is_some(),
        "auto policy should compute a rebuild cadence",
    );
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_exposes_runtime_state_cpu() {
    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    assert_eq!(policy.rebuild_every_tokens(), None);
    assert_eq!(policy.tokens_since_rebuild(), None);
    policy.invalidate();
    assert_eq!(policy.rebuild_every_tokens(), None);
    assert_eq!(policy.tokens_since_rebuild(), None);
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_roundtrip_single_symbol_cpu() {
    let dim = 128usize;
    let bit_width = 4_u8;
    let seed = 0xA11CE_u64;
    let input = vec![0.125_f32; dim];
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet = policy.encode(&outputs);
    let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));

    assert_eq!(decoded, expected);
    assert!(policy.rebuild_every_tokens().is_some());
    assert!(policy.tokens_since_rebuild().is_some());
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_roundtrip_tiny_dims_cpu() {
    let dims = [1usize, 2usize];
    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    for (i, dim) in dims.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0xB001_u64 + i as u64);
        let mut input = vec![0.0_f32; *dim];
        for value in &mut input {
            *value = rng.gen_range(-1.0_f32..1.0_f32);
        }
        let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
            &Default::default(),
            &input,
            3,
            0xCA11_u64 + i as u64,
            true,
        );
        let expected = read_fused_indices(&outputs);
        let packet = policy.encode(&outputs);
        let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
        assert_eq!(decoded, expected, "tiny dim mismatch for dim={dim}");
    }
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_roundtrip_uniform_like_cpu() {
    let dim = 192usize;
    let bit_width = 4_u8;
    let seed = 0xBEEFu64;
    let mut rng = StdRng::seed_from_u64(0x12345678);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet = policy.encode(&outputs);
    let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
    assert_eq!(decoded, expected);
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_handles_bit_width_and_dim_changes_cpu() {
    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let mut rng = StdRng::seed_from_u64(0xD1F7_u64);

    let mut input_a = vec![0.0_f32; 64];
    for value in &mut input_a {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs_a = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input_a,
        4,
        0x11_u64,
        true,
    );
    let expected_a = read_fused_indices(&outputs_a);
    let packet_a = policy.encode(&outputs_a);
    let decoded_a = read_u32_buffer(&packet_a.client, policy.decode(&packet_a));
    assert_eq!(decoded_a, expected_a);
    assert_eq!(policy.tokens_since_rebuild(), Some(1));

    let mut input_b = vec![0.0_f32; 96];
    for value in &mut input_b {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs_b = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input_b,
        4,
        0x22_u64,
        true,
    );
    let expected_b = read_fused_indices(&outputs_b);
    let packet_b = policy.encode(&outputs_b);
    let decoded_b = read_u32_buffer(&packet_b.client, policy.decode(&packet_b));
    assert_eq!(decoded_b, expected_b);
    assert_eq!(policy.tokens_since_rebuild(), Some(1));

    let outputs_c = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input_b,
        5,
        0x33_u64,
        true,
    );
    let expected_c = read_fused_indices(&outputs_c);
    let packet_c = policy.encode(&outputs_c);
    let decoded_c = read_u32_buffer(&packet_c.client, policy.decode(&packet_c));
    assert_eq!(decoded_c, expected_c);
    assert_eq!(policy.tokens_since_rebuild(), Some(1));
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_long_run_drift_roundtrip_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let mut rng = StdRng::seed_from_u64(0x1A2B3C4D_u64);

    for step in 0..128usize {
        let mut input = vec![0.0_f32; dim];
        for value in &mut input {
            if step < 64 {
                *value = rng.gen_range(-0.05_f32..0.05_f32);
            } else {
                let sign = if rng.gen_range(0_u32..2_u32) == 0 { -1.0_f32 } else { 1.0_f32 };
                *value = sign * rng.gen_range(0.6_f32..1.0_f32);
            }
        }
        let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
            &Default::default(),
            &input,
            bit_width,
            0x9000_u64 + step as u64,
            true,
        );
        let expected = read_fused_indices(&outputs);
        let packet = policy.encode(&outputs);
        let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
        assert_eq!(decoded, expected, "auto policy drift mismatch at step={step}");
    }
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_rebuild_boundary_cpu() {
    let dim = 80usize;
    let bit_width = 4_u8;
    let seed = 0x4455_u64;
    let mut rng = StdRng::seed_from_u64(0x55667788_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet0 = policy.encode(&outputs);
    let decoded0 = read_u32_buffer(&packet0.client, policy.decode(&packet0));
    assert_eq!(decoded0, expected);
    let rebuild_every = policy
        .rebuild_every_tokens()
        .expect("auto policy must set rebuild cadence after first encode");
    assert!((1..=256).contains(&rebuild_every));

    for _ in 0..rebuild_every.saturating_sub(1) {
        let packet = policy.encode(&outputs);
        let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
        assert_eq!(decoded, expected);
    }
    assert_eq!(policy.tokens_since_rebuild(), Some(rebuild_every));

    let packet_rebuild = policy.encode(&outputs);
    let decoded_rebuild = read_u32_buffer(&packet_rebuild.client, policy.decode(&packet_rebuild));
    assert_eq!(decoded_rebuild, expected);
    assert_eq!(
        policy.tokens_since_rebuild(),
        Some(1),
        "counter should reset after rebuild and increment for rebuilt encode"
    );
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_decode_before_encode_fallback_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0xA0A0_u64;
    let mut rng = StdRng::seed_from_u64(0x1020_3040_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let bitpacked = encode_device_bitpacked(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let decoded = read_u32_buffer(&bitpacked.client, policy.decode(&bitpacked));

    assert_eq!(decoded, expected);
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_invalidate_then_rebuild_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let mut rng = StdRng::seed_from_u64(0x2030_4050_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        0xB0B0_u64,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();

    let packet_first = policy.encode(&outputs);
    let decoded_first = read_u32_buffer(&packet_first.client, policy.decode(&packet_first));
    assert_eq!(decoded_first, expected);
    let cadence_before = policy.rebuild_every_tokens();
    assert!(cadence_before.is_some());
    assert_eq!(policy.tokens_since_rebuild(), Some(1));

    policy.invalidate();
    assert_eq!(policy.rebuild_every_tokens(), cadence_before);
    assert_eq!(policy.tokens_since_rebuild(), Some(0));

    let packet_second = policy.encode(&outputs);
    let decoded_second = read_u32_buffer(&packet_second.client, policy.decode(&packet_second));
    assert_eq!(decoded_second, expected);
    assert!(policy.rebuild_every_tokens().is_some());
    assert_eq!(policy.tokens_since_rebuild(), Some(1));
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_huffman_decode_without_written_bits_handle_uses_valid_bits_cpu() {
    let dim = 128usize;
    let bit_width = 4_u8;
    let mut rng = StdRng::seed_from_u64(0x3040_5060_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        0xC0C0_u64,
        true,
    );
    let expected = read_fused_indices(&outputs);
    let codebook = build_device_huffman_codebook(&outputs);
    let packet = encode_device_huffman_with_codebook(&outputs, &codebook);

    let mut packet_without_written = DeviceEncodedPacket::<cubecl::cpu::CpuRuntime> {
        client: packet.client.clone(),
        payload_words_handle: packet.payload_words_handle.clone(),
        valid_bits: packet.valid_bits,
        word_count: packet.word_count,
        dim: packet.dim,
        bit_width: packet.bit_width,
        seed: packet.seed,
        encoding: packet.encoding,
        huffman_parent_handle: packet.huffman_parent_handle.clone(),
        huffman_left_handle: packet.huffman_left_handle.clone(),
        huffman_right_handle: packet.huffman_right_handle.clone(),
        huffman_root_handle: packet.huffman_root_handle.clone(),
        huffman_written_bits_handle: None,
        huffman_codebook_generation: packet.huffman_codebook_generation,
        huffman_policy_id: packet.huffman_policy_id,
        huffman_codebook_fingerprint: packet.huffman_codebook_fingerprint,
        payload_crc32c: None,
    };
    packet_without_written.payload_crc32c = Some(packet_payload_crc32c(&packet_without_written));

    let decoded = read_u32_buffer(
        &packet_without_written.client,
        decode_device_indices_with_codebook(&packet_without_written, Some(&codebook)),
    );
    assert_eq!(decoded, expected);

    let packet_bytes = huffman_packet_wire_bytes(&packet_without_written);
    assert_eq!(packet_bytes, bits_to_bytes(packet_without_written.valid_bits as usize));
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_large_dim_stress_cpu() {
    let dim = 4096usize;
    let bit_width = 4_u8;
    let mut rng = StdRng::seed_from_u64(0x4050_6070_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        0xD0D0_u64,
        true,
    );
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet = policy.encode(&outputs);
    let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));

    assert_eq!(decoded, expected);
    assert!(policy.rebuild_every_tokens().is_some());
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
fn test_auto_policy_cadence_bounds_cpu() {
    let cases = [
        (64usize, 2_u8, 0xE001_u64),
        (96usize, 3_u8, 0xE002_u64),
        (128usize, 4_u8, 0xE003_u64),
        (256usize, 5_u8, 0xE004_u64),
    ];

    for (dim, bit_width, seed) in cases {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xF0F0_u64);
        let mut input = vec![0.0_f32; dim];
        for value in &mut input {
            *value = rng.gen_range(-1.0_f32..1.0_f32);
        }
        let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
            &Default::default(),
            &input,
            bit_width,
            seed,
            true,
        );
        let expected = read_fused_indices(&outputs);
        let mut policy = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
        let packet = policy.encode(&outputs);
        let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
        assert_eq!(decoded, expected);

        let cadence = policy
            .rebuild_every_tokens()
            .expect("auto policy must compute cadence after first encode");
        assert!(
            (1usize..=256usize).contains(&cadence),
            "cadence must stay in clamp bounds; got {cadence}"
        );
    }
}

#[cfg(all(feature = "cpu", feature = "experimental-huffman"))]
#[test]
#[should_panic(expected = "policy identity mismatch")]
fn test_auto_policy_decode_rejects_wrong_policy_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0xE100_u64;
    let mut rng = StdRng::seed_from_u64(0x51525354_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let outputs = launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(
        &Default::default(),
        &input,
        bit_width,
        seed,
        true,
    );

    let mut policy_a = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let packet = policy_a.encode(&outputs);

    let mut policy_b = AutoHuffmanCodebookPolicy::<cubecl::cpu::CpuRuntime>::new();
    let _ = policy_b.encode(&outputs);
    let _ = policy_b.decode(&packet);
}

#[cfg(feature = "cpu")]
#[test]
fn test_device_pipeline_fused_kernel_matches_staged_kernels_cpu() {
    let dim = 80usize;
    let bit_width = 4_u8;
    let seed = 0xABCD_u64;
    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let device = Default::default();

    let staged_outputs =
        launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(&device, &input, bit_width, seed, true);
    let staged_encoded = encode_device_entropy(&encode_device_bitpacked(&staged_outputs));
    let staged_indices = read_u32_buffer(
        &staged_encoded.client,
        decode_device_indices(&staged_encoded),
    );

    let (fused_outputs, fused_packet) = launch_turboquant_pipeline_device::<cubecl::cpu::CpuRuntime>(
        &device,
        &input,
        bit_width,
        seed,
        true,
        None,
        true,
    );
    let fused_indices = read_u32_buffer(&fused_packet.client, decode_device_indices(&fused_packet));

    assert_eq!(read_fused_indices(&fused_outputs), read_fused_indices(&staged_outputs));
    assert_eq!(fused_indices, staged_indices);
}

#[cfg(feature = "cpu")]
#[test]
fn test_device_pipeline_from_handle_stays_device_native_cpu() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0xFEED_u64;
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }
    let device = Default::default();

    let assets = prepare_turboquant_launch_assets::<cubecl::cpu::CpuRuntime>(
        &device, dim, bit_width, seed, None,
    );
    let input_handle = assets.client.create_from_slice(f32::as_bytes(&input));
    let (_state, encoded) = launch_turboquant_pipeline_device_from_handle(
        &assets,
        &input_handle,
        true,
        true,
    );
    let decoded_indices = read_u32_buffer(&encoded.client, decode_device_indices(&encoded));
    assert_eq!(decoded_indices.len(), dim);
}

#[cfg(feature = "cpu")]
#[test]
fn test_fluent_kernel_api_matches_direct_launch() {
    use crate::api::kernel::turboquant_kernel;

    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let dim = 48usize;
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-1.0_f32..1.0_f32);
    }

    let device = Default::default();
    let direct =
        launch_turboquant_fused_device::<cubecl::cpu::CpuRuntime>(&device, &input, 4, 77, true);
    let (mse_direct, qjl_direct) = read_fused_outputs(&direct);

    let fluent = turboquant_kernel::<cubecl::cpu::CpuRuntime>(&device, &input)
        .bit_width(4)
        .seed(77)
        .emit_qjl(true);
    let fluent_outputs = fluent.launch_device();
    let (mse_fluent, qjl_fluent) = read_fused_outputs(&fluent_outputs);

    assert_eq!(mse_fluent, mse_direct);
    assert_eq!(qjl_fluent, qjl_direct);
    assert!(fluent.validate_on_device(&fluent_outputs, 1e-6));
}


#[cfg(all(feature = "wgpu", feature = "wgpu-msl"))]
#[test]
fn test_wgpu_msl_runtime_gate() {
    fn assert_runtime<R: Runtime>() {}
    assert_runtime::<cubecl::wgpu::WgpuRuntime>();
}

#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
#[test]
fn test_fused_launch_wgpu_msl_runtime() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 1009_u64;
    let mut rng = StdRng::seed_from_u64(505);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-0.5_f32..0.5_f32);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let outputs =
        launch_turboquant_fused_device::<cubecl::wgpu::WgpuRuntime>(&device, &input, bit_width, seed, true);
    let (mse_rot, qjl) = read_fused_outputs(&outputs);
    let mse = invert_signed_permutation(&mse_rot, dim, seed ^ MSE_ROTATION_SALT);
    let packet = quantize_prod(&input, bit_width, seed);
    let mse_expected = dequantize_mse(&packet.mse);

    for i in 0..dim {
        let expected = mse_expected[i];
        assert!((mse[i] - expected).abs() <= 1e-3, "mse mismatch at {i}");
        let expected_sign = packet.qjl_signs[i] as f32;
        assert!((qjl[i] - expected_sign).abs() <= 1e-3, "qjl mismatch at {i}");
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
#[test]
fn test_fused_launch_wgpu_msl_runtime_device_validation() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 2009_u64;
    let mut rng = StdRng::seed_from_u64(1505);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-0.5_f32..0.5_f32);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);

    let outputs =
        launch_turboquant_fused_device::<cubecl::wgpu::WgpuRuntime>(&device, &input, bit_width, seed, true);

    let ok = validate_fused_outputs_on_device(&input, bit_width, seed, &outputs, 1e-3);
    assert!(ok, "on-device validation failed for wgpu-msl runtime");
}

#[cfg(all(feature = "wgpu", feature = "wgpu-msl", feature = "experimental-huffman", target_os = "macos"))]
#[test]
fn test_auto_policy_roundtrip_wgpu_msl() {
    let dim = 96usize;
    let bit_width = 4_u8;
    let seed = 0x7788_u64;
    let mut rng = StdRng::seed_from_u64(0x99887766_u64);
    let mut input = vec![0.0_f32; dim];
    for value in &mut input {
        *value = rng.gen_range(-0.8_f32..0.8_f32);
    }

    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    init_wgpu_msl_once(&device);
    let outputs =
        launch_turboquant_fused_device::<cubecl::wgpu::WgpuRuntime>(&device, &input, bit_width, seed, true);
    let expected = read_fused_indices(&outputs);

    let mut policy = AutoHuffmanCodebookPolicy::<cubecl::wgpu::WgpuRuntime>::new();
    let packet = policy.encode(&outputs);
    let decoded = read_u32_buffer(&packet.client, policy.decode(&packet));
    assert_eq!(decoded, expected);
}

#[cfg(all(feature = "wgpu", feature = "wgpu-msl", target_os = "macos"))]
fn init_wgpu_msl_once(device: &cubecl::wgpu::WgpuDevice) {
    static WGPU_INIT: OnceLock<()> = OnceLock::new();
    WGPU_INIT.get_or_init(|| {
        let _setup = cubecl::wgpu::init_setup::<cubecl::wgpu::Metal>(
            device,
            cubecl::wgpu::RuntimeOptions::default(),
        );
    });
}
