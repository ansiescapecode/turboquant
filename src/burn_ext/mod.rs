//! Burn backend extension for TurboQuant device kernels.
//!
//! This extension is intentionally device-only and only implemented for
//! `burn-cubecl` backends with `f32` float tensors.

use burn::tensor::{backend::Backend, DType, Shape, Tensor, TensorPrimitive};
use cubecl::server::Handle;

/// Burn backend extension trait for device-native TurboQuant operations.
pub trait TurboQuantBackendExt: Backend {
    /// Launch TurboQuant MSE fused kernel and return the device tensor output.
    fn turboquant_mse_device(tensor: Tensor<Self, 1>, bit_width: u8, seed: u64) -> Tensor<Self, 1>;

    /// Launch TurboQuant prod fused kernel and return the device tensor output.
    fn turboquant_prod_device(tensor: Tensor<Self, 1>, bit_width: u8, seed: u64)
        -> Tensor<Self, 1>;
}

/// User-facing Tensor wrapper for TurboQuant MSE kernel path.
pub fn turboquant_mse<B: TurboQuantBackendExt>(
    tensor: Tensor<B, 1>,
    bit_width: u8,
    seed: u64,
) -> Tensor<B, 1> {
    B::turboquant_mse_device(tensor, bit_width, seed)
}

/// User-facing Tensor wrapper for TurboQuant prod kernel path.
pub fn turboquant_prod<B: TurboQuantBackendExt>(
    tensor: Tensor<B, 1>,
    bit_width: u8,
    seed: u64,
) -> Tensor<B, 1> {
    B::turboquant_prod_device(tensor, bit_width, seed)
}

/// Encode Burn tensor MSE indices to a device bitpacked payload.
pub fn turboquant_mse_encode_bitpacked<R, I, BT>(
    tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>,
    bit_width: u8,
    seed: u64,
) -> crate::kernels::DeviceEncodedPacket<R>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    use crate::kernels::{
        encode_device_bitpacked, launch_turboquant_fused_device_from_handle,
        prepare_turboquant_launch_assets,
    };

    let primitive = tensor.into_primitive().tensor();
    let dim = primitive.shape.num_elements();
    let assets =
        prepare_turboquant_launch_assets::<R>(&primitive.device, dim, bit_width, seed, None);
    let outputs = launch_turboquant_fused_device_from_handle::<R>(&assets, &primitive.handle, true);
    encode_device_bitpacked(&outputs)
}

/// Encode Burn tensor MSE indices to a device entropy payload.
pub fn turboquant_mse_encode_entropy<R, I, BT>(
    tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>,
    bit_width: u8,
    seed: u64,
) -> crate::kernels::DeviceEncodedPacket<R>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    use crate::kernels::{
        encode_device_bitpacked, encode_device_entropy, launch_turboquant_fused_device_from_handle,
        prepare_turboquant_launch_assets,
    };
    let primitive = tensor.into_primitive().tensor();
    let dim = primitive.shape.num_elements();
    let assets =
        prepare_turboquant_launch_assets::<R>(&primitive.device, dim, bit_width, seed, None);
    let outputs = launch_turboquant_fused_device_from_handle::<R>(&assets, &primitive.handle, true);
    let bitpacked = encode_device_bitpacked(&outputs);
    encode_device_entropy(&bitpacked)
}

/// Build a reusable Huffman codebook for Burn tensor MSE indices.
pub fn turboquant_mse_build_huffman_codebook<R, I, BT>(
    tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>,
    bit_width: u8,
    seed: u64,
) -> crate::kernels::DeviceHuffmanCodebook<R>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    #[cfg(not(feature = "experimental-huffman"))]
    {
        let _ = (tensor, bit_width, seed);
        panic!("huffman is experimental; enable feature \"experimental-huffman\"");
    }
    #[cfg(feature = "experimental-huffman")]
    {
        use crate::kernels::{
            build_device_huffman_codebook, launch_turboquant_fused_device_from_handle,
            prepare_turboquant_launch_assets,
        };
        let primitive = tensor.into_primitive().tensor();
        let dim = primitive.shape.num_elements();
        let assets =
            prepare_turboquant_launch_assets::<R>(&primitive.device, dim, bit_width, seed, None);
        let outputs = launch_turboquant_fused_device_from_handle::<R>(&assets, &primitive.handle, true);
        build_device_huffman_codebook(&outputs)
    }
}

/// Encode Burn tensor MSE indices to Huffman payload using a shared codebook.
pub fn turboquant_mse_encode_entropy_with_codebook<R, I, BT>(
    tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>,
    bit_width: u8,
    seed: u64,
    codebook: &crate::kernels::DeviceHuffmanCodebook<R>,
) -> crate::kernels::DeviceEncodedPacket<R>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    #[cfg(not(feature = "experimental-huffman"))]
    {
        let _ = (tensor, bit_width, seed, codebook);
        panic!("huffman is experimental; enable feature \"experimental-huffman\"");
    }
    #[cfg(feature = "experimental-huffman")]
    {
        use crate::kernels::{
            encode_device_huffman_with_codebook, launch_turboquant_fused_device_from_handle,
            prepare_turboquant_launch_assets,
        };
        let primitive = tensor.into_primitive().tensor();
        let dim = primitive.shape.num_elements();
        let assets =
            prepare_turboquant_launch_assets::<R>(&primitive.device, dim, bit_width, seed, None);
        let outputs = launch_turboquant_fused_device_from_handle::<R>(&assets, &primitive.handle, true);
        encode_device_huffman_with_codebook(&outputs, codebook)
    }
}

/// Decode a device packet back to device-resident indices.
pub fn turboquant_mse_decode_indices<R: burn_cubecl::CubeRuntime>(
    packet: &crate::kernels::DeviceEncodedPacket<R>,
) -> Handle {
    crate::kernels::decode_device_indices(packet)
}

/// Decode a Huffman packet using an external shared codebook.
pub fn turboquant_mse_decode_indices_with_codebook<R: burn_cubecl::CubeRuntime>(
    packet: &crate::kernels::DeviceEncodedPacket<R>,
    codebook: &crate::kernels::DeviceHuffmanCodebook<R>,
) -> Handle {
    #[cfg(not(feature = "experimental-huffman"))]
    {
        let _ = (packet, codebook);
        panic!("huffman is experimental; enable feature \"experimental-huffman\"");
    }
    #[cfg(feature = "experimental-huffman")]
    {
        crate::kernels::decode_device_indices_with_codebook(packet, Some(codebook))
    }
}

impl<R, I, BT> TurboQuantBackendExt for burn_cubecl::CubeBackend<R, f32, I, BT>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    fn turboquant_mse_device(tensor: Tensor<Self, 1>, bit_width: u8, seed: u64) -> Tensor<Self, 1> {
        let primitive = tensor.into_primitive().tensor();
        let dim = primitive.shape.num_elements();
        let assets = crate::kernels::prepare_turboquant_launch_assets::<R>(
            &primitive.device,
            dim,
            bit_width,
            seed,
            None,
        );
        let outputs = crate::kernels::launch_turboquant_fused_device_from_handle::<R>(
            &assets,
            &primitive.handle,
            false,
        );
        let out = burn_cubecl::tensor::CubeTensor::new_contiguous(
            outputs.client,
            primitive.device,
            Shape::new([dim]),
            outputs.mse_handle,
            DType::F32,
        );
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    fn turboquant_prod_device(
        tensor: Tensor<Self, 1>,
        bit_width: u8,
        seed: u64,
    ) -> Tensor<Self, 1> {
        let primitive = tensor.into_primitive().tensor();
        let dim = primitive.shape.num_elements();
        let assets = crate::kernels::prepare_turboquant_launch_assets::<R>(
            &primitive.device,
            dim,
            bit_width,
            seed,
            None,
        );
        let outputs = crate::kernels::launch_turboquant_fused_device_from_handle::<R>(
            &assets,
            &primitive.handle,
            true,
        );
        let out = burn_cubecl::tensor::CubeTensor::new_contiguous(
            outputs.client,
            primitive.device,
            Shape::new([dim]),
            outputs.qjl_handle,
            DType::F32,
        );
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

#[cfg(all(
    test,
    feature = "burn-ext",
    feature = "wgpu",
    feature = "wgpu-msl",
    target_os = "macos"
))]
mod tests {
    use super::*;

    type B = burn_wgpu::Wgpu<f32, i32, u32>;

    fn sample_tensor() -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(
                vec![0.2_f32, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9],
                [8],
            ),
            &device,
        )
    }

    #[test]
    fn test_burn_ext_public_paths_wgpu_msl() {
        let tensor = sample_tensor();
        let bit_width = 4;
        let seed = 21;

        let _mse = turboquant_mse(tensor.clone(), bit_width, seed);
        let _prod = turboquant_prod(tensor.clone(), bit_width, seed);
        let packet_bp = turboquant_mse_encode_bitpacked::<burn_wgpu::WgpuRuntime, i32, u32>(
            tensor.clone(),
            bit_width,
            seed,
        );
        let packet_ent = turboquant_mse_encode_entropy::<burn_wgpu::WgpuRuntime, i32, u32>(
            tensor.clone(),
            bit_width,
            seed,
        );
        let _decoded_bp = turboquant_mse_decode_indices::<burn_wgpu::WgpuRuntime>(&packet_bp);
        let _decoded_ent = turboquant_mse_decode_indices::<burn_wgpu::WgpuRuntime>(&packet_ent);
    }

    #[cfg(feature = "experimental-huffman")]
    #[test]
    fn test_burn_ext_huffman_helpers_wgpu_msl() {
        let tensor = sample_tensor();
        let bit_width = 4;
        let seed = 31;
        let codebook = turboquant_mse_build_huffman_codebook::<burn_wgpu::WgpuRuntime, i32, u32>(
            tensor.clone(),
            bit_width,
            seed,
        );
        let packet = turboquant_mse_encode_entropy_with_codebook::<burn_wgpu::WgpuRuntime, i32, u32>(
            tensor, bit_width, seed, &codebook,
        );
        let _decoded = turboquant_mse_decode_indices_with_codebook::<burn_wgpu::WgpuRuntime>(
            &packet, &codebook,
        );
    }
}
