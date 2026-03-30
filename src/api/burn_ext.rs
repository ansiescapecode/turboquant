use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;

use crate::burn_ext::{turboquant_mse, turboquant_prod, TurboQuantBackendExt};
#[cfg(feature = "experimental-huffman")]
use crate::kernels::{
    build_device_huffman_codebook, decode_device_indices_with_codebook, encode_device_huffman,
    encode_device_huffman_with_codebook, AutoHuffmanCodebookPolicy, DeviceHuffmanCodebook,
    HuffmanCodebookReusePolicy,
};
use crate::kernels::{
    decode_device_indices, encode_device_bitpacked, encode_device_entropy,
    launch_turboquant_fused_device_from_handle,
    launch_turboquant_pipeline_device_from_handle_with_options,
    prepare_turboquant_launch_assets_with_options, DeviceEncodedPacket, DeviceFusedOutputs,
    DeviceLaunchAssets, TurboQuantKernelOptions, TurboQuantLaunchOverride,
};

/// Fluent Burn extension builder for `Tensor<B, 1>`.
#[derive(Debug)]
pub struct TurboQuantFluent<B: TurboQuantBackendExt> {
    tensor: Tensor<B, 1>,
    bit_width: u8,
    seed: u64,
}

impl<B: TurboQuantBackendExt> TurboQuantFluent<B> {
    /// Build fluent API with defaults: `bit_width=4`, `seed=0`.
    pub fn new(tensor: Tensor<B, 1>) -> Self {
        Self {
            tensor,
            bit_width: 4,
            seed: 0,
        }
    }

    /// Set total TurboQuant bit-width.
    /// Set total quantization bit-width.
    ///
    /// # Parameters
    ///
    /// - `bit_width`: Target bit-width in `[1, 8]`.
    ///
    /// # Returns
    ///
    /// Updated fluent builder.
    pub fn bit_width(mut self, bit_width: u8) -> Self {
        self.bit_width = bit_width;
        self
    }

    /// Set deterministic seed.
    /// Set deterministic seed used by permutation/projection generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Execute MSE path.
    pub fn mse(self) -> Tensor<B, 1> {
        turboquant_mse(self.tensor, self.bit_width, self.seed)
    }

    /// Execute product path.
    pub fn prod(self) -> Tensor<B, 1> {
        turboquant_prod(self.tensor, self.bit_width, self.seed)
    }
}

/// Device-native fluent builder specialized for Burn CubeBackend tensors.
#[derive(Debug)]
pub struct TurboQuantCubeFluent<R, I, BT>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>,
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
    emit_entropy: bool,
    single_kernel_packet: bool,
    launch_override: Option<TurboQuantLaunchOverride>,
}

impl<R, I, BT> TurboQuantCubeFluent<R, I, BT>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    /// Construct with defaults: bit_width=4, seed=0, emit_qjl=true.
    pub fn new(tensor: Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>) -> Self {
        Self {
            tensor,
            bit_width: 4,
            seed: 0,
            emit_qjl: true,
            emit_entropy: true,
            single_kernel_packet: true,
            launch_override: None,
        }
    }

    /// Set total quantization bit-width.
    pub fn bit_width(mut self, bit_width: u8) -> Self {
        self.bit_width = bit_width;
        self
    }

    /// Set deterministic seed used by the kernel path.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable QJL sign emission.
    pub fn emit_qjl(mut self, emit_qjl: bool) -> Self {
        self.emit_qjl = emit_qjl;
        self
    }

    /// Enable or disable entropy transform in pipeline launches.
    pub fn emit_entropy(mut self, emit_entropy: bool) -> Self {
        self.emit_entropy = emit_entropy;
        self
    }

    /// Select fused single-kernel packet path vs staged encode path.
    pub fn single_kernel_packet(mut self, single_kernel_packet: bool) -> Self {
        self.single_kernel_packet = single_kernel_packet;
        self
    }

    /// Override cube dimension for launch tuning.
    pub fn cube_dim(mut self, cube_dim: u32) -> Self {
        self.launch_override = Some(TurboQuantLaunchOverride { cube_dim });
        self
    }

    /// Keep existing ergonomic tensor output path on Burn tensors.
    pub fn mse(self) -> Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1> {
        turboquant_mse(self.tensor, self.bit_width, self.seed)
    }

    /// Keep existing ergonomic tensor output path on Burn tensors.
    pub fn prod(self) -> Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1> {
        turboquant_prod(self.tensor, self.bit_width, self.seed)
    }

    fn options(&self) -> TurboQuantKernelOptions {
        TurboQuantKernelOptions {
            emit_qjl: self.emit_qjl,
            emit_entropy: self.emit_entropy,
            single_kernel_packet: self.single_kernel_packet,
        }
    }

    fn primitive(&self) -> burn_cubecl::tensor::CubeTensor<R> {
        match self.tensor.clone().into_primitive() {
            TensorPrimitive::Float(t) => t,
            TensorPrimitive::QFloat(t) => {
                panic!(
                    "quantized Burn tensors are not supported in TurboQuantCubeFluent: {:?}",
                    t.dtype
                )
            }
        }
    }

    /// Prepare reusable launch assets for handle-based execution.
    pub fn prepare_assets(&self) -> DeviceLaunchAssets<R> {
        let primitive = self.primitive();
        let dim = primitive.shape.num_elements();
        prepare_turboquant_launch_assets_with_options::<R>(
            &primitive.device,
            dim,
            self.bit_width,
            self.seed,
            self.launch_override,
            self.options(),
        )
    }

    /// Launch fused TurboQuant and keep outputs device-resident.
    pub fn launch_device(&self) -> DeviceFusedOutputs<R> {
        let primitive = self.primitive();
        let assets = self.prepare_assets();
        launch_turboquant_fused_device_from_handle::<R>(&assets, &primitive.handle, self.emit_qjl)
    }

    /// Launch fused TurboQuant plus encoded packet pipeline on-device.
    pub fn launch_device_pipeline(&self) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
        let primitive = self.primitive();
        let assets = self.prepare_assets();
        launch_turboquant_pipeline_device_from_handle_with_options::<R>(
            &assets,
            &primitive.handle,
            self.options(),
        )
    }

    /// Launch and encode with bitpacking on-device.
    pub fn launch_device_bitpacked(&self) -> DeviceEncodedPacket<R> {
        let outputs = self.launch_device();
        self.encode_device(&outputs)
    }

    /// Launch and encode with bitpacking + entropy on-device.
    pub fn launch_device_entropy(&self) -> DeviceEncodedPacket<R> {
        let outputs = self.launch_device();
        let packed = self.encode_device(&outputs);
        self.entropy_device(&packed)
    }

    /// Encode fused outputs with fixed-width bitpacking on-device.
    pub fn encode_device(&self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        encode_device_bitpacked(outputs)
    }

    /// Apply the reversible delta-xor entropy transform to a packed packet.
    pub fn entropy_device(&self, packet: &DeviceEncodedPacket<R>) -> DeviceEncodedPacket<R> {
        encode_device_entropy(packet)
    }

    /// Encode with Huffman on-device.
    ///
    /// # Panics
    ///
    /// Panics when the `experimental-huffman` feature is not enabled.
    pub fn huffman_device(&self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        #[cfg(not(feature = "experimental-huffman"))]
        panic!("huffman is experimental; enable feature \"experimental-huffman\"");
        #[cfg(feature = "experimental-huffman")]
        encode_device_huffman(outputs)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn build_huffman_codebook(
        &self,
        outputs: &DeviceFusedOutputs<R>,
    ) -> DeviceHuffmanCodebook<R> {
        build_device_huffman_codebook(outputs)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_with_codebook(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        codebook: &DeviceHuffmanCodebook<R>,
    ) -> DeviceEncodedPacket<R> {
        encode_device_huffman_with_codebook(outputs, codebook)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_policy(&self, rebuild_every_tokens: usize) -> HuffmanCodebookReusePolicy<R> {
        HuffmanCodebookReusePolicy::new(rebuild_every_tokens)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_policy_auto(&self) -> AutoHuffmanCodebookPolicy<R> {
        AutoHuffmanCodebookPolicy::new()
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_with_policy(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        policy: &mut HuffmanCodebookReusePolicy<R>,
    ) -> DeviceEncodedPacket<R> {
        policy.encode(outputs)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_auto(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        policy: &mut AutoHuffmanCodebookPolicy<R>,
    ) -> DeviceEncodedPacket<R> {
        policy.encode(outputs)
    }

    /// Decode a packet into device-resident indices.
    ///
    /// # Panics
    ///
    /// Panics on integrity/invariant failures (fail-closed decode).
    pub fn decode_device(&self, packet: &DeviceEncodedPacket<R>) -> cubecl::server::Handle {
        decode_device_indices(packet)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_with_codebook(
        &self,
        packet: &DeviceEncodedPacket<R>,
        codebook: &DeviceHuffmanCodebook<R>,
    ) -> cubecl::server::Handle {
        decode_device_indices_with_codebook(packet, Some(codebook))
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_with_policy(
        &self,
        packet: &DeviceEncodedPacket<R>,
        policy: &mut HuffmanCodebookReusePolicy<R>,
    ) -> cubecl::server::Handle {
        policy.decode(packet)
    }

    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_auto(
        &self,
        packet: &DeviceEncodedPacket<R>,
        policy: &mut AutoHuffmanCodebookPolicy<R>,
    ) -> cubecl::server::Handle {
        policy.decode(packet)
    }

    /// Decode alias for ergonomic parity.
    pub fn decode_indices(&self, packet: &DeviceEncodedPacket<R>) -> cubecl::server::Handle {
        self.decode_device(packet)
    }
}

/// Extension trait to start fluent TurboQuant API from a tensor.
pub trait TurboQuantTensorFluentExt<B: TurboQuantBackendExt> {
    /// Begin fluent TurboQuant chain.
    fn turboquant(self) -> TurboQuantFluent<B>;
}

impl<B: TurboQuantBackendExt> TurboQuantTensorFluentExt<B> for Tensor<B, 1> {
    fn turboquant(self) -> TurboQuantFluent<B> {
        TurboQuantFluent::new(self)
    }
}

/// Extension trait for CubeBackend tensors to access full device-kernel API surface.
pub trait TurboQuantCubeTensorFluentExt<R, I, BT>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    /// Begin the CubeBackend fluent API chain.
    fn turboquant_cube(self) -> TurboQuantCubeFluent<R, I, BT>;
}

impl<R, I, BT> TurboQuantCubeTensorFluentExt<R, I, BT>
    for Tensor<burn_cubecl::CubeBackend<R, f32, I, BT>, 1>
where
    R: burn_cubecl::CubeRuntime,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    fn turboquant_cube(self) -> TurboQuantCubeFluent<R, I, BT> {
        TurboQuantCubeFluent::new(self)
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
                vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
                [8],
            ),
            &device,
        )
    }

    #[test]
    fn test_burn_fluent_basic_paths_wgpu_msl() {
        let tensor = sample_tensor();
        let _ = tensor.clone().turboquant().bit_width(4).seed(3).mse();
        let _ = tensor.turboquant().bit_width(4).seed(3).prod();
    }

    #[test]
    fn test_burn_cube_fluent_device_paths_wgpu_msl() {
        let tensor = sample_tensor();
        let cube = tensor
            .clone()
            .turboquant_cube()
            .bit_width(4)
            .seed(5)
            .emit_qjl(true)
            .emit_entropy(true)
            .single_kernel_packet(false)
            .cube_dim(2);
        let outputs = cube.launch_device();
        let _ = tensor.clone().turboquant_cube().bit_width(4).seed(5).mse();
        let _ = tensor.clone().turboquant_cube().bit_width(4).seed(5).prod();
        let _ = cube.prepare_assets();
        let _ = cube.launch_device_pipeline();
        let _ = cube.launch_device_bitpacked();
        let packet = cube.launch_device_entropy();
        let _ = cube.decode_device(&packet);
        let _ = cube.decode_indices(&packet);
        let _ = cube.encode_device(&outputs);
        let _ = cube.entropy_device(&packet);
    }

    #[cfg(feature = "experimental-huffman")]
    #[test]
    fn test_burn_cube_fluent_huffman_paths_wgpu_msl() {
        let tensor = sample_tensor();
        let cube = tensor
            .clone()
            .turboquant_cube()
            .bit_width(4)
            .seed(7)
            .emit_qjl(true);
        let outputs = cube.launch_device();

        let _ = cube.huffman_device(&outputs);
        let codebook = cube.build_huffman_codebook(&outputs);
        let packet = cube.huffman_device_with_codebook(&outputs, &codebook);
        let _ = cube.decode_device_with_codebook(&packet, &codebook);

        let mut policy = cube.huffman_policy(4);
        let packet_policy = cube.huffman_device_with_policy(&outputs, &mut policy);
        let _ = cube.decode_device_with_policy(&packet_policy, &mut policy);

        let mut auto = cube.huffman_policy_auto();
        let packet_auto = cube.huffman_device_auto(&outputs, &mut auto);
        let _ = cube.decode_device_auto(&packet_auto, &mut auto);
    }
}
