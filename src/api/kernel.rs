use cubecl::prelude::Runtime;

use crate::kernels::{
    decode_device_indices,
    encode_device_bitpacked, encode_device_entropy,
    launch_turboquant_fused_device_from_handle, launch_turboquant_fused_device_with_launch,
    launch_turboquant_pipeline_device_with_options, launch_turboquant_pipeline_device_from_handle_with_options,
    prepare_turboquant_launch_assets_with_options, read_fused_outputs, validate_fused_outputs_on_device,
    DeviceEncodedPacket, DeviceFusedOutputs, DeviceLaunchAssets, TurboQuantKernelOptions,
    TurboQuantLaunchOverride,
};
#[cfg(feature = "experimental-huffman")]
use crate::kernels::{
    AutoHuffmanCodebookPolicy, DeviceHuffmanCodebook, HuffmanCodebookReusePolicy,
    build_device_huffman_codebook, decode_device_indices_with_codebook, encode_device_huffman,
    encode_device_huffman_with_codebook,
};

/// Fluent kernel launch builder for strict-equivalence fused execution.
#[derive(Debug)]
pub struct TurboQuantKernelFluent<'a, R: Runtime> {
    device: &'a R::Device,
    input: &'a [f32],
    bit_width: u8,
    seed: u64,
    emit_qjl: bool,
    emit_entropy: bool,
    single_kernel_packet: bool,
    launch_override: Option<TurboQuantLaunchOverride>,
}

impl<'a, R: Runtime> TurboQuantKernelFluent<'a, R> {
    /// Construct with defaults: `bit_width=4`, `seed=0`, `emit_qjl=true`.
    pub fn new(device: &'a R::Device, input: &'a [f32]) -> Self {
        Self {
            device,
            input,
            bit_width: 4,
            seed: 0,
            emit_qjl: true,
            emit_entropy: true,
            single_kernel_packet: true,
            launch_override: None,
        }
    }

    /// Set total TurboQuant bit-width.
    pub fn bit_width(mut self, bit_width: u8) -> Self {
        self.bit_width = bit_width;
        self
    }

    /// Set deterministic seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable/disable QJL sign emission.
    pub fn emit_qjl(mut self, emit_qjl: bool) -> Self {
        self.emit_qjl = emit_qjl;
        self
    }

    /// Enable/disable on-device entropy transform for encoded pipeline payloads.
    pub fn emit_entropy(mut self, emit_entropy: bool) -> Self {
        self.emit_entropy = emit_entropy;
        self
    }

    /// Select between single-kernel packet fusion and staged packet kernels.
    pub fn single_kernel_packet(mut self, single_kernel_packet: bool) -> Self {
        self.single_kernel_packet = single_kernel_packet;
        self
    }

    /// Override cube dimension for launch tuning.
    pub fn cube_dim(mut self, cube_dim: u32) -> Self {
        self.launch_override = Some(TurboQuantLaunchOverride { cube_dim });
        self
    }

    /// Launch fused kernel and keep outputs on device.
    pub fn launch_device(&self) -> DeviceFusedOutputs<R> {
        launch_turboquant_fused_device_with_launch::<R>(
            self.device,
            self.input,
            self.bit_width,
            self.seed,
            self.emit_qjl,
            self.launch_override,
        )
    }

    /// Launch fused kernel (device-first default).
    pub fn launch(&self) -> DeviceFusedOutputs<R> {
        self.launch_device()
    }

    /// Explicit host export helper for device outputs.
    pub fn export_to_host(&self, outputs: &DeviceFusedOutputs<R>) -> (Vec<f32>, Vec<f32>) {
        read_fused_outputs(outputs)
    }

    /// Launch full device-resident pipeline and return encoded packet on device.
    pub fn launch_device_encoded(&self, entropy: bool) -> DeviceEncodedPacket<R> {
        let (_state, packet) = launch_turboquant_pipeline_device_with_options::<R>(
            self.device,
            self.input,
            self.bit_width,
            self.seed,
            self.launch_override,
            TurboQuantKernelOptions {
                emit_qjl: self.emit_qjl,
                emit_entropy: entropy,
                single_kernel_packet: self.single_kernel_packet,
            },
        );
        packet
    }

    /// Launch full device-resident pipeline using fluent entropy/CubeK options.
    pub fn launch_device_pipeline(&self) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
        launch_turboquant_pipeline_device_with_options::<R>(
            self.device,
            self.input,
            self.bit_width,
            self.seed,
            self.launch_override,
            TurboQuantKernelOptions {
                emit_qjl: self.emit_qjl,
                emit_entropy: self.emit_entropy,
                single_kernel_packet: self.single_kernel_packet,
            },
        )
    }

    /// Pre-build reusable device launch assets using current fluent options.
    pub fn prepare_assets(&self) -> DeviceLaunchAssets<R> {
        prepare_turboquant_launch_assets_with_options::<R>(
            self.device,
            self.input.len(),
            self.bit_width,
            self.seed,
            self.launch_override,
            TurboQuantKernelOptions {
                emit_qjl: self.emit_qjl,
                emit_entropy: self.emit_entropy,
                single_kernel_packet: self.single_kernel_packet,
            },
        )
    }

    /// Launch fused kernel from an existing device input handle.
    pub fn launch_from_handle(
        &self,
        assets: &DeviceLaunchAssets<R>,
        input_handle: &cubecl::server::Handle,
    ) -> DeviceFusedOutputs<R> {
        launch_turboquant_fused_device_from_handle(assets, input_handle, self.emit_qjl)
    }

    /// Launch pipeline from an existing device input handle with fluent options.
    pub fn launch_pipeline_from_handle(
        &self,
        assets: &DeviceLaunchAssets<R>,
        input_handle: &cubecl::server::Handle,
    ) -> (DeviceFusedOutputs<R>, DeviceEncodedPacket<R>) {
        launch_turboquant_pipeline_device_from_handle_with_options(
            assets,
            input_handle,
            TurboQuantKernelOptions {
                emit_qjl: self.emit_qjl,
                emit_entropy: self.emit_entropy,
                single_kernel_packet: self.single_kernel_packet,
            },
        )
    }

    /// Encode already-launched fused state on-device with fixed-width bitpacking.
    pub fn encode_device(&self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        encode_device_bitpacked(outputs)
    }

    /// Apply on-device entropy transform to a device packet.
    pub fn entropy_device(&self, packet: &DeviceEncodedPacket<R>) -> DeviceEncodedPacket<R> {
        encode_device_entropy(packet)
    }

    /// Apply on-device Huffman encoding to already-launched fused outputs.
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device(&self, outputs: &DeviceFusedOutputs<R>) -> DeviceEncodedPacket<R> {
        encode_device_huffman(outputs)
    }

    /// Build a reusable Huffman codebook for this output shape.
    #[cfg(feature = "experimental-huffman")]
    pub fn build_huffman_codebook(&self, outputs: &DeviceFusedOutputs<R>) -> DeviceHuffmanCodebook<R> {
        build_device_huffman_codebook(outputs)
    }

    /// Encode with a reusable Huffman codebook (packet-only payload).
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_with_codebook(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        codebook: &DeviceHuffmanCodebook<R>,
    ) -> DeviceEncodedPacket<R> {
        encode_device_huffman_with_codebook(outputs, codebook)
    }

    /// Create a reusable Huffman codebook policy (rebuild every N encodes).
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_policy(&self, rebuild_every_tokens: usize) -> HuffmanCodebookReusePolicy<R> {
        HuffmanCodebookReusePolicy::new(rebuild_every_tokens)
    }

    /// Create a fully automatic Huffman policy (no manual cadence tuning).
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_policy_auto(&self) -> AutoHuffmanCodebookPolicy<R> {
        AutoHuffmanCodebookPolicy::new()
    }

    /// Encode with an automatic shared-codebook policy.
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_with_policy(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        policy: &mut HuffmanCodebookReusePolicy<R>,
    ) -> DeviceEncodedPacket<R> {
        policy.encode(outputs)
    }

    /// Encode with a fully automatic shared-codebook policy.
    #[cfg(feature = "experimental-huffman")]
    pub fn huffman_device_auto(
        &self,
        outputs: &DeviceFusedOutputs<R>,
        policy: &mut AutoHuffmanCodebookPolicy<R>,
    ) -> DeviceEncodedPacket<R> {
        policy.encode(outputs)
    }

    /// Decode device packet back into device-resident indices handle.
    pub fn decode_device(&self, packet: &DeviceEncodedPacket<R>) -> cubecl::server::Handle {
        decode_device_indices(packet)
    }

    /// Decode Huffman payload with an external reusable codebook.
    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_with_codebook(
        &self,
        packet: &DeviceEncodedPacket<R>,
        codebook: &DeviceHuffmanCodebook<R>,
    ) -> cubecl::server::Handle {
        decode_device_indices_with_codebook(packet, Some(codebook))
    }

    /// Decode with an automatic shared-codebook policy.
    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_with_policy(
        &self,
        packet: &DeviceEncodedPacket<R>,
        policy: &mut HuffmanCodebookReusePolicy<R>,
    ) -> cubecl::server::Handle {
        policy.decode(packet)
    }

    /// Decode with a fully automatic shared-codebook policy.
    #[cfg(feature = "experimental-huffman")]
    pub fn decode_device_auto(
        &self,
        packet: &DeviceEncodedPacket<R>,
        policy: &mut AutoHuffmanCodebookPolicy<R>,
    ) -> cubecl::server::Handle {
        policy.decode(packet)
    }

    /// Validate device outputs against strict host-stage semantics.
    pub fn validate_on_device(&self, outputs: &DeviceFusedOutputs<R>, tolerance: f32) -> bool {
        validate_fused_outputs_on_device(self.input, self.bit_width, self.seed, outputs, tolerance)
    }

}

/// Start a fluent fused-kernel configuration chain.
pub fn turboquant_kernel<'a, R: Runtime>(
    device: &'a R::Device,
    input: &'a [f32],
) -> TurboQuantKernelFluent<'a, R> {
    TurboQuantKernelFluent::new(device, input)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use cubecl::CubeElement;

    #[test]
    fn test_kernel_fluent_device_paths_cpu() {
        let device = Default::default();
        let input = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let fluent = turboquant_kernel::<cubecl::cpu::CpuRuntime>(&device, &input)
            .bit_width(4)
            .seed(9)
            .emit_qjl(true)
            .emit_entropy(true)
            .single_kernel_packet(false)
            .cube_dim(4);

        let outputs = fluent.launch_device();
        let _ = fluent.launch();
        let (_mse, _qjl) = fluent.export_to_host(&outputs);
        let _bitpacked = fluent.encode_device(&outputs);
        let entropy = fluent.launch_device_encoded(true);
        let _decoded = fluent.decode_device(&entropy);
        let _ = fluent.validate_on_device(&outputs, 1e-5);

        let pipeline = fluent.launch_device_pipeline();
        let assets = fluent.prepare_assets();
        let input_handle = assets.client.create_from_slice(f32::as_bytes(&input));
        let from_handle = fluent.launch_from_handle(&assets, &input_handle);
        let (_state, packet_from_handle) = fluent.launch_pipeline_from_handle(&assets, &input_handle);
        let _decoded2 = fluent.decode_device(&packet_from_handle);
        let _ = fluent.export_to_host(&from_handle);
        let _ = pipeline;
    }

    #[cfg(feature = "experimental-huffman")]
    #[test]
    fn test_kernel_fluent_huffman_paths_cpu() {
        let device = Default::default();
        let input = vec![0.25_f32; 32];
        let fluent = turboquant_kernel::<cubecl::cpu::CpuRuntime>(&device, &input)
            .bit_width(4)
            .seed(11)
            .emit_qjl(true);
        let outputs = fluent.launch_device();

        let _packet = fluent.huffman_device(&outputs);
        let codebook = fluent.build_huffman_codebook(&outputs);
        let packet_cb = fluent.huffman_device_with_codebook(&outputs, &codebook);
        let _decoded_cb = fluent.decode_device_with_codebook(&packet_cb, &codebook);

        let mut policy = fluent.huffman_policy(4);
        let packet_policy = fluent.huffman_device_with_policy(&outputs, &mut policy);
        let _decoded_policy = fluent.decode_device_with_policy(&packet_policy, &mut policy);

        let mut auto = fluent.huffman_policy_auto();
        let packet_auto = fluent.huffman_device_auto(&outputs, &mut auto);
        let _decoded_auto = fluent.decode_device_auto(&packet_auto, &mut auto);
    }
}
