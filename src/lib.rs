//! TurboQuant crate entry points.
//!
//! The crate exposes:
//! - `kernels`: quantization/dequantization APIs and runtime launch helpers.
//! - `bench`: lightweight benchmark helpers.
//! - `api`: fluent public API builders.

pub mod api;
pub mod bench;
#[cfg(feature = "burn-ext")]
pub mod burn_ext;
pub mod kernels;
