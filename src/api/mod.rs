//! Public API layer for fluent, chainable interfaces.
//!
//! Core algorithm/reference implementations remain in `crate::kernels` and
//! `crate::burn_ext`. This module provides ergonomic builders on top.

pub mod kernel;

#[cfg(feature = "burn-ext")]
pub mod burn_ext;
