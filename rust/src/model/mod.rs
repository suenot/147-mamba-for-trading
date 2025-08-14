//! Mamba model implementation for trading.

pub mod mamba;
pub mod trading;

use serde::{Deserialize, Serialize};

/// Configuration for Mamba model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    /// Number of input features
    pub n_features: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of Mamba layers
    pub n_layers: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Expansion factor for inner dimension
    pub expand: usize,
    /// Number of output classes
    pub n_classes: usize,
    /// Dropout rate (for training)
    pub dropout: f64,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            n_features: 14,
            d_model: 64,
            n_layers: 4,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_classes: 3,
            dropout: 0.1,
        }
    }
}

impl MambaConfig {
    /// Create a small model configuration.
    pub fn small(n_features: usize) -> Self {
        Self {
            n_features,
            d_model: 32,
            n_layers: 2,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            n_classes: 3,
            dropout: 0.1,
        }
    }

    /// Create a medium model configuration.
    pub fn medium(n_features: usize) -> Self {
        Self {
            n_features,
            d_model: 64,
            n_layers: 4,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_classes: 3,
            dropout: 0.1,
        }
    }

    /// Create a large model configuration.
    pub fn large(n_features: usize) -> Self {
        Self {
            n_features,
            d_model: 128,
            n_layers: 6,
            d_state: 32,
            d_conv: 4,
            expand: 2,
            n_classes: 3,
            dropout: 0.1,
        }
    }

    /// Calculate the inner dimension.
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }
}
