//! # Mamba Trading
//!
//! A Rust implementation of the Mamba architecture for trading applications.
//!
//! ## Features
//!
//! - Efficient Mamba (Selective State Space) model implementation
//! - Data loading from Bybit cryptocurrency exchange
//! - Feature engineering for trading
//! - High-performance inference
//!
//! ## Example
//!
//! ```rust,no_run
//! use mamba_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load data from Bybit
//!     let client = BybitClient::new();
//!     let bars = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     // Compute features
//!     let fe = FeatureEngineer::new(true);
//!     let features = fe.compute_features(&bars);
//!
//!     // Create model
//!     let config = MambaConfig::default();
//!     let model = MambaTrading::new(config);
//!
//!     // Get prediction from last 60 bars
//!     let seq: Vec<Vec<f64>> = features[features.len()-60..].to_vec();
//!     let prediction = model.predict(&seq);
//!     println!("Signal: {:?}", prediction.signal);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::bybit::BybitClient;
    pub use crate::data::{features::FeatureEngineer, loader::DataLoader, OhlcvBar};
    pub use crate::model::{mamba::MambaBlock, trading::MambaTrading, MambaConfig};
}

// Re-exports
pub use api::bybit::BybitClient;
pub use data::{features::FeatureEngineer, loader::DataLoader, OhlcvBar};
pub use model::{mamba::MambaBlock, trading::MambaTrading, MambaConfig};
