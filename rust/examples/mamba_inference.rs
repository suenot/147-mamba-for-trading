//! Example: Mamba Model Inference
//!
//! This example demonstrates how to use the Mamba model
//! for trading signal prediction.
//!
//! Run with:
//! ```bash
//! cargo run --example mamba_inference
//! ```

use mamba_trading::{
    data::{features::FeatureEngineer, OhlcvBar, TradingSignal},
    model::{trading::MambaTrading, MambaConfig},
};
use ndarray::Array3;

fn main() {
    println!("Mamba Trading - Inference Example");
    println!("==================================\n");

    // Create synthetic market data
    println!("Generating synthetic market data...");
    let bars = generate_synthetic_data(200);
    println!("Generated {} bars\n", bars.len());

    // Compute features
    println!("Computing features...");
    let fe = FeatureEngineer::new(true);
    let features = fe.compute_features(&bars);
    let feature_names = fe.feature_names();
    println!("Computed {} features: {:?}\n", feature_names.len(), feature_names);

    // Create model
    println!("Creating Mamba model...");
    let config = MambaConfig::small(feature_names.len());
    println!("  d_model: {}", config.d_model);
    println!("  n_layers: {}", config.n_layers);
    println!("  d_state: {}", config.d_state);
    println!("  d_inner: {}\n", config.d_inner());

    let model = MambaTrading::new(config);

    // Prepare input sequence
    let lookback = 60;
    let start_idx = features.len() - lookback;
    let sequence: Vec<Vec<f64>> = features[start_idx..].to_vec();

    // Convert to tensor
    let mut x = Array3::<f64>::zeros((1, lookback, feature_names.len()));
    for (t, feat) in sequence.iter().enumerate() {
        for (i, &v) in feat.iter().enumerate() {
            x[[0, t, i]] = if v.is_nan() { 0.0 } else { v };
        }
    }

    // Get prediction
    println!("Running inference...");
    let predictions = model.generate_signals(&x, 0.5, 0.5);

    if let Some(pred) = predictions.first() {
        println!("\nPrediction Results:");
        println!("{:-<50}", "");
        println!(
            "  Signal:     {:?}",
            match pred.signal {
                TradingSignal::Buy => "BUY",
                TradingSignal::Hold => "HOLD",
                TradingSignal::Sell => "SELL",
            }
        );
        println!("  Confidence: {:.2}%", pred.confidence * 100.0);
        println!("\n  Probabilities:");
        println!("    Sell:  {:.2}%", pred.probabilities[0] * 100.0);
        println!("    Hold:  {:.2}%", pred.probabilities[1] * 100.0);
        println!("    Buy:   {:.2}%", pred.probabilities[2] * 100.0);
    }

    // Run batch inference
    println!("\n\nRunning batch inference on last 10 time steps...");
    println!("{:-<70}", "");
    println!(
        "{:<5} {:>10} {:>10} {:>12} {:>12} {:>12}",
        "Step", "Price", "Signal", "Sell%", "Hold%", "Buy%"
    );
    println!("{:-<70}", "");

    for i in (features.len() - 10)..features.len() {
        let start = i.saturating_sub(lookback - 1);
        let end = i + 1;

        if end - start < lookback {
            continue;
        }

        let seq: Vec<Vec<f64>> = features[start..end].to_vec();
        let prediction = model.predict(&seq);

        let signal_str = match prediction.signal {
            TradingSignal::Buy => "BUY",
            TradingSignal::Hold => "HOLD",
            TradingSignal::Sell => "SELL",
        };

        println!(
            "{:<5} {:>10.2} {:>10} {:>11.1}% {:>11.1}% {:>11.1}%",
            i,
            bars[i].close,
            signal_str,
            prediction.probabilities[0] * 100.0,
            prediction.probabilities[1] * 100.0,
            prediction.probabilities[2] * 100.0
        );
    }

    // Save model
    println!("\n\nSaving model...");
    let save_path = std::path::Path::new("mamba_model.json");
    if let Err(e) = model.save(save_path) {
        println!("Warning: Could not save model: {}", e);
    } else {
        println!("Model saved to {:?}", save_path);
    }

    println!("\nDone!");
}

/// Generate synthetic market data for demonstration.
fn generate_synthetic_data(n_bars: usize) -> Vec<OhlcvBar> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut bars = Vec::with_capacity(n_bars);
    let mut price: f64 = 100.0;
    let mut timestamp = 1704067200000_i64; // 2024-01-01 00:00:00 UTC

    for _ in 0..n_bars {
        // Random walk with slight upward drift
        let change: f64 = rng.gen_range(-2.0..2.5);
        price = (price + change).max(50.0);

        let volatility: f64 = rng.gen_range(0.5..3.0);
        let open: f64 = price - volatility * rng.gen_range(-0.5_f64..0.5);
        let close: f64 = price + volatility * rng.gen_range(-0.5_f64..0.5);
        let high = open.max(close) + volatility * rng.gen_range(0.0_f64..1.0);
        let low = open.min(close) - volatility * rng.gen_range(0.0_f64..1.0);
        let volume: f64 = rng.gen_range(1000.0..10000.0);

        bars.push(OhlcvBar {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });

        timestamp += 3600000; // 1 hour
        price = close;
    }

    bars
}
