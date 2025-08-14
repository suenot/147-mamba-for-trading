//! Trading-specific Mamba model implementation.

use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use super::mamba::MambaLayer;
use super::MambaConfig;
use crate::data::{Prediction, TradingSignal};

/// Complete Mamba model for trading signal prediction.
///
/// Architecture:
/// 1. Input projection (n_features -> d_model)
/// 2. Stack of Mamba layers
/// 3. Output normalization
/// 4. Classification head (d_model -> n_classes)
#[derive(Clone, Serialize, Deserialize)]
pub struct MambaTrading {
    /// Model configuration
    pub config: MambaConfig,

    /// Input projection weights
    input_proj_weight: Array2<f64>,
    input_proj_bias: Array1<f64>,

    /// Mamba layers
    layers: Vec<MambaLayer>,

    /// Output normalization
    output_norm_weight: Array1<f64>,
    output_norm_bias: Array1<f64>,

    /// Classification head
    head_weight1: Array2<f64>,
    head_bias1: Array1<f64>,
    head_weight2: Array2<f64>,
    head_bias2: Array1<f64>,

    eps: f64,
}

impl MambaTrading {
    /// Create a new trading model with the given configuration.
    pub fn new(config: MambaConfig) -> Self {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();

        // Input projection
        let in_scale = (2.0 / (config.n_features + config.d_model) as f64).sqrt();
        let in_dist = Normal::new(0.0, in_scale).unwrap();
        let input_proj_weight =
            Array2::from_shape_fn((config.n_features, config.d_model), |_| in_dist.sample(&mut rng));
        let input_proj_bias = Array1::zeros(config.d_model);

        // Mamba layers
        let layers: Vec<MambaLayer> = (0..config.n_layers)
            .map(|_| {
                MambaLayer::new(config.d_model, config.d_state, config.d_conv, config.expand)
            })
            .collect();

        // Output normalization
        let output_norm_weight = Array1::ones(config.d_model);
        let output_norm_bias = Array1::zeros(config.d_model);

        // Classification head (two-layer MLP)
        let hidden_dim = config.d_model / 2;
        let head_scale1 = (2.0 / (config.d_model + hidden_dim) as f64).sqrt();
        let head_scale2 = (2.0 / (hidden_dim + config.n_classes) as f64).sqrt();

        let head_dist1 = Normal::new(0.0, head_scale1).unwrap();
        let head_dist2 = Normal::new(0.0, head_scale2).unwrap();

        let head_weight1 =
            Array2::from_shape_fn((config.d_model, hidden_dim), |_| head_dist1.sample(&mut rng));
        let head_bias1 = Array1::zeros(hidden_dim);
        let head_weight2 =
            Array2::from_shape_fn((hidden_dim, config.n_classes), |_| head_dist2.sample(&mut rng));
        let head_bias2 = Array1::zeros(config.n_classes);

        Self {
            config,
            input_proj_weight,
            input_proj_bias,
            layers,
            output_norm_weight,
            output_norm_bias,
            head_weight1,
            head_bias1,
            head_weight2,
            head_bias2,
            eps: 1e-5,
        }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features of shape (batch, seq_len, n_features)
    /// * `return_all_steps` - If true, return predictions for all timesteps
    ///
    /// # Returns
    ///
    /// Logits of shape (batch, n_classes) or (batch, seq_len, n_classes)
    pub fn forward(&self, x: &Array3<f64>, return_all_steps: bool) -> Array2<f64> {
        let (batch, seq_len, _) = x.dim();

        // Input projection
        let mut h = self.linear_3d(x, &self.input_proj_weight, Some(&self.input_proj_bias));

        // Mamba layers
        for layer in &self.layers {
            h = layer.forward(&h);
        }

        // Output normalization
        h = self.layer_norm_3d(&h);

        // Get last timestep or all timesteps
        let features = if return_all_steps {
            h.clone()
        } else {
            // Take only the last timestep
            h.slice(ndarray::s![.., seq_len - 1, ..])
                .to_owned()
                .insert_axis(Axis(1))
        };

        // Classification head
        let hidden = self.gelu_3d(&self.linear_3d(
            &features,
            &self.head_weight1,
            Some(&self.head_bias1),
        ));
        let logits = self.linear_3d(&hidden, &self.head_weight2, Some(&self.head_bias2));

        // Reshape output
        if return_all_steps {
            let (b, s, c) = logits.dim();
            logits
                .into_shape((b * s, c))
                .expect("Reshape failed")
                .to_owned()
        } else {
            logits.slice(ndarray::s![.., 0, ..]).to_owned()
        }
    }

    /// Get prediction probabilities using softmax.
    pub fn predict_proba(&self, x: &Array3<f64>) -> Array2<f64> {
        let logits = self.forward(x, false);
        self.softmax(&logits)
    }

    /// Generate trading signals from model predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features
    /// * `buy_threshold` - Minimum probability for buy signal
    /// * `sell_threshold` - Minimum probability for sell signal
    ///
    /// # Returns
    ///
    /// Vector of predictions with signals and confidence
    pub fn generate_signals(
        &self,
        x: &Array3<f64>,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Vec<Prediction> {
        let probs = self.predict_proba(x);
        let (batch, _) = probs.dim();

        let mut predictions = Vec::with_capacity(batch);

        for b in 0..batch {
            let sell_prob = probs[[b, 0]];
            let hold_prob = probs[[b, 1]];
            let buy_prob = probs[[b, 2]];

            let (signal, confidence) = if buy_prob > buy_threshold && buy_prob > sell_prob {
                (TradingSignal::Buy, buy_prob)
            } else if sell_prob > sell_threshold && sell_prob > buy_prob {
                (TradingSignal::Sell, sell_prob)
            } else {
                (TradingSignal::Hold, hold_prob)
            };

            predictions.push(Prediction {
                signal,
                confidence,
                probabilities: [sell_prob, hold_prob, buy_prob],
            });
        }

        predictions
    }

    /// Predict on a single sequence.
    pub fn predict(&self, features: &[Vec<f64>]) -> Prediction {
        // Convert to 3D array with batch size 1
        let seq_len = features.len();
        let n_features = features.first().map(|f| f.len()).unwrap_or(0);

        let mut x = Array3::<f64>::zeros((1, seq_len, n_features));
        for (t, feat) in features.iter().enumerate() {
            for (i, &v) in feat.iter().enumerate() {
                x[[0, t, i]] = v;
            }
        }

        let predictions = self.generate_signals(&x, 0.5, 0.5);
        predictions.into_iter().next().unwrap_or(Prediction {
            signal: TradingSignal::Hold,
            confidence: 0.33,
            probabilities: [0.33, 0.34, 0.33],
        })
    }

    /// Load model weights from JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model: Self = serde_json::from_reader(reader)?;
        Ok(model)
    }

    /// Save model weights to JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    // Helper methods

    fn linear_3d(
        &self,
        x: &Array3<f64>,
        weight: &Array2<f64>,
        bias: Option<&Array1<f64>>,
    ) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();
        let out_dim = weight.dim().1;
        let mut output = Array3::<f64>::zeros((batch, seq_len, out_dim));

        for b in 0..batch {
            for t in 0..seq_len {
                let x_t = x.slice(ndarray::s![b, t, ..]);
                let mut out_t = x_t.dot(weight);
                if let Some(b) = bias {
                    out_t = &out_t + b;
                }
                output.slice_mut(ndarray::s![b, t, ..]).assign(&out_t);
            }
        }

        output
    }

    fn layer_norm_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let mut output = Array3::<f64>::zeros((batch, seq_len, d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                let x_t = x.slice(ndarray::s![b, t, ..]);
                let mean = x_t.mean().unwrap_or(0.0);
                let var = x_t.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
                let std = (var + self.eps).sqrt();

                for i in 0..d_model {
                    output[[b, t, i]] = (x[[b, t, i]] - mean) / std * self.output_norm_weight[i]
                        + self.output_norm_bias[i];
                }
            }
        }

        output
    }

    fn gelu_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        x.mapv(|v| 0.5 * v * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh()))
    }

    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch, classes) = x.dim();
        let mut output = Array2::<f64>::zeros((batch, classes));

        for b in 0..batch {
            let row = x.slice(ndarray::s![b, ..]);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();

            for (c, &exp_v) in exp_vals.iter().enumerate() {
                output[[b, c]] = exp_v / sum;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_trading_creation() {
        let config = MambaConfig::small(14);
        let model = MambaTrading::new(config);
        assert_eq!(model.config.n_features, 14);
        assert_eq!(model.config.d_model, 32);
    }

    #[test]
    fn test_mamba_trading_forward() {
        let config = MambaConfig::small(10);
        let model = MambaTrading::new(config);

        let x = Array3::<f64>::zeros((2, 20, 10)); // batch=2, seq_len=20, features=10
        let output = model.forward(&x, false);

        assert_eq!(output.dim(), (2, 3)); // batch=2, classes=3
    }

    #[test]
    fn test_generate_signals() {
        let config = MambaConfig::small(10);
        let model = MambaTrading::new(config);

        let x = Array3::<f64>::ones((1, 10, 10));
        let signals = model.generate_signals(&x, 0.5, 0.5);

        assert_eq!(signals.len(), 1);
        assert!(signals[0].confidence >= 0.0 && signals[0].confidence <= 1.0);
    }

    #[test]
    fn test_predict_single() {
        let config = MambaConfig::small(5);
        let model = MambaTrading::new(config);

        let features: Vec<Vec<f64>> = (0..10).map(|_| vec![0.1, 0.2, 0.3, 0.4, 0.5]).collect();
        let prediction = model.predict(&features);

        assert!(prediction.probabilities.iter().sum::<f64>() > 0.99);
    }
}
